"""Step 1 (OpenMC env): small MSRE-like graphite-moderated molten salt core.

Eigenvalue run of a bare core (fuel salt channels in graphite, Hastelloy-N
vessel). Every neutron and photon leaving the vessel's side wall is written to
a surface source. Also tallies kappa-fission so results can be scaled to power.

Geometry (cm): lattice of 5.08 cm graphite cells with r = 1.35 cm salt channels
inside r < 70, |z| < 80; vessel 70 < r < 72.5 plus 2.5 cm end plates; void outside.
"""

from pathlib import Path

import openmc

PARTICLES = 20_000
BATCHES = 60
INACTIVE = 15
HERE = Path(__file__).parent
RUN = HERE / "results" / "core"
RUN.mkdir(parents=True, exist_ok=True)

# MSRE fuel salt LiF-BeF2-ZrF4-UF4 (65-29.1-5-0.9 mol%), 33 wt% U-235, 99.995% Li-7
salt = openmc.Material(name="fuel salt")
salt.add_nuclide("Li7", 0.65 * 0.99995)
salt.add_nuclide("Li6", 0.65 * 0.00005)
salt.add_element("Be", 0.291)
salt.add_element("Zr", 0.05)
salt.add_nuclide("U235", 0.009 * 0.33282)  # 33 wt% as atom fraction
salt.add_nuclide("U238", 0.009 * 0.66718)
salt.add_element("F", 0.65 + 2 * 0.291 + 4 * 0.05 + 4 * 0.009)
salt.set_density("g/cm3", 2.27)

graphite = openmc.Material(name="graphite")
graphite.add_element("C", 1.0)
graphite.set_density("g/cm3", 1.86)
graphite.add_s_alpha_beta("c_Graphite")

hastelloy = openmc.Material(name="Hastelloy-N")
for element, fraction in {"Ni": 0.71, "Mo": 0.16, "Cr": 0.07, "Fe": 0.05, "Si": 0.01}.items():
    hastelloy.add_element(element, fraction, "wo")
hastelloy.set_density("g/cm3", 8.86)

pitch, n_cells = 5.08, 28
channel = openmc.ZCylinder(r=1.35)
cell_u = openmc.Universe(cells=[
    openmc.Cell(fill=salt, region=-channel),
    openmc.Cell(fill=graphite, region=+channel),
])
lattice = openmc.RectLattice()
lattice.lower_left = (-pitch * n_cells / 2,) * 2
lattice.pitch = (pitch, pitch)
lattice.universes = [[cell_u] * n_cells for _ in range(n_cells)]
lattice.outer = openmc.Universe(cells=[openmc.Cell(fill=graphite)])

core_r, core_bot, core_top = openmc.ZCylinder(r=70), openmc.ZPlane(-80), openmc.ZPlane(80)
vessel_r = openmc.ZCylinder(r=72.5, surface_id=100)
vessel_bot, vessel_top = openmc.ZPlane(-82.5), openmc.ZPlane(82.5)
boundary = openmc.Sphere(r=300, boundary_type="vacuum")

core_region = -core_r & +core_bot & -core_top
vessel_outer = -vessel_r & +vessel_bot & -vessel_top
core = openmc.Cell(name="core", fill=lattice, region=core_region)
vessel = openmc.Cell(name="vessel", fill=hastelloy, region=vessel_outer & ~core_region)
outside = openmc.Cell(name="outside", region=-boundary & ~vessel_outer)
geometry = openmc.Geometry([core, vessel, outside])

settings = openmc.Settings()
settings.particles = PARTICLES
settings.batches = BATCHES
settings.inactive = INACTIVE
settings.photon_transport = True
settings.source = openmc.IndependentSource(
    space=openmc.stats.Box((-70, -70, -80), (70, 70, 80)),
    constraints={"fissionable": True},
)
# Side wall only: anything leaving through the end plates moves away from the detector
settings.surf_source_write = {"surface_ids": [100], "cellfrom": vessel.id,
                              "max_particles": 20_000_000}

power = openmc.Tally(name="kappa-fission")
power.scores = ["kappa-fission"]

model = openmc.Model(geometry, openmc.Materials([salt, graphite, hastelloy]),
                     settings, openmc.Tallies([power]))
statepoint = model.run(cwd=RUN, threads=8, output=False)

with openmc.StatePoint(statepoint) as sp:
    print(f"k-eff = {sp.keff.n:.5f} +/- {sp.keff.s:.5f}")
    kf = sp.get_tally(name="kappa-fission").mean.ravel()[0]
    print(f"kappa-fission per source neutron: {kf / 1e6:.2f} MeV")
print(f"active histories: {PARTICLES * (BATCHES - INACTIVE)}")
