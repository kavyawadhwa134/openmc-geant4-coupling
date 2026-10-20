"""Step 1 (OpenMC env): simplified CMFX room model with a 2"x2" EJ-301 detector.

Recreates the OpenMC half of Ball et al., "Measurements of Fusion Yield on the
Centrifugal Mirror Fusion Experiment" (arXiv:2505.23047), Sec. 2.1.2:

* G, the zero-threshold efficiency: elastic (n,p) scatters in the detector per
  emitted D-D neutron, tallied versus incoming neutron energy (the paper's
  "scattering rate spectrum");
* a surface source of every neutron entering the detector, used by step 3 for
  a direct event-by-event Geant4 coupling.

Variance reduction is two-stage surface-source splitting. Stage A runs the full
room and records neutrons entering a 60 cm sphere around the detector. Stage B
replays each of them about 100 times inside that sphere only, whose surface is a
vacuum boundary so nothing is counted twice.

Taken from the paper: 2.45 MeV D-D source; plasma 60 cm long, radii 5-21 cm;
vessel wall radius 36.1 cm (75 cm diameter); central conductor; two magnets
1.45 m apart; a concrete floor and shield wall, with the 2"x2" detector on top
of the wall 3.1 m from the machine centre.

Assumed here (the paper refers to a thesis for these): every thickness, every
material mix, and the wall and floor sizes and positions. There is no air,
ceiling or other room walls.

Coordinates (cm): machine axis along z, y vertical; see LAYOUT below for the
room. Detector centre is 3.1 m from the machine centre in both layouts.

Usage: python step1_openmc_cmfx.py [particles_per_batch]
"""

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import openmc

PARTICLES = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
BATCHES = 20
MPI_RANKS = 8
HERE = Path(__file__).parent

ss304 = openmc.Material(name="SS304")
for element, fraction in {"Fe": 0.695, "Cr": 0.19, "Ni": 0.095, "Mn": 0.02}.items():
    ss304.add_element(element, fraction, "wo")
ss304.set_density("g/cm3", 8.0)

# MRI magnet homogenised: 30% steel + 20% copper by volume, rest cryostat void
magnet = openmc.Material(name="magnet")
for element, fraction in {"Fe": 0.398, "Cr": 0.109, "Ni": 0.054, "Mn": 0.011, "Cu": 0.428}.items():
    magnet.add_element(element, fraction, "wo")
magnet.set_density("g/cm3", 4.19)

concrete = openmc.Material(name="concrete")
for element, fraction in {
    "H": 0.022100, "C": 0.002484, "O": 0.574930, "Na": 0.015208,
    "Mg": 0.001266, "Al": 0.019953, "Si": 0.304627, "K": 0.010045,
    "Ca": 0.042951, "Fe": 0.006435,
}.items():
    concrete.add_element(element, fraction, "wo")
concrete.set_density("g/cm3", 2.3)

# EJ-301 (xylene): 4.82e22 H and 3.98e22 C atoms per cm3, 0.874 g/cm3 (Eljen data sheet)
ej301 = openmc.Material(name="EJ-301")
ej301.add_element("H", 4.82, "ao")
ej301.add_element("C", 3.98, "ao")
ej301.set_density("g/cm3", 0.874)

# Machine
conductor = -openmc.ZCylinder(r=2.5) & +openmc.ZPlane(-100) & -openmc.ZPlane(100)
vessel = (+openmc.ZCylinder(r=36.1) & -openmc.ZCylinder(r=37.1)
          & +openmc.ZPlane(-100) & -openmc.ZPlane(100))
end_plates = (-openmc.ZCylinder(r=37.1) & +openmc.ZCylinder(r=2.5)
              & ((+openmc.ZPlane(-101) & -openmc.ZPlane(-100))
                 | (+openmc.ZPlane(100) & -openmc.ZPlane(101))))
magnet_annulus = +openmc.ZCylinder(r=45) & -openmc.ZCylinder(r=100)
magnets = magnet_annulus & ((+openmc.ZPlane(47.5) & -openmc.ZPlane(97.5))
                            | (+openmc.ZPlane(-97.5) & -openmc.ZPlane(-47.5)))
# Room. LAYOUT "fig5" follows the paper's Fig. 5: floor top at y = -100, a
# single 30 cm shield wall (255 < x < 285, top y = 135), detector above it.
# LAYOUT "bunker" is an open-top box of 60 cm walls with the detector on the
# inner top edge, which gives it a clear view of the plasma (sensitivity case).
LAYOUT = os.environ.get("CMFX_LAYOUT", "fig5")
RESULTS = HERE / "results" / LAYOUT


def box(x0, x1, y0, top, z0, z1):
    return (+openmc.XPlane(x0) & -openmc.XPlane(x1) & +openmc.YPlane(y0)
            & -top & +openmc.ZPlane(z0) & -openmc.ZPlane(z1))


if LAYOUT == "fig5":
    # Detector centre read off Fig. 5, ~10 cm above the wall top (the cell
    # stands on its PMT, which is not modelled)
    floor_y, wall_top_y, det_c = -100.0, 135.0, (281.0, 147.0, 0.0)
    wall_top = openmc.YPlane(wall_top_y)
    wall = box(255, 285, floor_y, wall_top, -200, 200)
    det_bot = openmc.YPlane(det_c[1] - 2.54, surface_id=902)
else:
    floor_y, wall_top_y, det_c = -150.0, 100.0, (292.54, 102.54, 0.0)
    # Wall top and detector bottom are one shared surface, so a neutron passing
    # from concrete into the detector always crosses surface 902 and is recorded
    wall_top = openmc.YPlane(wall_top_y, surface_id=902)
    wall = (box(290, 350, floor_y, wall_top, -460, 460) | box(-350, -290, floor_y, wall_top, -460, 460)
            | box(-290, 290, floor_y, wall_top, 400, 460) | box(-290, 290, floor_y, wall_top, -460, -400))
    det_bot = wall_top
floor = (+openmc.YPlane(floor_y - 50) & -openmc.YPlane(floor_y) & +openmc.XPlane(-500)
         & -openmc.XPlane(500) & +openmc.ZPlane(-500) & -openmc.ZPlane(500))
# 2" x 2" detector, axis vertical
DET_C = det_c
det_side = openmc.YCylinder(x0=det_c[0], z0=0, r=2.54, surface_id=901)
det_top = openmc.YPlane(det_c[1] + 2.54, surface_id=903)
detector_region = -det_side & +det_bot & -det_top
boundary = openmc.Sphere(r=800, boundary_type="vacuum")

# Sphere around the detector for two-stage surface-source splitting
SPLIT_R = 60.0
split_sphere = openmc.Sphere(*DET_C, r=SPLIT_R, surface_id=950)

detector = openmc.Cell(name="detector", fill=ej301, region=detector_region)
cells = [
    openmc.Cell(name="conductor", fill=ss304, region=conductor),
    openmc.Cell(name="vessel", fill=ss304, region=vessel | end_plates),
    openmc.Cell(name="magnets", fill=magnet, region=magnets),
    openmc.Cell(name="floor", fill=concrete, region=floor),
    openmc.Cell(name="wall outside sphere", fill=concrete, region=wall & +split_sphere),
    openmc.Cell(name="wall inside sphere", fill=concrete, region=wall & -split_sphere),
    detector,
]
solid = conductor | vessel | end_plates | magnets | floor | wall | detector_region
cells += [
    openmc.Cell(name="void outside sphere", region=-boundary & +split_sphere & ~solid),
    openmc.Cell(name="void inside sphere", region=-split_sphere & ~solid),
]
geometry = openmc.Geometry(cells)
materials = openmc.Materials([ss304, magnet, concrete, ej301])

source = openmc.IndependentSource(
    space=openmc.stats.CylindricalIndependent(
        r=openmc.stats.PowerLaw(5.0, 21.0, 1),  # uniform per unit area in the annulus
        phi=openmc.stats.Uniform(0, 2 * np.pi),
        z=openmc.stats.Uniform(-30.0, 30.0),
    ),
    angle=openmc.stats.Isotropic(),
    energy=openmc.stats.Discrete([2.45e6], [1.0]),
    particle="neutron",
)

# Stage A: full room, record every neutron crossing into the sphere.
# A history that leaves and re-enters is recorded each time it enters.
STAGE_A = RESULTS / "stage_a"
STAGE_A.mkdir(parents=True, exist_ok=True)
settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.particles = PARTICLES
settings.batches = BATCHES
settings.source = source
settings.surf_source_write = {"surface_ids": [950], "max_particles": 50_000_000}
openmc.Model(geometry, materials, settings).run(
    cwd=STAGE_A, output=False, mpi_args=["mpiexec", "-n", str(MPI_RANKS)])
n_source = PARTICLES * BATCHES

with h5py.File(STAGE_A / "surface_source.h5") as f:
    a = f["source_bank"][()]
pos = np.column_stack([a["r"][k] for k in "xyz"])
u = np.column_stack([a["u"][k] for k in "xyz"])
inward = np.sum(u * (pos - np.array(DET_C)), axis=1) < 0
a, pos, u = a[inward], pos[inward], u[inward]
n_rows = len(a)
pos = pos + 1e-4 * u  # start 1 um inside, off the stage-B vacuum surface
openmc.write_source_file(
    [openmc.SourceParticle(r=tuple(p), u=tuple(d), E=e, wgt=w)
     for p, d, e, w in zip(pos, u, a["E"], a["wgt"])],
    STAGE_A / "sphere_source.h5",
)

# Stage B: inside the sphere only (vacuum surface, so nothing is counted twice),
# each recorded neutron replayed ~REPLAYS times.
REPLAYS = 100
STAGE_B = RESULTS / "stage_b"
STAGE_B.mkdir(parents=True, exist_ok=True)
sphere_b = openmc.Sphere(*DET_C, r=SPLIT_R, boundary_type="vacuum")
detector_b = openmc.Cell(name="detector", fill=ej301, region=detector_region)
geometry_b = openmc.Geometry([
    openmc.Cell(name="wall", fill=concrete, region=wall & -sphere_b),
    detector_b,
    openmc.Cell(name="void", region=-sphere_b & ~wall & ~detector_region),
])
settings_b = openmc.Settings()
settings_b.run_mode = "fixed source"
settings_b.batches = 20
settings_b.particles = max(1000, REPLAYS * n_rows // settings_b.batches)
settings_b.source = openmc.FileSource(path=(STAGE_A / "sphere_source.h5").resolve())
settings_b.surf_source_write = {"surface_ids": [901, 902, 903], "cellto": detector_b.id,
                                "max_particles": 50_000_000}

energy_edges = np.concatenate([np.logspace(-5, 5, 41), np.linspace(1.5e5, 2.6e6, 50)[1:]])
scatter = openmc.Tally(name="np scatter spectrum")
scatter.filters = [openmc.CellFilter([detector_b]), openmc.EnergyFilter(energy_edges)]
scatter.nuclides = ["H1"]
scatter.scores = ["elastic"]
statepoint = openmc.Model(geometry_b, materials, settings_b, openmc.Tallies([scatter])).run(
    cwd=STAGE_B, output=False, mpi_args=["mpiexec", "-n", str(MPI_RANKS)])
n_b = settings_b.particles * settings_b.batches

# FileSource samples rows uniformly, so per-source-neutron = per-B-history * rows / N_A
norm = n_rows / n_source
with openmc.StatePoint(statepoint) as sp:
    t = sp.get_tally(name="np scatter spectrum")
    s_mean, s_std = t.mean.ravel() * norm, t.std_dev.ravel() * norm

with h5py.File(STAGE_B / "surface_source.h5") as f:
    bank = f["source_bank"][()]

np.savez(
    RESULTS / "openmc_handoff.npz",
    energy_edges_ev=energy_edges, scatter_mean=s_mean, scatter_std=s_std,
    n_source=n_source, n_sphere_rows=n_rows, det_centre_cm=DET_C, layout=LAYOUT,
    energy_ev=bank["E"], position_cm=np.column_stack([bank["r"][k] for k in "xyz"]),
    direction=np.column_stack([bank["u"][k] for k in "xyz"]),
    weight=bank["wgt"] * norm / n_b,  # per source neutron
)

G, G_std = s_mean.sum(), np.sqrt(np.sum(s_std ** 2))
fast = energy_edges[1:] > 2.0e6
print(f"layout {LAYOUT}, detector at {DET_C}, {np.hypot(DET_C[0], DET_C[1]):.0f} cm from centre")
print(f"source neutrons: {n_source:.3e}; neutrons entering the {SPLIT_R:.0f} cm sphere: "
      f"{n_rows} ({a['wgt'].sum() / n_source:.3e} per source neutron)")
print(f"stage B histories: {n_b:.3e}; detector entries recorded: {len(bank)}")
print(f"G = {G:.4e} +/- {G_std:.1e} (stage B only) (n,p) scatters per source neutron "
      f"(paper: 1.021e-4); {100 * s_mean[fast].sum() / G:.1f}% of scatters above 2 MeV")
print(f"neutrons entering detector per source neutron: {bank['wgt'].sum() * norm / n_b:.3e}")
