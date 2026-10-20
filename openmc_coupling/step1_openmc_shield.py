"""Step 1 (OpenMC env): 14.1 MeV neutrons through a concrete wall.

Writes every particle leaving the wall's outer face (x = 40 cm) to a surface
source, then converts it to ``results/handoff.npz`` so the Geant4 step does not
need h5py. Also tallies heating in the plastic scintillator as a full-OpenMC
reference for the coupled result.

Geometry (cm), all |y|, |z| <= 50:
    beam at origin along +x -> concrete 10 < x < 40 -> gap -> plastic 40.5 < x < 45.5
"""

from pathlib import Path

import h5py
import numpy as np
import openmc

N_PARTICLES = 20_000
N_BATCHES = 10
# Photon energy deposited by its Compton electrons is booked under "electron"
PARTICLES = ["neutron", "photon", "electron", "positron"]
HERE = Path(__file__).parent
RUN = HERE / "results" / "openmc"
RUN.mkdir(parents=True, exist_ok=True)

# Ordinary concrete (PNNL compendium), mass fractions, 2.3 g/cm3
concrete = openmc.Material(name="concrete")
for element, fraction in {
    "H": 0.022100, "C": 0.002484, "O": 0.574930, "Na": 0.015208,
    "Mg": 0.001266, "Al": 0.019953, "Si": 0.304627, "K": 0.010045,
    "Ca": 0.042951, "Fe": 0.006435,
}.items():
    concrete.add_element(element, fraction, "wo")
concrete.set_density("g/cm3", 2.3)

# Same composition as Geant4's G4_PLASTIC_SC_VINYLTOLUENE
plastic = openmc.Material(name="plastic")
plastic.add_element("H", 0.085, "wo")
plastic.add_element("C", 0.915, "wo")
plastic.set_density("g/cm3", 1.032)

y0, y1 = openmc.YPlane(-50), openmc.YPlane(50)
z0, z1 = openmc.ZPlane(-50), openmc.ZPlane(50)
side = +y0 & -y1 & +z0 & -z1
wall_in, wall_out = openmc.XPlane(10), openmc.XPlane(40, surface_id=40)
det_in, det_out = openmc.XPlane(40.5), openmc.XPlane(45.5)
boundary = openmc.Sphere(r=150, boundary_type="vacuum")

wall = openmc.Cell(name="wall", fill=concrete, region=+wall_in & -wall_out & side)
detector = openmc.Cell(name="detector", fill=plastic, region=+det_in & -det_out & side)
void = openmc.Cell(name="void", region=-boundary & ~wall.region & ~detector.region)
geometry = openmc.Geometry([wall, detector, void])

settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.particles = N_PARTICLES
settings.batches = N_BATCHES
settings.photon_transport = True
settings.source = openmc.IndependentSource(
    space=openmc.stats.Point((0, 0, 0)),
    angle=openmc.stats.Monodirectional((1, 0, 0)),
    energy=openmc.stats.Discrete([14.1e6], [1.0]),
    particle="neutron",
)
settings.surf_source_write = {
    "surface_ids": [40],
    "cellfrom": wall.id,
    "max_particles": 10_000_000,
}

heating = openmc.Tally(name="detector heating")
heating.filters = [openmc.CellFilter([detector]), openmc.ParticleFilter(PARTICLES)]
heating.scores = ["heating"]

model = openmc.Model(geometry, openmc.Materials([concrete, plastic]), settings,
                     openmc.Tallies([heating]))
statepoint = model.run(cwd=RUN, threads=8, output=False)

with openmc.StatePoint(statepoint) as sp:
    t = sp.get_tally(name="detector heating")
    heat_mean = t.mean.ravel()  # eV per source neutron, one per PARTICLES entry
    heat_std = t.std_dev.ravel()

with h5py.File(RUN / "surface_source.h5") as f:
    bank = f["source_bank"][()]

# OpenMC particle codes: 0 neutron, 1 photon, 2 electron, 3 positron
names = np.array(["neutron", "gamma", "e-", "e+"])
np.savez(
    HERE / "results" / "handoff.npz",
    particle=names[bank["particle"]],
    energy_ev=bank["E"],
    position_cm=np.column_stack([bank["r"][k] for k in "xyz"]),
    direction=np.column_stack([bank["u"][k] for k in "xyz"]),
    weight=bank["wgt"],
    time_s=bank["time"],
    n_source=N_PARTICLES * N_BATCHES,
    openmc_heating_ev=heat_mean,
    openmc_heating_std_ev=heat_std,
)

n_source = N_PARTICLES * N_BATCHES
print(f"source neutrons: {n_source}")
for name in ("neutron", "gamma", "e-", "e+"):
    sel = names[bank["particle"]] == name
    if sel.any():
        print(f"{name:8s} leaving wall: {sel.sum():7d} "
              f"({bank['wgt'][sel].sum() / n_source:.4f} per source n), "
              f"mean E {bank['E'][sel].mean() / 1e6:.3f} MeV")
print("OpenMC heating in plastic per source n:")
for name, mean, std in zip(PARTICLES, heat_mean, heat_std):
    print(f"  {name:8s} {mean / 1e3:8.2f} +/- {std / 1e3:.2f} keV")
print(f"  total    {heat_mean.sum() / 1e3:8.2f} keV")
