"""Step 2 (OpenMC env): turn vessel leakage into a source for an ex-core He-3 tube.

The tube (axis along z, centre at x = 90 cm, |z| < 15 cm) is tiny compared with
the vessel, so few leaking particles would ever reach it. Two exact tricks:

* The core is (nearly) symmetric about z, so each leaked particle is copied K
  times at random angles about z, each copy carrying 1/K of the weight.
* Outside the vessel is void and the vessel is convex, so particles fly in
  straight lines. Only copies whose ray hits the tube are kept.

Then runs OpenMC on exactly those particles with only the tube present, scoring
He-3(n,p), as a reference for the Geant4 detector step. Writes results/handoff.npz.
"""

from pathlib import Path

import h5py
import numpy as np
import openmc

K = 256  # azimuthal copies per leaked particle
TUBE_X, GAS_R, WALL_R, HALF_L = 90.0, 1.22, 1.27, 15.0
HE3_DENSITY = 5.02e-4  # g/cm3, 4 atm at 293 K
HERE = Path(__file__).parent
CORE = HERE / "results" / "core"


def hits_tube(pos, u):
    """True where the ray pos + t*u (t > 0) crosses the tube's outer cylinder."""
    dx, dy = pos[:, 0] - TUBE_X, pos[:, 1]
    a = u[:, 0] ** 2 + u[:, 1] ** 2
    b = 2 * (dx * u[:, 0] + dy * u[:, 1])
    c = dx ** 2 + dy ** 2 - WALL_R ** 2
    disc = b ** 2 - 4 * a * c
    ok = (disc > 0) & (a > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    a_safe = np.where(ok, a, 1.0)
    t1 = np.maximum((-b - sq) / (2 * a_safe), 0.0)
    t2 = (-b + sq) / (2 * a_safe)
    z1, z2 = pos[:, 2] + t1 * u[:, 2], pos[:, 2] + t2 * u[:, 2]
    return ok & (t2 > 0) & (np.maximum(z1, z2) > -HALF_L) & (np.minimum(z1, z2) < HALF_L)


sp_path = sorted(CORE.glob("statepoint.*.h5"))[-1]
with openmc.StatePoint(sp_path) as sp:
    keff, keff_std = sp.keff.n, sp.keff.s
    kf_ev = sp.get_tally(name="kappa-fission").mean.ravel()[0]
    n_active = sp.n_particles * (sp.n_batches - sp.n_inactive)

with h5py.File(CORE / "surface_source.h5") as f:
    bank = f["source_bank"][()]
pos0 = np.column_stack([bank["r"][k] for k in "xyz"])
u0 = np.column_stack([bank["u"][k] for k in "xyz"])
E0, w0, ptype = bank["E"], bank["wgt"], bank["particle"]

rng = np.random.default_rng(0)
keep = {k: [] for k in ("orig", "pos", "u")}
for start in range(0, len(bank), 20_000):
    idx = np.arange(start, min(start + 20_000, len(bank)))
    rows = np.repeat(idx, K)
    phi = rng.uniform(0, 2 * np.pi, len(rows))
    c, s = np.cos(phi), np.sin(phi)
    p, u = pos0[rows], u0[rows]
    p = np.column_stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1], p[:, 2]])
    u = np.column_stack([c * u[:, 0] - s * u[:, 1], s * u[:, 0] + c * u[:, 1], u[:, 2]])
    hit = hits_tube(p, u)
    keep["orig"].append(rows[hit]); keep["pos"].append(p[hit]); keep["u"].append(u[hit])
orig = np.concatenate(keep["orig"])
pos, u = np.concatenate(keep["pos"]), np.concatenate(keep["u"])
energy, ptype_hit = E0[orig], ptype[orig]
weight = w0[orig] / K / n_active  # per core source neutron

# OpenMC reference: same particles, only the tube in the geometry
ref_dir = HERE / "results" / "detector_ref"
ref_dir.mkdir(parents=True, exist_ok=True)
types = {0: openmc.ParticleType.NEUTRON, 1: openmc.ParticleType.PHOTON}
openmc.write_source_file(
    [openmc.SourceParticle(r=tuple(pos[i]), u=tuple(u[i]), E=energy[i],
                           wgt=w0[orig[i]] / K, particle=types[int(ptype_hit[i])])
     for i in range(len(orig))],
    ref_dir / "tube_source.h5",
)
he3 = openmc.Material(name="He-3 gas")
he3.add_nuclide("He3", 1.0)
he3.set_density("g/cm3", HE3_DENSITY)
al = openmc.Material(name="Al")
al.add_element("Al", 1.0)
al.set_density("g/cm3", 2.70)
axis = openmc.ZCylinder(x0=TUBE_X, r=GAS_R)
outer = openmc.ZCylinder(x0=TUBE_X, r=WALL_R)
ends = +openmc.ZPlane(-HALF_L) & -openmc.ZPlane(HALF_L)
boundary = openmc.Sphere(r=300, boundary_type="vacuum")
gas = openmc.Cell(fill=he3, region=-axis & ends)
wall = openmc.Cell(fill=al, region=+axis & -outer & ends)
void = openmc.Cell(region=-boundary & ~(-outer & ends))
settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.particles = len(orig)
settings.batches = 20
settings.photon_transport = True
settings.source = openmc.FileSource(path=(ref_dir / "tube_source.h5").resolve())
np_tally = openmc.Tally(name="He3 (n,p)")
np_tally.filters = [openmc.CellFilter([gas])]
np_tally.scores = ["(n,p)"]
ref = openmc.Model(openmc.Geometry([gas, wall, void]), openmc.Materials([he3, al]),
                   settings, openmc.Tallies([np_tally]))
with openmc.StatePoint(ref.run(cwd=ref_dir, threads=8, output=False)) as sp:
    t = sp.get_tally(name="He3 (n,p)")
    # FileSource picks file rows uniformly: mean per sampled particle * rows / histories
    ref_np = t.mean.ravel()[0] * len(orig) / n_active
    ref_np_std = t.std_dev.ravel()[0] * len(orig) / n_active

source_per_watt = 1.0 / (kf_ev * 1.602176634e-19)  # core source neutrons per J
names = np.array(["neutron", "gamma"])
np.savez(
    HERE / "results" / "handoff.npz",
    particle=names[ptype_hit], energy_ev=energy, position_cm=pos, direction=u,
    weight=weight, orig=orig, n_leaked=len(bank), keff=keff, keff_std=keff_std,
    kf_ev=kf_ev, n_active=n_active, source_per_watt=source_per_watt,
    openmc_np=ref_np, openmc_np_std=ref_np_std,
    tube=np.array([TUBE_X, GAS_R, WALL_R, HALF_L, HE3_DENSITY]),
)

n_mask = ptype == 0
thermal = np.sum(w0[n_mask] * (E0[n_mask] < 0.625)) / np.sum(w0[n_mask])
print(f"k-eff {keff:.5f} +/- {keff_std:.5f}; {kf_ev / 1e6:.1f} MeV fission energy "
      f"per source neutron; {source_per_watt:.3e} source neutrons/s per W")
print(f"leaking through side wall per source neutron: "
      f"{w0[n_mask].sum() / n_active:.4f} n ({100 * thermal:.1f}% below 0.625 eV), "
      f"{w0[~n_mask].sum() / n_active:.4f} gamma")
print(f"rays hitting tube: {len(orig)} ({np.sum(ptype_hit == 0)} n, "
      f"{np.sum(ptype_hit == 1)} gamma) from {len(np.unique(orig))} leaked particles")
print(f"OpenMC He-3(n,p) in tube: {ref_np:.4e} +/- {ref_np_std:.1e} per source neutron "
      f"= {ref_np * source_per_watt:.3e} /s per W")
