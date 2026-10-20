"""Step 4 (any env with matplotlib): compare with Ball et al. (arXiv:2505.23047).

Paper, 2" EJ-301 on CMFX: G = 1.021e-4 (n,p) scatters per source neutron,
T = 0.0389 above threshold, total efficiency eps = G*T = 3.97e-6 counts per
source neutron.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "results" / os.environ.get("CMFX_LAYOUT", "fig5")
PAPER = {"G": 1.021e-4, "T": 0.0389, "eps": 3.97e-6}

h = np.load(RESULTS / "openmc_handoff.npz")
m = np.load(RESULTS / "response_matrix.npz")
d = np.load(RESULTS / "direct.npz")

S, S_std = h["scatter_mean"], h["scatter_std"]
G = S.sum()
G_std = np.sqrt(np.sum(S_std ** 2))

# Paper's method: scatter-rate spectrum x normalised response matrix
eps_matrix = np.sum(S * m["p_count"])
eps_matrix_std = np.sqrt(np.sum((S_std * m["p_count"]) ** 2))

# Direct coupling: one pulse per entering neutron
w, c = d["weight"], d["counted"]
eps_direct = np.sum(w * c)
per_entry = np.bincount(d["orig"], weights=w * c)
eps_direct_std = per_entry.std() * np.sqrt(len(per_entry))
# Recoils above 100 keV: n-p elastic is isotropic in the CM frame below ~10 MeV,
# so a scatter at energy E gives a recoil above 0.1 MeV with probability 1 - 0.1/E
G_fast_geant4 = np.sum(w * d["n_fast"])
e_mid = 0.5 * (h["energy_edges_ev"][1:] + h["energy_edges_ev"][:-1]) / 1e6
G_fast_openmc = np.sum(S * np.clip(1 - 0.1 / e_mid, 0, None))

rows = [
    ("G  (n,p) scatters / source n", PAPER["G"], G, G_std, None),
    ("T  above-threshold fraction", PAPER["T"], eps_matrix / G, None, eps_direct / G),
    ("eps counts / source n", PAPER["eps"], eps_matrix, eps_matrix_std, eps_direct),
]
print(f"{'':32s}{'paper':>11s}{'matrix (paper method)':>24s}{'direct coupling':>18s}")
for name, paper, ours, std, direct in rows:
    s = f"{ours:.3e}" + (f" +/- {std:.1e}" if std is not None else "")
    print(f"{name:32s}{paper:11.3e}{s:>24s}{(f'{direct:.3e}' if direct is not None else '-'):>18s}")
print(f"\neps direct = {eps_direct:.3e} +/- {eps_direct_std:.1e} (detector-stage statistics only)")
print(f"direct / matrix = {eps_direct / eps_matrix:.3f}")
print(f"(n,p) recoils > 100 keV per source neutron: OpenMC {G_fast_openmc:.3e}, "
      f"Geant4 on the same neutrons {G_fast_geant4:.3e} (ratio {G_fast_geant4 / G_fast_openmc:.3f})")
edges = h["energy_edges_ev"]
for lo in (1e-5, 1.0, 1e5, 1e6, 2.0e6):
    sel = edges[:-1] >= lo
    print(f"  share of scatters above {lo:8.0e} eV: {S[sel].sum() / G:6.1%}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
centres = np.sqrt(edges[1:] * edges[:-1])
leth = np.log(edges[1:] / edges[:-1])
ax1.stairs(S / leth, edges / 1e6, lw=1.5)
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("Incoming neutron energy [MeV]")
ax1.set_ylabel("(n,p) scatters per unit lethargy per source n")
ax1.set_title("OpenMC: scatter-rate spectrum in the 2\" EJ-301 cell")

le = m["light_edges"]
synthetic = S @ m["matrix"]
ax2.stairs(synthetic / np.diff(le), le, lw=1.5, label="matrix method (paper's approach)")
hist, _ = np.histogram(d["light"], bins=le, weights=w * (d["proton"] > 0.5 * d["total"]))
ax2.stairs(hist / np.diff(le), le, lw=1.5, label="direct event-by-event coupling")
ax2.axvline(0.2, color="r", ls="--", lw=1)
ax2.set_yscale("log"); ax2.set_xlim(0, 1.0)
ax2.set_xlabel("Light [MeVee]")
ax2.set_ylabel("Pulses per MeVee per source neutron")
ax2.set_title("Geant4: synthetic pulse-height spectrum")
ax2.legend()
fig.tight_layout()
fig.savefig(RESULTS / "cmfx_recreation.png", dpi=130)
print(f"wrote {RESULTS / 'cmfx_recreation.png'}")
