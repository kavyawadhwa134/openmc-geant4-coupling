"""Step 4 (any env with matplotlib): He-3 pulse-height and leakage spectra."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
r = np.load(HERE / "results" / "geant4_response.npz")
h = np.load(HERE / "results" / "handoff.npz")
edep, particle, weight = r["edep_kev"], r["particle"], r["weight"]
per_w = float(h["source_per_watt"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

edges = np.linspace(0, 900, 181)
for name in ("neutron", "gamma"):
    s = (particle == name) & (edep > 0)
    ax1.hist(edep[s], bins=edges, weights=weight[s] * per_w / np.diff(edges)[0],
             histtype="step", lw=1.5, label=f"{name}s reaching tube")
for e, label in ((191, "triton 191"), (573, "proton 573"), (764, "full 764")):
    ax1.axvline(e, color="grey", ls=":", lw=1)
    ax1.text(e, 0.98, f" {label}", rotation=90, va="top", fontsize=8,
             transform=ax1.get_xaxis_transform())
ax1.axvline(150, color="r", ls="--", lw=1)
ax1.text(150, 0.5, " 150 keV threshold", rotation=90, color="r", fontsize=8,
         transform=ax1.get_xaxis_transform())
ax1.set_yscale("log")
ax1.set_xlabel("Energy deposited in He-3 gas [keV]")
ax1.set_ylabel("Counts per keV per second per W")
ax1.set_title("Geant4: ex-core He-3 tube pulse-height spectrum")
ax1.legend(loc="upper left")

s = h["particle"] == "neutron"
e_edges = np.logspace(-3, 7.2, 103)  # eV
w = h["weight"][s] * per_w
counts, _ = np.histogram(h["energy_ev"][s], bins=e_edges, weights=w)
ax2.stairs(counts / np.log(e_edges[1:] / e_edges[:-1]), e_edges, lw=1.5, color="C0")
ax2.set_xscale("log")
ax2.set_xlabel("Neutron energy at the tube [eV]")
ax2.set_ylabel("Neutrons per unit lethargy per second per W")
ax2.set_title("OpenMC: neutrons leaving the vessel toward the tube")
ax2.axvline(0.625, color="grey", ls=":", lw=1)

fig.tight_layout()
fig.savefig(HERE / "results" / "msr_he3_response.png", dpi=130)
print("wrote results/msr_he3_response.png")
