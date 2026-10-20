"""Step 3 (any env with matplotlib): deposited-energy spectrum in the plastic."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
r = np.load(HERE / "results" / "geant4_response.npz")
edep, particle, weight = r["edep_kev"] / 1e3, r["particle"], r["weight"]

edges = np.linspace(0.05, 14, 140)
fig, ax = plt.subplots(figsize=(7, 4.5))
for name, label in (("neutron", "neutrons leaving wall"), ("gamma", "gammas leaving wall")):
    s = particle == name
    ax.hist(edep[s], bins=edges, weights=weight[s] / np.diff(edges)[0],
            histtype="step", lw=1.5, label=label)
ax.hist(edep, bins=edges, weights=weight / np.diff(edges)[0], histtype="step",
        color="k", lw=1, label="total")
ax.set_yscale("log")
ax.set_xlabel("Energy deposited in 5 cm plastic per event [MeV]")
ax.set_ylabel("Events per MeV per source neutron")
ax.set_title("OpenMC wall transport -> Geant4 detector response")
ax.legend()
fig.tight_layout()
fig.savefig(HERE / "results" / "deposited_energy_spectrum.png", dpi=130)
print("wrote results/deposited_energy_spectrum.png")
