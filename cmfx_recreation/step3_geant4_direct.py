"""Step 3 (geant4_python_application env): direct event-by-event coupling.

Starts every neutron that OpenMC recorded entering the detector (stage B
surface source) as a Geant4 event, REPLAYS times each, and counts pulses above
threshold. Unlike the paper's matrix method, one entering neutron gives at
most one pulse, however many times it scatters.

Also counts Geant4 (n,p) recoils above 100 keV per source neutron, a
code-to-code check of OpenMC's scatter rate on identical incoming neutrons.

Writes results/direct.npz.
"""

import os
from pathlib import Path

import awkward as ak
import numpy as np

import geant4_python_application as g4
from detector_model import CUT_COMMANDS, EVENT_FIELDS, counted, detector_gdml, event_light, smear

HERE = Path(__file__).parent
RESULTS = HERE / "results" / os.environ.get("CMFX_LAYOUT", "fig5")
REPLAYS = 10
BATCH = 20_000


def main():
    h = np.load(RESULTS / "openmc_handoff.npz")
    n = len(h["energy_ev"])
    rows = np.repeat(np.arange(n), REPLAYS)
    rng = np.random.default_rng(3)
    light = np.empty(len(rows)); total = np.empty(len(rows))
    proton = np.empty(len(rows)); n_p = np.empty(len(rows)); n_fast = np.empty(len(rows))
    centre = tuple(h["det_centre_cm"])  # same coordinates as the OpenMC model
    with g4.Application(gdml=detector_gdml(centre), physics="QGSP_BIC_HP", seed=4) as app:
        app.set_event_fields(EVENT_FIELDS)
        app.commands(CUT_COMMANDS)
        for i in range(0, len(rows), BATCH):
            r = rows[i:i + BATCH]
            pos, u = h["position_cm"][r], h["direction"][r]
            events = app.run(ak.Array({
                "particle": ["neutron"] * len(r), "energy": h["energy_ev"][r] / 1e3,
                "position": {k: pos[:, j] for j, k in enumerate("xyz")},
                "direction": {k: u[:, j] for j, k in enumerate("xyz")},
            }))
            sl = slice(i, i + len(r))
            total[sl], proton[sl], n_p[sl], n_fast[sl] = event_light(events)
            print(f"  {i + len(r)}/{len(rows)} events", flush=True)
    light = smear(total, rng)
    is_count = counted(light, total, proton)
    w = h["weight"][rows] / REPLAYS  # per source neutron
    np.savez(RESULTS / "direct.npz", light=light, total=total, proton=proton,
             n_p=n_p, n_fast=n_fast, counted=is_count, weight=w, energy_ev=h["energy_ev"][rows],
             orig=rows)


if __name__ == "__main__":
    main()
