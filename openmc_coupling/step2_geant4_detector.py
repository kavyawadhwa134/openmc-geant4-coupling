"""Step 2 (geant4_python_application env): detector response to OpenMC's particles.

Reads ``results/handoff.npz`` from step 1, starts each particle that left the
wall as one Geant4 event in the same plastic slab (40.5 < x < 45.5 cm), and
records the energy deposited per event. Only the plastic is modelled here; the
wall is OpenMC's job.

Usage: python step2_geant4_detector.py [max_primaries]
"""

import sys
import time
from pathlib import Path

import awkward as ak
import numpy as np

import geant4_python_application as g4

HERE = Path(__file__).parent
BATCH = 5_000
THRESHOLD_KEV = 100.0  # counting threshold on deposited energy

# Same coordinates as the OpenMC model (GDML lengths in mm)
DETECTOR_GDML = """<?xml version="1.0" encoding="utf-8"?>
<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:noNamespaceSchemaLocation="http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd">
  <solids>
    <box name="WorldSolid" x="3000" y="3000" z="3000" lunit="mm"/>
    <box name="PlasticSolid" x="50" y="1000" z="1000" lunit="mm"/>
  </solids>
  <structure>
    <volume name="PlasticLV">
      <materialref ref="G4_PLASTIC_SC_VINYLTOLUENE"/>
      <solidref ref="PlasticSolid"/>
    </volume>
    <volume name="World">
      <materialref ref="G4_Galactic"/>
      <solidref ref="WorldSolid"/>
      <physvol name="Plastic">
        <volumeref ref="PlasticLV"/>
        <position name="PlasticPosition" unit="mm" x="430" y="0" z="0"/>
      </physvol>
    </volume>
  </structure>
  <setup name="Default" version="1.0"><world ref="World"/></setup>
</gdml>
"""


def to_primaries(h, sel):
    """OpenMC hand-off rows -> the package's primaries array (keV, cm)."""
    pos, dirn = h["position_cm"][sel], h["direction"][sel]
    return ak.Array({
        "particle": h["particle"][sel].tolist(),
        "energy": h["energy_ev"][sel] / 1e3,
        "position": {k: pos[:, i] for i, k in enumerate("xyz")},
        "direction": {k: dirn[:, i] for i, k in enumerate("xyz")},
    })


def main():
    h = dict(np.load(HERE / "results" / "handoff.npz"))
    n_total = len(h["particle"])
    n_use = int(sys.argv[1]) if len(sys.argv) > 1 else n_total
    # A random subset keeps the neutron/gamma mix; rescale per-source results
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(n_total, size=n_use, replace=False))
    scale = n_total / n_use

    edep = np.empty(n_use)
    start = time.time()
    # n_threads=0 (serial) keeps event i aligned with primaries row i
    with g4.Application(gdml=DETECTOR_GDML, physics="QGSP_BIC_HP", seed=1) as app:
        app.set_event_fields({"step_energy", "step_volume"})
        for i in range(0, n_use, BATCH):
            chunk = rows[i:i + BATCH]
            events = app.run(to_primaries(h, chunk))
            assert len(events) == len(chunk)
            edep[i:i + len(chunk)] = ak.to_numpy(
                g4.Scoring.energy_deposit(events, volume="Plastic"))
            print(f"  {i + len(chunk):7d}/{n_use} events, {time.time() - start:6.1f} s",
                  flush=True)

    particle = h["particle"][rows]
    weight = h["weight"][rows] * scale / h["n_source"]  # per source neutron
    openmc_total_kev = h["openmc_heating_ev"].sum() / 1e3

    print(f"\nGeant4 primaries: {n_use} of {n_total} OpenMC surface particles")
    print("Per source neutron      deposited [keV]   counts > 100 keV   efficiency")
    for name in ("neutron", "gamma"):
        s = particle == name
        e = edep[s]
        print(f"  from {name:8s} {np.sum(e * weight[s]):16.1f} "
              f"{np.sum(weight[s] * (e > THRESHOLD_KEV)):18.4f} "
              f"{np.mean(e > THRESHOLD_KEV):12.3f}")
    total = np.sum(edep * weight)
    rel = np.std(edep * weight) * np.sqrt(n_use) / total
    print(f"  total          {total:16.1f} (+/- {100 * rel:.1f}%) "
          f"{np.sum(weight * (edep > THRESHOLD_KEV)):18.4f}")
    print(f"OpenMC full-model heating in plastic: {openmc_total_kev:.1f} keV per source "
          f"neutron -> Geant4/OpenMC = {total / openmc_total_kev:.3f}")
    print(f"Geant4 time: {time.time() - start:.0f} s")

    np.savez(HERE / "results" / "geant4_response.npz", edep_kev=edep,
             particle=particle, weight=weight, openmc_total_kev=openmc_total_kev)


if __name__ == "__main__":
    main()
