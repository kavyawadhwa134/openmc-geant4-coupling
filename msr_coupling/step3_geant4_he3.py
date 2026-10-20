"""Step 3 (geant4_python_application env): He-3 counter response in Geant4.

Starts every ray from results/handoff.npz that hits the tube as one Geant4
event, records the energy deposited in the He-3 gas, and counts events above a
150 keV threshold (the usual cut that rejects gammas). He-3(n,p) releases
764 keV shared by a 573 keV proton and a 191 keV triton.

Usage: python step3_geant4_he3.py [max_primaries]
"""

import sys
import time
from pathlib import Path

import awkward as ak
import numpy as np

import geant4_python_application as g4

HERE = Path(__file__).parent
BATCH = 10_000
THRESHOLD_KEV = 150.0


def detector_gdml(tube_x, gas_r, wall_r, half_l, density):
    mm = 10.0
    return f"""<?xml version="1.0" encoding="utf-8"?>
<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:noNamespaceSchemaLocation="http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd">
  <materials>
    <isotope name="He3iso" Z="2" N="3"><atom unit="g/mole" value="3.016"/></isotope>
    <element name="He3el"><fraction ref="He3iso" n="1.0"/></element>
    <material name="He3gas" state="gas">
      <D unit="g/cm3" value="{density}"/><fraction ref="He3el" n="1.0"/>
    </material>
  </materials>
  <solids>
    <box name="WorldSolid" x="4000" y="4000" z="4000" lunit="mm"/>
    <tube name="WallSolid" rmin="{gas_r * mm}" rmax="{wall_r * mm}" z="{2 * half_l * mm}"
          deltaphi="360" aunit="deg" lunit="mm"/>
    <tube name="GasSolid" rmin="0" rmax="{gas_r * mm}" z="{2 * half_l * mm}"
          deltaphi="360" aunit="deg" lunit="mm"/>
  </solids>
  <structure>
    <volume name="WallLV"><materialref ref="G4_Al"/><solidref ref="WallSolid"/></volume>
    <volume name="GasLV"><materialref ref="He3gas"/><solidref ref="GasSolid"/></volume>
    <volume name="World">
      <materialref ref="G4_Galactic"/>
      <solidref ref="WorldSolid"/>
      <physvol name="Wall"><volumeref ref="WallLV"/>
        <position name="WallPos" unit="mm" x="{tube_x * mm}" y="0" z="0"/></physvol>
      <physvol name="Gas"><volumeref ref="GasLV"/>
        <position name="GasPos" unit="mm" x="{tube_x * mm}" y="0" z="0"/></physvol>
    </volume>
  </structure>
  <setup name="Default" version="1.0"><world ref="World"/></setup>
</gdml>
"""


def to_primaries(h, sel):
    pos, dirn = h["position_cm"][sel], h["direction"][sel]
    return ak.Array({
        "particle": h["particle"][sel].tolist(),
        "energy": h["energy_ev"][sel] / 1e3,  # eV -> keV
        "position": {k: pos[:, i] for i, k in enumerate("xyz")},
        "direction": {k: dirn[:, i] for i, k in enumerate("xyz")},
    })


def main():
    h = dict(np.load(HERE / "results" / "handoff.npz"))
    n_total = len(h["particle"])
    n_use = min(int(sys.argv[1]), n_total) if len(sys.argv) > 1 else n_total
    rows = np.arange(n_use)
    scale = n_total / n_use

    edep = np.empty(n_use)
    start = time.time()
    # n_threads=0 (serial) keeps event i aligned with primaries row i
    with g4.Application(gdml=detector_gdml(*h["tube"]), physics="QGSP_BIC_HP", seed=1) as app:
        app.set_event_fields({"step_energy", "step_volume"})
        for i in range(0, n_use, BATCH):
            chunk = rows[i:i + BATCH]
            events = app.run(to_primaries(h, chunk))
            assert len(events) == len(chunk)
            edep[i:i + len(chunk)] = ak.to_numpy(
                g4.Scoring.energy_deposit(events, volume="Gas"))
            print(f"  {i + len(chunk):7d}/{n_use} events, {time.time() - start:6.1f} s",
                  flush=True)

    particle = h["particle"][rows]
    weight = h["weight"][rows] * scale  # per core source neutron
    counted = edep > THRESHOLD_KEV
    rate = np.sum(weight * counted)
    # Core-stage statistics: copies of one leaked particle are correlated, so
    # sum per leaked particle and take the spread over all leaked particles
    per_leaked = np.bincount(h["orig"][rows], weights=weight * counted,
                             minlength=int(h["n_leaked"]))
    rate_std = per_leaked.std() * np.sqrt(len(per_leaked))
    s_per_w = float(h["source_per_watt"])

    print(f"\nk-eff {float(h['keff']):.5f} +/- {float(h['keff_std']):.5f}")
    print(f"Geant4 primaries: {n_use} of {n_total} rays hitting the tube")
    for name in ("neutron", "gamma"):
        s = particle == name
        print(f"  {name:8s} {s.sum():7d} rays, {np.mean(edep[s] > 0):6.1%} deposit energy, "
              f"{np.mean(counted[s]):6.2%} above {THRESHOLD_KEV:.0f} keV")
    print(f"Geant4 counts > {THRESHOLD_KEV:.0f} keV: {rate:.4e} +/- {rate_std:.1e} per source "
          f"neutron = {rate * s_per_w:.3e} counts/s per W")
    ref, ref_std = float(h["openmc_np"]), float(h["openmc_np_std"])
    print(f"OpenMC He-3(n,p) reactions:  {ref:.4e} +/- {ref_std:.1e} per source neutron "
          f"-> Geant4/OpenMC = {rate / ref:.3f}")
    n_peak = (particle == "neutron") & (np.abs(edep - 764) < 15)
    print(f"neutron events within 15 keV of 764 keV: {n_peak.sum() / counted.sum():.1%} "
          f"of counts (rest is wall effect); max gamma deposit {edep[particle == 'gamma'].max():.0f} keV")
    print(f"Geant4 time: {time.time() - start:.0f} s")

    np.savez(HERE / "results" / "geant4_response.npz", edep_kev=edep,
             particle=particle, weight=weight)


if __name__ == "__main__":
    main()
