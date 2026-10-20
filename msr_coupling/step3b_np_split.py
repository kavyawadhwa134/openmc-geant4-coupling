"""Step 3b (geant4_python_application env): which Geant4 counts are He-3(n,p)?

Step 3 compares Geant4 events above 150 keV with OpenMC's He-3(n,p) reaction
rate, which are not quite the same quantity. This reruns the same primaries with
track information and splits each event by what happened in it:

* (n,p): a triton was produced. Only He-3(n,p)t makes tritons here (the wall is
  aluminium), so this tags the reaction OpenMC tallies.
* other: no triton, e.g. a fast neutron knocking a He-3 nucleus (elastic recoil)
  or a proton from Al(n,p) in the wall entering the gas.

It then compares the Geant4 (n,p) rate, above threshold and in total, with
OpenMC.

Usage: python step3b_np_split.py [max_primaries]
"""

import sys
import time
from collections import Counter
from pathlib import Path

import awkward as ak
import numpy as np

import geant4_python_application as g4
from step3_geant4_he3 import BATCH, THRESHOLD_KEV, detector_gdml, to_primaries

HERE = Path(__file__).parent


def rate_and_error(h, rows, weight, mask):
    """Weighted rate per source neutron; error grouped by leaked particle,
    because the angle copies of one leaked particle are correlated."""
    per_leaked = np.bincount(h["orig"][rows], weights=weight * mask,
                             minlength=int(h["n_leaked"]))
    return np.sum(weight * mask), per_leaked.std() * np.sqrt(len(per_leaked))


def main():
    h = dict(np.load(HERE / "results" / "handoff.npz"))
    n_total = len(h["particle"])
    n_use = min(int(sys.argv[1]), n_total) if len(sys.argv) > 1 else n_total
    rows = np.arange(n_use)
    weight = h["weight"][rows] * n_total / n_use  # per core source neutron

    edep = np.empty(n_use)
    has_triton = np.zeros(n_use, bool)
    has_he3_recoil = np.zeros(n_use, bool)
    other_kinds = Counter()  # what the counted non-(n,p) events contain
    start = time.time()
    with g4.Application(gdml=detector_gdml(*h["tube"]), physics="QGSP_BIC_HP", seed=1) as app:
        app.set_event_fields({"step_energy", "step_volume", "track_particle",
                              "track_creator_process"})
        for i in range(0, n_use, BATCH):
            chunk = rows[i:i + BATCH]
            events = app.run(to_primaries(h, chunk))
            assert len(events) == len(chunk)
            sl = slice(i, i + len(chunk))
            edep[sl] = ak.to_numpy(g4.Scoring.energy_deposit(events, volume="Gas"))
            particles = events.track.particle
            has_triton[sl] = ak.to_numpy(ak.any(particles == "triton", axis=1))
            has_he3_recoil[sl] = ak.to_numpy(ak.any(particles == "He3", axis=1))
            # Record the secondaries in counted events without a triton
            odd = (edep[sl] > THRESHOLD_KEV) & ~has_triton[sl]
            for parts, procs in zip(ak.to_list(particles[odd]),
                                    ak.to_list(events.track.creator_process[odd])):
                other_kinds[tuple(sorted({f"{p} ({c})" for p, c in zip(parts, procs)
                                          if c}))] += 1
            print(f"  {i + len(chunk):7d}/{n_use} events, {time.time() - start:6.1f} s",
                  flush=True)

    counted = edep > THRESHOLD_KEV
    is_n = h["particle"][rows] == "neutron"
    np_counted = counted & has_triton
    other_counted = counted & ~has_triton

    total, total_err = rate_and_error(h, rows, weight, counted)
    np_all, np_all_err = rate_and_error(h, rows, weight, has_triton)
    np_cnt, np_cnt_err = rate_and_error(h, rows, weight, np_counted)
    oth, oth_err = rate_and_error(h, rows, weight, other_counted)
    ref, ref_err = float(h["openmc_np"]), float(h["openmc_np_std"])

    def ratio(x, dx):
        r = x / ref
        return r, r * np.hypot(dx / x, ref_err / ref)

    print(f"\nGeant4 primaries: {n_use} of {n_total}; "
          f"{np.sum(has_triton)} events with a triton, {np.sum(counted)} counted")
    print("Per source neutron                         rate          vs OpenMC (n,p)")
    for label, x, dx in (
        ("all counts > 150 keV (step 3)", total, total_err),
        ("(n,p) events, any deposit", np_all, np_all_err),
        ("(n,p) events > 150 keV", np_cnt, np_cnt_err),
    ):
        r, dr = ratio(x, dx)
        print(f"  {label:38s} {x:.4e} +/- {dx:.1e}   {r:.3f} +/- {dr:.3f}")
    print(f"  {'other counts > 150 keV (no triton)':38s} {oth:.4e} +/- {oth_err:.1e}   "
          f"{100 * oth / total:.2f}% of counts")
    print(f"  {'OpenMC He-3(n,p)':38s} {ref:.4e} +/- {ref_err:.1e}")

    lost = has_triton & ~counted
    print(f"\n(n,p) events below threshold (wall effect): {lost.sum()} "
          f"({100 * np.sum(weight * lost) / np_all:.2f}% of (n,p) rate)")
    print(f"counted events with no triton: {other_counted.sum()} "
          f"({np.sum(other_counted & is_n)} from neutrons, "
          f"{np.sum(other_counted & ~is_n)} from gammas); "
          f"{np.sum(other_counted & has_he3_recoil)} contain a He-3 recoil")
    for kinds, n in other_kinds.most_common(6):
        print(f"  {n:5d} x {', '.join(kinds) or '(no secondaries)'}")
    print(f"Geant4 time: {time.time() - start:.0f} s")

    np.savez(HERE / "results" / "geant4_np_split.npz", edep_kev=edep,
             has_triton=has_triton, has_he3_recoil=has_he3_recoil, weight=weight,
             particle=h["particle"][rows])


if __name__ == "__main__":
    main()
