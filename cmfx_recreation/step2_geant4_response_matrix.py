"""Step 2 (geant4_python_application env): Geant4 response matrix, as in the paper.

For each OpenMC energy bin from 0.45 to 2.6 MeV, neutrons uniformly illuminate
the bare 2"x2" EJ-301 cell (isotropic inward flux). Each row is the pulse-height
distribution of events with at least one (n,p) scatter, normalised to 1. That
mirrors the paper's "each tallied scatter maps to one synthetic pulse".
Below 0.45 MeV the largest possible light (0.07 MeVee) is far under the
200 keVee threshold, so those rows count nothing.

Writes results/response_matrix.npz.
"""

import os
from pathlib import Path

import awkward as ak
import numpy as np

import geant4_python_application as g4
from detector_model import CUT_COMMANDS, EVENT_FIELDS, counted, detector_gdml, event_light, smear

HERE = Path(__file__).parent
RESULTS = HERE / "results" / os.environ.get("CMFX_LAYOUT", "fig5")
N_PER_BIN = 3000
R, H = 2.54, 5.08
LIGHT_EDGES = np.linspace(0, 1.0, 101)  # MeVee


def isotropic_inward(n, rng):
    """Points on the cell surface with cosine-weighted inward directions (cm)."""
    side, cap = 2 * np.pi * R * H, np.pi * R ** 2
    which = rng.choice(3, size=n, p=np.array([side, cap, cap]) / (side + 2 * cap))
    phi = rng.uniform(0, 2 * np.pi, n)
    rr = R * np.sqrt(rng.uniform(0, 1, n))
    pos = np.where(which[:, None] == 0,
                   np.column_stack([R * np.cos(phi), rng.uniform(-H / 2, H / 2, n), R * np.sin(phi)]),
                   np.column_stack([rr * np.cos(phi), np.where(which == 1, H / 2, -H / 2), rr * np.sin(phi)]))
    normal = np.where(which[:, None] == 0, np.column_stack([np.cos(phi), np.zeros(n), np.sin(phi)]),
                      np.column_stack([np.zeros(n), np.where(which == 1, 1.0, -1.0), np.zeros(n)]))
    # cosine-weighted direction about -normal
    mu, psi = np.sqrt(rng.uniform(0, 1, n)), rng.uniform(0, 2 * np.pi, n)
    helper = np.where(np.abs(normal[:, 1:2]) < 0.9, [[0, 1, 0]], [[1, 0, 0]])
    t1 = np.cross(normal, helper); t1 /= np.linalg.norm(t1, axis=1)[:, None]
    t2 = np.cross(normal, t1)
    st = np.sqrt(1 - mu ** 2)
    u = -mu[:, None] * normal + (st * np.cos(psi))[:, None] * t1 + (st * np.sin(psi))[:, None] * t2
    return pos + 1e-3 * normal, u


def main():
    edges = np.load(RESULTS / "openmc_handoff.npz")["energy_edges_ev"] / 1e6  # MeV
    rows = np.nonzero(edges[:-1] >= 0.45)[0]
    rng = np.random.default_rng(1)
    p_count = np.zeros(len(edges) - 1)
    matrix = np.zeros((len(edges) - 1, len(LIGHT_EDGES) - 1))
    with g4.Application(gdml=detector_gdml(), physics="QGSP_BIC_HP", seed=2) as app:
        app.set_event_fields(EVENT_FIELDS)
        app.commands(CUT_COMMANDS)  # no delta rays: each recoil keeps its own light
        for i in rows:
            pos, u = isotropic_inward(N_PER_BIN, rng)
            energy_kev = rng.uniform(edges[i], edges[i + 1], N_PER_BIN) * 1e3
            events = app.run(ak.Array({
                "particle": ["neutron"] * N_PER_BIN, "energy": energy_kev,
                "position": {k: pos[:, j] for j, k in enumerate("xyz")},
                "direction": {k: u[:, j] for j, k in enumerate("xyz")},
            }))
            total, proton, n_p, _ = event_light(events)
            hit = n_p > 0
            light = smear(total, rng)
            p_count[i] = np.mean(counted(light, total, proton)[hit])
            matrix[i], _ = np.histogram(light[hit], bins=LIGHT_EDGES)
            matrix[i] /= hit.sum()
            print(f"  E {edges[i]:.3f}-{edges[i + 1]:.3f} MeV: {hit.mean():5.1%} interact, "
                  f"P(count | interact) = {p_count[i]:.3f}", flush=True)
    np.savez(RESULTS / "response_matrix.npz", energy_edges_mev=edges,
             p_count=p_count, matrix=matrix, light_edges=LIGHT_EDGES)


if __name__ == "__main__":
    main()
