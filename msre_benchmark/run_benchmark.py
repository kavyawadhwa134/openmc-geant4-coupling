"""Run the simplified MSRE benchmark model (OpenMC env).

    python run_benchmark.py plot                 # geometry slices -> results/*.png
    python run_benchmark.py run T [particles] [noshield] [variant=NAME] [cold]
                               [xs=CROSS_SECTIONS_XML] [lib=LABEL]
                                                 # k-eff, graphite at T kelvin;
                                                 # NAME from msre_model.VARIANTS;
                                                 # cold: everything at 293.6 K
"""

import json
import os
import sys
from pathlib import Path

import openmc

import msre_model

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def plot():
    geometry, m = msre_model.build()
    z_mid = 0.5 * (msre_model.Z_LATTICE_BOT + msre_model.Z_LATTICE_TOP)
    colors = {m["salt"]: "orange", m["graphite"]: "dimgray", m["inor"]: "steelblue",
              m["lower_head"]: "peru", m["support"]: "silver", m["basket"]: "purple",
              m["shield"]: "seagreen"}
    plots = []
    for name, basis, origin, width, px in (
        ("xy_core", "xy", (0, 0, z_mid), (170, 170), (1700, 1700)),
        ("xy_center", "xy", (0, 0, z_mid), (25, 25), (800, 800)),
        ("xz_full", "xz", (0, msre_model.PITCH, z_mid), (340, 340), (1200, 1200)),
    ):
        p = openmc.Plot(name=name)
        p.basis, p.origin, p.width, p.pixels = basis, origin, width, px
        p.color_by, p.colors, p.filename = "material", colors, name
        plots.append(p)
    model = openmc.Model(geometry, openmc.Materials(m.values()))
    model.plots = openmc.Plots(plots)
    RESULTS.mkdir(exist_ok=True)
    model.plot_geometry(cwd=RESULTS, output=False)


def run(temperature, particles, thermal_shield=True, variant="base", cold=False,
        xs=None, lib=""):
    if xs:  # also steers how add_element expands natural elements
        os.environ["OPENMC_CROSS_SECTIONS"] = str(Path(xs).resolve())
        # openmc reads the variable at import, so update its config too
        openmc.config["cross_sections"] = os.environ["OPENMC_CROSS_SECTIONS"]
    model = msre_model.model(graphite_temperature=temperature, particles=particles,
                             thermal_shield=thermal_shield, variant=variant, cold=cold)
    model.materials.cross_sections = os.environ["OPENMC_CROSS_SECTIONS"]
    tag = ("" if thermal_shield else "_noshield") + ("" if variant == "base" else f"_{variant}")
    tag += ("_cold" if cold else "") + (f"_{lib}" if lib else "")
    run_dir = RESULTS / f"graphite_{temperature:.0f}K_{particles}{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # This OpenMC build has MPI but no OpenMP threading
    sp_path = model.run(cwd=run_dir, mpi_args=["mpiexec", "-n", "8"], output=False)
    with openmc.StatePoint(sp_path) as sp:
        k = sp.keff
        result = {"graphite_K": temperature, "particles": particles,
                  "thermal_shield": thermal_shield, "variant": variant,
                  "cold": cold, "cross_sections": os.environ["OPENMC_CROSS_SECTIONS"],
                  "batches": int(sp.n_batches), "inactive": int(sp.n_inactive),
                  "keff": float(k.n), "keff_std": float(k.s),
                  "runtime_s": float(sp.runtime["total"])}
    (run_dir / "keff.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


def arg(name, default):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(name + "=")), default)


if __name__ == "__main__":
    if sys.argv[1] == "plot":
        plot()
    else:
        run(float(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 20_000,
            thermal_shield="noshield" not in sys.argv,
            variant=arg("variant", "base"), cold="cold" in sys.argv,
            xs=arg("xs", None), lib=arg("lib", ""))
