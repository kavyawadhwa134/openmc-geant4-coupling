"""Compare the simplified OpenMC MSRE model with the benchmark and published codes.

Corrections for this model's simplifications use the published reactivity
effects of the same simplifications, in pcm, so the corrected value estimates
what the full benchmark geometry would give with this library.
"""

import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

BENCHMARK_K, BENCHMARK_SD = 0.99978, 0.00420  # report Table 4.2
HEADS_FLAT_VS_DOMED = 243  # report Table 4.9: flat heads at the vessel's height

# Published reactivity effect of each simplification used here (pcm, added
# back to approximate the full model)
CORRECTIONS = {
    "flow distributor omitted (report Table 4.9: -98)": +98,
    "sample basket homogenised (report Table 4.9: -37)": +37,
    "all rods withdrawn (Frontiers 2024 OpenMC CSG: +70)": -70,
}

# Published calculations of the same benchmark
PUBLISHED = [
    ("Serpent 2.1.30, ENDF/B-VII.1 (benchmark report 2020)", 1.02132),
    ("Serpent 2.1.30, JENDL-4.0 (benchmark report 2020)", 1.02061),
    ("Serpent 2.1.29, ENDF/B-VII.1 (2018 model, Frontiers 2024 table)", 1.03122),
    ("MCNP 6.1, ENDF/B-VII.1 (2018 model, Frontiers 2024 table)", 1.03065),
    ("KENO 6.2.3, ENDF/B-VII.1 (2018 model, Frontiers 2024 table)", 1.02537),
    ("OpenMC CSG, ENDF/B-VII.1 (Frontiers 2024)", 1.02142),
    ("OpenMC CSG, ENDF/B-VIII.0 (Frontiers 2024)", 1.02346),
]

# Sensitivity runs: (variant directory, reference directory, published value)
SENSITIVITY = [
    ("no thermal shield", "noshield", "uncalibrated", -885),
    ("flat heads at true vessel height", "heads_height", "uncalibrated", +243),
    ("support lattice all salt (bound)", "support_salt", "uncalibrated", None),
    ("edge stringers without channels (bound)", "edge_closed", "uncalibrated", None),
    ("basket -> graphite, basket INOR-8 x1.0", "basket_graphite", "uncalibrated", +1623),
    ("basket -> graphite, calibrated basket", "basket_graphite", "", +1623),
]


def load(tag=""):
    path = RESULTS / f"graphite_911K_20000{'_' + tag if tag else ''}" / "keff.json"
    return json.loads(path.read_text()) if path.exists() else None


def delta(a, b):
    """Reactivity-style difference a - b in pcm, with its 1-sigma."""
    d = 1e5 * (a["keff"] - b["keff"]) / b["keff"]
    s = 1e5 * (a["keff_std"] ** 2 + b["keff_std"] ** 2) ** 0.5 / b["keff"]
    return d, s


def pcm(k):
    return 1e5 * (k - BENCHMARK_K) / BENCHMARK_K


base = load()
print(f"Benchmark k-eff (experiment, simplification bias removed): "
      f"{BENCHMARK_K:.5f} +/- {BENCHMARK_SD:.5f}\n")

print("Sensitivity of this model                     this model        published")
for label, variant, ref, published in SENSITIVITY:
    a, b = load(variant), load(ref)
    if a and b:
        d, s = delta(a, b)
        pub = f"{published:+6d}" if published is not None else "     -"
        print(f"  {label:42s} {d:+6.0f} +/- {s:3.0f} pcm   {pub} pcm")

if base:
    k, sd = base["keff"], base["keff_std"]
    corrections = dict(CORRECTIONS)
    heads, uncal = load("heads_height"), load("uncalibrated")
    if heads and uncal:
        d, s = delta(heads, uncal)
        corrections[f"flat heads -> domes: {d:+.0f} measured - {HEADS_FLAT_VS_DOMED} "
                    f"published (+/- {s:.0f})"] = round(d - HEADS_FLAT_VS_DOMED)
    kc = k * (1 + sum(corrections.values()) * 1e-5)
    sigma_exp = 1e5 * BENCHMARK_SD / BENCHMARK_K
    print(f"\nThis work, OpenMC 0.15.1, ENDF/B-VII.0, simplified CSG model:")
    print(f"  as modelled:  {k:.5f} +/- {sd:.5f}   C/E-1 = {pcm(k):+5.0f} pcm")
    for what, d in corrections.items():
        print(f"    {d:+5d} pcm  {what}")
    print(f"  corrected:    {kc:.5f} +/- {sd:.5f}   C/E-1 = {pcm(kc):+5.0f} pcm "
          f"({pcm(kc) / sigma_exp:.1f} sigma_exp)")

    # Library effect: the ENDF/B-VII.1 files available here are room-temperature
    # only, so both libraries are compared cold and the difference is applied
    cold70, cold71 = load("cold_vii0"), load("cold_vii1")
    if cold70 and cold71:
        d, s = delta(cold71, cold70)
        k71 = kc * (1 + d * 1e-5)
        s71 = k71 * ((sd / kc) ** 2 + (s * 1e-5) ** 2) ** 0.5
        print(f"\n  Library effect at 293.6 K: VII.0 {cold70['keff']:.5f}, "
              f"VII.1 {cold71['keff']:.5f} -> {d:+.0f} +/- {s:.0f} pcm")
        print(f"  estimated with ENDF/B-VII.1: {k71:.5f} +/- {s71:.5f}   "
              f"C/E-1 = {pcm(k71):+5.0f} pcm ({pcm(k71) / sigma_exp:.1f} sigma_exp)")

print("\nPublished calculations                                              k-eff     C/E-1")
for label, k in PUBLISHED:
    print(f"  {label:66s} {k:.5f}  {pcm(k):+5.0f} pcm")
