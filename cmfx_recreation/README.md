# Recreating the CMFX OpenMC + Geant4 detector calibration

This folder recreates the in-silico calibration of a 2"×2" EJ-301 liquid
scintillator in:

> J. L. Ball et al., "Measurements of Fusion Yield on the Centrifugal Mirror
> Fusion Experiment", arXiv:2505.23047 (2025), Sec. 2.1.2.

The paper's efficiency is split into two factors, ε = G·T:
- **G**, computed in OpenMC: elastic (n,p) scatters in the detector per emitted D-D neutron.
- **T**, computed from a Geant4 response matrix plus light-output and resolution functions: the fraction of those scatters giving a pulse above 200 keVee.

Paper values: G = 1.021e-4, T = 0.0389, ε = 3.97e-6 counts per source neutron.

Here OpenMC runs through its Python API and Geant4 through
`geant4_python_application`. The paper's matrix method is reproduced and
compared with a direct event-by-event coupling, in which every neutron OpenMC
sends into the detector becomes a Geant4 event.

## Steps

```bash
# OpenMC environment (openmc + h5py + nuclear data; uses mpiexec -n 8)
export OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml
python step1_openmc_cmfx.py                 # ~4 min; CMFX_LAYOUT=bunker for the sensitivity case

# geant4_python_application environment: create a fresh venv first
python -m venv .venv && source .venv/bin/activate && pip install geant4-python-application
python step2_geant4_response_matrix.py      # paper's method
python step3_geant4_direct.py               # direct coupling

python step4_compare.py                     # any env with matplotlib
```

**Variance reduction.** The detector sees about 1e-5 of the source. Step 1
therefore uses two-stage surface-source splitting:
- **Stage A** runs the full room and records every neutron entering a 60 cm sphere around the detector.
- **Stage B** replays each of them about 100 times inside the sphere only, with a vacuum boundary so nothing is counted twice.

## Taken from the paper vs assumed

**Taken from the paper:**
- 2.45 MeV source in an annulus of radii 5–21 cm, 60 cm long
- vessel radius 36.1 cm, central conductor, magnets 1.45 m apart
- detector ~3.1 m from centre
- shield wall and floor positions read off Fig. 5
- 200 keVee threshold

**Assumed:**
- all thicknesses and material mixes, and the extent of the wall along the axis
- no air, ceiling or other hall walls; no PMT, housing or mu-metal
- isotropic illumination for the response matrix
- light output: the Cecil (1979) NE-213 proton curve
- resolution: Dietze–Klein form with typical 2" parameters
- PSD: replaced by requiring that protons produce most of the light

The paper's model geometry is in N. R. Schwartz's PhD thesis (UMD, 2024), and
its measured light-output and resolution functions are in Ball et al., RSI 95,
123514 (2024). Neither was used here.
