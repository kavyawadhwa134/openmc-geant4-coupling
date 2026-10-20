# Molten salt reactor → He-3 counter coupling example

A small MSRE-inspired core runs in OpenMC: LiF-BeF2-ZrF4-UF4 fuel salt in a
graphite lattice, inside a bare Hastelloy-N vessel, in eigenvalue mode. Every
neutron and gamma leaving the vessel's side wall is recorded. Geant4, through
`geant4_python_application`, then simulates an ex-core He-3 proportional
counter 90 cm from the core axis and produces its pulse-height spectrum.

Step 2 makes this cheap without biasing it:
- each leaked particle is copied at random angles about the core axis;
- only copies whose straight-line path hits the tube are kept (outside the vessel is void).

OpenMC also tallies He-3(n,p) on exactly those particles, which gives a
code-to-code check of the Geant4 detector stage.

```bash
# OpenMC environment (openmc, h5py, nuclear data with c_Graphite and photon libraries)
export OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml
python step1_openmc_core.py        # ~5 min
python step2_detector_source.py

# geant4_python_application environment: create a fresh venv first
python -m venv .venv && source .venv/bin/activate && pip install geant4-python-application
python step3_geant4_he3.py
python step3b_np_split.py          # optional: split counts into He-3(n,p) and other events

# anything with matplotlib
python step4_plot.py
```

Not modelled:
- fuel circulation, i.e. delayed-neutron precursors carried out of the core and gammas from the primary loop;
- temperature feedback;
- shielding around the detector;
- the 4-fold lattice asymmetry, which the azimuthal copies average over.
