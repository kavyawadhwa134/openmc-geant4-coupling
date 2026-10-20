"""Shared Geant4 detector model: a bare 2"x2" EJ-301 cell and its light response.

Light output and resolution are standard published NE-213/EJ-301 forms. The
paper used its own measured functions, which I could not obtain:

* protons: Cecil et al. (1979) NE-213, L = 0.83 E - 2.82 (1 - exp(-0.25 E^0.93)) MeVee
* carbon and other ions: L = 0.017 E;  electrons/positrons/photons: L = E
* resolution (Dietze & Klein form): FWHM/L = sqrt(a^2 + b^2/L + c^2/L^2),
  with a = 0.11, b = 0.10 MeV^0.5, c = 0.01 MeV (typical 2" NE-213 values)

A count needs light above 200 keVee, as in the paper. PSD is mimicked by
requiring that protons produce most of the light: gamma-like events are rejected.
"""

import awkward as ak
import numpy as np

THRESHOLD_MEVEE = 0.200
RES_A, RES_B, RES_C = 0.11, 0.10, 0.01

# Detector at the origin of its own world, axis along y as in the OpenMC model;
# the caller supplies the centre so OpenMC coordinates can be used unchanged.
def detector_gdml(centre_cm=(0.0, 0.0, 0.0)):
    x, y, z = (10.0 * c for c in centre_cm)
    return f"""<?xml version="1.0" encoding="utf-8"?>
<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:noNamespaceSchemaLocation="http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd">
  <define><rotation name="AxisAlongY" x="90" unit="deg"/></define>
  <materials>
    <element name="Hydrogen" formula="H" Z="1"><atom value="1.008"/></element>
    <element name="Carbon" formula="C" Z="6"><atom value="12.011"/></element>
    <!-- EJ-301: 4.82e22 H and 3.98e22 C per cm3 -->
    <material name="EJ301" state="liquid">
      <D value="0.874" unit="g/cm3"/>
      <fraction n="0.0923" ref="Hydrogen"/><fraction n="0.9077" ref="Carbon"/>
    </material>
  </materials>
  <solids>
    <box name="WorldSolid" x="20000" y="20000" z="20000" lunit="mm"/>
    <tube name="CellSolid" rmin="0" rmax="25.4" z="50.8" deltaphi="360" aunit="deg" lunit="mm"/>
  </solids>
  <structure>
    <volume name="CellLV"><materialref ref="EJ301"/><solidref ref="CellSolid"/></volume>
    <volume name="World">
      <materialref ref="G4_Galactic"/><solidref ref="WorldSolid"/>
      <physvol name="Detector"><volumeref ref="CellLV"/>
        <position name="CellPos" unit="mm" x="{x}" y="{y}" z="{z}"/>
        <rotationref ref="AxisAlongY"/></physvol>
    </volume>
  </structure>
  <setup name="Default" version="1.0"><world ref="World"/></setup>
</gdml>
"""


EVENT_FIELDS = {"track_particle", "track_initial_energy", "step_energy", "step_volume"}

# Geant4 also uses the range cut as the recoil-production threshold for protons
# (~100 keV per mm), so keep that cut tiny while suppressing delta rays.
CUT_COMMANDS = [
    "/run/setCut 1 cm",
    "/run/setCutForAGivenParticle proton 0.001 mm",
]


def proton_light(e_mev):
    return 0.83 * e_mev - 2.82 * (1.0 - np.exp(-0.25 * e_mev ** 0.93))


def event_light(events):
    """Per event: total light, proton light (MeVee), proton recoils, recoils > 100 keV."""
    tracks = events.track
    in_det = tracks.step.volume == "Detector"
    edep = ak.sum(tracks.step.energy * in_det, axis=-1) / 1e3  # MeV per track
    particle = tracks.particle
    is_p = particle == "proton"
    is_e = (particle == "e-") | (particle == "e+") | (particle == "gamma")
    light_p = ak.sum(ak.where(is_p, proton_light(edep), 0.0), axis=1)
    light_e = ak.sum(ak.where(is_e, edep, 0.0), axis=1)
    light_ion = ak.sum(ak.where(is_p | is_e, 0.0, 0.017 * edep), axis=1)
    total = ak.to_numpy(light_p + light_e + light_ion)
    fast_p = is_p & (tracks.initial_energy > 100.0)  # keV
    return (total, ak.to_numpy(light_p), ak.to_numpy(ak.sum(is_p, axis=1)),
            ak.to_numpy(ak.sum(fast_p, axis=1)))


def smear(light, rng):
    safe = np.maximum(light, 1e-6)
    fwhm = safe * np.sqrt(RES_A ** 2 + RES_B ** 2 / safe + RES_C ** 2 / safe ** 2)
    return np.where(light > 0, light + rng.normal(0, 1, len(light)) * fwhm / 2.355, 0.0)


def counted(light_smeared, light_total, light_proton):
    """Above threshold and neutron-like (PSD stand-in: protons give most of the light)."""
    return (light_smeared > THRESHOLD_MEVEE) & (light_proton > 0.5 * light_total)
