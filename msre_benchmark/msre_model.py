"""Simplified OpenMC model of the MSRE first-criticality benchmark (June 1965).

Source of every number: M. Fratoni, D. Shen, G. Ilas, J. Powers, "Molten Salt
Reactor Experiment Benchmark Evaluation", OSTI 1617123 (2020), the open report
behind the IRPhEP benchmark. Section / table numbers below refer to it.
Lengths are the report's "hot" dimensions (638 C), in cm.

Simplifications (the report's Table 4.9 gives the k-eff effect of most):
* no flow distributor (Table 4.9: -98 pcm)
* sample basket homogenised (Table 4.9: -37 pcm), its INOR-8 content
  calibrated to the report's basket worth (see BASKET_INOR_SCALE)
* all control rods withdrawn, thimbles empty (OpenMC CSG, Frontiers 2024: +70 pcm)
* vessel heads flat, with the salt volumes of Table 2.1 preserved (effect not
  published; the height-preserving flat variant is +243 pcm)
* horizontal graphite support lattice homogenised (75 % graphite, 25 % salt)
* edge stringers simply cut by the r = 70.285 cm cylinder (Fig. 3.6)
* no centring bridge, insulation, thermal-shield base or lid
* thermal shield = 60 % steel balls + 40 % water by volume; no H2O S(a,b)
  data in this library, so hydrogen is free gas
"""

import math

import openmc

# --- Dimensions (Table 3.1, Sections 2.1, 3.1, 3.4) -------------------------
PITCH = 5.08339  # stringer width = lattice pitch
CH_W, CH_L, CH_R = 1.01770, 3.05309, 0.50885  # fuel channel, round ends
R_LATTICE = 70.285
R_CAN_IN, R_CAN_OUT = 71.097, 71.737
R_VES_IN, R_VES_OUT = 74.299, 76.862
R_ROD_CHANNEL = 3.1992  # channel holding each thimble / the basket
EXPAND = 1 + 7.8e-6 * (1180.4 - 68)  # INOR-8, 20 C -> 638 C (Section 3.1)
R_THIMBLE_OUT = 2.54 * EXPAND
R_THIMBLE_IN = R_THIMBLE_OUT - 0.1651 * EXPAND
R_BASKET = 5.4287 / 2
THIMBLES = [(PITCH, PITCH), (-PITCH, PITCH), (-PITCH, -PITCH)]
BASKET = (PITCH, -PITCH)

H_SUPPORT = 2 * 2.54  # two layers of horizontal graphite blocks
H_GRAPHITE = 166.724  # graphite core height (Table 4.4)
VESSEL_AREA = math.pi * R_VES_IN**2
H_LOWER = 0.283e6 / (0.908 * VESSEL_AREA)  # lower head salt 0.283 m3, 90.8 % salt
H_UPPER = 0.297e6 / VESSEL_AREA  # upper head salt 0.297 m3
T_WALL = 2.54 * EXPAND
Z_LATTICE_BOT = H_SUPPORT
Z_LATTICE_TOP = H_SUPPORT + H_GRAPHITE
Z_PLENUM_BOT = -H_LOWER
Z_PLENUM_TOP = Z_LATTICE_TOP + H_UPPER
# Flat heads that keep the vessel height instead (Section 2.1: 233.90 cm cold,
# taken as outside height); the extra height is split between the two plenums
H_EXTRA = 233.90 * EXPAND - 2 * T_WALL - (Z_PLENUM_TOP - Z_PLENUM_BOT)

# The basket's INOR-8 content is only estimated from Section 2.1, and k is very
# sensitive to it. Scaled so that replacing the basket by graphite is worth
# +1623 pcm as in the report's Table 4.9 (1.0x gave +1318, 1.26x +1792, 1.6x +2012).
BASKET_INOR_SCALE = 1.17

# Sensitivity switches; the defaults are the base model
BASE = dict(h_lower=H_LOWER, h_upper=H_UPPER, support_graphite=0.75, edge_closed=False,
            basket_graphite=False, basket_inor_scale=BASKET_INOR_SCALE)
VARIANTS = {
    "base": {},
    "heads_height": dict(h_lower=H_LOWER + H_EXTRA / 2, h_upper=H_UPPER + H_EXTRA / 2),
    "support_salt": dict(support_graphite=0.0),
    "edge_closed": dict(edge_closed=True),
    # report Table 4.9 variant: basket position solid graphite (+1623 pcm there)
    "basket_graphite": dict(basket_graphite=True),
    "basket_inor_1.0": dict(basket_inor_scale=1.0),
    "basket_inor_1.26": dict(basket_inor_scale=1.26),
    "basket_inor_1.6": dict(basket_inor_scale=1.6),
}

# Thermal shield (Section 2.1): ID 236.22, OD 317.5, height 383.54
R_SHIELD_IN, R_SHIELD_OUT, H_SHIELD = 118.11, 158.75, 383.54

T_CORE = 911.0  # K
T_SHIELD = 305.0
T_COLD = 293.6  # for comparing libraries that only have room-temperature data


def materials(graphite_temperature=T_CORE, support_graphite=0.75, basket_inor_scale=1.0):
    # Fuel salt (Table 4.6 nominal, Table 3.4 uranium, Table 3.5 impurities)
    salt = openmc.Material(name="fuel salt", temperature=T_CORE)
    li6_atom = 0.00005
    m_li6, m_li7 = 6.015122, 7.016003
    li6_wt = li6_atom * m_li6 / (li6_atom * m_li6 + (1 - li6_atom) * m_li7)
    salt.add_nuclide("Li6", 10.957 * li6_wt, "wo")
    salt.add_nuclide("Li7", 10.957 * (1 - li6_wt), "wo")
    salt.add_element("Be", 6.349, "wo")
    salt.add_element("Zr", 11.101, "wo")
    for nuc, wt in {"U234": 0.014, "U235": 1.409, "U236": 0.006, "U238": 3.065}.items():
        salt.add_nuclide(nuc, wt, "wo")
    for el, ppm in {"Fe": 162, "Cr": 28, "Ni": 30, "O": 490}.items():
        salt.add_element(el, ppm * 1e-4, "wo")
    salt.add_element("F", 100 - (10.957 + 6.349 + 11.101 + 4.494 + 0.071), "wo")
    salt.set_density("g/cm3", 2.3275)

    # Graphite (Section 3.2, Table 3.6): 1.86 g/cm3, 0.8 ppm boron
    graphite = openmc.Material(name="graphite", temperature=graphite_temperature)
    graphite.add_element("C", 1 - 8e-7, "wo")
    graphite.add_element("B", 8e-7, "wo")
    graphite.set_density("g/cm3", 1.86)
    plain_graphite = graphite.clone()  # mix_materials refuses S(a,b); added after mixing
    graphite.add_s_alpha_beta("c_Graphite")

    # INOR-8 nominal 68Ni-17Mo-7Cr-5Fe (+0.06 C), renormalised; 8.7745 g/cm3
    inor = openmc.Material(name="INOR-8", temperature=T_CORE)
    for el, wt in {"Ni": 68, "Mo": 17, "Cr": 7, "Fe": 5, "C": 0.06}.items():
        inor.add_element(el, wt, "wo")
    inor.set_density("g/cm3", 8.7745)

    def mix(name, parts, sab=None):
        m = openmc.Material.mix_materials([p for p, _ in parts], [f for _, f in parts], "vo",
                                          name=name)
        m.temperature = T_CORE
        if sab:
            m.add_s_alpha_beta(sab)
        return m

    lower_head = mix("lower head", [(salt, 0.908), (inor, 0.092)])
    support = mix("support lattice", [(plain_graphite, support_graphite),
                                          (salt, 1 - support_graphite)], "c_Graphite")
    support.temperature = graphite_temperature

    # Sample basket (Section 2.1): 12 INOR-8 rods 0.635 cm, 15 graphite bars
    # 0.635 x 1.1938 cm, ~2.6 cm2 of 0.079 cm plate; the rest salt
    area = math.pi * R_BASKET**2
    f_inor = basket_inor_scale * (12 * math.pi * 0.3175**2 + 2.6) / area
    f_graphite = 15 * 0.635 * 1.1938 / area
    basket = mix("sample basket", [(plain_graphite, f_graphite), (inor, f_inor),
                                   (salt, 1 - f_inor - f_graphite)], "c_Graphite")
    basket.temperature = graphite_temperature

    steel = openmc.Material(name="steel")
    steel.add_element("Fe", 98.8, "wo")
    steel.add_element("Mn", 1.0, "wo")
    steel.add_element("C", 0.2, "wo")
    steel.set_density("g/cm3", 7.85)
    water = openmc.Material(name="water")
    water.add_element("H", 2, "ao")
    water.add_element("O", 1, "ao")
    water.set_density("g/cm3", 0.995)
    shield = openmc.Material.mix_materials([steel, water], [0.6, 0.4], "vo",
                                           name="thermal shield")
    shield.temperature = T_SHIELD

    return dict(salt=salt, graphite=graphite, inor=inor, lower_head=lower_head,
                support=support, basket=basket, shield=shield)


def stringer_universe(m):
    """One graphite stringer with half fuel channels on its four faces."""
    half = PITCH / 2
    straight = CH_L / 2 - CH_R
    channels = None
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cx, cy = sx * half, sy * half
        if sx:  # channel on an x face: narrow in x, long in y
            box = (+openmc.XPlane(cx - CH_W / 2) & -openmc.XPlane(cx + CH_W / 2)
                   & +openmc.YPlane(-straight) & -openmc.YPlane(straight))
            ends = (-openmc.ZCylinder(x0=cx, y0=straight, r=CH_R)
                    | -openmc.ZCylinder(x0=cx, y0=-straight, r=CH_R))
        else:
            box = (+openmc.YPlane(cy - CH_W / 2) & -openmc.YPlane(cy + CH_W / 2)
                   & +openmc.XPlane(-straight) & -openmc.XPlane(straight))
            ends = (-openmc.ZCylinder(x0=straight, y0=cy, r=CH_R)
                    | -openmc.ZCylinder(x0=-straight, y0=cy, r=CH_R))
        region = box | ends
        channels = region if channels is None else channels | region
    fuel = openmc.Cell(name="channel", fill=m["salt"], region=channels)
    graphite = openmc.Cell(name="stringer", fill=m["graphite"], region=~channels)
    return openmc.Universe(cells=[fuel, graphite])


def build(graphite_temperature=T_CORE, thermal_shield=True, variant="base"):
    v = {**BASE, **VARIANTS[variant]}
    m = materials(graphite_temperature, v["support_graphite"], v["basket_inor_scale"])
    z_plenum_bot, z_plenum_top = -v["h_lower"], Z_LATTICE_TOP + v["h_upper"]

    # Graphite lattice, clipped later by the r = 70.285 cm cylinder
    n = int(math.ceil(R_LATTICE / PITCH)) + 1
    lattice = openmc.RectLattice(name="stringers")
    lattice.pitch = (PITCH, PITCH)
    lattice.lower_left = (-(n + 0.5) * PITCH, -(n + 0.5) * PITCH)
    stringer = stringer_universe(m)
    solid = openmc.Universe(cells=[openmc.Cell(fill=m["graphite"])])

    def cell_universe(i, j):
        # edge_closed: stringers cut by the lattice cylinder have no channels
        if v["edge_closed"]:
            far = math.hypot(abs(i - n) * PITCH + PITCH / 2, abs(j - n) * PITCH + PITCH / 2)
            if far > R_LATTICE:
                return solid
        return stringer

    lattice.universes = [[cell_universe(i, j) for i in range(2 * n + 1)]
                         for j in range(2 * n + 1)]
    lattice.outer = stringer

    z_lat_bot, z_lat_top = openmc.ZPlane(Z_LATTICE_BOT), openmc.ZPlane(Z_LATTICE_TOP)
    z0 = openmc.ZPlane(0.0)
    z_pl_bot, z_pl_top = openmc.ZPlane(z_plenum_bot), openmc.ZPlane(z_plenum_top)
    z_ves_bot = openmc.ZPlane(z_plenum_bot - T_WALL)
    z_ves_top = openmc.ZPlane(z_plenum_top + T_WALL)

    r_lat = openmc.ZCylinder(r=R_LATTICE)
    r_can_in, r_can_out = openmc.ZCylinder(r=R_CAN_IN), openmc.ZCylinder(r=R_CAN_OUT)
    r_ves_in, r_ves_out = openmc.ZCylinder(r=R_VES_IN), openmc.ZCylinder(r=R_VES_OUT)

    # Central channels: three thimbles (empty, rods withdrawn) and the basket
    channel_regions, cells = [], []
    for x, y in THIMBLES + [BASKET]:
        channel_regions.append(-openmc.ZCylinder(x0=x, y0=y, r=R_ROD_CHANNEL))
    in_channels = channel_regions[0] | channel_regions[1] | channel_regions[2] | channel_regions[3]
    core_z = +z_lat_bot & -z_lat_top

    for x, y in THIMBLES:
        t_out = openmc.ZCylinder(x0=x, y0=y, r=R_THIMBLE_OUT)
        t_in = openmc.ZCylinder(x0=x, y0=y, r=R_THIMBLE_IN)
        ch = openmc.ZCylinder(x0=x, y0=y, r=R_ROD_CHANNEL)
        cells += [
            openmc.Cell(name="thimble inside", region=-t_in & core_z),
            openmc.Cell(name="thimble wall", fill=m["inor"], region=+t_in & -t_out & core_z),
            openmc.Cell(name="channel salt", fill=m["salt"], region=+t_out & -ch & core_z),
        ]
    bx, by = BASKET
    b_out = openmc.ZCylinder(x0=bx, y0=by, r=R_BASKET)
    b_ch = openmc.ZCylinder(x0=bx, y0=by, r=R_ROD_CHANNEL)
    if v["basket_graphite"]:
        cells.append(openmc.Cell(name="basket graphite", fill=m["graphite"],
                                 region=-b_ch & core_z))
    else:
        cells += [
            openmc.Cell(name="basket", fill=m["basket"], region=-b_out & core_z),
            openmc.Cell(name="basket salt", fill=m["salt"], region=+b_out & -b_ch & core_z),
        ]

    cells += [
        openmc.Cell(name="graphite lattice", fill=lattice,
                    region=-r_lat & core_z & ~in_channels),
        openmc.Cell(name="lattice edge salt", fill=m["salt"],
                    region=+r_lat & -r_can_in & core_z),
        openmc.Cell(name="support lattice", fill=m["support"],
                    region=-r_can_in & +z0 & -z_lat_bot),
        openmc.Cell(name="core can", fill=m["inor"],
                    region=+r_can_in & -r_can_out & +z0 & -z_lat_top),
        openmc.Cell(name="downcomer", fill=m["salt"],
                    region=+r_can_out & -r_ves_in & +z0 & -z_lat_top),
        openmc.Cell(name="lower head", fill=m["lower_head"],
                    region=-r_ves_in & +z_pl_bot & -z0),
        openmc.Cell(name="upper head", fill=m["salt"],
                    region=-r_ves_in & +z_lat_top & -z_pl_top),
        openmc.Cell(name="vessel", fill=m["inor"],
                    region=(-r_ves_out & +z_ves_bot & -z_ves_top)
                    & ~(-r_ves_in & +z_pl_bot & -z_pl_top)),
    ]

    # Thermal shield, centred on the vessel; void between them (insulation)
    z_mid = 0.5 * (z_plenum_bot + z_plenum_top)
    s_bot, s_top = openmc.ZPlane(z_mid - H_SHIELD / 2), openmc.ZPlane(z_mid + H_SHIELD / 2)
    s_in, s_out = openmc.ZCylinder(r=R_SHIELD_IN), openmc.ZCylinder(r=R_SHIELD_OUT)
    boundary = openmc.Sphere(z0=z_mid, r=260.0, boundary_type="vacuum")
    shield_region = +s_in & -s_out & +s_bot & -s_top
    vessel_outer = -r_ves_out & +z_ves_bot & -z_ves_top
    void = -boundary & ~vessel_outer
    if thermal_shield:
        cells.append(openmc.Cell(name="thermal shield", fill=m["shield"], region=shield_region))
        void = void & ~shield_region
    # else: report Table 4.9 variant with no thermal shield or insulation
    cells.append(openmc.Cell(name="cell void", region=void))

    geometry = openmc.Geometry(cells)
    return geometry, m


def model(graphite_temperature=T_CORE, particles=20_000, batches=150, inactive=30,
          thermal_shield=True, variant="base", cold=False):
    geometry, m = build(graphite_temperature, thermal_shield, variant)
    if cold:  # every material at room temperature, nearest data within 5 K
        for mat in m.values():
            mat.temperature = T_COLD
    settings = openmc.Settings()
    settings.particles = particles
    settings.batches = batches
    settings.inactive = inactive
    settings.temperature = (
        {"method": "nearest", "tolerance": 5.0, "default": T_COLD} if cold
        else {"method": "interpolation", "default": T_CORE})
    z_mid = 0.5 * (Z_LATTICE_BOT + Z_LATTICE_TOP)
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box((-50, -50, z_mid - 60), (50, 50, z_mid + 60)),
        constraints={"fissionable": True})
    used = {c.fill for c in geometry.get_all_material_cells().values()}
    return openmc.Model(geometry, openmc.Materials([x for x in m.values() if x in used]),
                        settings)
