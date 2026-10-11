"""Shared geometry for the TX optics subassembly (workstream T, DevLog-008). Pure Python (no bpy).

Used by make_pic_texture.py (Pillow rasterizer) and build_optics_tx.py (Blender) so that pads, ports and bond
landing points on the 3D model match the PIC layout texture exactly.

Units: mm. Subassembly frame (DevLog-005 Sec. 3): origin = centre of the 2x4 eye-plate outer surface (dome bases),
+X = optical axis out of the eye, +Z up. PIC face normal = +X, PIC optical edge faces +Y.
PIC layout coordinates: u along the optical edge (= subassembly Z), v = depth from the optical edge
(subassembly Y = Y_EDGE - v). Grades: see research/tx_optics_refs.md (A spec, B vendor/paper, C photo, D estimate).
"""
import math

# ------------------------------------------------------------------ materials (1310 nm)
N_FS = 1.447        # B fused silica (Malitson Sellmeier) - collimator block, staircase prism
N_BK7 = 1.504       # B N-BK7 class (Schott Sellmeier) - rhomboids, eye plate
LAM = 1.311e-3      # mm, DR8 1311 nm (research)

# ------------------------------------------------------------------ PIC die
PIC_W = 7.0         # B along the optical edge (Credo Carmel8 7 x 7.5)
PIC_L = 7.5         # B depth
PIC_T = 0.75        # D (0.15-0.78)
HALF_W = PIC_W / 2
LANE_PITCH = 0.78   # D 8 MZMs across the die
LANE_U = [(k - 3.5) * LANE_PITCH for k in range(8)]
EDGE_PITCH = 0.25   # B FAU / MT pitch 250 um
OUT_U = [(i - 3.5) * EDGE_PITCH for i in range(8)]       # 1x8 output SSCs (lane k -> output k)
LASER_U = [-2.5, 2.5]                                    # C PSM8 photo laser sites at +/-2.5 mm
LOOP_U = [(-1.375, -1.125), (1.125, 1.375)]              # D loopback alignment SSC pairs (on the 0.25 grid, lenslets 1 and 12)
SSC_LEN = 0.25      # D inverse-taper SSC length
MZM_LEN = 2.4       # B (low) TW electrode length (papers 2-2.5)
V_RF_PAD = (7.20, 7.38)
V_EL = (4.15, 6.55)         # TW electrode span; RF enters at v=6.55 (back), light in the arms travels -v
V_TERM = (4.01, 4.09)       # on-chip termination resistors
V_HTR = (3.58, 4.00)        # thermal phase shifters on both arms
V_MMI_OUT = (3.46, 3.50)
V_MPD = (3.20, 3.30)
V_FAN = (0.70, 3.10)        # output fan-in S-bends
V_LEVELS0, V_LEVEL_STEP = 2.45, 0.042   # M2 horizontal DC buses (12 per side)
RF_PAD_PITCH = 0.27
RF_PAD_U = [-3.24 + RF_PAD_PITCH * j for j in range(25)]   # G S S G S S G ... (9 G shared + 16 S)
RF_PAD_KIND = ["G" if j % 3 == 0 else "S" for j in range(25)]
DC_PAD_V = [3.25 + 0.17 * j for j in range(14)]           # per side: 12 lane nets + GND + MPD bias
DC_PAD_U = 3.38                                           # pad centre |u|
LASER_CH_U = 3.04                                         # laser feed waveguide channel |u|
ARM_DU = 0.04                                             # arm half separation (under the S lines)
EL = dict(G=(0.10, 0.20), S=(0.015, 0.065))               # |u - u_k| spans of G and S lines


def lane_nets(k):
    """DC nets of lane k: (name, kind). Order defines bus level and pad index."""
    return [("HTR%da" % (k + 1), "htr"), ("HTR%db" % (k + 1), "htr"), ("MPD%d" % (k + 1), "mpd")]


def side_of(k):
    return -1 if k < 4 else 1


def dc_assign():
    """Per side: list of (net, lane, kind, m) with m = bus level / pad index (0 = lowest v, outermost run)."""
    out = {-1: [], 1: []}
    for s in (-1, 1):
        lanes = [0, 1, 2, 3] if s < 0 else [7, 6, 5, 4]          # outer lane first -> lowest level
        m = 0
        for k in lanes:
            for net, kind in lane_nets(k):
                out[s].append(dict(net=net, lane=k, kind=kind, m=m))
                m += 1
        out[s].append(dict(net="GND_%s" % ("L" if s < 0 else "R"), lane=None, kind="gnd", m=m))
        out[s].append(dict(net="MPD_BIAS_%s" % ("L" if s < 0 else "R"), lane=None, kind="bias", m=m + 1))
    return out


def run_u(s, m):
    """Side vertical run column for bus m (outermost = lowest pad)."""
    return s * (3.29 - 0.014 * m)


def level_v(m):
    return V_LEVELS0 + V_LEVEL_STEP * m


def drop_u(k, kind_idx):
    """Device drop column for net kind_idx (0 HTRa, 1 HTRb, 2 MPD anode) of lane k."""
    return LANE_U[k] + (-0.13, 0.13, 0.25)[kind_idx]


def sbend(p0, p1, n=24):
    """Cosine S-bend from p0 to p1 (u, v), monotonic in v."""
    (u0, v0), (u1, v1) = p0, p1
    return [(u0 + (u1 - u0) * (1 - math.cos(math.pi * t)) / 2, v0 + (v1 - v0) * t) for t in [i / n for i in range(n + 1)]]


def arc(c, r, a0, a1, n=24):
    return [(c[0] + r * math.cos(a0 + (a1 - a0) * i / n), c[1] + r * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


def waveguides():
    """List of polylines (u, v) for Si waveguides; plus MMI rects and SSC tapers."""
    wg, mmi, ssc = [], [], []
    v_tree = (7.03, 6.93, 6.80, 6.72, 6.66, 6.62)
    for g, s in enumerate((-1, 1)):
        ul = LASER_U[g]
        uc = sum(LANE_U[4 * g:4 * g + 4]) / 4                      # tree root
        # laser SSC -> outer channel -> back of the die -> root
        p = sbend((ul, SSC_LEN), (s * LASER_CH_U, 0.95))
        p += [(s * LASER_CH_U, 6.80)]
        p += arc((s * (LASER_CH_U - 0.22), 6.80), 0.22, 0 if s > 0 else math.pi, math.pi / 2 if s > 0 else math.pi / 2, 12)[1:]
        p += [(uc + s * 0.15, v_tree[0] - 0.0), ]
        p += arc((uc + s * 0.15, v_tree[0] - 0.08), 0.08, math.pi / 2, math.pi / 2 + s * math.pi / 2, 8)[1:]
        wg.append(p)
        u_root = uc + s * 0.15 - s * 0.08
        mmi.append((u_root - 0.02, v_tree[1] - 0.04, u_root + 0.02, v_tree[1]))
        wg.append([(u_root, v_tree[0] - 0.08), (u_root, v_tree[1])])
        lv2 = [(LANE_U[4 * g] + LANE_U[4 * g + 1]) / 2, (LANE_U[4 * g + 2] + LANE_U[4 * g + 3]) / 2]
        for u2, pair in zip(lv2, ((4 * g, 4 * g + 1), (4 * g + 2, 4 * g + 3))):
            wg.append(sbend((u_root + (0.008 if u2 > u_root else -0.008), v_tree[1] - 0.04), (u2, v_tree[2])))
            mmi.append((u2 - 0.02, v_tree[3], u2 + 0.02, v_tree[2]))
            for k in pair:
                wg.append(sbend((u2 + (0.008 if LANE_U[k] > u2 else -0.008), v_tree[3]), (LANE_U[k], v_tree[4])))
    for k, uk in enumerate(LANE_U):
        mmi.append((uk - 0.02, V_EL[1] + 0.03, uk + 0.02, 6.66))                        # MZM input MMI
        for a in (-1, 1):
            arm = sbend((uk + a * 0.008, V_EL[1] + 0.03), (uk + a * ARM_DU, V_EL[1] - 0.01), 8)
            arm += [(uk + a * ARM_DU, V_HTR[0] - 0.02)]
            arm += sbend((uk + a * ARM_DU, V_HTR[0] - 0.02), (uk + a * 0.008, V_MMI_OUT[1]), 8)[1:]
            wg.append(arm)
        mmi.append((uk - 0.02, V_MMI_OUT[0], uk + 0.02, V_MMI_OUT[1]))
        out = [(uk, V_MMI_OUT[0]), (uk, V_FAN[1])] + sbend((uk, V_FAN[1]), (OUT_U[k], V_FAN[0]), 40)[1:]
        out += [(OUT_U[k], SSC_LEN)]
        wg.append(out)
        # 2 % monitor tap (D) -> Ge PD
        wg.append(sbend((uk + 0.006, 3.40), (uk + 0.14, V_MPD[1]), 10))
        ssc.append((OUT_U[k], 0.0, SSC_LEN))
    for ul in LASER_U:
        ssc.append((ul, 0.0, SSC_LEN))
    for (ua, ub) in LOOP_U:
        c = ((ua + ub) / 2, 0.50)
        r = abs(ub - ua) / 2
        wg.append([(ua, SSC_LEN), (ua, 0.50)] + arc(c, r, math.pi if ua < ub else 0, 0 if ua < ub else math.pi, 16)[1:] +
                  [(ub, SSC_LEN)])
        ssc.append((ua, 0.0, SSC_LEN))
        ssc.append((ub, 0.0, SSC_LEN))
    return wg, mmi, ssc


def rf_metal():
    """Polygons (list of (u, v)) of the GSSG electrodes + feeds + pads; termination resistor rects."""
    polys, res = [], []
    for k, uk in enumerate(LANE_U):
        spans = [(-EL["G"][1], -EL["G"][0], 3 * k), (-EL["S"][1], -EL["S"][0], 3 * k + 1),
                 (EL["S"][0], EL["S"][1], 3 * k + 2), (EL["G"][0], EL["G"][1], 3 * k + 3)]
        for (a, b, j) in spans:
            polys.append([(uk + a, V_EL[0]), (uk + b, V_EL[0]), (uk + b, V_EL[1]), (uk + a, V_EL[1])])
            pu = RF_PAD_U[j]
            w = 0.045 if RF_PAD_KIND[j] == "S" else 0.07
            polys.append([(uk + a, V_EL[1]), (uk + b, V_EL[1]), (pu + w, V_RF_PAD[0] - 0.12), (pu + w, V_RF_PAD[0]),
                          (pu - w, V_RF_PAD[0]), (pu - w, V_RF_PAD[0] - 0.12)])
        # termination: S1 -> G1 and S2 -> G2 (2 x 50 ohm, 100 ohm differential, D)
        res.append((uk - EL["G"][0] - 0.004, V_TERM[0], uk - EL["S"][1] + 0.004, V_TERM[1]))
        res.append((uk + EL["S"][1] - 0.004, V_TERM[0], uk + EL["G"][0] + 0.004, V_TERM[1]))
        # electrode extensions down to the termination
        for (a, b) in ((-EL["G"][1], -EL["G"][0]), (-EL["S"][1], -EL["S"][0]), (EL["S"][0], EL["S"][1]), (EL["G"][0], EL["G"][1])):
            polys.append([(uk + a, V_TERM[0] - 0.01), (uk + b, V_TERM[0] - 0.01), (uk + b, V_EL[0]), (uk + a, V_EL[0])])
    pads = [(u - 0.07, V_RF_PAD[0], u + 0.07, V_RF_PAD[1]) for u in RF_PAD_U]
    return polys, res, pads


def heaters():
    """Meander polylines over both arms of each lane (TiN, D)."""
    out = []
    for uk in LANE_U:
        for a in (-1, 1):
            u0 = uk + a * ARM_DU
            pts = []
            n = 14
            for i in range(n + 1):
                v = V_HTR[0] + (V_HTR[1] - V_HTR[0]) * i / n
                pts.append((u0 + (0.011 if i % 2 else -0.011), v))
            out.append(pts)
    return out


def dc_routing():
    """M1 verticals, M2 horizontals, vias, side pads, Ge PDs. Returns dict of lists."""
    m1, m2, vias, pads, ge = [], [], [], [], []
    asg = dc_assign()
    for s in (-1, 1):
        lanes = [0, 1, 2, 3] if s < 0 else [4, 5, 6, 7]
        for d in asg[s]:
            m = d["m"]
            pv = DC_PAD_V[m]
            pads.append((s * DC_PAD_U - 0.07, pv - 0.06, s * DC_PAD_U + 0.07, pv + 0.06, d["net"]))
            ur = run_u(s, m)
            vl = level_v(m)
            m1.append([(ur, vl), (ur, pv), (s * (DC_PAD_U - 0.07), pv)])     # side run up to the pad
            vias.append((ur, vl))
            if d["kind"] == "gnd":                                         # to the outer G rail of the outer lane
                uo = LANE_U[lanes[0] if s < 0 else lanes[-1]] + s * 0.17
                m2.append([(ur, vl), (uo, vl)])
                m1.append([(uo, vl), (uo, V_TERM[0] - 0.005)])
                vias += [(uo, vl), (uo, V_TERM[0] - 0.005)]
                continue
            if d["kind"] == "bias":                                        # MPD cathode bias bus across the side
                uend = LANE_U[lanes[-1] if s < 0 else lanes[0]] + 0.16
                m2.append([(ur, vl), (uend, vl)])
                for k in lanes:
                    uc = LANE_U[k] + 0.16
                    m1.append([(uc, V_MPD[0]), (uc, vl)])
                    vias.append((uc, vl))
                continue
            k = d["lane"]
            if d["kind"] == "htr":
                ki = 1 if d["net"].endswith("b") else 0
                ud = drop_u(k, ki)
                ua = LANE_U[k] + (-ARM_DU if ki == 0 else ARM_DU)
                m1.append([(ua, V_HTR[0]), (ud, V_HTR[0] - 0.03), (ud, vl)])
                ug = LANE_U[k] + (-EL["G"][0] - 0.02 if ki == 0 else EL["G"][0] + 0.02)
                m1.append([(ua, V_HTR[1]), (ug, V_HTR[1] + 0.006)])           # far end tied to the G rail (D)
                vias.append((ug, V_HTR[1] + 0.006))
            else:
                ud = drop_u(k, 2)
                m1.append([(LANE_U[k] + 0.20, (V_MPD[0] + V_MPD[1]) / 2), (ud, (V_MPD[0] + V_MPD[1]) / 2), (ud, vl)])
            vias.append((ud, vl))
            m2.append([(ud, vl), (ur, vl)])
    for uk in LANE_U:
        ge.append((uk + 0.12, V_MPD[0], uk + 0.20, V_MPD[1]))
    return dict(m1=m1, m2=m2, vias=vias, pads=pads, ge=ge)


def fiducials():
    """Crosses (u, v, arm), verniers, circles."""
    crosses = [(-3.2, 0.45, 0.08), (3.2, 0.45, 0.08), (-3.33, 6.15, 0.06), (3.33, 6.15, 0.06)]
    circles = [(-1.95, 0.62, 0.09), (1.95, 0.62, 0.09)]
    verniers = [(-2.05, 0.30), (1.85, 0.30)]
    return dict(crosses=crosses, circles=circles, verniers=verniers)


# ------------------------------------------------------------------ 3D stack (subassembly frame, mm)
X_W = -1.70                 # PIC top surface = waveguide plane (D; set by the output-optics stack below)
X_PIC0 = X_W - PIC_T
EPOXY = 0.02                # D die-attach / bench-attach bond line
X_CAR1 = X_PIC0 - EPOXY     # carrier cavity floor (PIC + bench sit in a cavity)
X_CTOP = X_W - 0.05         # carrier top surface outside the cavity: pads ~level with the PIC pads (short bonds)
CAR_T = 0.60                # D ceramic/laminate carrier
X_CAR0 = X_CAR1 - CAR_T
Y_EDGE = -2.25              # PIC optical edge (set so column 0 clears the staircase entrance)
Y_BACK = Y_EDGE - PIC_L


def pic_to_xyz(u, v, x=X_W):
    return (x, Y_EDGE - v, u)


# laser train along -Y into the PIC at u = +/-2.5 (Focuslight 18-01342 collimator, isolator, 18-01339 focuser)
LENS1 = dict(L=1.2, W=0.6, T=0.5, CA=0.5, R=0.83, EFL=0.34, WD=0.194, part="Focuslight 18-01342 class")   # A
LENS2 = dict(L=1.0, W=0.6, T=0.5, CA=0.5, R=1.10, EFL=0.45, WD=0.30, part="Focuslight 18-01339 class")    # B
ISO = dict(H=0.60, W=0.60, L=0.55, pol=0.10, hwp=0.10)                                                            # D
DFB = dict(L=1.0, W=0.30, T=0.10)          # B/D 1 mm cavity, 0.3 wide, 0.1 thick (research)
SUB = dict(L=1.40, W=0.80, T=0.25)         # D AlN chip-on-submount
TRAIN = {}
_y = Y_EDGE
TRAIN["lens2_flat"] = _y + LENS2["WD"]
TRAIN["lens2_apex"] = TRAIN["lens2_flat"] + LENS2["T"]
TRAIN["iso0"] = TRAIN["lens2_apex"] + 0.10
TRAIN["iso1"] = TRAIN["iso0"] + ISO["L"]
TRAIN["lens1_apex"] = TRAIN["iso1"] + 0.10
TRAIN["lens1_flat"] = TRAIN["lens1_apex"] + LENS1["T"]
TRAIN["dfb_facet"] = TRAIN["lens1_flat"] + LENS1["WD"]
TRAIN["dfb_back"] = TRAIN["dfb_facet"] + DFB["L"]
TRAIN["sub0"] = TRAIN["dfb_facet"] - 0.04
TRAIN["sub1"] = TRAIN["sub0"] + SUB["L"]
X_SUB1 = X_W - DFB["T"]                  # emitter ~at the die top (p-side up, D)
X_BENCH1 = X_SUB1 - SUB["T"]
BENCH = dict(y0=Y_EDGE + 0.03, y1=TRAIN["sub1"] + 0.30, z=3.40)

# output optics
COLL = dict(L=0.80, y0=Y_EDGE, x0=X_BENCH1 + 0.02, x1=X_W + 0.30, z=1.60, lid_l=0.60)   # FAU-attach collimator
COLS_Y = [(c - 1.5) * 0.75 for c in range(4)]          # D eye lenslet pitch 0.75 (hero)
ROWS_Z = [-0.375, 0.375]
STAIR = dict(y0=Y_EDGE + COLL["L"] + 0.03 + 0.10, xb=X_W - 0.15, xt=X_W + 0.20)
RH = dict(s=0.20, x0=X_W + 0.25)                       # rhomboid aperture and entrance plane
EYE = dict(t=1.20, M=3.5, a_out=0.35, a_in=0.11, W=3.2, H=1.7)   # Galilean eye plate


def lane_map():
    """lane k (0..7, TX1..TX8) -> output z, column, row, dz (row shift by rhomboid)."""
    out = []
    for k in range(8):
        z = OUT_U[k]
        if k < 4:
            c, r = k, 0
        else:
            c, r = k - 4, 1
        out.append(dict(lane=k, name="TX%d" % (k + 1), z_in=z, col=c, row=r, y_out=COLS_Y[c], z_out=ROWS_Z[r],
                        dz=ROWS_Z[r] - z))
    return out


def gaussian_design():
    """Paraxial Gaussian numbers for the output and laser trains (method in research/tx_optics_refs.md 3.2)."""
    w0 = 4.6e-3                                         # SSC mode (MFD 9.2 um), D
    L_red = COLL["L"] / N_FS
    zr0 = math.pi * w0 ** 2 / LAM
    w_c = w0 * math.sqrt(1 + (L_red / zr0) ** 2)
    R_coll = (N_FS - 1) * L_red
    zr_c = math.pi * w_c ** 2 / LAM
    # path from the collimator lens to the eye plate inner face (air-equivalent, worst lane)
    lm = lane_map()
    paths = []
    for d in lm:
        y_path = d["y_out"] - (Y_EDGE + COLL["L"])
        x_path = (-EYE["t"]) - X_W
        paths.append(y_path + x_path + abs(d["dz"]))
    p_max = max(paths)
    w_eye_in = w_c * math.sqrt(1 + (p_max / zr_c) ** 2)
    tn = EYE["t"] / N_BK7
    D1 = tn / (EYE["M"] - 1)
    R_in = (N_BK7 - 1) * D1
    R_out = (N_BK7 - 1) * (D1 + tn)
    w_out = EYE["M"] * w_c
    w_out_max = EYE["M"] * w_eye_in
    zr_out = math.pi * w_out ** 2 / LAM
    clip_in = math.exp(-2 * EYE["a_in"] ** 2 / w_eye_in ** 2)
    clip_out = math.exp(-2 * EYE["a_out"] ** 2 / (EYE["M"] * w_eye_in) ** 2)
    # laser train
    w_dfb = 1.0e-3                                      # D assumed (0.8-1.1 um from a 25-35 deg FWHM far field)
    w_coll_laser = LAM * LENS1["EFL"] / (math.pi * w_dfb)
    w_ssc_laser = w_dfb * LENS2["EFL"] / LENS1["EFL"]
    return dict(
        ssc_w0_mm=w0, coll_len_mm=COLL["L"], coll_lens_R_mm=R_coll, collimated_w_mm=w_c, collimated_zR_mm=zr_c,
        max_path_to_eye_mm=p_max, w_at_eye_inner_mm=w_eye_in, eye_t_mm=EYE["t"], eye_M=EYE["M"],
        eye_R_in_concave_mm=R_in, eye_R_out_convex_mm=R_out, eye_out_w_mm=w_out, eye_out_w_max_mm=w_out_max, eye_out_zR_mm=zr_out,
        eye_clip_inner=clip_in, eye_clip_outer=clip_out, tir_staircase_crit_deg=math.degrees(math.asin(1 / N_BK7)),
        tir_rhomboid_crit_deg=math.degrees(math.asin(1 / N_BK7)), dfb_w0_mm=w_dfb,
        laser_collimated_w_mm=w_coll_laser, laser_ssc_w_mm=w_ssc_laser,
        lens_sag_lens1_mm=LENS1["R"] - math.sqrt(LENS1["R"] ** 2 - (LENS1["CA"] / 2) ** 2),
        lens_sag_lens2_mm=LENS2["R"] - math.sqrt(LENS2["R"] ** 2 - (LENS2["CA"] / 2) ** 2),
    )
