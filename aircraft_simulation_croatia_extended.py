# aircraft_simulation_croatia.py
# North-up / East-right. Triangular corridor meshes.
# Weather comms persist ≥10 min; hard/soft zones for robust local rerouting.

import os, time, uuid, math, random, csv
from math import sqrt

# =================== Batch controls ===================
BATCH_MODE = os.getenv("BATCH_MODE", "0") == "1"
BATCH_SEEDS_STR = os.getenv("BATCH_SEEDS", "21-50")
BATCH_SCENARIOS = [s.strip() for s in os.getenv("BATCH_SCENARIOS", "no_comms,instant,delayed").split(",") if s.strip()]
RUN_TAG_DEFAULT = os.getenv("RUN_TAG", "croatia_6")
CSV_DIR_DEFAULT = os.getenv("CSV_DIR", "results_final")
ALERT_LATENCY_SIM_S_DEFAULT = float(os.getenv("ALERT_LATENCY_SIM_S", "300.0"))

ROW_MODE = os.getenv("ROW_MODE", "fixed").lower()
if ROW_MODE not in {"fixed", "sampled"}:
    raise ValueError(f"ROW_MODE must be 'fixed' or 'sampled', got {ROW_MODE}")

# -------- Headless toggle BEFORE pyplot --------
import matplotlib
NO_PLOT = os.getenv("NO_PLOT", "0") == "1"
if NO_PLOT:
    matplotlib.use("Agg")
    matplotlib.interactive(False)
else:
    matplotlib.interactive(True)

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib import patheffects as pe

# =================== Utils ===================
def parse_seed_set(spec: str):
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = a.strip(); b = b.strip()
            if a and b:
                out.update(range(int(a), int(b) + 1))
            elif a and not b:
                out.add(int(a))
            elif b and not a:
                out.add(int(b))
        else:
            out.add(int(part))
    return sorted(out)

def ci_halfwidth_mean(x, z=1.96):
    x = [float(v) for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"), n)
    mean = sum(x) / n
    s = (sum((v - mean) ** 2 for v in x) / (n - 1)) ** 0.5
    hw = z * s / sqrt(n)
    return (mean, hw, n)

def kpi1_target(mean_delay):
    if mean_delay != mean_delay:
        return float("nan")
    return max(0.5, abs(mean_delay) * 0.05)

def ensure_schema(csv_path, header_cols):
    import pandas as pd
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header_cols)
        return
    try:
        df = pd.read_csv(csv_path)
        if list(df.columns) != header_cols:
            ts = time.strftime("%Y%m%d-%H%M%S")
            backup = f"{csv_path}.bak_{ts}"
            os.replace(csv_path, backup)
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(header_cols)
    except Exception:
        ts = time.strftime("%Y%m%d-%H%M%S")
        backup = f"{csv_path}.bak_{ts}"
        os.replace(csv_path, backup)
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header_cols)

            
def isa_troposphere_rho(h_m: float) -> float:
    T0 = 288.15   # K
    P0 = 101325.0 # Pa
    L  = -0.0065  # K/m (tropospheric lapse rate)
    g  = 9.80665  # m/s²
    R  = 287.058  # J/(kg·K)
    h  = max(0.0, float(h_m))
    if h <= 11000.0:
        T = T0 + L*h
        # avoid division by zero when very close to tropopause
        P = P0 * (T/T0) ** (-g/(L*R))
        rho = P/(R*T)
        return float(rho)
    # simple clamp for >11km (not used here)
    return float(isa_troposphere_rho(11000.0))

# --- Map your altitude colors to representative VFR altitudes over islands (ft -> m) ---
# Close-in, reasonably spaced, typical low VFR: 1500/2500/3500 ft (~457/762/1067 m).
ALT_COLOR_TO_ALT_M = {
    'red':        1500.0 * 0.3048,
    'lightgreen': 2500.0 * 0.3048,
    'royalblue':  3500.0 * 0.3048,
}

# --- Simple parasitic drag proxy via CdA (C_D0 * S) per aircraft class (m²) ------------
# Keep it simple, order-of-magnitude only; we use this *only* for relative DOC. :contentReference[oaicite:3]{index=3}
MODEL_CDA_M2 = {
    "C172-class": 0.55,
    "PA-28-class": 0.52,
    "DA40-class": 0.48,
}

# --- Wind used ONLY for DOC/time outputs (not for animation) ---------------------------
# "north-easternly wind at 50 km/h" interpreted as blowing TOWARD the south-east (arrow to SE).
WIND_SPEED_KMH = 50.0
WIND_DIR_DEG   = 135.0  # 0°=East, 90°=South in our North-up/East-right (y increases to South)
import math as _math
WIND_VX_KMH = WIND_SPEED_KMH * _math.cos(_math.radians(WIND_DIR_DEG))  # +x East
WIND_VY_KMH = WIND_SPEED_KMH * _math.sin(_math.radians(WIND_DIR_DEG))  # +y South

# ====== CSV schemas ======
header_cols_results = [
    "run_id","scenario","seed",
    "aircraft","model","cruise (km/h)",
    "rerouted (weather) (yes/no)","VFR slowdown (yes/no)","VFR reacted to (altitudes)",
    "affected by weather (yes/no)","used jet stream (yes/no)","jet stream dir (E/W/none)",
    "initial flightpath (km)","final flightpath (km)","flightpath delta (km)",
    "initial flight time (min)","final airborne time (min)","airborne delta (min)","altitude",
    "coordination latency (min)"
]

header_cols_runs = [
    "run_id","scenario","seed","n_aircraft","n_weather_reroute","n_affected",
    "n_vfr","n_js_yes","js_e","js_w","weather_avoid_rate",
    "mean_delay_min","p95_delay_min","first_alert_min","last_reroute_min","reroute_span_min",
    "run_tag","alert_latency_sim_s"
]

# =================== One-run simulation ===================
def run_single(seed:int,
               scenario:str,
               run_tag:str = RUN_TAG_DEFAULT,
               csv_dir:str = CSV_DIR_DEFAULT,
               alert_latency_sim_s:float = ALERT_LATENCY_SIM_S_DEFAULT):

    SEED = int(seed)
    random.seed(SEED)

    SCENARIO = scenario.lower()
    if SCENARIO not in {"no_comms","instant","delayed"}:
        raise ValueError(f"Unknown SCENARIO={SCENARIO}")

    WEATHER_COMMS_ENABLED = SCENARIO in {"instant","delayed"}
    CLOUD_ALERT_MODE = "instant" if SCENARIO != "delayed" else "delayed"
    ALERT_LATENCY_SIM_S = float(alert_latency_sim_s)

    RUN_TAG = str(run_tag if run_tag else time.strftime("%Y%m%d-%H%M%S"))
    RUN_ID = f"{SCENARIO}_seed{SEED}_{RUN_TAG}_{uuid.uuid4().hex[:6]}"

    CSV_DIR = csv_dir
    os.makedirs(CSV_DIR, exist_ok=True)
    RUN_DIR = os.path.join(CSV_DIR, RUN_ID); os.makedirs(RUN_DIR, exist_ok=True)

    RUN_CSV = os.path.join(RUN_DIR, f"aircraft_report_{RUN_ID}.csv")
    CSV_MASTER = os.path.join(CSV_DIR, "master_results.csv")
    RUNS_MASTER = os.path.join(CSV_DIR, "master_runs.csv")

    ensure_schema(CSV_MASTER, header_cols_results)
    ensure_schema(RUNS_MASTER, header_cols_runs)

    # -------- Radio log --------
    COMM_LOG_PATH = os.path.join(RUN_DIR, f"comms_{RUN_ID}.txt")
    def log_radio(sim_min: float, msg: str):
        # why: persistent textual output of traffic comms
        line = f"[{sim_min:6.2f} min] {msg}"
        with open(COMM_LOG_PATH, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")
        print(line)

    # -------- Time mapping --------
    SIM_SEC_PER_FRAME = float(os.getenv("SIM_DT", "6.0"))
    ANIM_INTERVAL_MS = int(os.getenv("ANIM_INTERVAL_MS", "50"))
    DISPLAY_DT = float(os.getenv("DT_DISPLAY", "0.1"))

    # =================== Geometry ===================
    G = nx.DiGraph()
    positions = {}

    fig, ax = plt.subplots()
    plt.subplots_adjust(right=0.75)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 240)
    ax.set_ylim(240, 0)

    GEO_WAYPOINTS = {
        "RIJEKA": (45.21917814937265, 14.569623776754668),
        "PULA":   (44.89368944731554, 13.919768179802649),
        "ZADAR":  (44.10298517307362, 15.353993297350277),
        "SPLIT":  (43.53795874389773, 16.296065139190997),
        "BRAC":   (43.28826378413314, 16.678878341451515),
        "ANCONA": (43.62017568889805, 13.37043074386936),
        "LOSINJ":  (44.67564024790982, 14.376931308495495),
        "KORNATI": (43.96139448278266, 15.091246526920678),
    }
    WAYPOINT_ONLY = {"LOSINJ", "KORNATI"}

    lats = [lat for (lat,lon) in GEO_WAYPOINTS.values()]
    lons = [lon for (lat,lon) in GEO_WAYPOINTS.values()]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    lat_mid = 0.5*(lat_min+lat_max)
    cos_phi = math.cos(math.radians(lat_mid))
    MARGIN = 10.0
    avail = 240.0 - 2*MARGIN
    dx = (lon_max - lon_min) * cos_phi
    dy = (lat_max - lat_min)
    s = min(avail/(dx if dx>0 else 1.0), avail/(dy if dy>0 else 1.0))
    KM_PER_DEG_LAT = 111.0
    KM_PER_DEG_LON = KM_PER_DEG_LAT * cos_phi
    pixels_per_km_x = s / KM_PER_DEG_LON
    pixels_per_km_y = s / KM_PER_DEG_LAT
    pixels_per_km = 0.5*(pixels_per_km_x + pixels_per_km_y)

    def geo_to_xy(lat, lon):
        x = MARGIN + s * ((lon - lon_min)*cos_phi)
        y = 240.0 - (MARGIN + s * (lat - lat_min))
        return (x, y)

    def xy_to_geo(x, y):
        lon = ((x - MARGIN) / s) / cos_phi + lon_min
        lat = ((240.0 - y - MARGIN) / s) + lat_min
        return (lat, lon)

    for name,(lat,lon) in GEO_WAYPOINTS.items():
        x,y = geo_to_xy(lat, lon)
        positions[name] = (x,y)
        kind = "wp_only" if name in WAYPOINT_ONLY else "wp"
        G.add_node(name, pos=(x,y), kind=kind, corr=set())

    ROUTE_OPTIONS = [
        ["ANCONA","KORNATI","ZADAR","SPLIT"],
        ["ANCONA","PULA","RIJEKA"],
        ["BRAC","SPLIT","ZADAR","LOSINJ","PULA"],
        ["BRAC","SPLIT","ZADAR","RIJEKA"],
        ["BRAC","KORNATI","ANCONA"],
        ["BRAC","KORNATI","LOSINJ","PULA"],
    ]
    ROUTE_OPTIONS = ROUTE_OPTIONS + [list(reversed(r)) for r in ROUTE_OPTIONS]

    ROUTES = {
        "R1": ["BRAC","SPLIT","ZADAR","RIJEKA"],
        "R2": ["BRAC","SPLIT","ZADAR","LOSINJ","PULA"],
        "R3": ["BRAC","KORNATI","LOSINJ","PULA"],
        "R4": ["ANCONA","KORNATI","ZADAR"],
        "R5": ["ANCONA","PULA","RIJEKA"],
        "R6": ["BRAC","KORNATI","ANCONA"],
    }

    LATERAL_OFFSETS_KM = [-9.0, 0.0, +9.0]
    STEP_KM_ALONG      = 5.0
    EXTEND_KM          = 12.0

    def _unit(vx, vy):
        L = math.hypot(vx, vy)
        return (vx/L, vy/L) if L>0 else (0.0,0.0)
    def _left_normal(vx, vy): return (-vy, vx)
    def _interp_segment(p0, p1, step_px):
        x0,y0 = p0; x1,y1 = p1
        L = math.hypot(x1-x0, y1-y0)
        if L < 1e-9:
            yield (x0,y0); return
        n = max(1, int(round(L/step_px)))
        for k in range(n):
            t = k/n
            yield (x0+(x1-x0)*t, y0+(y1-y0)*t)
        yield (x1,y1)

    def add_node_if_absent(name, xy, corr_id=None, kind="mesh"):
        if name not in G:
            G.add_node(name, pos=xy, kind=kind, corr=set())
        if corr_id:
            G.nodes[name]["corr"].add(corr_id)

    def connect(a,b, corr_id=None):
        if a==b: return
        (x0,y0)=positions[a]; (x1,y1)=positions[b]
        dist_km = math.hypot((x1-x0),(y1-y0))/pixels_per_km
        if dist_km <= 0: return
        G.add_edge(a,b,weight=dist_km, corr=set())
        G.add_edge(b,a,weight=dist_km, corr=set())
        if corr_id:
            G.edges[a,b]["corr"].add(corr_id)
            G.edges[b,a]["corr"].add(corr_id)
            G.nodes[a]["corr"].add(corr_id)
            G.nodes[b]["corr"].add(corr_id)

    def _extend_endpoints(poly_nodes, extend_km):
        pts = [positions[n] for n in poly_nodes]
        if len(pts) < 2: return pts
        x0,y0 = pts[0]; x1,y1 = pts[1]
        vx,vy = x0-x1, y0-y1; ux,uy = _unit(vx,vy)
        pre = (x0 + ux*(extend_km*pixels_per_km), y0 + uy*(extend_km*pixels_per_km))
        xN1,yN1 = pts[-2]; xN,yN = pts[-1]
        vx2,vy2 = xN - xN1, yN - yN1; ux2,uy2 = _unit(vx2,vy2)
        post = (xN + ux2*(extend_km*pixels_per_km), yN + uy2*(extend_km*pixels_per_km))
        return [pre] + pts + [post]

    def _sign_for_side_towards_point(poly_nodes, ref_xy):
        pts = [positions[n] for n in poly_nodes]
        if len(pts) < 2: return +1
        mid = len(pts)//2
        p0, p1 = pts[mid-1], pts[mid]
        vx,vy = p1[0]-p0[0], p1[1]-p0[1]
        tx,ty = _unit(vx,vy)
        nxL,nyL = _left_normal(tx,ty)
        mx,my = 0.5*(p0[0]+p1[0]), 0.5*(p0[1]+p1[1])
        rx,ry = ref_xy[0]-mx, ref_xy[1]-my
        dot = rx*nxL + ry*nyL
        return +1 if dot >= 0 else -1

    def build_corridor(corr_id, poly_nodes, step_km, lateral_offsets, keep_side_point=None):
        center=[]
        ext_pts = _extend_endpoints(poly_nodes, EXTEND_KM)
        for i in range(len(ext_pts)-1):
            p0=ext_pts[i]; p1=ext_pts[i+1]
            for pt in _interp_segment(p0,p1, step_km*pixels_per_km):
                if not center or pt!=center[-1]:
                    center.append(pt)
        allowed_sign = None
        if keep_side_point is not None:
            allowed_sign = _sign_for_side_towards_point(poly_nodes, keep_side_point)

        rails=[]
        for idx,d_km in enumerate(lateral_offsets):
            if allowed_sign is not None and d_km != 0.0 and math.copysign(1.0, d_km) != allowed_sign:
                continue
            rail=[]
            for k,(cx,cy) in enumerate(center):
                if k+1 < len(center):
                    dx,dy = center[k+1][0]-cx, center[k+1][1]-cy
                else:
                    dx,dy = cx-center[k-1][0], cy-center[k-1][1]
                tx,ty = _unit(dx,dy)
                nxL,nyL = _left_normal(tx,ty)
                ox,oy = cx + nxL*(d_km*pixels_per_km), cy + nyL*(d_km*pixels_per_km)
                name=f"{corr_id}_rail{idx}_k{k}"
                add_node_if_absent(name,(ox,oy), corr_id=corr_id, kind="mesh")
                positions[name]=(ox,oy)
                rail.append(name)
            for a,b in zip(rail[:-1], rail[1:]):
                connect(a,b, corr_id=corr_id)
            rails.append(rail)

        if len(rails) >= 2:
            m = min(len(r) for r in rails)
            for k in range(m):
                for rL, rR in zip(rails[:-1], rails[1:]):
                    connect(rL[k], rR[k], corr_id=corr_id)
            for k in range(m-1):
                for rL, rR in zip(rails[:-1], rails[1:]):
                    connect(rL[k],   rR[k+1], corr_id=corr_id)
                    connect(rL[k+1], rR[k],   corr_id=corr_id)

        if center and rails:
            for wp in poly_nodes:
                wx, wy = positions[wp]
                k_near = min(range(len(center)), key=lambda k: (center[k][0]-wx)**2 + (center[k][1]-wy)**2)
                for r in rails:
                    if 0 <= k_near < len(r):
                        connect(wp, r[k_near], corr_id=corr_id)

    losinj_latlon = GEO_WAYPOINTS["LOSINJ"]
    losinj_xy = geo_to_xy(*losinj_latlon)
    positions["LOSINJ"] = losinj_xy
    for rid, poly in ROUTES.items():
        if rid == "R1":
            build_corridor(rid, poly, STEP_KM_ALONG, LATERAL_OFFSETS_KM, keep_side_point=losinj_xy)
        else:
            build_corridor(rid, poly, STEP_KM_ALONG, LATERAL_OFFSETS_KM, keep_side_point=None)

    centerline_edges = []
    for rid, poly in ROUTES.items():
        for a, b in zip(poly[:-1], poly[1:]):
            connect(a, b, corr_id=rid)
            centerline_edges.append((a, b)); centerline_edges.append((b, a))

    mesh_nodes = [n for n,d in G.nodes(data=True) if d.get("kind")=="mesh"]
    for wp in GEO_WAYPOINTS.keys():
        if G.degree(wp) == 0 and mesh_nodes:
            wx,wy = positions[wp]
            nearest = min(mesh_nodes, key=lambda m: (positions[m][0]-wx)**2 + (positions[m][1]-wy)**2)
            connect(wp, nearest)

    print(f"[mesh] nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    # -------- Weather/sensing --------
    cloud_speed_kmph = 17.0
    KMH_TO_PPF = (1.0/3600.0) * pixels_per_km * float(os.getenv("SIM_DT", "6.0"))
    cloud_pixels_per_frame = cloud_speed_kmph * KMH_TO_PPF

    CLOUD_RADIUS_KM = 7.0
    CLOUD_RADIUS_PX = CLOUD_RADIUS_KM * pixels_per_km

    SIGHT_AHEAD_KM = 12.0
    DETECT_ENTER_MARGIN_KM = 2.0
    DETECT_EXIT_MARGIN_KM  = 4.0
    CLOUD_CLEAR_FRAMES     = 14
    ENTER_R_PX = (CLOUD_RADIUS_KM + DETECT_ENTER_MARGIN_KM) * pixels_per_km
    EXIT_R_PX  = (CLOUD_RADIUS_KM + DETECT_EXIT_MARGIN_KM)  * pixels_per_km

    # reroute margin remains, but now becomes the soft zone
    SOFT_MARGIN_KM = 2.0
    SOFT_MARGIN_PX = SOFT_MARGIN_KM * pixels_per_km

    COMM_MARK_AHEAD_KM = 30.0

    def jet_factor(u,v): return 1.0
    def jet_use_dir_for_path(_): return ("no","none")

    def weight_with_pref(u,v,attrs, corr_pref=None, penalty_factor=20.0):
        base = attrs.get('weight')
        if base is None:
            (x1,y1) = positions[u]; (x2,y2) = positions[v]
            base = math.hypot(x2-x1, y2-y1)/pixels_per_km
        if corr_pref is not None:
            e_corr = attrs.get("corr", set())
            if corr_pref not in e_corr:
                base *= penalty_factor
        return float(base)

    def point_to_segment_distance(px,py,x1,y1,x2,y2):
        vx,vy = x2-x1, y2-y1; wx,wy = px-x1, py-y1
        seg2 = vx*vx+vy*vy
        if seg2==0: return math.hypot(px-x1, py-y1)
        t = max(0.0, min(1.0, (wx*vx+wy*vy)/seg2))
        qx,qy = x1+t*vx, y1+t*vy
        return math.hypot(px-qx, py-qy)

    def cloud_detected_in_front(u, v, start_frac, cx, cy, radius_px, sight_km):
        (x0, y0) = positions[u]; (x1, y1) = positions[v]
        dist_px = math.hypot(x1 - x0, y1 - y0)
        edge_km = dist_px / pixels_per_km
        if edge_km <= 1e-9:
            return False
        df = min(1.0 - start_frac, sight_km / edge_km)
        if df <= 1e-9:
            return False
        sx = x0 + (x1 - x0) * start_frac
        sy = y0 + (y1 - y0) * start_frac
        ex = x0 + (x1 - x0) * (start_frac + df)
        ey = y0 + (y1 - y0) * (start_frac + df)
        return point_to_segment_distance(cx, cy, sx, sy, ex, ey) <= radius_px

    def _choose_next_v(u, came_from):
        mesh_nodes_set = {n for n,d in G.nodes(data=True) if d.get("kind")=="mesh"}
        cand = [n for n in G.successors(u) if n in mesh_nodes_set] or list(G.successors(u))
        non_back = [n for n in cand if n != came_from]
        return random.choice(non_back or cand or [u])

    def _recalc_segment(u,v):
        ux,uy = positions[u]; vx,vy = positions[v]
        dx,dy = vx-ux, vy-uy
        L = math.hypot(dx,dy)
        return ux,uy,dx,dy,L

    def edge_fraction(u, v, x, y):
        x0,y0 = positions[u]; x1,y1 = positions[v]
        vx,vy = x1-x0, y1-y0
        seg2 = vx*vx + vy*vy
        if seg2 <= 0.0: return 0.0
        t = ((x-x0)*vx + (y-y0)*vy) / seg2
        return max(0.0, min(1.0, t))

    def pick_disturbed_edge_centerline():
        if not centerline_edges:
            raise RuntimeError("No centerline edges available.")
        u,v = random.choice(centerline_edges)
        return (u,v)
    disturbed_edge = pick_disturbed_edge_centerline()

    # -------- Initialize moving cloud --------
    cloud_u, cloud_v = disturbed_edge
    cloud_prev = None
    ux,uy = positions[cloud_u]; vx,vy = positions[cloud_v]
    seg_dx,seg_dy = vx-ux, vy-uy; seg_len = math.hypot(seg_dx,seg_dy)
    cloud_s = seg_len*0.5
    cx0, cy0 = (ux + seg_dx*(cloud_s/seg_len), uy + seg_dy*(cloud_s/seg_len)) if seg_len>0 else positions[cloud_u]

    # -------- Aircraft setup --------
    MODELS = [{"name":"C172-class","cruise_kmph":240.0},
              {"name":"PA-28-class","cruise_kmph":195.0},
              {"name":"DA40-class","cruise_kmph":275.0}]
    MODEL_CODE = {"C172-class":"C","PA-28-class":"P","DA40-class":"D"}
    WEATHER_SLOW_FACTOR = 0.70

    REROUTE_HALO_CLOUD = 'black'
    REROUTE_HALO_VFR   = 'mediumorchid'
    ALT_BLUE = 'royalblue'

    # ---- VFR proximity params ----
    VFR_PROX_KM = float(os.getenv("VFR_PROX_KM", "3.0"))
    VFR_HEADON_FACTOR = float(os.getenv("VFR_HEADON_FACTOR", "0.75"))
    VFR_TAIL_FACTOR   = float(os.getenv("VFR_TAIL_FACTOR", "0.90"))
    VFR_RELEASE_GRACE_FRAMES = 2

    def effective_ppf(u,v, cruise_kmph, slowed=False, vfr_factor=1.0):
        base_kmph = cruise_kmph * (WEATHER_SLOW_FACTOR if slowed else 1.0)
        return (base_kmph * (1.0/3600.0) * pixels_per_km * float(os.getenv("SIM_DT", "6.0"))) * float(vfr_factor)

    def interpolate_path(path_nodes, cruise_kmph, slow_edges=None, vfr_penalty=None):
        if slow_edges is None: slow_edges=set()
        if vfr_penalty is None: vfr_penalty={}
        coords=[]
        for j in range(len(path_nodes)-1):
            u,v = path_nodes[j], path_nodes[j+1]
            if u == v:  continue
            x0,y0 = positions[u]; x1,y1 = positions[v]
            dist_px = math.hypot(x1-x0, y1-y0)
            vfr = vfr_penalty.get((u,v),1.0)
            ppf = effective_ppf(u,v, cruise_kmph, slowed=((u,v) in slow_edges), vfr_factor=vfr)
            steps = max(int(dist_px/ppf),1)
            for t in range(steps):
                coords.append((x0+(x1-x0)*t/steps, y0+(y1-y0)*t/steps))
        coords.append(positions[path_nodes[-1]])
        return coords, None

    def interpolate_path_from(path_nodes, start_edge_idx, start_frac, cruise_kmph, slow_edges=None, vfr_penalty=None):
        if slow_edges is None: slow_edges=set()
        if vfr_penalty is None: vfr_penalty={}
        coords=[]
        u = path_nodes[start_edge_idx]; v = path_nodes[start_edge_idx+1]
        if u != v:
            x0,y0 = positions[u]; x1,y1 = positions[v]
            dist_px = math.hypot(x1-x0, y1-y0)
            vfr = vfr_penalty.get((u,v),1.0)
            ppf = effective_ppf(u,v, cruise_kmph, slowed=((u,v) in slow_edges), vfr_factor=vfr)
            steps = max(int(dist_px/ppf),1)
            start_frac = max(0.0, min(0.999999, float(start_frac)))
            xcur = x0+(x1-x0)*start_frac; ycur = y0+(y1-y0)*start_frac
            coords.append((xcur,ycur))
            start_idx_new = min(int(math.floor(start_frac*steps)), steps-1)
            for t in range(start_idx_new+1, steps):
                coords.append((x0+(x1-x0)*t/steps, y0+(y1-y0)*t/steps))
        for j in range(start_edge_idx+1, len(path_nodes)-1):
            u = path_nodes[j]; v = path_nodes[j+1]
            if u == v:  continue
            x0,y0 = positions[u]; x1,y1 = positions[v]
            dist_px = math.hypot(x1-x0, y1-y0)
            vfr = vfr_penalty.get((u,v),1.0)
            ppf = effective_ppf(u,v, cruise_kmph, slowed=((u,v) in slow_edges), vfr_factor=vfr)
            steps = max(int(dist_px/ppf),1)
            for t in range(steps):
                coords.append((x0+(x1-x0)*t/steps, y0+(y1-y0)*t/steps))
        coords.append(positions[path_nodes[-1]])
        return coords

    def compress_consecutive_duplicates(nodes):
        out = []
        for n in nodes:
            if not out or n != out[-1]:
                out.append(n)
        return out

    def remove_pingpong(nodes):
        out=[]
        for n in nodes:
            out.append(n)
            if len(out)>=3 and out[-1]==out[-3]:
                out.pop(-2)
        return out

    # ===== Initial routing =====
    N_AIRCRAFT = 24
    DEPARTURE_INTERVAL_SIM_S = 240.0
    frames_per_departure = max(1, int(round(DEPARTURE_INTERVAL_SIM_S/float(os.getenv("SIM_DT","6.0")))))
    seqs = [random.choice(ROUTE_OPTIONS)[:] for _ in range(N_AIRCRAFT)]

    start_end_pairs=[]
    side_index=[]; side_flag=[]
    for i, seq in enumerate(seqs):
        start_end_pairs.append((seq[0], seq[-1]))
        side_index.append(i)
        side_flag.append('L2R')
    aircraft_delays = [i*frames_per_departure for i in range(N_AIRCRAFT)]

    # -------- Alerts (edge segments + zones) --------
    def undirected(e): return frozenset(e)

    alert_segments = {}
    alert_payload = {}
    alert_segments = {}
    alert_payload = {}
    originator_until = {}

    # ≥ 10 min persistence
    ALERT_MIN_PERSIST_S = 10*60.0
    ALERT_HOLD_SIM_S = max(ALERT_MIN_PERSIST_S, 2*DEPARTURE_INTERVAL_SIM_S)

    first_alert_sim_sec = None
    weather_reroute_times_sim_s = []

    def ping_alert_segment(u, v, start_frac, now_sim_sec, cx, cy, r_px, who_callsign):
        nonlocal first_alert_sim_sec
        if not WEATHER_COMMS_ENABLED:
            return
        e_ud = undirected((u, v))
        t_act = now_sim_sec if CLOUD_ALERT_MODE=='instant' else now_sim_sec + float(ALERT_LATENCY_SIM_S)

        prev = alert_payload.get(e_ud)
        if prev is None or t_act < prev["t"]:
            alert_payload[e_ud] = {"t": t_act, "cx": cx, "cy": cy, "r_px": r_px, "who": who_callsign}

        expire_time = t_act + ALERT_HOLD_SIM_S
        originator_until[who_callsign] = max(originator_until.get(who_callsign, 0.0), expire_time)

        x0,y0 = positions[u]; x1,y1 = positions[v]
        edge_km = math.hypot(x1-x0, y1-y0)/pixels_per_km
        if edge_km <= 1e-9:
            return
        seg_len_km = COMM_MARK_AHEAD_KM
        t0 = max(0.0, float(start_frac))
        t1 = min(1.0, t0 + seg_len_km/edge_km)
        alert_segments.setdefault(e_ud, []).append({
            "t": t_act,
            "expire": expire_time,
            "u": u, "v": v,
            "t0": t0, "t1": t1,
            "cx": cx, "cy": cy, "r_px": r_px,
            "who": who_callsign
        })
        if first_alert_sim_sec is None:
            first_alert_sim_sec = t_act

    def active_edge_segments(now_sim_sec):
        out=[]
        for _eud, lst in alert_segments.items():
            for meta in lst:
                if now_sim_sec >= meta["t"] and now_sim_sec <= meta["expire"]:
                    out.append((meta["u"], meta["v"], meta["t0"], meta["t1"], meta["cx"], meta["cy"], meta["r_px"], meta["who"], meta["t"]))
        out.sort(key=lambda x: x[-1], reverse=True)  # newest first
        return out

    # Return hard & soft zones (centers are same, radii differ)
    def active_alert_zones2(now_sim_sec):
        if not WEATHER_COMMS_ENABLED:
            return ([], [])
        hard=[]; soft=[]
        for meta in alert_payload.values():
            t = meta["t"]
            if now_sim_sec >= t and (now_sim_sec - t) <= ALERT_HOLD_SIM_S:
                hard.append((meta["cx"], meta["cy"], meta["r_px"]))                          # must-avoid
                soft.append((meta["cx"], meta["cy"], meta["r_px"] + SOFT_MARGIN_PX))         # penalty zone
        return (hard, soft)

    # -------- Altitudes + VFR --------
    ALT_BLUE = 'royalblue'
    def make_alt_pool(n):
        base = ['red']*4 + ['lightgreen']*3 + [ALT_BLUE]*3
        if n < len(base):
            pool = random.sample(base, k=n)
        elif n > len(base):
            pool = base + [random.choice(['red','lightgreen',ALT_BLUE]) for _ in range(n-len(base))]
        else:
            pool = base[:]
        random.shuffle(pool)
        return pool

    alt_seq_by_side = {'L2R': make_alt_pool(len([s for s in side_flag if s=='L2R'])),
                       'R2L': make_alt_pool(len([s for s in side_flag if s=='R2L']))}

    # -------- Plot (mesh & nodes) --------
    pos = nx.get_node_attributes(G,'pos')
    nx.draw_networkx_edges(G, pos, edge_color='gainsboro', width=1.0, arrows=False, ax=ax)
    mesh_nodes_plot = [n for n,d in G.nodes(data=True) if d.get("kind")=="mesh"]
    if mesh_nodes_plot:
        nx.draw_networkx_nodes(G, pos, nodelist=mesh_nodes_plot, node_color='lightgray', node_size=10, ax=ax)

    airport_nodes = [n for n in GEO_WAYPOINTS if n not in WAYPOINT_ONLY]
    waypoint_only_nodes = list(WAYPOINT_ONLY)
    nx.draw_networkx_nodes(G, pos, nodelist=airport_nodes,
                           node_color='limegreen', node_size=220, linewidths=0.8, edgecolors='black', ax=ax)
    if waypoint_only_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=waypoint_only_nodes,
                               node_color='#FFA500', node_shape='s',
                               node_size=180, linewidths=0.8, edgecolors='black', ax=ax)

    for rid, poly in ROUTES.items():
        for a, b in zip(poly[:-1], poly[1:]):
            xa,ya = positions[a]; xb,yb = positions[b]
            ax.plot([xa,xb],[ya,yb], color='silver', linewidth=3, alpha=0.9, zorder=2)

    label_pos = {n: (positions[n][0], positions[n][1]-4) for n in G.nodes}
    nx.draw_networkx_labels(G, label_pos, labels={n:n for n in G.nodes if G.nodes[n].get("kind")!="mesh"},
                            font_size=7, ax=ax)

    # -------- Build aircraft --------
    aircraft_paths = []
    aircraft_dots = []
    reroute_halos_cloud = []
    reroute_halos_vfr = []
    slow_overlays = []
    reroute_lines_cloud = []
    type_labels = []

    MAX_ALERT_OVERLAYS = 12
    alert_overlays = []
    for _ in range(MAX_ALERT_OVERLAYS):
        ln, = ax.plot([], [], color='gold', linewidth=4, alpha=0.9)
        ln.set_animated(True)
        alert_overlays.append(ln)

    travel_times = [None] * len(start_end_pairs)
    initial_lengths = []
    initial_time_min_planned = []
    final_lengths = [None] * len(start_end_pairs)

    def weight_none_pref(u,v,attrs): return weight_with_pref(u,v,attrs,None)

    def shortest_path_via(seq_nodes):
        path=[]
        for a,b in zip(seq_nodes[:-1], seq_nodes[1:]):
            try:
                seg = nx.shortest_path(G, a, b, weight=weight_none_pref)
            except nx.NetworkXNoPath:
                connect(a, b)
                seg = nx.shortest_path(G, a, b, weight=weight_none_pref)
            if not path:
                path.extend(seg)
            else:
                path.extend(seg[1:])
        return compress_consecutive_duplicates(path)

    FL_CODE = {"red":"FLR", "lightgreen":"FLG", ALT_BLUE:"FLB"}

    for i in range(len(seqs)):
        model = random.choice(MODELS)
        multi_seq = seqs[i][:]
        path = shortest_path_via(multi_seq)

        coords, _ = interpolate_path(path, model["cruise_kmph"])

        total_km = 0.0
        for k in range(len(path)-1):
            u, v = path[k], path[k+1]
            total_km += (G.edges[u, v]['weight'] if G.has_edge(u,v)
                         else math.hypot(positions[v][0]-positions[u][0], positions[v][1]-positions[u][1])/pixels_per_km)
        initial_lengths.append(round(total_km, 2))

        plan_min = 0.0
        for k in range(len(path) - 1):
            u, v = path[k], path[k + 1]
            dist_km = (G.edges[u, v]['weight'] if G.has_edge(u,v)
                       else math.hypot(positions[v][0]-positions[u][0], positions[v][1]-positions[u][1])/pixels_per_km)
            v_eff_kmph = model["cruise_kmph"] * jet_factor(u, v)
            plan_min += (dist_km / v_eff_kmph) * 60.0
        initial_time_min_planned.append(plan_min)

        alt_color = alt_seq_by_side[side_flag[i]][side_index[i]]
        alt_m = ALT_COLOR_TO_ALT_M.get(alt_color, 2500.0*0.3048)
        rho_at_alt = isa_troposphere_rho(alt_m)
        cda_m2 = MODEL_CDA_M2.get(model["name"], 0.52)
        callsign = f"A{i+1:02d}{FL_CODE.get(alt_color,'FL?')}"

        for j in range(len(coords)-1):
            x0,y0 = coords[j]; x1,y1 = coords[j+1]
            ax.plot([x0,x1],[y0,y1], color=alt_color, linewidth=2, alpha=0.18)

        aircraft_paths.append({
            'original_path': path,
            'coords': coords,
            'dest': multi_seq[-1],
            'model_name': model["name"],
            'model_code': MODEL_CODE[model["name"]],
            'cruise_kmph': model["cruise_kmph"],
            'slowed_edges': set(),
            'vfr_penalty': {},
            'had_vfr': False,
            'vfr_active': False,
            'vfr_react_colors': set(),
            'had_reroute_cloud': False,
            'affected': False,
            'alt_color': alt_color,
            'cloud_seen': {},
            'ac_coord_latency_min': None,
            'reroute_time_min': None,
            'last_plan_frame': -10**9,
            'active_zone_sig': None,
            'lock_idx': None,
            'callsign': callsign,
            'broadcasted_edges': set(),
            'ack_logged': False,
            'vfr_current_edge': None,
            'vfr_active_until_frame': -1,
            'alt_m': alt_m,
            'rho_at_alt': rho_at_alt,
            'cda_m2': cda_m2,
        })

    # ---------- Artists ----------
    for ac in aircraft_paths:
        dot, = ax.plot([], [], marker='o', markersize=14, color=ac['alt_color'])
        dot.set_animated(True)
        aircraft_dots.append(dot)

        halo_cloud, = ax.plot([], [], marker='o', markersize=16, markerfacecolor='none',
                              markeredgecolor='black', markeredgewidth=2, alpha=0.95)
        halo_cloud.set_animated(True)
        reroute_halos_cloud.append(halo_cloud)

        halo_vfr, = ax.plot([], [], marker='o', markersize=18, markerfacecolor='none',
                            markeredgecolor='mediumorchid', markeredgewidth=2, alpha=0.95)
        halo_vfr.set_animated(True)
        reroute_halos_vfr.append(halo_vfr)

        slow_line, = ax.plot([], [], linewidth=3, alpha=0.80, color='gold')
        slow_line.set_animated(True)
        slow_overlays.append(slow_line)

        line_c, = ax.plot([], [], color='dimgray', linewidth=2, alpha=0.85)
        line_c.set_animated(True)
        reroute_lines_cloud.append(line_c)

        lbl = ax.text(0,0, ac['model_code'], ha='center', va='center',
                      fontsize=8, fontweight='bold', color='white', visible=False)
        lbl.set_path_effects([pe.withStroke(linewidth=2.0, foreground='black', alpha=0.85)])
        lbl.set_animated(True)
        type_labels.append(lbl)

    class TypeLetterHandler(HandlerBase):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height,
                           fontsize, trans):
            letter = orig_handle.get_text()
            txt = Text(x=xdescent + width / 2.0, y=ydescent + height / 2.0,
                       text=letter, ha='center', va='center',
                       fontsize=fontsize, color='white', fontweight='bold')
            txt.set_path_effects([pe.withStroke(linewidth=1.2, foreground='black', alpha=0.95)])
            txt.set_transform(trans)
            return [txt]

    legend_elements = [
        Line2D([0],[0], marker='o', color='w', label='Altitude: Red', markerfacecolor='red', markersize=6),
        Line2D([0],[0], marker='o', color='w', label='Altitude: Light Green', markerfacecolor='lightgreen', markersize=6),
        Line2D([0],[0], marker='o', color='w', label='Altitude: Blue', markerfacecolor=ALT_BLUE, markersize=6),
        Line2D([0],[0], marker='o', color='w', label='Reroute: Cloud',
               markerfacecolor='none', markeredgecolor='black', markeredgewidth=2, markersize=10),
        Line2D([0],[0], color='dimgray', lw=2, label='Rerouted path (cloud)'),
        Line2D([0],[0], marker='o', color='w', label='VFR Collision Avoidance',
               markerfacecolor='none', markeredgecolor='mediumorchid', markeredgewidth=2, markersize=12),
        Ellipse((0,0), width=18, height=12, facecolor='gray', edgecolor='black', alpha=0.4, label='Weather cell'),
        Line2D([0],[0], marker='s', color='w', label='Waypoint only',
               markerfacecolor='#FFA500', markeredgecolor='black', markersize=8),
    ]
    type_C = Text(0, 0, 'C', label='Cessna C172-class')
    type_P = Text(0, 0, 'P', label='Piper PA-28-class')
    type_D = Text(0, 0, 'D', label='Diamond DA40-class')
    legend_elements.extend([type_C, type_P, type_D])
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=True, borderaxespad=0., handletextpad=0.6, handler_map={Text: TypeLetterHandler()} )
    plt.title(f"Aircraft Simulation — {'No Weather Communication' if SCENARIO=='no_comms' else ('Instant' if SCENARIO=='instant' else 'Delayed')}")
    plt.axis("off")

    def _edge_len_km(u, v):
        if G.has_edge(u, v) and 'weight' in G.edges[u, v]:
            return float(G.edges[u, v]['weight'])
        (x1, y1), (x2, y2) = positions[u], positions[v]
        return float(math.hypot(x2 - x1, y2 - y1) / pixels_per_km)

    def _edge_unit_heading(u, v):
        (x1, y1), (x2, y2) = positions[u], positions[v]
        dx, dy = (x2 - x1), (y2 - y1)  # +x East, +y South (screen coords)
        L = math.hypot(dx, dy) or 1.0
        return (dx / L, dy / L)

    def _wind_along_kmh(u, v):
        ux, uy = _edge_unit_heading(u, v)
        return WIND_VX_KMH * ux + WIND_VY_KMH * uy

    cloud = plt.Circle((cx0,cy0), radius=CLOUD_RADIUS_PX, facecolor='gray',
                       edgecolor='black', lw=1.0, alpha=0.45)
    cloud.set_animated(True); ax.add_patch(cloud)
    sim_time_text = ax.text(0.02,0.95,'', transform=ax.transAxes); sim_time_text.set_animated(True)

    sim_done = False
    ani = None

    # ---- Zone pruning cache ----
    _cached_zone_sig = None
    _cached_G2 = None
    _edge_zone_hit_cache_hard = {}
    _edge_zone_hit_cache_soft = {}

    def _reset_zone_cache():
        nonlocal _edge_zone_hit_cache_hard, _edge_zone_hit_cache_soft
        _edge_zone_hit_cache_hard = {}
        _edge_zone_hit_cache_soft = {}

    def edge_hits_zones_cached(u, v, zones, cache):
        key = (u, v, tuple((round(zx,1), round(zy,1), round(zr,1)) for (zx,zy,zr) in zones))
        if key in cache:
            return cache[key]
        x0, y0 = positions[u]; x1, y1 = positions[v]
        hit = any(point_to_segment_distance(zx, zy, x0, y0, x1, y1) <= zr for (zx, zy, zr) in zones)
        cache[key] = hit
        return hit

    def _t_along(ax_, ay_, bx_, by_, px, py):
        vx, vy = bx_ - ax_, by_ - ay_
        L = math.hypot(vx, vy) or 1.0
        ux, uy = vx / L, vy / L
        return (px - ax_) * ux + (py - ay_) * uy

    def all_aircraft_done():
        return all(t is not None for t in travel_times)

    # -------- Update --------
    def update(frame):
        nonlocal sim_done, ani, cloud_u, cloud_v, cloud_prev, cloud_s, seg_dx, seg_dy, seg_len
        nonlocal _cached_zone_sig, _cached_G2

        real_t = frame*DISPLAY_DT
        sim_t = frame*float(os.getenv("SIM_DT","6.0"))
        sim_time_text.set_text(f"Real Time: {real_t:5.1f}s | Simulated Time: {sim_t/60.0:5.1f} min")

        # move cloud
        cloud_s += cloud_pixels_per_frame
        while cloud_s >= seg_len and seg_len > 0:
            overflow = cloud_s - seg_len
            cloud_prev = cloud_u; cloud_u = cloud_v; cloud_v = _choose_next_v(cloud_u, cloud_prev)
            ux,uy,seg_dx,seg_dy,seg_len = _recalc_segment(cloud_u, cloud_v)
            cloud_s = overflow
        if seg_len > 0:
            ux,uy = positions[cloud_u]
            cx = ux + seg_dx*(cloud_s/seg_len); cy = uy + seg_dy*(cloud_s/seg_len)
            cloud.center = (cx,cy)
        else:
            cx,cy = cloud.center

        # zones
        hard_zones, soft_zones = active_alert_zones2(sim_t)
        zone_sig = (
            tuple(sorted((round(zx,1), round(zy,1), round(zr,1)) for (zx,zy,zr) in hard_zones)),
            tuple(sorted((round(zx,1), round(zy,1), round(zr,1)) for (zx,zy,zr) in soft_zones)),
        )
        seg_alerts = active_edge_segments(sim_t)

        # render latest alert edge-subsegments (visual)
        for ln in alert_overlays: ln.set_data([], [])
        for idx, (u,v,t0,t1, *_rest) in enumerate(seg_alerts[:MAX_ALERT_OVERLAYS]):
            x0,y0 = positions[u]; x1,y1 = positions[v]
            sx = x0 + (x1-x0)*t0; sy = y0 + (y1-y0)*t0
            ex = x0 + (x1-x0)*t1; ey = y0 + (y1-y0)*t1
            alert_overlays[idx].set_data([sx,ex],[sy,ey])

        # occupancy & current states (for VFR and path indexing)
        edge_progress = {}
        cur_pos = []
        cur_heading = []
        cur_edge = []
        for i, ac in enumerate(aircraft_paths):
            delay = aircraft_delays[i]
            if frame < delay:
                type_labels[i].set_visible(False)
                cur_pos.append(None); cur_heading.append(None); cur_edge.append(None)
                continue
            idx = frame - delay
            path = ac['original_path']
            if idx >= len(ac['coords']):
                cur_pos.append(None); cur_heading.append(None); cur_edge.append(None)
                continue
            x,y = ac['coords'][idx]
            aircraft_dots[i].set_data([x],[y]); type_labels[i].set_position((x,y)); type_labels[i].set_visible(True)
            total=0; found=False
            for j in range(len(path)-1):
                u,v = path[j], path[j+1]
                if u==v: continue
                x0,y0 = positions[u]; x1,y1 = positions[v]
                dist_px = math.hypot(x1-x0, y1-y0)
                vfr_here = ac['vfr_penalty'].get((u,v),1.0)
                ppf = effective_ppf(u,v, ac['cruise_kmph'], slowed=((u,v) in ac['slowed_edges']), vfr_factor=vfr_here)
                steps = max(int(dist_px/ppf),1)
                if total <= idx < total+steps:
                    frac = (idx-total)/float(steps)
                    edge_progress[(u,v, ac['alt_color'])] = edge_progress.get((u,v, ac['alt_color']), []) + [(i, frac)]
                    hx,hy = x1-x0, y1-y0
                    hL = math.hypot(hx,hy) or 1.0
                    cur_pos.append((x,y))
                    cur_heading.append((hx/hL, hy/hL))
                    cur_edge.append((j,u,v,frac))
                    found=True
                    break
                total += steps
            if not found:
                cur_pos.append(None); cur_heading.append(None); cur_edge.append(None)

        # --------- VFR proximity (distance-based) ---------
        prox_px = VFR_PROX_KM * pixels_per_km
        for i, ac in enumerate(aircraft_paths):
            if cur_pos[i] is None: 
                continue
            ac['vfr_active'] = False
            xi, yi = cur_pos[i]; hi = cur_heading[i]; cei = cur_edge[i]
            if cei is None: 
                continue
            _, u_i, v_i, frac_i = cei
            for j, bc in enumerate(aircraft_paths):
                if i == j or bc['alt_color'] != ac['alt_color']:
                    continue
                pj = cur_pos[j]
                if pj is None: 
                    continue
                xj,yj = pj
                dpx = math.hypot(xj - xi, yj - yi)
                if dpx > prox_px:
                    continue
                dot = hi[0]*cur_heading[j][0] + hi[1]*cur_heading[j][1]
                factor = VFR_HEADON_FACTOR if dot < 0 else VFR_TAIL_FACTOR
                cur = ac['vfr_penalty'].get((u_i, v_i), 1.0)
                if factor < cur:
                    ac['vfr_penalty'][(u_i, v_i)] = factor
                    ac['had_vfr'] = True
                    ac['vfr_react_colors'].add(ac['alt_color'])
                    path = ac['original_path']; cruise = ac['cruise_kmph']
                    tail = interpolate_path_from(path, cur_edge[i][0], frac_i, cruise,
                                                 slow_edges=ac['slowed_edges'], vfr_penalty=ac['vfr_penalty'])
                    idx_now = frame - aircraft_delays[i]
                    ac['coords'] = ac['coords'][:idx_now] + tail
                ac['vfr_active'] = True
                ac['vfr_current_edge'] = (u_i, v_i)
                ac['vfr_active_until_frame'] = frame + VFR_RELEASE_GRACE_FRAMES
                break

        # VFR halos + cloud slow/broadcast (leaders)
        for i, ac in enumerate(aircraft_paths):
            delay = aircraft_delays[i]
            if frame < delay: 
                continue
            idx = frame - delay
            path = ac['original_path']; cruise = ac['cruise_kmph']
            if idx >= len(ac['coords']): 
                continue
            x,y = ac['coords'][idx]
            if ac['vfr_active'] or frame < ac['vfr_active_until_frame']:
                reroute_halos_vfr[i].set_data([x],[y])
            else:
                reroute_halos_vfr[i].set_data([], [])

            total=0
            for j in range(len(path)-1):
                u = path[j]; v = path[j+1]
                if u==v:  continue
                x0,y0 = positions[u]; x1,y1 = positions[v]
                dist_px = math.hypot(x1-x0, y1-y0)
                is_slow = (u,v) in ac['slowed_edges']
                vfr_here = ac['vfr_penalty'].get((u,v),1.0)
                ppf = effective_ppf(u,v, cruise, slowed=is_slow, vfr_factor=vfr_here)
                steps = max(int(dist_px/ppf),1)
                inside = (total < idx < total+steps)
                at_start = (idx == total)
                if inside or at_start:
                    start_frac = edge_fraction(u, v, x, y)
                    detected_enter = cloud_detected_in_front(u,v, start_frac, cx,cy, ENTER_R_PX, SIGHT_AHEAD_KM)
                    if detected_enter and (u,v) not in ac['slowed_edges']:
                        ac['slowed_edges'].add((u,v)); ac['affected']=True
                        ping_alert_segment(u, v, start_frac, sim_t, cx, cy, CLOUD_RADIUS_PX, ac['callsign'])
                        if (u,v) not in ac['broadcasted_edges']:
                            lat, lon = xy_to_geo(cx, cy)
                            log_radio(sim_t/60.0, f"{ac['callsign']}: encountered weather disturbance at {lat:.5f}, {lon:.5f}. Avoid.")
                            ac['broadcasted_edges'].add((u,v))
                        tail = interpolate_path_from(path, j, start_frac, cruise,
                                                     slow_edges=ac['slowed_edges'], vfr_penalty=ac['vfr_penalty'])
                        ac['coords'] = ac['coords'][:idx] + tail
                    if (u,v) in ac['slowed_edges']:
                        # hysteresis clear
                        still_see = cloud_detected_in_front(u,v, start_frac, cx,cy, EXIT_R_PX, SIGHT_AHEAD_KM)
                        last_seen = ac['cloud_seen'].get((u,v), -10**9)
                        if (not still_see) and (frame - last_seen) >= CLOUD_CLEAR_FRAMES:
                            ac['slowed_edges'].discard((u,v))
                            tail = interpolate_path_from(path, j, start_frac, cruise,
                                                         slow_edges=ac['slowed_edges'], vfr_penalty=ac['vfr_penalty'])
                            ac['coords'] = ac['coords'][:idx] + tail
                    if (u,v) in ac['slowed_edges']:
                        slow_overlays[i].set_data([x0,x1],[y0,y1]); aircraft_dots[i].set_markersize(16)
                    else:
                        slow_overlays[i].set_data([],[]); aircraft_dots[i].set_markersize(14)
                    break
                total += steps

        # --------- Rerouting (block-based with hard/soft zones) ---------
        if hard_zones or soft_zones or seg_alerts:
            BUFFER_KM   = 15.0
            FORWARD_KM  = 35.0
            BUFFER_PX   = BUFFER_KM * pixels_per_km
            FORWARD_PX  = FORWARD_KM * pixels_per_km

            # Longer cooldown to reduce thrashing under bursty alerts
            PLAN_COOLDOWN_FRAMES = 24

            def edge_hits_hard(u, v):
                return edge_hits_zones_cached(u, v, hard_zones, _edge_zone_hit_cache_hard)

            def edge_hits_soft(u, v):
                return edge_hits_zones_cached(u, v, soft_zones, _edge_zone_hit_cache_soft)

            def soft_penalty_km(u, v):
                # why: allow slight touch; penalize proportionally inside soft zone
                x0,y0 = positions[u]; x1,y1 = positions[v]
                pen = 0.0
                for (zx,zy,zr) in soft_zones:
                    d = point_to_segment_distance(zx, zy, x0, y0, x1, y1)
                    if d <= zr:
                        # linear penalty up to ~3x segment length fraction (tunable)
                        overlap = (zr - d) / max(zr, 1e-6)
                        seg_km = math.hypot(x1-x0, y1-y0)/pixels_per_km
                        pen += seg_km * (1.5 * overlap)
                return pen

            # Rebuild pruned graph on hard zones only
            if zone_sig != _cached_zone_sig:
                _reset_zone_cache()
                G2 = G.copy()
                to_remove = [(a, b) for (a, b, attrs) in G2.edges(data=True) if edge_hits_hard(a, b)]
                G2.remove_edges_from(to_remove)
                _cached_G2 = G2
                _cached_zone_sig = zone_sig
            else:
                G2 = _cached_G2

            ANGLE_PENALTY_KM = 25.0

             # Freeze sim time for checks
            now_sim_sec = sim_t

            for i, ac in enumerate(aircraft_paths):
                delay = aircraft_delays[i]
                if frame < delay:
                    continue
 
        #leader immunity — if this aircraft originated any still-active alert, do NOT reroute it.
        #It will still experience slowdown via 'slowed_edges'.
                if originator_until.get(ac['callsign'], 0.0) > now_sim_sec:
            # keep flying through with reduced speed; skip the reroute branch entirely
                    continue
 
                idx_now = frame - delay
                coords = ac['coords']
                if idx_now >= len(coords):
                    continue

                path = ac['original_path']; cruise = ac['cruise_kmph']

                # locate current edge & pos
                total=0; j_cur=None; xcur=ycur=None
                for j in range(len(path)-1):
                    u,v = path[j], path[j+1]
                    if u==v: continue
                    x0,y0 = positions[u]; x1,y1 = positions[v]
                    dist_px = math.hypot(x1-x0, y1-y0)
                    vfr = ac['vfr_penalty'].get((u,v),1.0)
                    ppf = effective_ppf(u,v, cruise, slowed=((u,v) in ac['slowed_edges']), vfr_factor=vfr)
                    steps = max(int(dist_px/ppf),1)
                    if total <= idx_now < total+steps:
                        j_cur=j; xcur,ycur = coords[idx_now]; break
                    total += steps
                if j_cur is None:
                    continue

                cur_u, cur_v = path[j_cur], path[j_cur+1]
                hx, hy = positions[cur_v][0]-positions[cur_u][0], positions[cur_v][1]-positions[cur_u][1]
                hL = math.hypot(hx,hy) or 1.0
                hx, hy = hx/hL, hy/hL

                # corridor preference (stay on same corridor mesh)
                corr_inter = G.nodes[cur_u].get("corr", set()).intersection(G.nodes[cur_v].get("corr", set()))
                corr_pref = next(iter(corr_inter)) if corr_inter else None

                # Stability gate: only replan if forward path hits a HARD zone
                forward_hits_hard = any(edge_hits_hard(path[k], path[k+1]) for k in range(j_cur, len(path)-1))
                if not forward_hits_hard:
                    continue
                if ac['active_zone_sig'] == zone_sig and (frame - ac['last_plan_frame'] < PLAN_COOLDOWN_FRAMES):
                    continue
                if ac['lock_idx'] is not None and j_cur < ac['lock_idx']-1:
                    continue  # locked until we pass rejoin

                # Build contiguous affected block using HARD/alerts only (keeps reroute local)
                def edge_alerted(a,b):
                    return edge_hits_hard(a,b) or any({a,b} == {u,v} for (u,v,*_) in seg_alerts)
                k0 = None; k1 = None
                for k in range(j_cur, len(path)-1):
                    a,b = path[k], path[k+1]
                    if edge_alerted(a,b):
                        k0 = k; break
                if k0 is None:
                    continue
                k1 = k0
                for k in range(k0+1, len(path)-1):
                    a,b = path[k], path[k+1]
                    if edge_alerted(a,b):
                        k1 = k
                    else:
                        break
                rejoin_idx = min(k1+1, len(path)-1)
                rejoin_node = path[rejoin_idx]

                # pick start anchor near current edge, not far behind
                def _t_along(ax_, ay_, bx_, by_, px, py):
                    vx, vy = bx_ - ax_, by_ - ay_
                    L = math.hypot(vx, vy) or 1.0
                    ux, uy = vx / L, vy / L
                    return (px - ax_) * ux + (py - ay_) * uy
                ax0,ay0 = positions[cur_u]; ax1,ay1 = positions[cur_v]
                cand = list({cur_u, cur_v,
                             *G.predecessors(cur_u), *G.successors(cur_u),
                             *G.predecessors(cur_v), *G.successors(cur_v)})
                cand = [n for n in cand if n in G.nodes]
                if corr_pref:
                    cand = [n for n in cand if corr_pref in G.nodes[n].get("corr", set())]
                cand = [n for n in cand if _t_along(ax0,ay0,ax1,ay1, positions[n][0], positions[n][1]) > -STEP_KM_ALONG*pixels_per_km*1.5]
                if not cand: cand=[cur_u,cur_v]
                def _score(n):
                    dx,dy = positions[n][0]-xcur, positions[n][1]-ycur
                    d = math.hypot(dx,dy); L = d or 1.0
                    cx_,cy_ = dx/L, dy/L
                    cosang = max(-1.0, min(1.0, cx_*hx + cy_*hy))
                    return d/pixels_per_km + ANGLE_PENALTY_KM*(1.0 - cosang)
                start_anchor = min(cand, key=_score)

                # local strip around current edge
                local_nodes = set([start_anchor, rejoin_node])
                for n, d in G.nodes(data=True):
                    if corr_pref and (corr_pref not in d.get("corr", set())): 
                        continue
                    if d.get("kind") != "mesh" and n not in {start_anchor, rejoin_node}:
                        continue
                    px,py = positions[n]
                    # Allow a generous forward window but keep lateral buffer tight
                    vx, vy = ax1-ax0, ay1-ay0
                    Lseg = math.hypot(vx,vy) or 1.0
                    # distance to infinite line along current edge
                    t = ((px-ax0)*(vx/Lseg) + (py-ay0)*(vy/Lseg))
                    # clamp to a window around the current edge
                    if -FORWARD_PX <= t <= (Lseg + FORWARD_PX):
                        # lateral distance
                        nx_, ny_ = -(vy/Lseg), (vx/Lseg)
                        dlat = abs((px-ax0)*nx_ + (py-ay0)*ny_)
                        if dlat <= BUFFER_PX:
                            local_nodes.add(n)

                # local graph with HARD-removed edges
                G2_local = _cached_G2.subgraph(local_nodes).copy()

                def w_fn(u,v,attrs):
                    base = weight_with_pref(u,v,attrs, corr_pref=corr_pref, penalty_factor=25.0)
                    if edge_hits_soft(u,v):
                        base += soft_penalty_km(u,v)  
                    return base

                # detour path
                try:
                    detour_nodes = nx.shortest_path(G2_local, start_anchor, rejoin_node, weight=w_fn)
                except Exception:
                    try:
                        detour_nodes = nx.shortest_path(_cached_G2, start_anchor, rejoin_node, weight=w_fn)
                    except Exception:
                        continue

                # prefix to anchor
                if start_anchor in path:
                    idx_anchor = path.index(start_anchor)
                    prefix = path[:idx_anchor+1]
                else:
                    prefix = path[:j_cur+1] + detour_nodes[:1]

                new_path = compress_consecutive_duplicates(prefix + detour_nodes[1:] + path[rejoin_idx+1:])
                new_path = remove_pingpong(new_path)

                # splice from current (xcur,ycur)
                best_k = 0; best_d = float('inf')
                for k in range(len(new_path)-1):
                    u2,v2 = new_path[k], new_path[k+1]
                    d = point_to_segment_distance(xcur, ycur, positions[u2][0], positions[u2][1], positions[v2][0], positions[v2][1])
                    if d < best_d:
                        best_d = d; best_k = k
                start_frac_new = edge_fraction(new_path[best_k], new_path[best_k + 1], xcur, ycur)
                tail_coords = interpolate_path_from(new_path, best_k, start_frac_new, cruise,
                                                    slow_edges=ac['slowed_edges'], vfr_penalty=ac['vfr_penalty'])

                # ACK once per aircraft per reroute burst
                if not ac['ack_logged']:
                    leader = seg_alerts[0][7] if seg_alerts else None
                    # rough direction for message
                    if len(new_path) > best_k+1:
                        x2,y2 = positions[new_path[best_k+1]]
                        direction = "east" if (x2 - xcur) > 0 else "west"
                    else:
                        direction = "west"
                    if leader:
                        log_radio(sim_t/60.0, f"thank you {leader}, {ac['callsign']} rerouting {direction}.")
                    else:
                        log_radio(sim_t/60.0, f"{ac['callsign']} rerouting {direction}.")
                    ac['ack_logged'] = True

                ac['coords'] = ac['coords'][:idx_now] + tail_coords
                ac['original_path'] = new_path
                ac['had_reroute_cloud'] = True
                ac['last_plan_frame'] = frame
                ac['active_zone_sig'] = zone_sig
                try:
                    ac['lock_idx'] = new_path.index(rejoin_node)
                except ValueError:
                    ac['lock_idx'] = None

                # draw reroute path tail
                xs,ys = [],[]
                for k in range(best_k, len(new_path)):
                    xs.append(positions[new_path[k]][0]); ys.append(positions[new_path[k]][1])
                reroute_lines_cloud[i].set_data(xs,ys)
                weather_reroute_times_sim_s.append(sim_t)
                if ac.get('reroute_time_min') is None:
                    ac['reroute_time_min'] = sim_t/60.0
                    if first_alert_sim_sec is not None:
                        ac['ac_coord_latency_min'] = (sim_t - first_alert_sim_sec)/60.0

        # draw cloud reroute halos
        for i, ac in enumerate(aircraft_paths):
            delay = aircraft_delays[i]
            if frame < delay: 
                reroute_halos_cloud[i].set_data([], [])
                continue
            idx = frame - delay
            if idx < len(ac['coords']) and ac['had_reroute_cloud']:
                x,y = ac['coords'][idx]
                reroute_halos_cloud[i].set_data([x],[y])
            else:
                reroute_halos_cloud[i].set_data([], [])

        # finish detection
        for i, ac in enumerate(aircraft_paths):
            delay = aircraft_delays[i]
            if frame < delay: 
                continue
            idx = frame - delay
            if idx >= len(ac['coords']) and travel_times[i] is None:
                travel_times[i] = sim_t
                # final length recompute on current path
                path = ac['original_path']
                fin_len = 0.0
                for k in range(len(path)-1):
                    u,v = path[k], path[k+1]
                    if u==v:  continue
                    fin_len += (G.edges[u,v]['weight'] if G.has_edge(u,v)
                                else math.hypot(positions[v][0]-positions[u][0], positions[v][1]-positions[u][1])/pixels_per_km)
                final_lengths[i] = round(fin_len, 2)

        if not sim_done and all(v is not None for v in travel_times):
            sim_done = True
            if not NO_PLOT and ani is not None:
                ani.event_source.stop()

        return (aircraft_dots + reroute_halos_cloud + reroute_halos_vfr +
                slow_overlays + reroute_lines_cloud + alert_overlays + type_labels +
                [sim_time_text, cloud])

    # -------- Drive the sim --------
    def frame_gen():
        f = 0
        while not sim_done:
            yield f
            f += 1

    ani = animation.FuncAnimation(
        fig, update, frames=frame_gen(),
        interval=ANIM_INTERVAL_MS, blit=True, repeat=False, cache_frame_data=False
    )

    if NO_PLOT:
        f = 0
        while not sim_done:
            update(f); f += 1
        plt.close(fig)
    else:
        plt.show()

    # -------- CSV output --------
    def yesno(b): return "yes" if b else "no"
    def join_or_none(s): return "none" if not s else ",".join(sorted(s))

    with open(RUN_CSV,"w",newline="") as f:
        w=csv.writer(f); w.writerow(header_cols_results)
        for i in range(len(start_end_pairs)):
            init_len = initial_lengths[i]
            fin_len  = final_lengths[i] if final_lengths[i] is not None else init_len
            delta_len = fin_len - init_len

            init_time_min = initial_time_min_planned[i]
            fin_time_min  = (travel_times[i]/60.0) if travel_times[i] is not None else None
            dep_delay_min = aircraft_delays[i]*float(os.getenv("SIM_DT","6.0"))/60.0
            final_airborne_min = None if fin_time_min is None else (fin_time_min - dep_delay_min)
            airborne_delta_min = None if final_airborne_min is None else (final_airborne_min - init_time_min)

            ac = aircraft_paths[i]
            used_js, js_dir = jet_use_dir_for_path(ac['original_path'])
            ac_lat = ac.get('ac_coord_latency_min')

            w.writerow([RUN_ID, SCENARIO, SEED,
                        f"Aircraft {i+1}", ac['model_name'], round(ac['cruise_kmph'],1),
                        yesno(ac['had_reroute_cloud']), yesno(ac['had_vfr']), join_or_none(ac['vfr_react_colors']),
                        yesno(ac['affected']), used_js, js_dir,
                        round(init_len,2), round(fin_len,2), round(delta_len,2),
                        round(init_time_min,2),
                        None if final_airborne_min is None else round(final_airborne_min,2),
                        None if airborne_delta_min is None else round(airborne_delta_min,2),
                        ac['alt_color'],
                        None if ac_lat is None else round(ac_lat,3)])

    with open(CSV_MASTER, "a", newline="") as mf:
        mw = csv.writer(mf)
        with open(RUN_CSV, "r", newline="") as r:
            next(r)
            for row in csv.reader(r):
                mw.writerow(row)

    DOC_CSV = os.path.join(RUN_DIR, f"doc_report_{RUN_ID}.csv")
    with open(DOC_CSV, "w", newline="") as fdoc:
        w = csv.writer(fdoc)
        w.writerow([
            "run_id","scenario","seed",
            "aircraft","callsign","model",
            "alt_color","altitude_m","rho_kgpm3","CdA_m2","cruise_kmph",
            "mean_wind_along_kmh","time_hr_wind","drag_work_MJ","DOC_index"
        ])

        ALPHA_MJ  = 1.0   # weight for drag work [MJ]
        BETA_TIME = 20.0  # weight for time [h]

        for i, ac in enumerate(aircraft_paths):
            path = ac['original_path']
            if not path or len(path) < 2:
                continue

            rho = float(ac['rho_at_alt'])
            cda = float(ac['cda_m2'])
            cruise = float(ac['cruise_kmph'])
            slowed = set(ac.get('slowed_edges', set()))

            total_km = 0.0
            sum_w_al = 0.0
            time_hr_wind = 0.0
            work_J = 0.0

            for u, v in zip(path[:-1], path[1:]):
                if u == v:
                    continue
                seg_km = _edge_len_km(u, v)
                if seg_km <= 0:
                    continue
                total_km += seg_km

                w_par = _wind_along_kmh(u, v)  # along-track wind [km/h]
                sum_w_al += w_par * seg_km     # length-weighted accumulator

                v_air_kmh = cruise * (WEATHER_SLOW_FACTOR if (u, v) in slowed else 1.0)
                v_air_mps = max(1.0, v_air_kmh * (1000.0/3600.0))

                v_gnd_kmh = max(5.0, v_air_kmh + w_par)  # clamp to avoid negative/zero
                time_hr_wind += seg_km / v_gnd_kmh

                s_m = seg_km * 1000.0
                work_J += 0.5 * rho * (v_air_mps**2) * cda * s_m

            mean_w_al = (sum_w_al/total_km) if total_km > 0 else 0.0
            work_MJ = work_J / 1.0e6
            doc_index = ALPHA_MJ*work_MJ + BETA_TIME*time_hr_wind

            w.writerow([
                RUN_ID, SCENARIO, SEED,
                f"Aircraft {i+1}", ac['callsign'], ac['model_name'],
                ac['alt_color'], round(ac['alt_m'],1), round(rho,4), round(cda,3), round(cruise,1),
                round(mean_w_al,2), round(time_hr_wind,3), round(work_MJ,3), round(doc_index,3)
            ])

    print("DOC report:", os.path.abspath(DOC_CSV))

    # -------- RUN-LEVEL summary --------
    import statistics as _stats

    n_aircraft = len(start_end_pairs)
    n_weather_reroute = sum(1 for ac in aircraft_paths if ac['had_reroute_cloud'])
    n_affected = sum(1 for ac in aircraft_paths if ac['affected'])
    n_vfr = sum(1 for ac in aircraft_paths if ac['had_vfr'])
    n_js_yes = 0; js_e = 0; js_w = 0

    denom_exposed = n_weather_reroute + n_affected
    weather_avoid_rate = (n_weather_reroute / denom_exposed) if denom_exposed>0 else None

    delays = []
    for i in range(n_aircraft):
        init_time = initial_time_min_planned[i]
        fin_time_min = (travel_times[i]/60.0) if travel_times[i] is not None else None
        dep_delay_min = aircraft_delays[i]*float(os.getenv("SIM_DT","6.0"))/60.0
        final_airborne_min = None if fin_time_min is None else (fin_time_min - dep_delay_min)
        if final_airborne_min is not None:
            delays.append(final_airborne_min - init_time)

    mean_delay = _stats.mean(delays) if delays else None
    p95_delay = _stats.quantiles(delays, n=20)[18] if len(delays)>=20 else None

    first_alert_sim_sec_out = None
    if alert_payload:
        first_alert_sim_sec_out = min(v["t"] for v in alert_payload.values())
    first_alert_min = None if first_alert_sim_sec_out is None else (first_alert_sim_sec_out/60.0)
    last_reroute_min = None if not weather_reroute_times_sim_s else (max(weather_reroute_times_sim_s)/60.0)
    reroute_span_min = None if (first_alert_min is None or last_reroute_min is None) else (last_reroute_min - first_alert_min)

    with open(RUNS_MASTER, "a", newline="") as fr:
        wr = csv.writer(fr)
        wr.writerow([
            RUN_ID, SCENARIO, SEED, n_aircraft, n_weather_reroute, n_affected,
            n_vfr, n_js_yes, js_e, js_w,
            None if weather_avoid_rate is None else round(weather_avoid_rate,4),
            None if mean_delay is None else round(mean_delay,3),
            None if p95_delay is None else round(p95_delay,3),
            None if first_alert_min is None else round(first_alert_min,3),
            None if last_reroute_min is None else round(last_reroute_min,3),
            None if reroute_span_min is None else round(reroute_span_min,3),
            RUN_TAG, ALERT_LATENCY_SIM_S
        ])

    print("Per-run CSV:", os.path.abspath(RUN_CSV))
    print("Master CSV:", os.path.abspath(CSV_MASTER))
    print("Master RUNS CSV:", os.path.abspath(RUNS_MASTER))
    print("Radio Log:", os.path.abspath(COMM_LOG_PATH))

# =================== Batch runner ===================
def run_batch_and_ci():
    import pandas as pd
    import math
    t0 = time.time()

    seeds = parse_seed_set(BATCH_SEEDS_STR)
    scenarios = [s.strip() for s in os.getenv("BATCH_SCENARIOS", "no_comms,instant,delayed").split(",") if s.strip()]
    print(f"Batch run: seeds={seeds} scenarios={scenarios} RUN_TAG={RUN_TAG_DEFAULT}")

    for s in seeds:
        for sc in scenarios:
            run_single(seed=s, scenario=sc, run_tag=RUN_TAG_DEFAULT, csv_dir=CSV_DIR_DEFAULT,
                       alert_latency_sim_s=ALERT_LATENCY_SIM_S_DEFAULT)

    runs_path = os.path.join(CSV_DIR_DEFAULT, "master_runs.csv")
    if not os.path.exists(runs_path):
        print("No master_runs.csv found, skipping CI summary.")
        dur_min = (time.time() - t0) / 60.0
        print(f"\n=== DONE: batch '{RUN_TAG_DEFAULT}' completed in {dur_min:.1f} min ===")
        return

    runs = pd.read_csv(runs_path)
    df_batch = runs[runs.get("run_tag", "").eq(RUN_TAG_DEFAULT)]
    if df_batch.empty:
        df_batch = runs[(runs["seed"].isin(seeds)) & (runs["scenario"].isin(scenarios))]

    print("\n=== CI summary for current batch ===")
    for scen in scenarios:
        df = df_batch[df_batch["scenario"] == scen]
        if df.empty:
            print(f"[scen={scen}] no rows found for this batch")
            continue
        m1, hw1, _ = ci_halfwidth_mean([float(v) for v in df["mean_delay_min"] if not pd.isna(v)])
        tgt1 = kpi1_target(m1)
        m2, hw2, _ = ci_halfwidth_mean([float(v) for v in df["weather_avoid_rate"] if not pd.isna(v)])
        kpi1_ok = (hw1 == hw1) and (tgt1 == tgt1) and (hw1 <= tgt1)
        kpi2_ok = (hw2 == hw2) and (hw2 <= 0.03)
        print(f"[{scen}] n={len(df)}  KPI1 delay = {m1:.2f} ± {hw1:.2f} min (≤ {tgt1:.2f}) -> {'OK' if kpi1_ok else 'NO'};  "
              f"KPI2 avoid = {100*m2:.2f}% ± {100*hw2:.2f} pp -> {'OK' if kpi2_ok else 'NO'}")

    dur_min = (time.time() - t0) / 60.0
    print(f"\n=== DONE: batch '{RUN_TAG_DEFAULT}' completed in {dur_min:.1f} min ===")

# =================== Entry point ===================
if __name__ == "__main__":
    if BATCH_MODE:
        _t0 = time.perf_counter()
        run_batch_and_ci()
        _mins = (time.perf_counter() - _t0) / 60.0
        print(f"\n=== DONE: batch '{RUN_TAG_DEFAULT}' completed in {_mins:.1f} min ===")
    else:
        _seed_env = os.getenv("SEED", "1").lower()
        SEED = random.randrange(1, 10**9) if _seed_env == "random" else int(_seed_env)
        SCENARIO = os.getenv("SCENARIO", "delayed").lower()
        run_single(seed=SEED, scenario=SCENARIO,
                   run_tag=os.getenv("RUN_TAG", time.strftime("%Y%m%d-%H%M%S")),
                   csv_dir=os.getenv("CSV_DIR", CSV_DIR_DEFAULT),
                   alert_latency_sim_s=float(os.getenv("ALERT_LATENCY_SIM_S", str(ALERT_LATENCY_SIM_S_DEFAULT))))
        print("\n=== DONE: single run finished ===")
