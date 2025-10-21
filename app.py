# app.py — Tablero Puebla Pro (dark only, map-first, overlays)
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import json, hashlib, re
from pathlib import Path
from datetime import date, timedelta
from joblib import load as joblib_load
# Vecindad geográfica
from shapely.geometry import shape
from shapely.errors import ShapelyError

# =========================
# Config de página
# =========================
st.set_page_config(
    page_title="Tablero Puebla Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"  # sidebar abierto por defecto
)

# =========================
# Tema oscuro global (CSS + Altair)
# =========================
def apply_dark_theme():
    def _rad_dark():
        return {
            "config": {
                "view": {"stroke": "transparent"},
                "background": "#0e1117",
                "style": {"text": {"font": "Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"}},
                "axisX": {
                    "labelColor": "#e5e7eb", "titleColor": "#e5e7eb",
                    "gridColor": "#1f2937", "tickColor": "#6b7280"
                },
                "axisY": {
                    "labelColor": "#e5e7eb", "titleColor": "#e5e7eb",
                    "gridColor": "#1f2937", "tickColor": "#6b7280"
                },
                "legend": {"labelColor": "#e5e7eb", "titleColor": "#e5e7eb"},
                "header": {"labelColor": "#e5e7eb", "titleColor": "#e5e7eb"},
                "range": { "category": ["#60a5fa","#fbbf24","#34d399","#f87171","#a78bfa","#22d3ee","#fb7185"] },
                "mark": {"color": "#60a5fa"}
            }
        }
    alt.themes.register("rad_dark", _rad_dark)
    alt.themes.enable("rad_dark")

    st.markdown(
        """
        <style>
          :root { --bg:#0e1117; --surface:#111827; --text:#e5e7eb; --muted:#9ca3af; --border:#1f2937; --primary:#60a5fa; }
          .stApp { background: var(--bg); color: var(--text); }
          .block-container { padding-top: 1.0rem !important; padding-bottom: .6rem !important; }
          h1 { font-size: clamp(1.6rem, 3.8vw, 2.4rem) !important; line-height: 1.15; margin-bottom: .6rem; }
          [data-testid="stHorizontalBlock"] { align-items: center; }
          [data-testid="stSidebar"] { background:#0b0e14; color:var(--text); border-right:1px solid var(--border); }

          /* Inputs base */
          .stSelectbox div[data-baseweb="select"],
          .stTextInput input, .stDateInput input, .stNumberInput input {
            background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: 10px;
          }

          /* Radios como "pills" (mismo look que tabs) */
          .stRadio > div[role="radiogroup"] { display:flex; gap:.5rem; flex-wrap:wrap; }
          .stRadio > div[role="radiogroup"] label {
            background: var(--surface); color: var(--text);
            border:1px solid var(--border); padding:.40rem .80rem; border-radius:10px; cursor:pointer;
          }
          .stRadio > div[role="radiogroup"] label:hover { border-color: var(--primary); }
          .stRadio > div[role="radiogroup"] label:has(input:checked) {
            border-color: var(--primary);
            box-shadow: 0 0 0 1px var(--primary) inset;
          }

          /* Tabs */
          .stTabs [role="tablist"] { gap: .5rem; }
          .stTabs [role="tab"]{
            background: var(--surface); color: var(--text); border: 1px solid var(--border);
            padding:.4rem .8rem; border-radius:10px;
          }
          .stTabs [role="tab"][aria-selected="true"]{ border-color: var(--primary); }

          /* DataFrame / Métricas */
          .stDataFrame table thead th { background:#0f172a; color:#e5e7eb; }
          div[data-testid="stMetricValue"] { color: var(--text); }
        </style>
        """,
        unsafe_allow_html=True,
    )

apply_dark_theme()
MAP_STYLE = "mapbox://styles/mapbox/dark-v10"

# =========================
# Header (branding)
# =========================
c1, c2 = st.columns([1, 6], vertical_alignment="center")
with c1:
    st.markdown(
        """
        <div style="font-weight:800;font-size:22px;letter-spacing:2px;
                    padding:8px 12px;border:1px solid rgba(255,255,255,.08);
                    border-radius:12px;display:inline-block;background:rgba(255,255,255,.04);">
        CHR
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.title("Tablero Puebla — v2.8 (map-first, dark only + overlays)")

# ---------- Paths ----------
DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "datos_puebla.csv"
GEOJSON_PATH = DATA_DIR / "secciones_puebla.geojson"
MODEL_PATH = DATA_DIR / "model.joblib"
ALTA_PRIORIDAD_CSV = DATA_DIR / "alta_prioridad.csv"

# Overlays (ya convertidos con mapshaper)
OVERLAY_FILES = {
    "Municipios":            "data/overlays/municipios.geojson",
    "Distritos (federales)": "data/overlays/distritos_federales.geojson",
    "Distritos (locales)":   "data/overlays/distritos_locales.geojson",
    "Lagos":                 "data/overlays/lagos.geojson",
    "Ríos":                  "data/overlays/rios.geojson",
    "Vialidad":              "data/overlays/vialidad.geojson",
    "Ferrocarril":           "data/overlays/ferrocarril.geojson",
    "Hospitales":            "data/overlays/hospitales.geojson",
    "Escuelas":              "data/overlays/escuelas.geojson",
    "Mercados":              "data/overlays/mercados.geojson",
    "Centros comerciales":   "data/overlays/centros_comerciales.geojson",
}

# ---------- Paleta ----------
PALETTE = {
    "Alta": "#ff4d4f", "Media": "#f0a800", "Baja": "#3aaed8",
    "Default": "#9aa0a6", "Out": "#2a2e35"
}

# ---------- Utils ----------
@st.cache_data(show_spinner=False)
def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    if "FECHA" in df.columns:
        df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce").dt.date
    for c in ["NUM_PROMOVIDOS","PAUTA_INVERTIDA","VISUALIZACIONES"]:
        if c not in df.columns:
            df[c] = 0
    if "SECCION" in df.columns:
        df["SECCION"] = (df["SECCION"].astype(str)
                         .str.replace(r"[^\d]", "", regex=True)
                         .str.lstrip("0"))
    return df

@st.cache_data(show_spinner=False)
def load_geojson(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def hex_to_rgba(hex_color: str, alpha: int = 150):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return [r, g, b, alpha]

def hhash(path: Path) -> str:
    if not path.exists(): return "—"
    m = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1<<16)
            if not chunk: break
            m.update(chunk)
    return m.hexdigest()[:12]

def ensure_files():
    if not CSV_PATH.exists():
        st.error("Falta `data/datos_puebla.csv`.")
        st.stop()
    if not GEOJSON_PATH.exists():
        st.error("Falta `data/secciones_puebla.geojson`.")
        st.stop()

def natural_sorted_options(values: pd.Series):
    s = values.dropna().astype(str)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all():
        df = pd.DataFrame({"raw": s, "num": num})
        return ["Todos"] + df.sort_values("num")["raw"].tolist()
    else:
        def as_num(x):
            try: return float(x)
            except: return np.inf
        return ["Todos"] + sorted(s.tolist(), key=lambda x: (np.isinf(as_num(x)), as_num(x), x))

def view_state_puebla_estado():
    return pdk.ViewState(latitude=19.40, longitude=-98.20, zoom=6.7, pitch=0)

def view_state_puebla_capital():
    return pdk.ViewState(latitude=19.0379, longitude=-98.2035, zoom=11.2, pitch=0)

@st.cache_data(show_spinner=False)
def extract_base_features(geojson_obj: dict):
    feats = []
    for f in geojson_obj.get("features", []):
        geom = f.get("geometry")
        props = f.get("properties", {})
        if not geom:
            continue
        feats.append({"type": "Feature",
                      "geometry": geom,
                      "properties": {"SECCION": str(props.get("SECCION",""))}})
    return {"type": "FeatureCollection", "features": feats}

# =========================
# Carga
# =========================
ensure_files()
df = load_table(CSV_PATH)
geojson_raw = load_geojson(GEOJSON_PATH)
geojson_base = extract_base_features(geojson_raw)

# =========================
# Elevar PRIORIDAD a "Alta" desde alta_prioridad.csv (silencioso)
# =========================
if ALTA_PRIORIDAD_CSV.exists():
    try:
        df_alta = pd.read_csv(ALTA_PRIORIDAD_CSV)
        df_alta.columns = [c.strip().upper() for c in df_alta.columns]
        cand_cols = [c for c in df_alta.columns if c == "SECCION" or "SECC" in c]
        if cand_cols:
            col = cand_cols[0]
            secciones_alta = (df_alta[col].astype(str)
                              .str.replace(r"[^\d]", "", regex=True)
                              .str.lstrip("0"))
            secciones_alta = set(secciones_alta.dropna().tolist())
            if "PRIORIDAD" not in df.columns:
                df["PRIORIDAD"] = ""
            df.loc[df["SECCION"].astype(str).isin(secciones_alta), "PRIORIDAD"] = "Alta"
    except Exception:
        pass  # Silencioso

# =========================
# Prioridades por vecindad (Altas -> vecinas Media; resto Baja)
# =========================
@st.cache_data(show_spinner=False)
def build_geo_index(geojson_fc: dict):
    """
    Devuelve:
      - sec2geom: dict SECCION -> (geom_shapely, bbox=(minx,miny,maxx,maxy))
      - disponibles: set de secciones con geometría válida
    """
    sec2geom = {}
    for f in geojson_fc.get("features", []):
        props = f.get("properties", {})
        sec = str(props.get("SECCION", "")).strip()
        geom = f.get("geometry")
        if not sec or not geom:
            continue
        try:
            g = shape(geom)
            if not g.is_valid or g.is_empty:
                continue
            sec2geom[sec] = (g, g.bounds)  # (geom, (minx,miny,maxx,maxy))
        except (ShapelyError, Exception):
            continue
    return sec2geom, set(sec2geom.keys())

def bbox_overlap(b1, b2):
    return (b1[0] <= b2[2]) and (b1[2] >= b2[0]) and (b1[1] <= b2[3]) and (b1[3] >= b2[1])

sec2geom, secciones_con_geom = build_geo_index(geojson_base)
alta_set = set(df.loc[df.get("PRIORIDAD","").astype(str).str.upper() == "ALTA", "SECCION"].astype(str))

media_set = set()
if alta_set:
    altas_geom = [(s, sec2geom[s][0], sec2geom[s][1]) for s in alta_set if s in sec2geom]
    for sec, (geom, bbox) in sec2geom.items():
        if sec in alta_set:
            continue
        # Prefiltro rápido con bbox
        candidatos = [ (sA, gA) for (sA, gA, bA) in altas_geom if bbox_overlap(bbox, bA) ]
        if not candidatos:
            continue
        # Contacto directo (comparten frontera). Si quieres incluir vértice, cambia a intersects()
        for _, gA in candidatos:
            try:
                if geom.touches(gA):
                    media_set.add(sec)
                    break
            except Exception:
                pass

# Regla final: Alta se conserva; vecinas -> Media; resto -> Baja (si estaban vacías o inválidas)
if "PRIORIDAD" not in df.columns:
    df["PRIORIDAD"] = ""
df["SECCION"] = df["SECCION"].astype(str)

df.loc[df["SECCION"].isin(list(media_set)) & ~df["SECCION"].isin(list(alta_set)), "PRIORIDAD"] = "Media"
mask_resto = df["SECCION"].isin(list(secciones_con_geom)) & ~df["SECCION"].isin(list(alta_set)) & ~df["SECCION"].isin(list(media_set))
df.loc[mask_resto & (df["PRIORIDAD"].isin(["", " ", None]) | ~df["PRIORIDAD"].isin(["Alta","Media","Baja"])), "PRIORIDAD"] = "Baja"

# =========================
# Sidebar: filtros + switches de overlays
# =========================
with st.sidebar:
    st.markdown("### Filtros")

    # Fecha (robusto)
    start_date, end_date = None, None
    if "FECHA" in df.columns and df["FECHA"].notna().any():
        min_date = df["FECHA"].min()
        max_date = df["FECHA"].max()
        picked = st.date_input(
            "Periodo",
            value=(max(min_date, max_date - timedelta(days=7)), max_date),
            min_value=min_date, max_value=max_date,
            key="periodo_rango"
        )
        if isinstance(picked, tuple) and len(picked) == 2:
            sd, ed = picked
        else:
            sd = ed = picked
        if sd and ed and sd > ed: sd, ed = ed, sd
        start_date, end_date = sd, ed
    else:
        st.caption("No hay fechas (FECHA): se muestra todo.")

    # Listas (las originales)
    prioridades = ["Todos"] + (sorted(df["PRIORIDAD"].dropna().astype(str).unique()) if "PRIORIDAD" in df.columns else [])
    municipios  = natural_sorted_options(df["MUNICIPIO"]) if "MUNICIPIO" in df.columns else ["Todos"]
    distritos   = natural_sorted_options(df["DISTRITO LOCAL"]) if "DISTRITO LOCAL" in df.columns else ["Todos"]

    sel_prio = st.selectbox("Prioridad", prioridades, index=0)
    sel_mun  = st.selectbox("Municipio", municipios, index=0)
    sel_dist = st.selectbox("Distrito Local", distritos, index=0)

    st.markdown("---")
    map_action = st.radio(
        "Enfoque del mapa",
        options=["Puebla estado", "Puebla capital", "Centrar en datos filtrados"],
        index=0,
        horizontal=True
    )

    # --- Capas extra (overlays) ---
    st.markdown("### Capas extra (overlays)")
    show = {}
    for name in ["Municipios","Distritos (federales)","Distritos (locales)","Lagos",
                 "Ríos","Vialidad","Ferrocarril","Hospitales","Escuelas","Mercados","Centros comerciales"]:
        show[name] = st.toggle(name, value=False)

# =========================
# Aplicar filtros
# =========================
df_f = df.copy()
if start_date and end_date and "FECHA" in df_f.columns:
    df_f = df_f[(df_f["FECHA"] >= start_date) & (df_f["FECHA"] <= end_date)]
if "PRIORIDAD" in df_f.columns and sel_prio != "Todos":
    df_f = df_f[df_f["PRIORIDAD"].astype(str) == str(sel_prio)]
if "MUNICIPIO" in df_f.columns and sel_mun != "Todos":
    df_f = df_f[df_f["MUNICIPIO"].astype(str) == str(sel_mun)]
if "DISTRITO LOCAL" in df_f.columns and sel_dist != "Todos":
    df_f = df_f[df_f["DISTRITO LOCAL"].astype(str) == str(sel_dist)]

# =========================
# Predicción: SCORE (si hay modelo)
# =========================
def score_to_band(series, labels=("Baja","Media","Alta","Muy alta"), q=4):
    s = pd.Series(series).astype(float)
    if s.empty or s.nunique() == 1:
        return pd.Series([labels[1]] * len(s), index=s.index)
    try:
        return pd.qcut(s, q=q, labels=labels, duplicates="drop")
    except Exception:
        unique_bins = max(2, min(q, s.nunique()))
        bins = np.linspace(s.min(), s.max(), num=unique_bins + 1)
        use_labels = labels[:unique_bins]
        return pd.cut(s, bins=bins, labels=use_labels, include_lowest=True, duplicates="drop")

model, feature_names = None, None
if MODEL_PATH.exists():
    try:
        bundle = joblib_load(MODEL_PATH)
        model = bundle.get("model")
        feature_names = bundle.get("features", ["NUM_PROMOVIDOS","PAUTA_INVERTIDA","VISUALIZACIONES"])
    except Exception:
        model = None

needed = ["NUM_PROMOVIDOS","PAUTA_INVERTIDA","VISUALIZACIONES"]
for c in needed:
    if c not in df_f.columns: df_f[c] = 0
X = df_f[needed].copy()
Xlog = np.log1p(X)

if model is not None and len(df_f) > 0:
    try:
        df_f["SCORE"] = model.predict_proba(Xlog)[:,1]
    except Exception:
        raw = model.predict(Xlog)
        df_f["SCORE"] = (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)
    df_f["RANGO"] = score_to_band(df_f["SCORE"])

# =====================================================
# MAPA (PRIMERO) — pinta secciones y añade overlays
# =====================================================
# Lookups
lookup_prio, lookup_score = {}, {}
if "SECCION" in df_f.columns:
    if "PRIORIDAD" in df_f.columns:
        for r in df_f[["SECCION","PRIORIDAD"]].itertuples(index=False):
            lookup_prio[str(r.SECCION)] = str(r.PRIORIDAD)
    if "SCORE" in df_f.columns:
        for r in df_f[["SECCION","SCORE"]].itertuples(index=False):
            lookup_score[str(r.SECCION)] = float(r.SCORE)

# Pintado inicial por prioridad
features = []
for f in geojson_base["features"]:
    sec = f["properties"]["SECCION"]
    props = {"SECCION": sec}
    if sec in lookup_prio:
        col = PALETTE.get(lookup_prio[sec], PALETTE["Default"])
        fill = hex_to_rgba(col, 160)
        label = lookup_prio[sec]
    else:
        fill = hex_to_rgba(PALETTE["Out"], 100)
        label = "Fuera de filtro"
    props["__FILL__"]  = fill
    props["__LABEL__"] = label
    features.append({"type": "Feature", "geometry": f["geometry"], "properties": props})
geojson_painted = {"type": "FeatureCollection", "features": features}

# Vista inicial
if map_action == "Puebla estado":
    vstate = view_state_puebla_estado()
elif map_action == "Puebla capital":
    vstate = view_state_puebla_capital()
else:  # Centrar en datos filtrados
    xs, ys = [], []
    for f in features:
        if f["properties"]["__LABEL__"] == "Fuera de filtro": 
            continue
        g = f["geometry"]; t = g.get("type"); coords = g.get("coordinates")
        if not coords: continue
        if t == "Polygon":
            for x,y in coords[0]:
                xs.append(x); ys.append(y)
        elif t == "MultiPolygon":
            for x,y in coords[0][0]:
                xs.append(x); ys.append(y)
    if xs and ys:
        vstate = pdk.ViewState(latitude=float(np.mean(ys)), longitude=float(np.mean(xs)), zoom=10.2, pitch=0)
    else:
        vstate = view_state_puebla_estado()

# Capa principal (secciones)
layer = pdk.Layer(
    "GeoJsonLayer",
    data=geojson_painted,
    pickable=True, auto_highlight=True,
    stroked=True, filled=True, opacity=1.0,
    get_fill_color="properties.__FILL__",
    get_line_color=[180,180,200,120],
    line_width_min_pixels=1,
)
tooltip = {
    "html":"<b>Sección:</b> {SECCION}<br/><b>Dato:</b> {__LABEL__}",
    "style":{"backgroundColor":"rgba(0,0,0,.85)","color":"#fff"}
}

# ==== Helpers overlays ====
@st.cache_data(show_spinner=False)
def _load_gj(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)
        feats = gj.get("features") or []
        return gj if feats else None
    except Exception:
        return None

def _first_coord(geom):
    if not geom:
        return None
    t = geom.get("type","")
    c = geom.get("coordinates")
    if not c:
        return None
    try:
        if t == "Point":            return c
        if t == "MultiPoint":       return c[0]
        if t == "LineString":       return c[0]
        if t == "MultiLineString":  return c[0][0]
        if t == "Polygon":          return c[0][0]
        if t == "MultiPolygon":     return c[0][0][0]
    except Exception:
        return None
    return None

# ==== Construcción de capas extra según los switches ====
extra_layers = []

# Polígonos
if show.get("Municipios"):
    gj = _load_gj(OVERLAY_FILES["Municipios"])
    if gj:
        extra_layers.append(pdk.Layer(
            "GeoJsonLayer", gj,
            pickable=False, stroked=True, filled=True, opacity=0.18,
            get_fill_color=[60, 180, 75, 50],
            get_line_color=[60, 180, 75, 160],
            line_width_min_pixels=1
        ))

if show.get("Distritos (federales)"):
    gj = _load_gj(OVERLAY_FILES["Distritos (federales)"])
    if gj:
        extra_layers.append(pdk.Layer(
            "GeoJsonLayer", gj,
            pickable=False, stroked=True, filled=False, opacity=1.0,
            get_line_color=[255, 215, 0, 180],
            line_width_min_pixels=1.2
        ))

if show.get("Distritos (locales)"):
    gj = _load_gj(OVERLAY_FILES["Distritos (locales)"])
    if gj:
        extra_layers.append(pdk.Layer(
            "GeoJsonLayer", gj,
            pickable=False, stroked=True, filled=False, opacity=1.0,
            get_line_color=[255, 105, 180, 180],
            line_width_min_pixels=1.2
        ))

if show.get("Lagos"):
    gj = _load_gj(OVERLAY_FILES["Lagos"])
    if gj:
        extra_layers.append(pdk.Layer(
            "GeoJsonLayer", gj,
            pickable=False, stroked=True, filled=True, opacity=0.25,
            get_fill_color=[64, 156, 255, 90],
            get_line_color=[64, 156, 255, 180],
            line_width_min_pixels=1
        ))

# Líneas
if show.get("Ríos"):
    gj = _load_gj(OVERLAY_FILES["Ríos"])
    if gj:
        extra_layers.append(pdk.Layer(
            "GeoJsonLayer", gj,
            pickable=False, stroked=True, filled=False, opacity=0.85,
            get_line_color=[80, 150, 255, 180],
            line_width_min_pixels=1
        ))

if show.get("Vialidad"):
    gj = _load_gj(OVERLAY_FILES["Vialidad"])
    if gj:
        extra_layers.append(pdk.Layer(
            "GeoJsonLayer", gj,
            pickable=False, stroked=True, filled=False, opacity=0.5,
            get_line_color=[170, 170, 170, 160],
            line_width_min_pixels=0.6
        ))

if show.get("Ferrocarril"):
    gj = _load_gj(OVERLAY_FILES["Ferrocarril"])
    if gj:
        extra_layers.append(pdk.Layer(
            "GeoJsonLayer", gj,
            pickable=False, stroked=True, filled=False, opacity=0.9,
            get_line_color=[255, 140, 0, 200],
            line_width_min_pixels=1
        ))

# Puntos (POIs)
def _add_points_layer(name, color_rgba):
    gj = _load_gj(OVERLAY_FILES[name])
    if not gj:
        return
    pts = []
    for f in gj.get("features", []):
        c = _first_coord(f.get("geometry"))
        if c:
            pts.append({"position": c, "name": name})
    if not pts:
        return
    extra_layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=pts,
        get_position="position",
        get_radius=60,
        radius_units="meters",
        get_fill_color=color_rgba,
        pickable=True
    ))

if show.get("Hospitales"):          _add_points_layer("Hospitales",          [255, 99, 132, 200])
if show.get("Escuelas"):            _add_points_layer("Escuelas",            [100, 181, 246, 200])
if show.get("Mercados"):            _add_points_layer("Mercados",            [72, 187, 120, 200])
if show.get("Centros comerciales"): _add_points_layer("Centros comerciales", [255, 206, 86,  200])

# === Arma el Deck sumando la capa principal 'layer' + extras ===
deck = pdk.Deck(
    layers=[layer] + extra_layers,
    initial_view_state=vstate,
    map_style=MAP_STYLE,
    tooltip=tooltip
)
st.pydeck_chart(deck, use_container_width=True)

# Selector de coloreo debajo del mapa
color_mode = st.radio("Colorear por", ["Prioridad", "Score (si hay modelo)"], horizontal=True)
if color_mode.startswith("Score") and len(lookup_score) > 0:
    features2 = []
    for f in geojson_base["features"]:
        sec = f["properties"]["SECCION"]
        props = {"SECCION": sec}
        if sec in lookup_score:
            val = float(lookup_score[sec])
            r = int(255*val); g = int(80*(1-val)+50); b = int(120*(1-val)+50)
            props["__FILL__"]  = [r,g,b,160]
            props["__LABEL__"] = f"Score {val:.2f}"
        else:
            props["__FILL__"]  = [150,150,160,90]
            props["__LABEL__"] = "Sin score"
        features2.append({"type": "Feature", "geometry": f["geometry"], "properties": props})
    deck2 = pdk.Deck(
        layers=[pdk.Layer(
            "GeoJsonLayer", {"type":"FeatureCollection","features":features2},
            pickable=True, auto_highlight=True, stroked=True, filled=True, opacity=1.0,
            get_fill_color="properties.__FILL__",
            get_line_color=[180,180,200,120],
            line_width_min_pixels=1)],
        initial_view_state=vstate, map_style=MAP_STYLE, tooltip=tooltip
    )
    st.pydeck_chart(deck2, use_container_width=True)

st.divider()

# ================= TABS =================
tab_prev, tab_day, tab_res = st.tabs(["📋 Previa (Campaña)", "🗳️ Día D", "📈 Resultados"])

# ================= PREVIA =================
with tab_prev:
    st.subheader("Métricas clave y resumen")
    def safe_sum(col):
        return int(pd.to_numeric(df_f.get(col, pd.Series([])), errors="coerce").fillna(0).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Secciones", df_f["SECCION"].nunique() if "SECCION" in df_f.columns else len(df_f))
    m2.metric("Promovidos", safe_sum("NUM_PROMOVIDOS"))
    m3.metric("Pauta invertida", f"${safe_sum('PAUTA_INVERTIDA'):,}")
    m4.metric("Visualizaciones", f"{safe_sum('VISUALIZACIONES'):,}")

    st.markdown("### Gráficas")
    cols = st.columns(2)

    if "PRIORIDAD" in df_f.columns and df_f["PRIORIDAD"].notna().any():
        pr = df_f["PRIORIDAD"].astype(str).value_counts().rename_axis("PRIORIDAD").reset_index(name="SECCIONES")
        order_map = {"Alta": 0, "Media": 1, "Baja": 2}
        pr["ORD"] = pr["PRIORIDAD"].map(order_map).fillna(99)
        pr = pr.sort_values(["ORD", "SECCIONES"], ascending=[True, False])
        pr["COLOR"] = pr["PRIORIDAD"].map(lambda x: PALETTE.get(x, PALETTE["Default"]))
        chart_prio = (alt.Chart(pr)
            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
            .encode(
                x=alt.X("SECCIONES:Q", title="Secciones"),
                y=alt.Y("PRIORIDAD:N", sort=pr["PRIORIDAD"].tolist(), title=None),
                color=alt.Color("PRIORIDAD:N", scale=alt.Scale(range=list(pr["COLOR"])), legend=None),
                tooltip=["PRIORIDAD","SECCIONES"]
            ).properties(title="Secciones por prioridad", height=240))
        cols[0].altair_chart(chart_prio, use_container_width=True)

    if "MUNICIPIO" in df_f.columns:
        top_m = (df_f["MUNICIPIO"].astype(str).value_counts()
                 .rename_axis("MUNICIPIO").reset_index(name="SECCIONES")
                 .head(8).sort_values("SECCIONES"))
        chart_mun = (alt.Chart(top_m)
            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
            .encode(
                x=alt.X("SECCIONES:Q", title="Secciones"),
                y=alt.Y("MUNICIPIO:N", sort=None, title=None),
                color=alt.value("#5C7CFA"),
                tooltip=["MUNICIPIO","SECCIONES"]
            ).properties(title="Top 8 municipios", height=260))
        cols[1].altair_chart(chart_mun, use_container_width=True)

    # Predicción + simulador
    st.markdown("### Predicción (baseline) y simulación")
    left, right = st.columns([2,1])

    if model is not None and len(df_f) > 0:
        with left:
            st.dataframe(df_f[["SECCION","SCORE","RANGO"]].head(20), use_container_width=True)
            if hasattr(model, "feature_importances_") and feature_names:
                fi = pd.DataFrame({"feature": feature_names,
                                   "importance": getattr(model, "feature_importances_")}).sort_values("importance")
            else:
                fi = pd.DataFrame({"feature": feature_names or needed,
                                   "importance": [1/len(needed)]*len(needed)})
            chart_fi = (alt.Chart(fi).mark_bar()
                        .encode(x=alt.X("importance:Q", title="Importancia"),
                                y=alt.Y("feature:N", sort=None, title="Variable"),
                                tooltip=["feature","importance"])
                        .properties(title="Explicabilidad", height=240))
            cols[0].altair_chart(chart_fi, use_container_width=True)

        with right:
            st.markdown("**Simulador**")
            mult_pauta = st.slider("x Pauta", 0.5, 2.0, 1.0, 0.1)
            mult_promos = st.slider("x Promovidos", 0.5, 2.0, 1.0, 0.1)
            mult_views = st.slider("x Visualizaciones", 0.5, 2.0, 1.0, 0.1)
            Xsim = X.copy()
            Xsim["PAUTA_INVERTIDA"] *= mult_pauta
            Xsim["NUM_PROMOVIDOS"] *= mult_promos
            Xsim["VISUALIZACIONES"] *= mult_views
            Xsim_log = np.log1p(Xsim)
            try:
                proba_sim = model.predict_proba(Xsim_log)[:,1]
            except Exception:
                raw2 = model.predict(Xsim_log)
                proba_sim = (raw2 - raw2.min())/(raw2.max()-raw2.min()+1e-6)
            if "SCORE" in df_f.columns:
                delta = (proba_sim - df_f["SCORE"]).sum()
                st.metric("Impacto simulado (Σ Δ score)", f"{float(delta):.2f}")
    else:
        st.info("Entrena `data/model.joblib` con `python train_model.py` para activar predicción y simulación.")

# ================= DÍA D =================
with tab_day:
    st.subheader("Monitoreo operativo y countdown")
    ELECTION_DATE = date(2027, 6, 6)  # estimado (ajustar cuando sea oficial)
    days_left = (ELECTION_DATE - date.today()).days
    c = st.columns(3)
    c[0].metric("Días para la elección municipal (estimado)", f"{max(days_left,0)}")
    c[1].metric("% casillas instaladas", "—")
    c[2].metric("Incidencias reportadas", "—")
    st.caption("Actualiza la fecha cuando la autoridad electoral la publique oficialmente.")

# ================= RESULTADOS =================
with tab_res:
    st.subheader("Resultados preliminares (plantillas)")
    st.write("Listo para integrar actas, PREP y comparativos.")
    st.caption("Incluye huella de datos para trazabilidad.")

# ---------- Footer ----------
st.markdown("---")
st.caption(f"Huella datos_puebla.csv: `{hhash(CSV_PATH)}` · Modelo: {'OK' if MODEL_PATH.exists() else 'N/D'}")
