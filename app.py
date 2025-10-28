# app.py — Tablero Puebla Pro (OSM siempre activo) — Filtros por formulario + Tema Claro/Oscuro

import json
import hashlib
from pathlib import Path
from datetime import date, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
from joblib import load as joblib_load
from shapely.geometry import shape
from shapely.errors import ShapelyError

# -------------------------- Config básica --------------------------
st.set_page_config(
    page_title="Tablero Puebla Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------- Helpers tema --------------------------
def _alt_theme(is_dark: bool):
    txt = "#e5e7eb" if is_dark else "#111827"
    bg = "#0e1117" if is_dark else "#ffffff"
    grid = "#1f2937" if is_dark else "#e5e7eb"
    tick = "#9aa0a6" if is_dark else "#9ca3af"
    primary = "#60a5fa" if is_dark else "#3b82f6"
    return {
        "config": {
            "view": {"stroke": "transparent"},
            "background": bg,
            "style": {"text": {"font": "Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif", "fill": txt}},
            "axisX": {"labelColor": txt, "titleColor": txt, "gridColor": grid, "tickColor": tick},
            "axisY": {"labelColor": txt, "titleColor": txt, "gridColor": grid, "tickColor": tick},
            "legend": {"labelColor": txt, "titleColor": txt},
            "header": {"labelColor": txt, "titleColor": txt},
            "range": {"category": ["#60a5fa", "#fbbf24", "#34d399", "#f87171", "#a78bfa", "#22d3ee", "#fb7185"]},
            "mark": {"color": primary},
        }
    }

def _emit_css_vars(theme: str):
    if theme == "dark":
        vars_css = """
:root{
  --bg:#0e1117; --surface:#111827; --text:#e5e7eb; --muted:#9aa0a6; --border:#1f2937;
  --primary:#60a5fa; --primary-weak:#0b1220; --chip:#111b2d;
}
"""
    else:
        vars_css = """
:root{
  --bg:#ffffff; --surface:#f8fafc; --text:#0f172a; --muted:#64748b; --border:#e5e7eb;
  --primary:#3b82f6; --primary-weak:#dbeafe; --chip:#eef2ff;
}
"""
    st.markdown(
        f"""
<style>
{vars_css}
.stApp {{ background: var(--bg); color: var(--text); }}
.block-container {{ padding-top: 1rem !important; padding-bottom:.6rem !important; max-width: 1400px; }}
h1 {{ font-size: clamp(1.6rem, 3.6vw, 2.4rem)!important; line-height:1.15; margin: .2rem 0 .75rem 0; }}
[data-testid="stSidebar"]{{ background: var(--surface); color: var(--text); border-right:1px solid var(--border); }}
[data-testid="stSidebar"] .stButton>button, .stForm button[kind="primary"]{{
  width:100%; border-radius:10px; border:1px solid var(--border);
  background:var(--primary); color:#fff; font-weight:600;
}}
[data-testid="stSidebar"] .stButton>button:hover, .stForm button[kind="primary"]:hover{{ filter:brightness(0.95); }}
.stSelectbox div[data-baseweb="select"], .stDateInput input, .stTextInput input {{
  background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:12px;
}}
.stRadio [role="radiogroup"]>div{{ background:var(--bg); border:1px solid var(--border); border-radius:12px; padding:.25rem .5rem; }}
.stTabs [role="tab"]{{
  background:var(--surface); color: var(--text); border:1px solid var(--border);
  padding:.45rem .8rem; border-radius:10px; margin-right:.4rem;
}}
.stTabs [role="tab"][aria-selected="true"]{{
  border-color: var(--primary);
  box-shadow:0 0 0 3px color-mix(in srgb, var(--primary) 20%, transparent);
}}
.stDataFrame table thead th {{ background: color-mix(in srgb, var(--surface) 70%, black); color:var(--text); }}
div[data-testid="stMetricValue"] {{ color: var(--text); }}
.kicker{{
  font-weight:800;font-size:22px;letter-spacing:2px;padding:8px 12px;
  border:1px solid var(--border);border-radius:12px;display:inline-block;background:var(--chip);
}}
.stDeckGlJsonChart{{ border-radius:12px; overflow:hidden; border:1px solid var(--border); }}
</style>
        """,
        unsafe_allow_html=True,
    )

def apply_theme(theme: str):
    # CSS variables
    _emit_css_vars(theme)
    # Altair theme
    try:
        alt.themes.register("theme_dark", lambda: _alt_theme(True))
        alt.themes.register("theme_light", lambda: _alt_theme(False))
    except Exception:
        pass
    alt.themes.enable("theme_dark" if theme == "dark" else "theme_light")

# -------------------------- Paths & Const --------------------------
APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "data"
CSV_PATH = DATA_DIR / "datos_puebla.csv"
GEOJSON_PATH = DATA_DIR / "secciones_puebla.geojson"
MODEL_PATH = DATA_DIR / "model.joblib"
ALTA_PRIORIDAD_CSV = DATA_DIR / "alta_prioridad.csv"

# Overlays base (admin)
OVER_MUN = DATA_DIR / "overlays" / "municipios.geojson"
OVER_DL  = DATA_DIR / "overlays" / "distritos_locales.geojson"
OVER_DF  = DATA_DIR / "overlays" / "distritos_federales.geojson"

PALETTE = {"Alta":"#ff4d4f","Media":"#f0a800","Baja":"#3aaed8","Default":"#9aa0a6","Out":"#2a2e35"}

# -------------------------- Estado inicial de tema --------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # valor por defecto

# Aplica el tema actual
apply_theme(st.session_state.theme)

# -------------------------- Helpers varios --------------------------
def ensure_files():
    if not CSV_PATH.exists():
        st.error("Falta data/datos_puebla.csv"); st.stop()
    if not GEOJSON_PATH.exists():
        st.error("Falta data/secciones_puebla.geojson"); st.stop()

@st.cache_data(show_spinner=False)
def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    if "FECHA" in df.columns:
        df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce").dt.date
    for c in ["NUM_PROMOVIDOS", "PAUTA_INVERTIDA", "VISUALIZACIONES"]:
        if c not in df.columns: df[c] = 0
    if "SECCION" in df.columns:
        df["SECCION"] = (df["SECCION"].astype(str).str.replace(r"[^\d]","",regex=True).str.lstrip("0"))
    return df

@st.cache_data(show_spinner=False)
def load_geojson(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

@st.cache_data(show_spinner=False)
def extract_base_features(geojson_obj: dict):
    feats = []
    for f in geojson_obj.get("features", []):
        geom = f.get("geometry"); props = f.get("properties", {})
        if geom:
            feats.append({"type":"Feature","geometry":geom,"properties":{"SECCION":str(props.get("SECCION",""))}})
    return {"type":"FeatureCollection","features":feats}

def hex_to_rgba(hex_color: str, alpha=150):
    h = hex_color.lstrip("#"); r,g,b = (int(h[i:i+2],16) for i in (0,2,4)); return [r,g,b,alpha]

def hhash(path: Path) -> str:
    if not path.exists(): return "—"
    m = hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(1<<16), b""): m.update(chunk)
    return m.hexdigest()[:12]

def view_state_puebla_estado(): return pdk.ViewState(latitude=19.40, longitude=-98.20, zoom=6.7, pitch=0)
def view_state_puebla_capital(): return pdk.ViewState(latitude=19.0379, longitude=-98.2035, zoom=11.2, pitch=0)
def _initial_view(map_action="Puebla estado"): return view_state_puebla_capital() if map_action=="Puebla capital" else view_state_puebla_estado()

def read_fc(path: Path):
    if not path.exists(): return None
    try:
        obj = json.load(open(path,"r",encoding="utf-8"))
        if isinstance(obj, dict) and obj.get("type")=="FeatureCollection":
            feats = obj.get("features", [])
            return obj if isinstance(feats,list) and len(feats)>0 else None
        return None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def build_geo_index(fc: dict):
    sec2geom = {}
    for f in fc.get("features", []):
        sec = str(f.get("properties", {}).get("SECCION","")).strip()
        geom = f.get("geometry")
        if not sec or not geom: continue
        try:
            g = shape(geom)
            if g.is_valid and not g.is_empty: sec2geom[sec] = (g, g.bounds)
        except (ShapelyError, Exception):
            continue
    return sec2geom, set(sec2geom.keys())

def bbox_overlap(b1,b2): return (b1[0]<=b2[2]) and (b1[2]>=b2[0]) and (b1[1]<=b2[3]) and (b1[3]>=b2[1])

# -------------------------- Carga base --------------------------
ensure_files()
df = load_table(CSV_PATH)
geojson_raw  = load_geojson(GEOJSON_PATH)
geojson_base = extract_base_features(geojson_raw)

mun_fc  = read_fc(OVER_MUN)
dloc_fc = read_fc(OVER_DL)
dfed_fc = read_fc(OVER_DF)

# -------------------------- Priorización --------------------------
if ALTA_PRIORIDAD_CSV.exists():
    try:
        df_alta = pd.read_csv(ALTA_PRIORIDAD_CSV)
        df_alta.columns = [c.strip().upper() for c in df_alta.columns]
        col = next((c for c in df_alta.columns if c=="SECCION" or "SECC" in c), None)
        if col:
            altas = (df_alta[col].astype(str).str.replace(r"[^\d]","",regex=True).str.lstrip("0")).dropna().tolist()
            if "PRIORIDAD" not in df.columns: df["PRIORIDAD"] = ""
            df.loc[df["SECCION"].astype(str).isin(altas), "PRIORIDAD"] = "Alta"
    except Exception:
        pass

sec2geom, secciones_validas = build_geo_index(geojson_base)
alta_set  = set(df.loc[df.get("PRIORIDAD","").astype(str).str.upper()=="ALTA","SECCION"].astype(str))
media_set = set()

if alta_set:
    altas_geom = [(s, sec2geom[s][0], sec2geom[s][1]) for s in alta_set if s in sec2geom]
    for sec, (geom,bbox) in sec2geom.items():
        if sec in alta_set: continue
        cand = [(sA,gA) for (sA,gA,bA) in altas_geom if bbox_overlap(bbox,bA)]
        for _, gA in cand:
            try:
                if geom.touches(gA): media_set.add(sec); break
            except Exception:
                pass

if "PRIORIDAD" not in df.columns: df["PRIORIDAD"] = ""
df["SECCION"] = df["SECCION"].astype(str)
df.loc[df["SECCION"].isin(media_set) & ~df["SECCION"].isin(alta_set), "PRIORIDAD"] = "Media"
resto = df["SECCION"].isin(secciones_validas) & ~df["SECCION"].isin(alta_set | media_set)
df.loc[resto & (~df["PRIORIDAD"].isin(["Alta","Media","Baja"])), "PRIORIDAD"] = "Baja"

# ---- Pintado secciones
lookup_prio = {}
if "SECCION" in df.columns and "PRIORIDAD" in df.columns:
    for r in df[["SECCION","PRIORIDAD"]].itertuples(index=False):
        lookup_prio[str(r.SECCION)] = str(r.PRIORIDAD)

features = []
for f in extract_base_features(geojson_raw)["features"]:
    sec = f["properties"]["SECCION"]; props = {"SECCION": sec}
    if sec in lookup_prio:
        col = PALETTE.get(lookup_prio[sec], PALETTE["Default"])
        fill = hex_to_rgba(col, 170); label = lookup_prio[sec]
    else:
        fill = hex_to_rgba(PALETTE["Out"], 110); label = "Fuera de filtro"
    props["__FILL__"] = fill; props["__LABEL__"] = label
    features.append({"type":"Feature","geometry":f["geometry"],"properties":props})
geojson_secciones_painted = {"type":"FeatureCollection","features":features}

# -------------------------- Estado de filtros aplicados --------------------------
def _default_filters():
    f = {}
    if "FECHA" in df.columns and df["FECHA"].notna().any():
        min_d, max_d = df["FECHA"].min(), df["FECHA"].max()
        f["periodo"] = (max(min_d, max_d - timedelta(days=7)), max_d)
    else:
        f["periodo"] = None
    f["sel_prioridad"] = "Todos"
    f["sel_municipio"] = "Todos"
    f["sel_dfederal"]  = "Todos"
    f["sel_dlocal"]    = "Todos"
    f["sel_map_action"]= "Puebla estado"
    return f

if "active_filters" not in st.session_state:
    st.session_state.active_filters = _default_filters()

# -------------------------- Sidebar --------------------------
with st.sidebar:
    # Switch de tema
    st.markdown("#### Apariencia")
    theme_choice = st.toggle("🌗 Modo oscuro", value=(st.session_state.theme=="dark"))
    new_theme = "dark" if theme_choice else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        apply_theme(new_theme)
        st.rerun()

    st.markdown("### Filtros")

    # Preparamos opciones
    prioridades = ["Todos"] + (sorted(df["PRIORIDAD"].dropna().astype(str).unique()) if "PRIORIDAD" in df.columns else [])
    municipios = ["Todos"] + (df.get("MUNICIPIO", pd.Series([], dtype=str)).dropna().astype(str).sort_values().unique().tolist())
    distritos_fed = ["Todos"] + (df.get("DISTRITO FEDERAL", pd.Series([], dtype=str)).dropna().astype(str).sort_values().unique().tolist())
    distritos_loc = ["Todos"] + (df.get("DISTRITO LOCAL", pd.Series([], dtype=str)).dropna().astype(str).sort_values().unique().tolist())

    # Tomamos los valores actualmente activos para precargar el formulario
    AF = st.session_state.active_filters
    default_periodo = AF.get("periodo", None)

    with st.form("filters_form", clear_on_submit=False):
        # Periodo
        start_date, end_date = None, None
        if "FECHA" in df.columns and df["FECHA"].notna().any():
            min_d, max_d = df["FECHA"].min(), df["FECHA"].max()
            if isinstance(default_periodo, tuple) and len(default_periodo) == 2:
                init_val = (default_periodo[0], default_periodo[1])
            else:
                init_val = (max(min_d, max_d - timedelta(days=7)), max_d)
            picked = st.date_input("Periodo", value=init_val, min_value=min_d, max_value=max_d, key="periodo_form")
            sd, ed = (picked if isinstance(picked, tuple) and len(picked)==2 else (picked, picked))
            if sd and ed and sd>ed: sd, ed = ed, sd
            start_date, end_date = sd, ed
        else:
            st.caption("No hay FECHA: se muestra todo.")

        sel_prio = st.selectbox("Prioridad", prioridades, index=(prioridades.index(AF["sel_prioridad"]) if AF["sel_prioridad"] in prioridades else 0), key="sel_prioridad_form")
        sel_mun  = st.selectbox("Municipio", municipios, index=(municipios.index(AF["sel_municipio"]) if AF["sel_municipio"] in municipios else 0), key="sel_municipio_form")
        sel_dfed = st.selectbox("Distrito federal", distritos_fed, index=(distritos_fed.index(AF["sel_dfederal"]) if AF["sel_dfederal"] in distritos_fed else 0), key="sel_dfederal_form")
        sel_dloc = st.selectbox("Distrito local", distritos_loc, index=(distritos_loc.index(AF["sel_dlocal"]) if AF["sel_dlocal"] in distritos_loc else 0), key="sel_dlocal_form")

        st.markdown("---")
        map_action = st.radio("Enfoque del mapa", ["Puebla estado","Puebla capital"], index=(0 if AF.get("sel_map_action","Puebla estado")=="Puebla estado" else 1), horizontal=True, key="sel_map_action_form")
        st.markdown("---")

        submitted = st.form_submit_button("Aplicar filtros ✅")

    # Botón para recargar datos (limpia cachés y fuerza un solo rerun)
    if st.button("🔄 Recargar datos"):
        st.cache_data.clear()
        st.rerun()

# Si el usuario presionó Aplicar filtros, actualizamos los activos
if submitted:
    st.session_state.active_filters = {
        "periodo": (start_date, end_date) if start_date and end_date else None,
        "sel_prioridad": sel_prio,
        "sel_municipio": sel_mun,
        "sel_dfederal": sel_dfed,
        "sel_dlocal": sel_dloc,
        "sel_map_action": map_action,
    }

# -------------------------- Aplicar filtros activos a df --------------------------
AF = st.session_state.active_filters
df_f = df.copy()

if AF.get("periodo") and "FECHA" in df_f.columns:
    sd, ed = AF["periodo"]
    df_f = df_f[(df_f["FECHA"]>=sd) & (df_f["FECHA"]<=ed)]
if "PRIORIDAD" in df_f.columns and AF["sel_prioridad"]!="Todos":
    df_f = df_f[df_f["PRIORIDAD"].astype(str)==str(AF["sel_prioridad"])]
if "MUNICIPIO" in df_f.columns and AF["sel_municipio"]!="Todos":
    df_f = df_f[df_f["MUNICIPIO"].astype(str)==str(AF["sel_municipio"])]
if "DISTRITO FEDERAL" in df_f.columns and AF["sel_dfederal"]!="Todos":
    df_f = df_f[df_f["DISTRITO FEDERAL"].astype(str)==str(AF["sel_dfederal"])]
if "DISTRITO LOCAL" in df_f.columns and AF["sel_dlocal"]!="Todos":
    df_f = df_f[df_f["DISTRITO LOCAL"].astype(str)==str(AF["sel_dlocal"])]

# -------------------------- Modelo (opcional) --------------------------
def score_to_band(series, labels=("Baja","Media","Alta","Muy alta"), q=4):
    s = pd.Series(series).astype(float)
    if s.empty or s.nunique()<=1:
        return pd.Series([labels[1]]*len(s), index=s.index)
    try:
        return pd.qcut(s, q=q, labels=labels, duplicates="drop")
    except Exception:
        uq = max(2, min(q, s.nunique()))
        bins = np.linspace(s.min(), s.max(), num=uq+1)
        return pd.cut(s, bins=bins, labels=labels[:uq], include_lowest=True, duplicates="drop")

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

if model is not None and len(df_f)>0:
    try:
        df_f["SCORE"] = model.predict_proba(Xlog)[:,1]
    except Exception:
        raw = model.predict(Xlog)
        df_f["SCORE"] = (raw - raw.min())/(raw.max()-raw.min()+1e-6)
    df_f["RANGO"] = score_to_band(df_f["SCORE"])

# -------------------------- Enriquecer tooltips overlays --------------------------
def _first_nonempty(d: dict, keys):
    for k in keys:
        v = d.get(k)
        if v is None: continue
        if isinstance(v, dict): v = v.get("num") or v.get("id") or v.get("value")
        s = str(v).strip()
        if s and s not in {"-","—","None"}: return s
    return None

def enrich_municipios(fc):
    if not fc: return fc
    for f in fc.get("features", []):
        p = f.setdefault("properties", {})
        name = p.get("NOM_MUN") or p.get("NOMBRE") or p.get("NAME") or p.get("MUN_NAME") or p.get("MUNICIPIO_N") or p.get("MUNNOM")
        mid  = p.get("MUNICIPIO") or p.get("CVE_MUN") or p.get("MUN") or p.get("MUNID")
        p["__MUN_NAME"] = str(name or "—"); p["__MUN_ID"] = str(mid or "—")
    return fc

def enrich_distritos_locales(fc):
    if not fc: return fc
    for f in fc.get("features", []):
        p = f.setdefault("properties", {})
        num = _first_nonempty(p, ["DISTRITO","NUMERO","NUM_DIST","CVE_LOC","CVE_DIST","DIST_LOC","DIST","DL","ID","IDDIST","NOM_DIST","DLOC"])
        p["__DL_NUM"] = num if num else "—"
    return fc

def enrich_distritos_federales(fc):
    if not fc: return fc
    for f in fc.get("features", []):
        p = f.setdefault("properties", {})
        num = _first_nonempty(p, ["DISTRITO","NUMERO","CVE_FED","CVE_DIST","DIST_FED","DIST","DF","ID","IDDIST","NOM_DIST","DFED"])
        p["__DF_NUM"] = num if num else "—"
    return fc

mun_fc  = enrich_municipios(mun_fc)
dloc_fc = enrich_distritos_locales(dloc_fc)
dfed_fc = enrich_distritos_federales(dfed_fc)

# -------------------------- Render con OSM (tema-aware) --------------------------
def basemap_url(theme: str) -> str:
    # Carto basemaps (OSM-based). Usamos subdominio 'a' para simplificar.
    if theme == "dark":
        return "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
    else:
        return "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"

def theme_line_color(theme: str):
    # Oscuro: líneas claras / Claro: líneas oscuras
    return [230, 235, 245, 220] if theme == "dark" else [50, 60, 70, 200]

def theme_fill_gray(theme: str, alpha=40):
    # Relleno tenue para overlays administrativos
    return [255, 255, 255, alpha] if theme == "light" else [30, 35, 45, alpha+20]

LINE_WIDTH = 1.2

def render_map(layers, vstate, theme: str):
    layers_final = [
        pdk.Layer(
            "TileLayer",
            data=basemap_url(theme),
            minZoom=0, maxZoom=19, tileSize=256,
        )
    ]
    layers_final.extend(layers)
    st.pydeck_chart(
        pdk.Deck(layers=layers_final, initial_view_state=vstate, map_style=None),
        use_container_width=True
    )

# -------------------------- Header --------------------------
c1, c2 = st.columns([1, 6], vertical_alignment="center")
with c1: st.markdown('<div class="kicker">CHR</div>', unsafe_allow_html=True)
with c2: st.title("Tablero Puebla — v4.0 (tema claro/oscuro + OSM)")

# -------------------------- MAPAS --------------------------
tab_sec, tab_mun, tab_dloc, tab_dfed = st.tabs(["Secciones","Municipios","Distritos locales","Distritos federales"])

with tab_sec:
    vstate = _initial_view(st.session_state.active_filters.get("sel_map_action","Puebla estado"))
    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson_secciones_painted,
        pickable=True, auto_highlight=True,
        stroked=True, filled=True, opacity=1.0,
        get_fill_color="properties.__FILL__",
        get_line_color=theme_line_color(st.session_state.theme),
        line_width_min_pixels=LINE_WIDTH
    )
    render_map([layer], vstate, st.session_state.theme)

with tab_mun:
    if not mun_fc:
        st.warning(f"No se encontró {OVER_MUN}.")
    else:
        vstate = _initial_view(st.session_state.active_filters.get("sel_map_action","Puebla estado"))
        layer = pdk.Layer(
            "GeoJsonLayer",
            data=mun_fc, pickable=True, auto_highlight=True,
            stroked=True, filled=True, opacity=0.18,
            get_fill_color=theme_fill_gray(st.session_state.theme, alpha=40),
            get_line_color=theme_line_color(st.session_state.theme),
            line_width_min_pixels=LINE_WIDTH
        )
        render_map([layer], vstate, st.session_state.theme)

with tab_dloc:
    if not dloc_fc:
        st.warning(f"No se encontró {OVER_DL}.")
    else:
        vstate = _initial_view(st.session_state.active_filters.get("sel_map_action","Puebla estado"))
        layer = pdk.Layer(
            "GeoJsonLayer",
            data=dloc_fc, pickable=True, auto_highlight=True,
            stroked=True, filled=True, opacity=0.18,
            get_fill_color=theme_fill_gray(st.session_state.theme, alpha=40),
            get_line_color=theme_line_color(st.session_state.theme),
            line_width_min_pixels=LINE_WIDTH
        )
        render_map([layer], vstate, st.session_state.theme)

with tab_dfed:
    if not dfed_fc:
        st.warning(f"No se encontró {OVER_DF}.")
    else:
        vstate = _initial_view(st.session_state.active_filters.get("sel_map_action","Puebla estado"))
        layer = pdk.Layer(
            "GeoJsonLayer",
            data=dfed_fc, pickable=True, auto_highlight=True,
            stroked=True, filled=True, opacity=0.18,
            get_fill_color=theme_fill_gray(st.session_state.theme, alpha=40),
            get_line_color=theme_line_color(st.session_state.theme),
            line_width_min_pixels=LINE_WIDTH
        )
        render_map([layer], vstate, st.session_state.theme)

st.divider()

# -------------------------- TABS de contenido --------------------------
tab_prev, tab_day, tab_res = st.tabs(["📋 Previa (Campaña)","🗳️ Día D","📈 Resultados"])

with tab_prev:
    st.subheader("Métricas clave y resumen")
    def safe_sum(col): return int(pd.to_numeric(df_f.get(col, pd.Series([])), errors="coerce").fillna(0).sum())
    c = st.columns(4)
    c[0].metric("Secciones", df_f["SECCION"].nunique() if "SECCION" in df_f.columns else len(df_f))
    c[1].metric("Promovidos", safe_sum("NUM_PROMOVIDOS"))
    c[2].metric("Pauta invertida", f"${safe_sum('PAUTA_INVERTIDA'):,}")
    c[3].metric("Visualizaciones", f"{safe_sum('VISUALIZACIONES'):,}")

    if "PRIORIDAD" in df_f.columns and df_f["PRIORIDAD"].notna().any():
        pr = (df_f["PRIORIDAD"].astype(str).value_counts().rename_axis("PRIORIDAD").reset_index(name="SECCIONES"))
        order_map = {"Alta":0,"Media":1,"Baja":2}
        pr["ORD"] = pr["PRIORIDAD"].map(order_map).fillna(99)
        pr = pr.sort_values(["ORD","SECCIONES"], ascending=[True,False])
        pr["COLOR"] = pr["PRIORIDAD"].map(lambda x: PALETTE.get(x, PALETTE["Default"]))
        chart = (
            alt.Chart(pr)
            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
            .encode(
                x=alt.X("SECCIONES:Q", title="Secciones"),
                y=alt.Y("PRIORIDAD:N", sort=pr["PRIORIDAD"].tolist(), title=None),
                color=alt.Color("PRIORIDAD:N", scale=alt.Scale(range=list(pr["COLOR"])), legend=None),
                tooltip=["PRIORIDAD","SECCIONES"],
            ).properties(title="Secciones por prioridad", height=240)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No hay columna PRIORIDAD para graficar.")

with tab_day:
    st.subheader("Monitoreo operativo y countdown")
    ELECTION_DATE = date(2027, 6, 6)
    days_left = (ELECTION_DATE - date.today()).days
    c = st.columns(3)
    c[0].metric("Días para la elección (estimado)", f"{max(days_left, 0)}")
    c[1].metric("% casillas instaladas", "—")
    c[2].metric("Incidencias reportadas", "—")
    st.caption("Actualiza la fecha cuando la autoridad la publique oficialmente.")

with tab_res:
    st.subheader("Resultados preliminares (plantillas)")
    st.write("Listo para integrar actas, PREP y comparativos.")
    st.caption("Incluye huella de datos para trazabilidad.")

st.markdown("---")
st.caption(f"Huella datos_puebla.csv: {hhash(CSV_PATH)} · Modelo: {'OK' if MODEL_PATH.exists() else 'N/D'}")
