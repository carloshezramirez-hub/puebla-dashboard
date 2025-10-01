import streamlit as st
import pandas as pd
import pydeck as pdk
import altair as alt
import json
from pathlib import Path

# ================== CONFIG BÁSICA ==================
st.set_page_config(
    page_title="Tablero Puebla",
    page_icon="📊",
    layout="wide"
)

# Paleta de colores (look pro y consistente)
PALETTE = {
    "Alta": "#d7191c",
    "Media": "#fdae61",
    "Baja": "#abd9e9",
    "Otra": "#2c7fb8",
    "Default": "#bdbdbd"
}

# ================== AUTENTICACIÓN SIMPLE ==================
# La contraseña se guarda en Secrets: [auth] password="..."
APP_PASSWORD = st.secrets.get("auth", {}).get("password")

def ask_password():
    st.markdown("""
    <div style="display:flex;min-height:60vh;align-items:center;justify-content:center;">
      <div style="max-width:420px;width:100%;padding:24px;border:1px solid #eee;border-radius:16px;background:white;box-shadow:0 6px 24px rgba(0,0,0,.06);">
        <h3 style="margin-top:0;margin-bottom:8px;font-weight:700;">Acceso</h3>
        <p style="margin-top:0;color:#666;">Ingresa la contraseña para ver el tablero.</p>
    """, unsafe_allow_html=True)
    with st.form("login", clear_on_submit=False):
        pwd = st.text_input("Contraseña", type="password")
        ok = st.form_submit_button("Entrar")
    st.markdown("</div></div>", unsafe_allow_html=True)
    return ok, pwd

if APP_PASSWORD:
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if not st.session_state.auth_ok:
        ok, pwd = ask_password()
        if ok:
            if pwd == APP_PASSWORD:
                st.session_state.auth_ok = True
                st.rerun()
            else:
                st.error("Contraseña incorrecta.")
                st.stop()
else:
    st.warning("No hay contraseña configurada (Secrets). Te recomendamos agregar una.")

# ================== RUTAS DE ARCHIVOS ==================
DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "datos_puebla.csv"              # <- tu tabla principal
GEOJSON_PATH = DATA_DIR / "secciones_puebla.geojson"  # <- tu mapa de secciones

# ================== CARGA CON CACHÉ (RÁPIDO) ==================
@st.cache_data(show_spinner=False)
def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.stop()
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def load_geojson(path: Path) -> dict:
    if not path.exists():
        st.stop()
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    return gj

@st.cache_data(show_spinner=False)
def hex_to_rgba(hex_color: str, alpha: int = 140):
    """Convierte '#RRGGBB' a [R,G,B,A] con alfa (0-255)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return [r, g, b, alpha]

# ================== CARGA DE DATOS ==================
try:
    df = load_table(CSV_PATH)
except Exception:
    st.error("No se encontró `data/datos_puebla.csv`. Coloca tus datos allí y recarga.")
    st.stop()

try:
    geojson_obj = load_geojson(GEOJSON_PATH)
except Exception:
    st.error("No se encontró `data/secciones_puebla.geojson`. Coloca tu GeoJSON allí y recarga.")
    st.stop()

# ================== SIDEBAR: FILTROS ==================
with st.sidebar:
    st.markdown("### Filtros")
    # Listas seguras
    prioridades = ["Todos"]
    if "PRIORIDAD" in df.columns:
        prioridades += sorted([str(x) for x in df["PRIORIDAD"].dropna().unique()])

    municipios = ["Todos"]
    if "MUNICIPIO" in df.columns:
        municipios += sorted([str(x) for x in df["MUNICIPIO"].dropna().unique()])

    distritos = ["Todos"]
    if "DISTRITO LOCAL" in df.columns:
        distritos += sorted([str(x) for x in df["DISTRITO LOCAL"].dropna().unique()])

    sel_prio = st.selectbox("Prioridad", prioridades, index=0)
    sel_mun = st.selectbox("Municipio", municipios, index=0)
    sel_dist = st.selectbox("Distrito Local", distritos, index=0)

# Aplica filtros
df_f = df.copy()
if "PRIORIDAD" in df_f.columns and sel_prio != "Todos":
    df_f = df_f[df_f["PRIORIDAD"].astype(str) == str(sel_prio)]
if "MUNICIPIO" in df_f.columns and sel_mun != "Todos":
    df_f = df_f[df_f["MUNICIPIO"].astype(str) == str(sel_mun)]
if "DISTRITO LOCAL" in df_f.columns and sel_dist != "Todos":
    df_f = df_f[df_f["DISTRITO LOCAL"].astype(str) == str(sel_dist)]

# ================== MÉTRICAS + RESUMEN ==================
st.markdown("## Métricas clave y resumen")

def safe_sum(col):
    return int(pd.to_numeric(df_f.get(col, pd.Series([])), errors="coerce").fillna(0).sum())

num_seccionales = len(df_f)
num_promovidos = safe_sum("NUM_PROMOVIDOS")
pauta_invertida = safe_sum("PAUTA_INVERTIDA")
num_visualizaciones = safe_sum("VISUALIZACIONES")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Secciones", num_seccionales)
m2.metric("Promovidos", num_promovidos)
m3.metric("Pauta invertida", pauta_invertida)
m4.metric("Visualizaciones", num_visualizaciones)

st.caption("El tablero muestra únicamente información (no permite subir archivos). Los filtros están a la izquierda.")

# ================== GRÁFICOS ==================
st.markdown("## Gráficos")

# 1) Barras por Prioridad
if "PRIORIDAD" in df_f.columns:
    prio_counts = df_f["PRIORIDAD"].astype(str).value_counts().reset_index()
    prio_counts.columns = ["PRIORIDAD", "CUENTA"]
    # Mapea colores por prioridad
    prio_counts["COLOR"] = prio_counts["PRIORIDAD"].map(lambda x: PALETTE.get(x, PALETTE["Default"]))
    chart1 = (
        alt.Chart(prio_counts)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("PRIORIDAD:N", sort="-y", title="Prioridad"),
            y=alt.Y("CUENTA:Q", title="Secciones"),
            color=alt.Color("PRIORIDAD:N", scale=alt.Scale(range=list(prio_counts["COLOR"])), legend=None),
            tooltip=["PRIORIDAD", "CUENTA"]
        )
        .properties(title="Secciones por prioridad")
    )
    st.altair_chart(chart1, use_container_width=True)

# 2) Dona por Municipio (Top 8)
if "MUNICIPIO" in df_f.columns:
    mun_counts = df_f["MUNICIPIO"].astype(str).value_counts().reset_index().head(8)
    mun_counts.columns = ["MUNICIPIO", "CUENTA"]
    chart2 = (
        alt.Chart(mun_counts)
        .mark_arc(innerRadius=60)
        .encode(
            theta="CUENTA:Q",
            color=alt.Color("MUNICIPIO:N", legend=None),
            tooltip=["MUNICIPIO", "CUENTA"]
        )
        .properties(title="Municipios (Top 8)")
    )
    st.altair_chart(chart2, use_container_width=True)

# ================== MAPA ==================
st.markdown("## Mapa de secciones (Puebla)")

# Prepara un dict {SECCION -> PRIORIDAD} del df filtrado
prio_by_seccion = {}
if "SECCION" in df_f.columns and "PRIORIDAD" in df_f.columns:
    tmp = df_f[["SECCION", "PRIORIDAD"]].dropna()
    prio_by_seccion = {str(r.SECCION): str(r.PRIORIDAD) for r in tmp.itertuples(index=False)}

# Enriquecer cada feature con COLOR según PRIORIDAD del filtro
features = geojson_obj.get("features", [])
for feat in features:
    props = feat.get("properties", {})
    sec = str(props.get("SECCION", ""))
    prioridad = prio_by_seccion.get(sec)  # puede ser None si está filtrado fuera
    if prioridad is None:
        # Si la sección no cumple el filtro, la pintamos gris muy claro
        color = hex_to_rgba("#E0E0E0", alpha=80)
    else:
        color = hex_to_rgba(PALETTE.get(prioridad, PALETTE["Default"]), alpha=140)
    props["__FILL_COLOR__"] = color
    props["__PRIORIDAD__"] = prioridad if prioridad else "Fuera de filtro"

# Capa GeoJSON
layer = pdk.Layer(
    "GeoJsonLayer",
    data=geojson_obj,
    pickable=True,
    stroked=True,
    filled=True,
    get_fill_color="properties.__FILL_COLOR__",
    get_line_color=[70, 70, 90, 180],
    line_width_min_pixels=1,
)

# Vista centrada en Puebla
view_state = pdk.ViewState(latitude=19.0379, longitude=-98.2035, zoom=9.4)

r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={
        "html": "<b>Sección:</b> {SECCION}<br/><b>Prioridad:</b> {__PRIORIDAD__}",
        "style": {"backgroundColor": "white", "color": "#111"}
    },
    map_style="light"
)
st.pydeck_chart(r, use_container_width=True)

# Leyenda simple
st.markdown("#### Leyenda")
legend_cols = st.columns(4)
legend_items = ["Alta", "Media", "Baja", "Fuera de filtro"]
legend_colors = [PALETTE["Alta"], PALETTE["Media"], PALETTE["Baja"], "#E0E0E0"]
for c, name, colhex in zip(legend_cols, legend_items, legend_colors):
    c.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;">
        <div style="width:14px;height:14px;border-radius:4px;background:{colhex};border:1px solid #999;"></div>
        <span style="font-size:0.9rem;">{name}</span>
    </div>
    """, unsafe_allow_html=True)
