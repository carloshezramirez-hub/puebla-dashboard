import streamlit as st
import pandas as pd
import pydeck as pdk
import altair as alt
import json
from pathlib import Path
from datetime import datetime, timedelta

# ================== CONFIG GENERAL ==================
st.set_page_config(page_title="Tablero Puebla", page_icon="📊", layout="wide")

# Logo simple (texto) y encabezado
c1, c2 = st.columns([1, 6])
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
    st.title("Tablero Puebla")

# Paleta (pensada para fondo oscuro)
PALETTE = {
    "Alta": "#ff4d4f",    # rojo vivo
    "Media": "#f0a800",   # ámbar
    "Baja": "#3aaed8",    # azul claro
    "Default": "#9aa0a6", # gris
    "Out": "#2a2e35"      # gris muy oscuro para fuera de filtro
}

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "datos_puebla.csv"
GEOJSON_PATH = DATA_DIR / "secciones_puebla.geojson"

# ================== CARGA CON CACHÉ ==================
@st.cache_data(show_spinner=False)
def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    # Normaliza fecha si existe
    if "FECHA" in df.columns:
        df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce").dt.date
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

def ensure_files():
    if not CSV_PATH.exists():
        st.error("No se encontró `data/datos_puebla.csv`. Colócalo y recarga.")
        st.stop()
    if not GEOJSON_PATH.exists():
        st.error("No se encontró `data/secciones_puebla.geojson`. Colócalo y recarga.")
        st.stop()

ensure_files()
df = load_table(CSV_PATH)
geojson_obj = load_geojson(GEOJSON_PATH)

# ================== SIDEBAR: FILTROS ==================
with st.sidebar:
    st.markdown("### Filtros")

    # 1) Periodo (por defecto últimos 7 días) — solo aplica si hay FECHA
    today = datetime.utcnow().date()
    default_start = today - timedelta(days=7)
    if "FECHA" in df.columns and df["FECHA"].notna().any():
        min_date = df["FECHA"].min()
        max_date = df["FECHA"].max()
        start_date, end_date = st.date_input(
            "Periodo",
            value=(max(default_start, min_date), max_date),
            min_value=min_date,
            max_value=max_date
        )
    else:
        st.caption("No se detectó columna FECHA: se mostrará todo el periodo.")
        start_date, end_date = None, None

    # 2) Filtros por prioridad, municipio y distrito
    prioridades = ["Todos"] + sorted(df["PRIORIDAD"].dropna().astype(str).unique()) if "PRIORIDAD" in df.columns else ["Todos"]
    municipios  = ["Todos"] + sorted(df["MUNICIPIO"].dropna().astype(str).unique()) if "MUNICIPIO" in df.columns else ["Todos"]
    distritos   = ["Todos"] + sorted(df["DISTRITO LOCAL"].dropna().astype(str).unique()) if "DISTRITO LOCAL" in df.columns else ["Todos"]

    sel_prio = st.selectbox("Prioridad", prioridades, index=0)
    sel_mun  = st.selectbox("Municipio", municipios, index=0)
    sel_dist = st.selectbox("Distrito Local", distritos, index=0)

    st.markdown("---")
    # Controles de mapa “dinámicos”
    map_action = st.radio(
        "Enfoque del mapa",
        options=["Puebla capital", "Centrar en datos filtrados", "Libre"],
        index=0
    )

# ================== APLICAR FILTROS ==================
df_f = df.copy()

# Periodo
if start_date and end_date and "FECHA" in df_f.columns:
    df_f = df_f[(df_f["FECHA"] >= start_date) & (df_f["FECHA"] <= end_date)]

# Dimensiones
if "PRIORIDAD" in df_f.columns and sel_prio != "Todos":
    df_f = df_f[df_f["PRIORIDAD"].astype(str) == str(sel_prio)]
if "MUNICIPIO" in df_f.columns and sel_mun != "Todos":
    df_f = df_f[df_f["MUNICIPIO"].astype(str) == str(sel_mun)]
if "DISTRITO LOCAL" in df_f.columns and sel_dist != "Todos":
    df_f = df_f[df_f["DISTRITO LOCAL"].astype(str) == str(sel_dist)]

# ================== TABS (mejor UX) ==================
tab1, tab2, tab3 = st.tabs(["📌 Métricas y resumen", "📈 Gráficos", "🗺️ Mapa"])

# ================== MÉTRICAS + RESUMEN ==================
with tab1:
    st.subheader("Métricas clave (periodo y filtros aplicados)")

    def safe_sum(col):
        return int(pd.to_numeric(df_f.get(col, pd.Series([])), errors="coerce").fillna(0).sum())

    num_seccionales    = len(df_f)
    num_promovidos     = safe_sum("NUM_PROMOVIDOS")
    pauta_invertida    = safe_sum("PAUTA_INVERTIDA")
    num_visualizaciones= safe_sum("VISUALIZACIONES")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Secciones", num_seccionales)
    m2.metric("Promovidos", num_promovidos)
    m3.metric("Pauta invertida", pauta_invertida)
    m4.metric("Visualizaciones", num_visualizaciones)

    st.markdown("#### ¿Qué significa cada cosa?")
    st.write(
        "- **Pauta invertida**: el gasto total en difusión/publicidad en el periodo seleccionado.\n"
        "- **Visualizaciones**: cuántas veces se vieron tus piezas/contenidos en el periodo.\n"
        "- **Promovidos**: personas confirmadas a favor/activadas (suma en el periodo).\n"
        "- **Secciones**: número de filas/segmentos (tras filtros).\n"
        "Si **no** tienes columna `FECHA`, estas métricas serán sobre **todo** tu histórico."
    )

# ================== GRÁFICOS ==================
with tab2:
    st.subheader("Gráficos (más intuitivos)")

    # A) Barras horizontales por Prioridad (más claro que la anterior)
    if "PRIORIDAD" in df_f.columns:
        prio_counts = (
            df_f["PRIORIDAD"].astype(str)
            .value_counts()
            .rename_axis("PRIORIDAD").reset_index(name="SECCIONES")
        )
        # Orden lógico: Alta > Media > Baja > otras
        order_map = {"Alta": 0, "Media": 1, "Baja": 2}
        prio_counts["ORD"] = prio_counts["PRIORIDAD"].map(order_map).fillna(99)
        prio_counts = prio_counts.sort_values(["ORD", "SECCIONES"], ascending=[True, False])

        # Colores por prioridad
        prio_counts["COLOR"] = prio_counts["PRIORIDAD"].map(lambda x: PALETTE.get(x, PALETTE["Default"]))

        chart_prio = (
            alt.Chart(prio_counts)
            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
            .encode(
                x=alt.X("SECCIONES:Q", title="Número de secciones"),
                y=alt.Y("PRIORIDAD:N", sort=prio_counts["PRIORIDAD"].tolist(), title=None),
                color=alt.Color("PRIORIDAD:N", scale=alt.Scale(range=list(prio_counts["COLOR"])), legend=None),
                tooltip=["PRIORIDAD", "SECCIONES"]
            )
            .properties(title="Secciones por prioridad (más intuitivo)")
        )
        st.altair_chart(chart_prio, use_container_width=True)
    else:
        st.info("No hay columna PRIORIDAD para graficar.")

    # B) Top Municipios (barras horizontales, no dona)
    if "MUNICIPIO" in df_f.columns:
        top_mun = (
            df_f["MUNICIPIO"].astype(str)
            .value_counts()
            .rename_axis("MUNICIPIO").reset_index(name="SECCIONES")
            .head(8)
        )
        chart_mun = (
            alt.Chart(top_mun.sort_values("SECCIONES"))
            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
            .encode(
                x=alt.X("SECCIONES:Q", title="Número de secciones"),
                y=alt.Y("MUNICIPIO:N", sort=None, title=None),
                color=alt.value("#5C7CFA"),
                tooltip=["MUNICIPIO", "SECCIONES"]
            )
            .properties(title="Top 8 municipios por número de secciones")
        )
        st.altair_chart(chart_mun, use_container_width=True)
    else:
        st.info("No hay columna MUNICIPIO para graficar.")

# ================== MAPA ==================
with tab3:
    st.subheader("Mapa de secciones (modo oscuro)")

    # Mapeo de prioridad por SECCION para colorear
    prio_by_seccion = {}
    if "SECCION" in df_f.columns and "PRIORIDAD" in df_f.columns:
        prio_by_seccion = {
            str(r.SECCION): str(r.PRIORIDAD)
            for r in df_f[["SECCION", "PRIORIDAD"]].dropna().itertuples(index=False)
        }

    # Enriquecer propiedades con color y label de prioridad
    features = geojson_obj.get("features", [])
    for feat in features:
        props = feat.get("properties", {})
        sec = str(props.get("SECCION", ""))
        prioridad = prio_by_seccion.get(sec)
        if prioridad:
            color = hex_to_rgba(PALETTE.get(prioridad, PALETTE["Default"]), alpha=160)
            props["__PRIORIDAD__"] = prioridad
        else:
            color = hex_to_rgba(PALETTE["Out"], alpha=140)
            props["__PRIORIDAD__"] = "Fuera de filtro"
        props["__FILL_COLOR__"] = color

    # Calcular vista
    def center_on_puebla_capital():
        return pdk.ViewState(latitude=19.0379, longitude=-98.2035, zoom=11.2, pitch=0, bearing=0)

    def bbox_center_of_filtered(gj: dict, active_only: bool = True):
        # Centra en las features que están dentro del filtro (las que no quedaron 'Fuera de filtro')
        lats, lons = [], []
        for f in gj.get("features", []):
            pr = f.get("properties", {})
            if active_only and pr.get("__PRIORIDAD__") == "Fuera de filtro":
                continue
            geom = f.get("geometry", {})
            coords = geom.get("coordinates")
            if not coords:
                continue
            # Soporte para Polygons y MultiPolygons
            def collect(cs):
                for ring in cs[0]:
                    lons.append(ring[0]); lats.append(ring[1])
            if geom.get("type") == "Polygon":
                for ring in geom["coordinates"][0]:
                    lons.append(ring[0]); lats.append(ring[1])
            elif geom.get("type") == "MultiPolygon":
                for poly in geom["coordinates"]:
                    for ring in poly[0]:
                        lons.append(ring[0]); lats.append(ring[1])
        if lats and lons:
            return pdk.ViewState(latitude=sum(lats)/len(lats), longitude=sum(lons)/len(lons), zoom=10.5)
        return center_on_puebla_capital()

    if map_action == "Puebla capital":
        view_state = center_on_puebla_capital()
    elif map_action == "Centrar en datos filtrados":
        view_state = bbox_center_of_filtered(geojson_obj, active_only=True)
    else:
        # Libre: deja un zoom cómodo al estado
        view_state = pdk.ViewState(latitude=19.0, longitude=-97.95, zoom=8.3)

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson_obj,
        pickable=True,
        auto_highlight=True,
        stroked=True,
        filled=True,
        get_fill_color="properties.__FILL_COLOR__",
        get_line_color=[180, 180, 200, 120],
        line_width_min_pixels=1,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>Sección:</b> {SECCION}<br/><b>Prioridad:</b> {__PRIORIDAD__}",
            "style": {"backgroundColor": "rgba(0,0,0,.85)", "color": "#fff"}
        },
        map_style="dark"
    )
    st.pydeck_chart(deck, use_container_width=True)

    # Leyenda
    st.markdown("#### Leyenda")
    cols = st.columns(4)
    legend = [("Alta", PALETTE["Alta"]), ("Media", PALETTE["Media"]), ("Baja", PALETTE["Baja"]), ("Fuera de filtro", "#2a2e35")]
    for c, (name, colhex) in zip(cols, legend):
        c.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:8px;">
              <div style="width:14px;height:14px;border-radius:4px;background:{colhex};border:1px solid rgba(255,255,255,.2);"></div>
              <span style="font-size:0.9rem;">{name}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
