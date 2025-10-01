import streamlit as st
import pandas as pd
import pydeck as pdk
import altair as alt
import json

st.set_page_config(page_title="Tablero Puebla", layout="wide")

# =============== Ayudas rápidas en la interfaz ===============
with st.sidebar:
    st.title("Tablero Puebla (Demo)")
    st.markdown("""
Sube tu **Excel o CSV** con columnas como:
- DISTRITO FEDERAL
- DISTRITO LOCAL
- TIPO DE SECCION
- MUNICIPIO
- NOMBRE DEL MUNICIPIO
- TIPO DE COLONIA
- NOMBRE DE LA COLONIA
- CP
- PRIORIDAD
Opcionales (si tienes): **NUM_PROMOVIDOS**, **PAUTA_INVERTIDA**, **VISUALIZACIONES**, **DEMOGRAFIA**, **PREF_ELECTORAL**.
""")

# =============== Cargar datos ===============
st.subheader("1) Datos")
uploaded = st.file_uploader("Sube tu archivo (.xlsx o .csv):", type=["xlsx", "csv"])

def read_file(file):
    if file.name.lower().endswith(".xlsx"):
        return pd.read_excel(file, engine="openpyxl")
    else:
        return pd.read_csv(file)

if uploaded:
    df = read_file(uploaded)
else:
    # Datos de demo si no subes nada
    df = pd.read_csv("data_demo.csv")

st.dataframe(df.head(20))

# Normalizar nombres de columnas (tolerante a mayúsculas/minúsculas y espacios)
def norm_cols(d):
    d = d.copy()
    d.columns = [c.strip().upper() for c in d.columns]
    return d

df = norm_cols(df)

# =============== Filtros ===============
st.subheader("2) Filtros")
col1, col2, col3 = st.columns(3)

prioridades = ["Todos"] + sorted([str(x) for x in df["PRIORIDAD"].dropna().unique()]) if "PRIORIDAD" in df.columns else ["Todos"]
municipios = ["Todos"] + sorted([str(x) for x in df["MUNICIPIO"].dropna().unique()]) if "MUNICIPIO" in df.columns else ["Todos"]
distritos = ["Todos"] + sorted([str(x) for x in df["DISTRITO LOCAL"].dropna().unique()]) if "DISTRITO LOCAL" in df.columns else ["Todos"]

sel_prio = col1.selectbox("Prioridad", prioridades, index=0)
sel_mun  = col2.selectbox("Municipio", municipios, index=0)
sel_dist = col3.selectbox("Distrito Local", distritos, index=0)

df_f = df.copy()
if "PRIORIDAD" in df_f.columns and sel_prio != "Todos":
    df_f = df_f[df_f["PRIORIDAD"].astype(str) == str(sel_prio)]
if "MUNICIPIO" in df_f.columns and sel_mun != "Todos":
    df_f = df_f[df_f["MUNICIPIO"].astype(str) == str(sel_mun)]
if "DISTRITO LOCAL" in df_f.columns and sel_dist != "Todos":
    df_f = df_f[df_f["DISTRITO LOCAL"].astype(str) == str(sel_dist)]

st.caption(f"Filas después de filtros: {len(df_f)}")

# =============== Métricas clave ===============
st.subheader("3) Métricas clave")
m1, m2, m3, m4 = st.columns(4)

def safe_sum(col):
    return int(pd.to_numeric(df_f.get(col, pd.Series([])), errors="coerce").fillna(0).sum())

num_seccionales = len(df_f)
num_promovidos = safe_sum("NUM_PROMOVIDOS")
pauta_invertida = safe_sum("PAUTA_INVERTIDA")
num_visualizaciones = safe_sum("VISUALIZACIONES")

m1.metric("Número de seccionales", num_seccionales)
m2.metric("Número de promovidos", num_promovidos)
m3.metric("Pauta invertida", pauta_invertida)
m4.metric("Visualizaciones", num_visualizaciones)

# =============== Gráficos simples ===============
st.subheader("4) Gráficos")
charts_row = st.columns(2)

# Barras por Prioridad
if "PRIORIDAD" in df_f.columns:
    prio_counts = df_f["PRIORIDAD"].astype(str).value_counts().reset_index()
    prio_counts.columns = ["PRIORIDAD", "CUENTA"]
    chart1 = alt.Chart(prio_counts).mark_bar().encode(
        x=alt.X("PRIORIDAD:N", sort="-y"),
        y="CUENTA:Q",
        tooltip=["PRIORIDAD", "CUENTA"]
    ).properties(title="Secciones por Prioridad")
    charts_row[0].altair_chart(chart1, use_container_width=True)

# Pie por Municipio (top 10)
if "MUNICIPIO" in df_f.columns:
    mun_counts = df_f["MUNICIPIO"].astype(str).value_counts().reset_index().head(10)
    mun_counts.columns = ["MUNICIPIO", "CUENTA"]
    chart2 = alt.Chart(mun_counts).mark_arc().encode(
        theta="CUENTA:Q",
        color="MUNICIPIO:N",
        tooltip=["MUNICIPIO", "CUENTA"]
    ).properties(title="Top 10 Municipios (por número de filas)")
    charts_row[1].altair_chart(chart2, use_container_width=True)

# Serie (si hay VISUALIZACIONES por FECHA opcional)
if "FECHA" in df_f.columns and "VISUALIZACIONES" in df_f.columns:
    st.subheader("Serie de tiempo (Visualizaciones por fecha)")
    tmp = df_f.copy()
    tmp["FECHA"] = pd.to_datetime(tmp["FECHA"], errors="coerce")
    series = tmp.groupby("FECHA", dropna=True)["VISUALIZACIONES"].sum().reset_index()
    chart3 = alt.Chart(series).mark_line(point=True).encode(
        x="FECHA:T",
        y="VISUALIZACIONES:Q",
        tooltip=["FECHA","VISUALIZACIONES"]
    )
    st.altair_chart(chart3, use_container_width=True)

# =============== Mapa de secciones ===============
st.subheader("5) Mapa de secciones (Puebla)")

st.markdown("""
**¿De dónde saco el mapa (.geojson)?**  
Para producción, usa la cartografía de **secciones electorales** del INE/IEE.  
Mientras tanto, este demo usa un archivo pequeño `puebla_secciones_demo.geojson` con 2 polígonos para que veas la app funcionando.
""")

# Selección de archivo geojson
gfile = st.file_uploader("Sube tu archivo GeoJSON de secciones", type=["geojson"], key="geojsonuploader")
if gfile:
    geojson_obj = json.load(gfile)
else:
    with open("puebla_secciones_demo.geojson", "r", encoding="utf-8") as f:
        geojson_obj = json.load(f)

# Si tu data tiene una columna que "relacione" con el GeoJSON (por ejemplo SECCION),
# puedes mostrar tooltip con datos tabulares filtrados.
# En el demo asumimos que en el GeoJSON hay una propiedad "SECCION".
prop_key = "SECCION"

# Construir capa GeoJSON
layer = pdk.Layer(
    "GeoJsonLayer",
    data=geojson_obj,
    pickable=True,
    stroked=True,
    filled=True,
    get_fill_color=[180, 180, 200, 120],
    get_line_color=[60, 60, 120, 180],
    line_width_min_pixels=1,
)

# Vista inicial centrada en Puebla (aprox)
view_state = pdk.ViewState(latitude=19.0379, longitude=-98.2035, zoom=9)

r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={
        "html": "<b>Sección:</b> {SECCION}",
        "style": {"backgroundColor": "white", "color": "black"}
    },
    map_style="light"
)
st.pydeck_chart(r, use_container_width=True)

# =============== Resumen claro y visual ===============
st.subheader("6) Resumen")
st.write(f"""
- Secciones (filtradas): **{num_seccionales}**  
- Promovidos totales: **{num_promovidos}**  
- Pauta invertida total: **{pauta_invertida}**  
- Visualizaciones totales: **{num_visualizaciones}**  
""")

st.info("Más adelante: integrar sugerencias con IA (ChatGPT) usando la API de OpenAI o funciones de Streamlit. Cuando quieras, te dejo el bloque listo.")
