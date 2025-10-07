import streamlit as st
import pandas as pd
import pydeck as pdk
import altair as alt
import json, hashlib
from pathlib import Path
from datetime import datetime, timedelta

# ================== CONFIG ==================
st.set_page_config(page_title="Tablero Puebla", page_icon="📊", layout="wide")

# Header con logo CHR
c1, c2 = st.columns([1, 6], vertical_alignment="center")
with c1:
    st.markdown("""
    <div style="font-weight:800;font-size:22px;letter-spacing:2px;
                padding:8px 12px;border:1px solid rgba(255,255,255,.08);
                border-radius:12px;display:inline-block;background:rgba(255,255,255,.04);">
    CHR
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.title("Tablero Puebla · V3")

# Paleta (modo oscuro)
PALETTE = {
    "Alta":   "#ff4d4f",
    "Media":  "#f0a800",
    "Baja":   "#3aaed8",
    "Default":"##9aa0a6".replace("##","#"),
    "Out":    "#2a2e35"
}

DATA_DIR = Path("data")
CSV_MAIN = DATA_DIR / "datos_puebla.csv"
GJ_PATH  = DATA_DIR / "secciones_puebla.geojson"
CSV_AD   = DATA_DIR / "ad_spend.csv"
CSV_SOC  = DATA_DIR / "social_sentiment.csv"

# ================== HELPERS ==================
@st.cache_data(show_spinner=False)
def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    if "FECHA" in df.columns:
        df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce").dt.date
    return df

@st.cache_data(show_spinner=False)
def load_geojson(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def hex_to_rgba(hex_color: str, alpha: int = 160):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return [r, g, b, alpha]

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_files():
    if not CSV_MAIN.exists():
        st.error("Falta `data/datos_puebla.csv`.")
        st.stop()
    if not GJ_PATH.exists():
        st.error("Falta `data/secciones_puebla.geojson`.")
        st.stop()

ensure_files()
df = load_df(CSV_MAIN)
geojson_obj = load_geojson(GJ_PATH)
df_ad  = load_df(CSV_AD) if CSV_AD.exists() else pd.DataFrame()
df_soc = load_df(CSV_SOC) if CSV_SOC.exists() else pd.DataFrame()

# ================== SIDEBAR: FILTROS ==================
with st.sidebar:
    st.markdown("### Filtros")

    # Periodo por defecto = últimos 7 días (si hay FECHA)
    today = datetime.utcnow().date()
    default_start = today - timedelta(days=7)

    if "FECHA" in df.columns and df["FECHA"].notna().any():
        min_d, max_d = df["FECHA"].min(), df["FECHA"].max()
        start_date, end_date = st.date_input(
            "Periodo",
            value=(max(default_start, min_d), max_d),
            min_value=min_d, max_value=max_d
        )
    else:
        start_date, end_date = None, None
        st.caption("No se detectó FECHA → se considera todo el histórico.")

    prioridades = ["Todos"] + sorted(df["PRIORIDAD"].dropna().astype(str).unique()) if "PRIORIDAD" in df.columns else ["Todos"]
    municipios  = ["Todos"] + sorted(df["MUNICIPIO"].dropna().astype(str).unique()) if "MUNICIPIO" in df.columns else ["Todos"]
    distritos   = ["Todos"] + sorted(df["DISTRITO LOCAL"].dropna().astype(str).unique()) if "DISTRITO LOCAL" in df.columns else ["Todos"]

    sel_prio = st.selectbox("Prioridad", prioridades, index=0)
    sel_mun  = st.selectbox("Municipio", municipios, index=0)
    sel_dist = st.selectbox("Distrito Local", distritos, index=0)

    st.markdown("---")
    map_action = st.radio("Enfoque del mapa", ["Puebla capital", "Centrar en datos filtrados", "Libre"], index=0)
    color_metric = st.selectbox(
        "Color del mapa por",
        ["PRIORIDAD", "PAUTA_INVERTIDA", "VISUALIZACIONES", "SENTIMIENTO (si hay)"],
        index=0
    )

# ================== APLICAR FILTROS ==================
df_f = df.copy()
if start_date and end_date and "FECHA" in df_f.columns:
    df_f = df_f[(df_f["FECHA"] >= start_date) & (df_f["FECHA"] <= end_date)]
if "PRIORIDAD" in df_f.columns and sel_prio != "Todos":
    df_f = df_f[df_f["PRIORIDAD"].astype(str) == str(sel_prio)]
if "MUNICIPIO" in df_f.columns and sel_mun != "Todos":
    df_f = df_f[df_f["MUNICIPIO"].astype(str) == str(sel_mun)]
if "DISTRITO LOCAL" in df_f.columns and sel_dist != "Todos":
    df_f = df_f[df_f["DISTRITO LOCAL"].astype(str) == str(sel_dist)]

# Si hay social sentiment, agregamos índice por sección (promedio simple pos-neg)
if not df_soc.empty and "SECCION" in df_soc.columns:
    soc = df_soc.copy()
    for c in ["SENT_POS","SENT_NEG","SENT_NEU"]:
        if c not in soc.columns: soc[c] = 0
    soc["SENT_INDEX"] = (pd.to_numeric(soc["SENT_POS"], errors="coerce").fillna(0)
                         - pd.to_numeric(soc["SENT_NEG"], errors="coerce").fillna(0))
    if "FECHA" in soc.columns and start_date and end_date:
        soc = soc[(soc["FECHA"] >= start_date) & (soc["FECHA"] <= end_date)]
    soc_agg = soc.groupby("SECCION", as_index=False)["SENT_INDEX"].mean()
    df_f = df_f.merge(soc_agg, on="SECCION", how="left")
else:
    df_f["SENT_INDEX"] = pd.NA

# ================== TABS ==================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Métricas y resumen", "📈 Gráficos", "🗺️ Mapa", "🧠 Insights de IA", "🔐 Auditoría"
])

# ---- MÉTRICAS ----
with tab1:
    st.subheader("Métricas (periodo y filtros aplicados)")
    def safe_sum(col): return int(pd.to_numeric(df_f.get(col, pd.Series([])), errors="coerce").fillna(0).sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Secciones", len(df_f))
    m2.metric("Promovidos", safe_sum("NUM_PROMOVIDOS"))
    m3.metric("Pauta invertida", safe_sum("PAUTA_INVERTIDA"))
    m4.metric("Visualizaciones", safe_sum("VISUALIZACIONES"))
    st.caption("Def.: Pauta= gasto; Visualizaciones= vistas; Promovidos= activados/confirmados.")

# ---- GRÁFICOS ----
with tab2:
    st.subheader("Gráficas (claras e intuitivas)")

    # Prioridad → barras horizontales
    if "PRIORIDAD" in df_f.columns:
        prio = (df_f["PRIORIDAD"].astype(str).value_counts()
                .rename_axis("PRIORIDAD").reset_index(name="SECCIONES"))
        order_map = {"Alta": 0, "Media": 1, "Baja": 2}
        prio["ORD"] = prio["PRIORIDAD"].map(order_map).fillna(99)
        prio = prio.sort_values(["ORD","SECCIONES"], ascending=[True, False])
        prio["COLOR"] = prio["PRIORIDAD"].map(lambda x: PALETTE.get(x, PALETTE["Default"]))
        chart1 = (alt.Chart(prio)
                  .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
                  .encode(
                      x=alt.X("SECCIONES:Q", title="Número de secciones"),
                      y=alt.Y("PRIORIDAD:N", sort=prio["PRIORIDAD"].tolist(), title=None),
                      color=alt.Color("PRIORIDAD:N", scale=alt.Scale(range=list(prio["COLOR"])), legend=None),
                      tooltip=["PRIORIDAD","SECCIONES"]
                  ).properties(title="Secciones por prioridad"))
        st.altair_chart(chart1, use_container_width=True)
    else:
        st.info("No hay PRIORIDAD para graficar.")

    # Top municipios
    if "MUNICIPIO" in df_f.columns:
        topm = (df_f["MUNICIPIO"].astype(str).value_counts()
                .rename_axis("MUNICIPIO").reset_index(name="SECCIONES").head(8))
        chart2 = (alt.Chart(topm.sort_values("SECCIONES"))
                  .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
                  .encode(
                      x=alt.X("SECCIONES:Q", title="Número de secciones"),
                      y=alt.Y("MUNICIPIO:N", sort=None, title=None),
                      color=alt.value("#5C7CFA"),
                      tooltip=["MUNICIPIO","SECCIONES"]
                  ).properties(title="Top 8 municipios"))
        st.altair_chart(chart2, use_container_width=True)
    else:
        st.info("No hay MUNICIPIO para graficar.")

# ---- MAPA ----
with tab3:
    st.subheader("Mapa (oscuro, dinámico y coloreado por métrica)")

    # Construir valor de color por sección
    prio_by_sec = {}
    if "SECCION" in df_f.columns and "PRIORIDAD" in df_f.columns:
        prio_by_sec = {str(r.SECCION): str(r.PRIORIDAD) for r in df_f[["SECCION","PRIORIDAD"]].dropna().itertuples(index=False)}

    # Índice por métrica
    metric_map = {}
    if color_metric == "PRIORIDAD":
        for sec, pr in prio_by_sec.items():
            metric_map[sec] = pr
    elif color_metric == "PAUTA_INVERTIDA":
        for r in df_f[["SECCION","PAUTA_INVERTIDA"]].dropna().itertuples(index=False):
            metric_map[str(r.SECCION)] = float(r.PAUTA_INVERTIDA)
    elif color_metric == "VISUALIZACIONES":
        for r in df_f[["SECCION","VISUALIZACIONES"]].dropna().itertuples(index=False):
            metric_map[str(r.SECCION)] = float(r.VISUALIZACIONES)
    else:  # Sentimiento
        for r in df_f[["SECCION","SENT_INDEX"]].dropna().itertuples(index=False):
            metric_map[str(r.SECCION)] = float(r.SENT_INDEX)

    # Bins para números (cuantiles) → gradiente morado
    def color_for_value(val: float):
        # tres cortes en cuantiles: bajo/medio/alto (opción simple)
        return hex_to_rgba("#7C5CFF", 90) if val is None else (
               hex_to_rgba("#A08BFF", 140) if val <= q1 else
               hex_to_rgba("#7C5CFF", 170) if val <= q2 else
               hex_to_rgba("#5838FF", 200))

    numeric_mode = color_metric != "PRIORIDAD"
    q1=q2=None
    if numeric_mode:
        vals = [v for v in metric_map.values() if isinstance(v,(int,float))]
        if len(vals)>=3:
            s = pd.Series(vals).astype(float)
            q1, q2 = s.quantile(0.5), s.quantile(0.8)
        else:
            q1 = q2 = None

    # Enriquecer features con color
    for feat in geojson_obj.get("features", []):
        props = feat.setdefault("properties", {})
        sec = str(props.get("SECCION",""))
        if numeric_mode:
            v = metric_map.get(sec, None)
            props["__PRIORIDAD__"] = "—"
            props["__METRIC__"] = v
            color = color_for_value(v) if q1 is not None else hex_to_rgba("#7C5CFF", 150)
        else:
            pr = metric_map.get(sec)
            props["__PRIORIDAD__"] = pr if pr else "Fuera de filtro"
            props["__METRIC__"] = pr
            color = hex_to_rgba(PALETTE.get(pr, PALETTE["Out"]), 160)
        props["__FILL_COLOR__"] = color

    # Vista
    def center_puebla_city():
        return pdk.ViewState(latitude=19.0379, longitude=-98.2035, zoom=11.2)
    def center_filtered(gj: dict):
        lats,lons=[],[]
        for f in gj.get("features",[]):
            pr=f.get("properties",{})
            if pr.get("__PRIORIDAD__")=="Fuera de filtro" and not numeric_mode:
                continue
            geom=f.get("geometry",{})
            if geom.get("type")=="Polygon":
                for x,y in geom["coordinates"][0]: lons.append(x); lats.append(y)
            elif geom.get("type")=="MultiPolygon":
                for poly in geom["coordinates"]:
                    for x,y in poly[0]: lons.append(x); lats.append(y)
        if lats and lons:
            return pdk.ViewState(latitude=sum(lats)/len(lats), longitude=sum(lons)/len(lons), zoom=10.5)
        return center_puebla_city()

    if map_action=="Puebla capital":
        view = center_puebla_city()
    elif map_action=="Centrar en datos filtrados":
        view = center_filtered(geojson_obj)
    else:
        view = pdk.ViewState(latitude=19.0, longitude=-97.95, zoom=8.3)

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson_obj,
        pickable=True,
        auto_highlight=True,
        stroked=True,
        filled=True,
        get_fill_color="properties.__FILL_COLOR__",
        get_line_color=[220,220,230,120],
        line_width_min_pixels=1,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip={"html": "<b>Sección:</b> {SECCION}<br/><b>Valor:</b> {__METRIC__}",
                 "style": {"backgroundColor":"rgba(0,0,0,.85)","color":"#fff"}},
        map_style="dark"
    )
    st.pydeck_chart(deck, use_container_width=True)

# ---- INSIGHTS DE IA (REGLAS EXPLICABLES) ----
with tab4:
    st.subheader("Recomendaciones automáticas (explicables)")
    # Reglas simples basadas en señales (sin APIs externas)
    insights = []

    # 1) Eficiencia de pauta: si gasto alto pero vistas por peso bajas
    if "PAUTA_INVERTIDA" in df_f.columns and "VISUALIZACIONES" in df_f.columns:
        g = df_f.copy()
        g["CPV"] = pd.to_numeric(g["PAUTA_INVERTIDA"], errors="coerce") / \
                   pd.to_numeric(g["VISUALIZACIONES"], errors="coerce").replace({0:pd.NA})
        avg_cpv = g["CPV"].median(skipna=True)
        if avg_cpv and avg_cpv>0:
            worst = g.sort_values("CPV", ascending=False).head(3)[["MUNICIPIO","DISTRITO LOCAL","CPV"]]
            if len(worst)>0:
                insights.append("💡 *Optimiza creatividades/segmentos*: hay zonas con **CPV** alto (gasto por vista). Prueba nuevas piezas y recorta inventario de bajo desempeño.")

    # 2) Sentimiento: si hay social_sentiment y SENT_INDEX<0 en varias secciones
    if "SENT_INDEX" in df_f.columns and df_f["SENT_INDEX"].notna().any():
        neg_share = (df_f["SENT_INDEX"]<0).mean()
        if neg_share>0.3:
            insights.append("💡 *Ajusta el mensaje*: hay sentimiento negativo en varias secciones. Prueba mensajes de contraste suave y refuerza validaciones de logro/sencillez.")

    # 3) Cobertura por prioridad: si muchas secciones 'Alta' sin vistas
    if "PRIORIDAD" in df_f.columns and "VISUALIZACIONES" in df_f.columns:
        sub = df_f[df_f["PRIORIDAD"].astype(str)=="Alta"]
        if not sub.empty and pd.to_numeric(sub["VISUALIZACIONES"], errors="coerce").fillna(0).median()==0:
            insights.append("💡 *Asignación rápida*: prioriza inversión mínima en secciones **Alta** sin impresiones para cobertura 100% en 48–72h.")

    # 4) Sugerencia de mix (inspirado en adopción IA/CTV 2024)
    insights.append("💡 *Mix sugerido*: testea **CTV + YouTube** para alcance incremental geolocalizado, con creatividades autogeneradas (múltiples variaciones) y rotación por desempeño. (Tendencia campañas 2024).")  # basado en prensa sectorial
    st.write("\n\n".join(insights) if insights else "No se detectaron señales claras. Con más datos (ad_spend, social_sentiment) mejoran las recomendaciones.")

    st.caption("Inspirado en prácticas reales: IA para escalar creatividades, segmentación y lectura de sentimiento/engagement en elecciones 2024, y modelos predictivos IBERO-UNAM tras eventos clave. ")
    st.caption("Fuentes: Digiday (ads + IA, escalamiento y targeting) y WIRED/IBERO (predictivo de voto).")

# ---- AUDITORÍA ----
with tab5:
    st.subheader("Trazabilidad de datos (huellas SHA-256)")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**datos_puebla.csv**")
        st.code(file_hash(CSV_MAIN))
        st.markdown("**secciones_puebla.geojson**")
        st.code(file_hash(GJ_PATH))
    with cols[1]:
        if CSV_AD.exists():
            st.markdown("**ad_spend.csv**")
            st.code(file_hash(CSV_AD))
        if CSV_SOC.exists():
            st.markdown("**social_sentiment.csv**")
            st.code(file_hash(CSV_SOC))
    st.info("Estas huellas permiten verificar que los datasets no se alteraron (paso previo a un esquema blockchain).")
