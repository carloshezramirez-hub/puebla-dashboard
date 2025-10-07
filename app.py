import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import json, hashlib
from pathlib import Path
from datetime import datetime, timedelta
from joblib import load as joblib_load

st.set_page_config(page_title="Tablero Puebla Pro", page_icon="📊", layout="wide")

# ---------- Branding ----------
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
    st.title("Tablero Puebla — v2.5 Pro")

# ---------- Paths ----------
DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "datos_puebla.csv"
GEOJSON_PATH = DATA_DIR / "secciones_puebla.geojson"
MODEL_PATH = DATA_DIR / "model.joblib"

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

ensure_files()
df = load_table(CSV_PATH)
geojson_obj = load_geojson(GEOJSON_PATH)

def score_to_band(series, labels=("Baja","Media","Alta","Muy alta"), q=4):
    import pandas as pd
    import numpy as np
    s = pd.Series(series).astype(float)

    # Si no hay datos o todos los scores son iguales, asigna banda media
    if s.empty or s.nunique() == 1:
        return pd.Series([labels[1]] * len(s), index=s.index)

    # Intentar quantiles con duplicados permitidos
    try:
        return pd.qcut(s, q=q, labels=labels, duplicates="drop")
    except Exception:
        # Fallback: cortes lineales con el nº de bins posible
        unique_bins = max(2, min(q, s.nunique()))
        bins = np.linspace(s.min(), s.max(), num=unique_bins + 1)
        use_labels = labels[:unique_bins]
        return pd.cut(s, bins=bins, labels=use_labels, include_lowest=True, duplicates="drop")


# ---------- Sidebar: filtros ----------
with st.sidebar:
    st.markdown("### Filtros")

# --------- Fecha robusta (maneja NaT, orden invertido y acota a límites) ---------
    start_date, end_date = None, None
    if "FECHA" in df.columns:
        valid_dates = df["FECHA"].dropna()
    else:
        valid_dates = pd.Series([], dtype="datetime64[ns]")

    if len(valid_dates) == 0:
        st.caption("No hay fechas válidas (columna FECHA vacía o ausente): se mostrará todo el histórico.")
    else:
        min_date = valid_dates.min()
        max_date = valid_dates.max()
        # Si por alguna razón vinieran invertidas
        if min_date > max_date:
            min_date, max_date = max_date, min_date

        # Por defecto: últimos 7 días o desde min_date si es más reciente
        default_end = max_date
        default_start = max(min_date, max_date - timedelta(days=7))

        # Construir el input de rango
        picked = st.date_input(
            "Periodo",
            value=(default_start, default_end),
            min_value=min_date,
            max_value=max_date
        )

        # Normalizar lo que regresa el widget (puede venir 1 fecha o 2)
        if isinstance(picked, tuple) and len(picked) == 2:
            sd, ed = picked
        else:
            # Si el usuario selecciona una sola fecha, usamos [fecha, fecha]
            sd = ed = picked

        # Si el usuario invierte el rango, lo ordenamos
        if sd and ed and sd > ed:
            sd, ed = ed, sd

        # Acotar a los límites por seguridad
        if sd and sd < min_date:
            sd = min_date
        if ed and ed > max_date:
            ed = max_date

        start_date, end_date = sd, ed


    prioridades = ["Todos"] + (sorted(df["PRIORIDAD"].dropna().astype(str).unique()) if "PRIORIDAD" in df.columns else [])
    municipios  = ["Todos"] + (sorted(df["MUNICIPIO"].dropna().astype(str).unique()) if "MUNICIPIO" in df.columns else [])
    distritos   = ["Todos"] + (sorted(df["DISTRITO LOCAL"].dropna().astype(str).unique()) if "DISTRITO LOCAL" in df.columns else [])

    sel_prio = st.selectbox("Prioridad", prioridades, index=0)
    sel_mun  = st.selectbox("Municipio", municipios, index=0)
    sel_dist = st.selectbox("Distrito Local", distritos, index=0)

    st.markdown("---")
    map_action = st.radio("Enfoque del mapa",
                          options=["Puebla capital", "Centrar en datos filtrados", "Libre"],
                          index=0)

# ---------- Aplicar filtros ----------
df_f = df.copy()
if start_date and end_date and "FECHA" in df_f.columns:
    df_f = df_f[(df_f["FECHA"] >= start_date) & (df_f["FECHA"] <= end_date)]
if "PRIORIDAD" in df_f.columns and sel_prio != "Todos":
    df_f = df_f[df_f["PRIORIDAD"].astype(str) == str(sel_prio)]
if "MUNICIPIO" in df_f.columns and sel_mun != "Todos":
    df_f = df_f[df_f["MUNICIPIO"].astype(str) == str(sel_mun)]
if "DISTRITO LOCAL" in df_f.columns and sel_dist != "Todos":
    df_f = df_f[df_f["DISTRITO LOCAL"].astype(str) == str(sel_dist)]

# ---------- Tabs ----------
tab_prev, tab_day, tab_res = st.tabs(["📋 Previa (Campaña)", "🗳️ Día D", "📈 Resultados"])

# ================= PREVIA =================
with tab_prev:
    st.subheader("Métricas clave y resumen")
    def safe_sum(col):
        return int(pd.to_numeric(df_f.get(col, pd.Series([])), errors="coerce").fillna(0).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Secciones", len(df_f))
    m2.metric("Promovidos", safe_sum("NUM_PROMOVIDOS"))
    m3.metric("Pauta invertida", safe_sum("PAUTA_INVERTIDA"))
    m4.metric("Visualizaciones", safe_sum("VISUALIZACIONES"))

    st.markdown("### Gráficas")
    cols = st.columns(2)

    if "PRIORIDAD" in df_f.columns:
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
            ).properties(title="Secciones por prioridad"))
        cols[0].altair_chart(chart_prio, use_container_width=True)

    if "MUNICIPIO" in df_f.columns:
        top_m = (df_f["MUNICIPIO"].astype(str).value_counts().rename_axis("MUNICIPIO")
                 .reset_index(name="SECCIONES").head(8).sort_values("SECCIONES"))
        chart_mun = (alt.Chart(top_m)
            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
            .encode(
                x=alt.X("SECCIONES:Q", title="Secciones"),
                y=alt.Y("MUNICIPIO:N", sort=None, title=None),
                color=alt.value("#5C7CFA"),
                tooltip=["MUNICIPIO","SECCIONES"]
            ).properties(title="Top 8 municipios"))
        cols[1].altair_chart(chart_mun, use_container_width=True)

    st.markdown("### Predicción (baseline) y simulación")
    left, right = st.columns([2,1])

    model = None
    feature_names = None
    if MODEL_PATH.exists():
        try:
            bundle = joblib_load(MODEL_PATH)
            model = bundle.get("model")
            feature_names = bundle.get("features", ["NUM_PROMOVIDOS","PAUTA_INVERTIDA","VISUALIZACIONES"])
        except Exception:
            st.warning("No se pudo cargar model.joblib. Entrena con train_model.py.")

    needed = ["NUM_PROMOVIDOS","PAUTA_INVERTIDA","VISUALIZACIONES"]
    for c in needed:
        if c not in df_f.columns:
            df_f[c] = 0
    X = df_f[needed].copy()
    Xlog = np.log1p(X)

    if model is not None and len(df_f) > 0:
        try:
            proba = model.predict_proba(Xlog)[:,1]
        except Exception:
            raw = model.predict(Xlog)
            proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)
        df_f["SCORE"] = proba
        df_f["RANGO"] = score_to_band(df_f["SCORE"])

        with left:
            st.success(f"Predicción generada para {len(df_f)} secciones.")
            st.dataframe(df_f[["SECCION","SCORE","RANGO"]].head(20), use_container_width=True)
            if hasattr(model, "feature_importances_") and feature_names:
                fi = pd.DataFrame({"feature": feature_names,
                                   "importance": getattr(model, "feature_importances_")}).sort_values("importance")
            else:
                fi = pd.DataFrame({"feature": feature_names or needed,
                                   "importance": [1/len(needed)]*len(needed)})
            chart_fi = (alt.Chart(fi).mark_bar().encode(
                x=alt.X("importance:Q", title="Importancia"),
                y=alt.Y("feature:N", sort=None, title="Variable"),
                tooltip=["feature","importance"]
            ).properties(title="Explicabilidad"))
            st.altair_chart(chart_fi, use_container_width=True)

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
            delta = proba_sim - proba
            gain = float(delta.sum())
            st.metric("Impacto simulado (Σ Δ score)", f"{gain:.2f}")
    else:
        st.info("Entrena `data/model.joblib` con `python train_model.py` para activar predicción y simulación.")

    # ----- Mapa -----
    st.markdown("### Mapa (colorear por prioridad o por score)")
    color_mode = st.radio("Colorear por", ["Prioridad", "Score (si hay modelo)"], horizontal=True)

    lookup = {}
    if "SECCION" in df_f.columns:
        if color_mode.startswith("Score") and "SCORE" in df_f.columns:
            for r in df_f[["SECCION","SCORE"]].itertuples(index=False):
                lookup[str(r.SECCION)] = float(r.SCORE)
        elif "PRIORIDAD" in df_f.columns:
            for r in df_f[["SECCION","PRIORIDAD"]].itertuples(index=False):
                lookup[str(r.SECCION)] = str(r.PRIORIDAD)

    feats = geojson_obj.get("features", [])
    for f in feats:
        props = f.get("properties", {})
        sec = str(props.get("SECCION",""))
        if color_mode.startswith("Score") and sec in lookup:
            val = float(lookup[sec])
            r = int(255*val); g = int(80*(1-val)+50); b = int(120*(1-val)+50)
            props["__LABEL__"] = f"Score {val:.2f}"
            props["__FILL__"] = [r,g,b,160]
        else:
            pr = str(lookup.get(sec,""))
            col = PALETTE.get(pr, PALETTE["Out"] if pr=="" else PALETTE["Default"])
            props["__LABEL__"] = pr if pr else "Fuera de filtro"
            props["__FILL__"] = hex_to_rgba(col, 160)

    def view_puebla():
        return pdk.ViewState(latitude=19.0379, longitude=-98.2035, zoom=11.2)

    def view_filtered(gj: dict):
        lats,lons=[],[]
        for f in gj.get("features",[]):
            pr = f.get("properties",{})
            if pr.get("__LABEL__")=="Fuera de filtro": continue
            g = f.get("geometry",{})
            if g.get("type")=="Polygon":
                for ring in g["coordinates"][0]:
                    lons.append(ring[0]); lats.append(ring[1])
            elif g.get("type")=="MultiPolygon":
                for poly in g["coordinates"]:
                    for ring in poly[0]:
                        lons.append(ring[0]); lats.append(ring[1])
        if lats and lons:
            return pdk.ViewState(latitude=float(np.mean(lats)), longitude=float(np.mean(lons)), zoom=10.5)
        return view_puebla()

    if map_action=="Puebla capital":
        vstate = view_puebla()
    elif map_action=="Centrar en datos filtrados":
        vstate = view_filtered(geojson_obj)
    else:
        vstate = pdk.ViewState(latitude=19.0, longitude=-97.95, zoom=8.3)

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson_obj,
        pickable=True, auto_highlight=True,
        stroked=True, filled=True,
        get_fill_color="properties.__FILL__",
        get_line_color=[180,180,200,120],
        line_width_min_pixels=1,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=vstate,
        tooltip={"html":"<b>Sección:</b> {SECCION}<br/><b>Dato:</b> {__LABEL__}",
                 "style":{"backgroundColor":"rgba(0,0,0,.85)","color":"#fff"}},
        map_style="dark"
    )
    st.pydeck_chart(deck, use_container_width=True)

# ================= DÍA D =================
with tab_day:
    st.subheader("Monitoreo operativo (plantillas)")
    c = st.columns(3)
    c[0].metric("% casillas instaladas", "—")
    c[1].metric("Incidencias reportadas", "—")
    c[2].metric("Participación estimada", "—")
    st.caption("Esta sección se alimenta con CSVs/Forms/Bots (configurable).")

# ================= RESULTADOS =================
with tab_res:
    st.subheader("Resultados preliminares (plantillas)")
    st.write("Listo para integrar actas, PREP y comparativos.")
    st.caption("Incluye huella de datos para trazabilidad.")

# ---------- Footer ----------
st.markdown("---")
st.caption(f"Huella datos_puebla.csv: `{hhash(CSV_PATH)}` · Modelo: {'OK' if MODEL_PATH.exists() else 'N/D'}")