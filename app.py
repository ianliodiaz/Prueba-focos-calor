"""
FIREN'T — Calendario armonizado de actividad de fuego (MODIS + VIIRS) para Argentina
NASA Space Apps Challenge 2026 · Desafío CONAE "Harmonization of MODIS and VIIRS Hot Spots"

Para correrla en la computadora:  streamlit run app.py
"""
import datetime as dt
import json
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

VERSION_APP = "v6"
RAIZ = Path(__file__).parent
_encontrados = sorted(RAIZ.rglob("serie_armonizada.csv"), key=lambda p: len(p.parts))
CARPETA = _encontrados[0].parent if _encontrados else RAIZ / "datos"
ARCHIVOS_FOCOS = sorted(RAIZ.rglob("focos_*.parquet"))

PAIS, AREA, RESTO = "Argentina (todo el país)", "Área propia", "Resto del país"
INICIO_VIIRS = pd.Timestamp("2012-01-20")

st.set_page_config(page_title="FIREN'T · Focos de calor en Argentina", page_icon="🔥", layout="wide")

# ================================================================ idioma
if "idioma" not in st.session_state:
    st.session_state.idioma = "ES"
EN = st.session_state.idioma == "EN"


def tr(es, en):
    "Devuelve el texto en el idioma elegido."
    return en if EN else es


MESES = tr("Ene Feb Mar Abr May Jun Jul Ago Sep Oct Nov Dic", "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec").split()
MESES_LARGOS = tr("enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre",
                  "January February March April May June July August September October November December").split()
NOMBRE_ZONA_EN = {PAIS: "Argentina (whole country)", "Gran Chaco": "Gran Chaco", "Delta del Paraná": "Paraná Delta",
                  "Sierras de Córdoba": "Córdoba Hills", "Corrientes e Iberá": "Corrientes & Iberá",
                  "Patagonia andina": "Andean Patagonia", RESTO: "Rest of the country", AREA: "Custom area"}
NOMBRE_MEDIDA = {"focos": tr("Focos", "Hotspots"), "dias": tr("Días con fuego", "Fire-days"),
                 "frp": tr("Energía (FRP, MW)", "Energy (FRP, MW)")}
UNIDAD = {"focos": tr("focos", "hotspots"), "dias": tr("días-celda", "cell-days"), "frp": "MW"}
EXPLICA_MEDIDA = {
    "focos": tr("Cantidad de detecciones satelitales (píxeles con fuego).",
                "Number of satellite detections (pixels with fire)."),
    "dias": tr("Pares (celda de 25 km, día) con al menos un foco: un incendio cuenta igual aunque el satélite lo "
               "divida en muchos píxeles.",
               "Pairs (25 km cell, day) with at least one hotspot: a fire counts the same even if the satellite "
               "splits it into many pixels."),
    "frp": tr("Suma de la potencia radiativa del fuego, en megavatios: mide energía, no cantidad de píxeles.",
              "Sum of fire radiative power, in megawatts: it measures energy, not number of pixels."),
}
NOMBRE_METODO_EN = {"Factor fijo por región": "Fixed factor per region",
                    "Regresión log por región y mes": "Log regression by region and month",
                    "Mapeo de cuantiles": "Quantile mapping", "Random Forest": "Random Forest"}


def zona_txt(z):
    return NOMBRE_ZONA_EN.get(z, z) if EN else z


def metodo_txt(m):
    return NOMBRE_METODO_EN.get(m, m) if EN else m


def miles(x, dec=0):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    s = f"{x:,.{dec}f}"
    return s if EN else s.replace(",", "X").replace(".", ",").replace("X", ".")


def abreviar(x):
    if not np.isfinite(x):
        return ""
    if x >= 10000:
        return f"{x / 1000:.0f}k"
    if x >= 1000:
        s = f"{x / 1000:.1f}k"
        return s if EN else s.replace(".", ",")
    return f"{x:.0f}"


def lista(items):
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + tr(" y ", " and ") + items[-1]


# ================================================================ estilo
TINTA, PAPEL, BRASA, AMBAR, AGUA, CENIZA = "#16232E", "#EEF2F4", "#E4572E", "#F2A541", "#2C7DA0", "#6B7B86"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..125,300..800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea, label {{
  font-family: 'Archivo', system-ui, sans-serif !important; }}
[data-testid="stAppViewContainer"] {{ background: {PAPEL}; }}
[data-testid="stSidebar"] {{ background: {TINTA}; }}
[data-testid="stSidebar"] * {{ color: #DCE4E9 !important; }}
[data-testid="stSidebar"] [data-baseweb="select"] *, [data-testid="stSidebar"] input {{ color: {TINTA} !important; }}
[data-testid="stSidebar"] hr {{ border-color: #2E4252; }}
.block-container {{ padding-top: 3.4rem; max-width: 1320px; }}
h1, h2, h3 {{ font-stretch: 118%; letter-spacing: -0.01em; color: {TINTA}; }}
h1 {{ font-weight: 800; font-size: 2.35rem !important; line-height: 1.08; margin-bottom: .2rem; }}
.zona-sub {{ color: {CENIZA}; font-size: 1.02rem; margin: 0 0 .6rem 0; }}
.estado {{ font-size: 1.25rem; line-height: 1.45; color: {TINTA}; max-width: 62ch; margin: .2rem 0 1.1rem 0; }}
.estado b {{ font-weight: 700; }}
.marca {{ font-stretch: 125%; font-weight: 800; font-size: 1.5rem; letter-spacing: .01em; color: #FFFFFF !important; }}
.marca span {{ color: {BRASA} !important; }}
[data-testid="stMetric"] {{ border-left: 3px solid {BRASA}; padding: .15rem 0 .15rem .8rem; }}
[data-testid="stMetricLabel"] p {{ color: {CENIZA} !important; font-size: .9rem; }}
[data-testid="stMetricValue"] {{ font-stretch: 112%; font-weight: 700; color: {TINTA}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid #CBD5DB; }}
.stTabs [data-baseweb="tab"] {{ padding: .5rem 0; font-weight: 600; color: {CENIZA}; }}
.stTabs [aria-selected="true"] {{ color: {TINTA} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {BRASA}; height: 3px; }}
.nota {{ color: {CENIZA}; font-size: .92rem; max-width: 80ch; }}
.metodo h3 {{ margin-top: 1.6rem; }}
.metodo p, .metodo li {{ max-width: 78ch; line-height: 1.6; }}
:focus-visible {{ outline: 2px solid {BRASA} !important; outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>""", unsafe_allow_html=True)

pio.templates["firent"] = go.layout.Template(layout=dict(
    font=dict(family="Archivo, system-ui, sans-serif", color=TINTA, size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[BRASA, AGUA, AMBAR, CENIZA, "#8E6C8A", "#4F9D69"],
    xaxis=dict(gridcolor="#D8E0E5", zeroline=False, linecolor="#B9C5CC"),
    yaxis=dict(gridcolor="#D8E0E5", zeroline=False, linecolor="#B9C5CC"),
    hoverlabel=dict(font=dict(family="Archivo, sans-serif")),
    legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=40, b=10, l=10, r=10)))
pio.templates.default = "firent"
CONFIG_GRAF = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
               "toImageButtonOptions": {"format": "png", "scale": 2}}


def grafico(fig, **kw):
    st.plotly_chart(fig, width="stretch", config=CONFIG_GRAF, **kw)


# ================================================================ datos
@st.cache_data
def cargar():
    serie = pd.read_csv(CARPETA / "serie_armonizada.csv")
    grilla = pd.read_csv(CARPETA / "grilla_mensual.csv")
    regiones = json.loads((CARPETA / "regiones.json").read_text(encoding="utf-8"))
    metricas = json.loads((CARPETA / "metricas.json").read_text(encoding="utf-8"))
    return serie, grilla, regiones, metricas


@st.cache_data
def cargar_focos():
    if not ARCHIVOS_FOCOS:
        return None
    f = pd.concat([pd.read_parquet(a) for a in ARCHIVOS_FOCOS], ignore_index=True)
    f["fecha"] = pd.to_datetime(f["fecha"])
    return f


NECESARIOS = ["serie_armonizada.csv", "grilla_mensual.csv", "regiones.json", "metricas.json"]
faltan = [f for f in NECESARIOS if not (CARPETA / f).exists()]
if faltan:
    st.caption(f"Versión de la página: {VERSION_APP}")
    hay = sorted(p.relative_to(RAIZ).as_posix() for p in RAIZ.rglob("*") if p.is_file() and ".git" not in p.parts)
    st.error("No encuentro estos archivos de resultados: " + ", ".join(faltan) + ". "
             "Súbanlos al repositorio de GitHub (los archivos de la carpeta `salida` de Drive).")
    st.write("Archivos que sí encuentro en el repositorio:", hay[:50])
    st.stop()

serie, grilla, REGIONES, MET = cargar()
focos = cargar_focos()
FECHA_FIN = pd.Timestamp(MET["fecha_fin"])
FACTORES = MET["factores_focos"]
FACTOR_PAIS = MET.get("factor_pais_focos", 3.35)
anios = sorted(serie.anio.unique())

CIUDADES = [("Buenos Aires", -34.60, -58.38), ("Córdoba", -31.42, -64.18), ("Rosario", -32.95, -60.65),
            ("Mendoza", -32.89, -68.84), ("San Miguel de Tucumán", -26.82, -65.22), ("Salta", -24.79, -65.41),
            ("San Salvador de Jujuy", -24.19, -65.30), ("Santa Fe", -31.63, -60.70), ("Paraná", -31.73, -60.53),
            ("Resistencia", -27.45, -58.99), ("Corrientes", -27.47, -58.83), ("Posadas", -27.37, -55.90),
            ("Formosa", -26.18, -58.17), ("Santiago del Estero", -27.79, -64.26), ("Catamarca", -28.47, -65.78),
            ("La Rioja", -29.41, -66.86), ("San Juan", -31.54, -68.54), ("San Luis", -33.30, -66.34),
            ("Santa Rosa", -36.62, -64.29), ("Neuquén", -38.95, -68.06), ("Bahía Blanca", -38.72, -62.27),
            ("Mar del Plata", -38.00, -57.56), ("Viedma", -40.81, -63.00), ("San Carlos de Bariloche", -41.13, -71.31),
            ("El Bolsón", -41.96, -71.53), ("Esquel", -42.91, -71.32), ("Rawson", -43.30, -65.10),
            ("Comodoro Rivadavia", -45.86, -67.48), ("Río Gallegos", -51.62, -69.22), ("Ushuaia", -54.80, -68.30)]


@st.cache_data
def conteo_diario(region, bbox, k):
    "Focos por día y sensor de una zona, y su versión armonizada (antes de VIIRS: MODIS × factor)."
    f = cargar_focos()
    if region == AREA:
        f = f[f.lat.between(bbox[0], bbox[1]) & f.lon.between(bbox[2], bbox[3])]
    elif region != PAIS:
        f = f[f.region == region]
    d = f.groupby(["fecha", "sensor"], observed=True).size().unstack("sensor")
    d = d.reindex(pd.date_range("2003-01-01", FECHA_FIN), fill_value=0).fillna(0)
    for s in ["MODIS", "VIIRS"]:
        if s not in d:
            d[s] = 0
    d["equiv"] = np.where(d.index >= INICIO_VIIRS, d.VIIRS, d.MODIS * k)
    return d


def region_que_contiene(lat, lon):
    for nombre, (la0, la1, lo0, lo1) in REGIONES.items():
        if la0 <= lat <= la1 and lo0 <= lon <= lo1:
            return nombre
    return RESTO


@st.cache_data
def serie_area(bbox):
    """Serie mensual armonizada de un área propia, calculada a partir de los focos individuales."""
    la0, la1, lo0, lo1 = bbox
    f = cargar_focos()
    sub = f[f.lat.between(la0, la1) & f.lon.between(lo0, lo1)]
    m = (sub.groupby([sub.fecha.dt.year.rename("anio"), sub.fecha.dt.month.rename("mes"), "sensor"], observed=True)
         .size().unstack("sensor"))
    idx = pd.MultiIndex.from_product([anios, range(1, 13)], names=["anio", "mes"])
    m = m.reindex(idx).fillna(0)
    for c in ["MODIS", "VIIRS"]:
        if c not in m:
            m[c] = 0.0
    comun = m.loc[(slice(2012, 2025), slice(None)), :]
    if comun.MODIS.sum() >= 300:
        k, origen = comun.VIIRS.sum() / comun.MODIS.sum(), "propio"
    else:
        k, origen = FACTORES.get(region_que_contiene((la0 + la1) / 2, (lo0 + lo1) / 2), FACTOR_PAIS), "region"
    d = m.reset_index().rename(columns={"MODIS": "focos_modis", "VIIRS": "focos_viirs"})
    d["focos_viirs"] = np.where(d.anio >= 2012, d.focos_viirs, np.nan)
    d["focos_equiv"] = np.where(d.anio >= 2012, d.focos_viirs, d.focos_modis * k)
    futuro = d.anio * 12 + d.mes > FECHA_FIN.year * 12 + FECHA_FIN.month
    d.loc[futuro, ["focos_modis", "focos_viirs", "focos_equiv"]] = np.nan
    d["region"], d["fuente"] = AREA, np.where(d.anio >= 2012, "VIIRS", "MODIS armonizado")
    c = 0.05                                     # grilla fina (~5 km) para el mapa del área
    g = sub.assign(lat_c=(np.floor(sub.lat / c) * c + c / 2).round(3), lon_c=(np.floor(sub.lon / c) * c + c / 2).round(3),
                   anio=sub.fecha.dt.year, mes=sub.fecha.dt.month)
    g = g.groupby(["lat_c", "lon_c", "anio", "mes", "sensor"], observed=True).size().unstack("sensor", fill_value=0)
    for s in ["MODIS", "VIIRS"]:
        if s not in g:
            g[s] = 0
    g = g.reset_index()
    g["focos_equiv"] = np.where(g.anio >= 2012, g.VIIRS, g.MODIS * k)
    g = g[g.focos_equiv > 0][["lat_c", "lon_c", "anio", "mes", "focos_equiv"]].assign(region=AREA)
    return d, g, float(k), origen, len(sub)


# ================================================================ área dibujada en un mapa (se aplica antes de los controles)
if "bbox_pendiente" in st.session_state:
    la0, la1, lo0, lo1 = st.session_state.pop("bbox_pendiente")
    st.session_state.update({"lat_n": la1, "lat_s": la0, "lon_o": lo0, "lon_e": lo1, "tipo_zona_val": "area"})
    for l_ in ("ES", "EN"):
        st.session_state.pop(f"tipo_zona_{l_}", None)
for k_, v_ in {"lat_n": -30.5, "lat_s": -32.5, "lon_o": -65.2, "lon_e": -64.0}.items():
    st.session_state.setdefault(k_, v_)

# ================================================================ barra lateral
# Los controles cuyo texto cambia con el idioma guardan su valor aparte, para no perderlo al cambiar de idioma
OTRO = "ES" if EN else "EN"


def control(func, nombre, opciones, **kw):
    st.session_state.pop(f"{nombre}_{OTRO}", None)
    previo = st.session_state.get(f"{nombre}_val", opciones[0])
    v = func(options=opciones, index=opciones.index(previo) if previo in opciones else 0,
             key=f"{nombre}_{st.session_state.idioma}", **kw)
    st.session_state[f"{nombre}_val"] = v
    return v

st.sidebar.markdown("<div class='marca'>FIREN<span>'</span>T</div>", unsafe_allow_html=True)
st.sidebar.caption(tr("Calendario armonizado de focos de calor · Argentina", "Harmonized hotspot calendar · Argentina")
                   + f" · {VERSION_APP}")
st.sidebar.radio("🌐 Idioma / Language", ["ES", "EN"], key="idioma", horizontal=True)

opciones_zona = ["region", "area"] if focos is not None else ["region"]
tipo_zona = control(st.sidebar.radio, "tipo_zona", opciones_zona, label=tr("Zona", "Area"), horizontal=True,
                    format_func=lambda x: tr("Región", "Region") if x == "region" else tr("Área propia", "Custom area"))
bbox = None
if tipo_zona == "region":
    region = control(st.sidebar.selectbox, "region", [PAIS] + list(REGIONES) + [RESTO], label=tr("Región", "Region"),
                     format_func=zona_txt)
else:
    region = AREA
    st.sidebar.caption(tr("Dibujá un rectángulo en el **Mapa** o escribí los límites en grados decimales "
                          "(sur y oeste son negativos).",
                          "Draw a rectangle on the **Map** or type the limits in decimal degrees "
                          "(south and west are negative)."))
    s1, s2 = st.sidebar.columns(2)
    s1.number_input(tr("Latitud norte", "North latitude"), -56.0, -21.0, step=0.1, format="%.2f", key="lat_n")
    s2.number_input(tr("Latitud sur", "South latitude"), -56.0, -21.0, step=0.1, format="%.2f", key="lat_s")
    s1.number_input(tr("Longitud oeste", "West longitude"), -74.0, -53.0, step=0.1, format="%.2f", key="lon_o")
    s2.number_input(tr("Longitud este", "East longitude"), -74.0, -53.0, step=0.1, format="%.2f", key="lon_e")
    ss = st.session_state
    bbox = (min(ss.lat_n, ss.lat_s), max(ss.lat_n, ss.lat_s), min(ss.lon_o, ss.lon_e), max(ss.lon_o, ss.lon_e))
    if bbox[0] == bbox[1] or bbox[2] == bbox[3]:
        st.sidebar.error(tr("El área no puede tener ancho o alto cero.", "The area cannot have zero width or height."))
        st.stop()

if region == AREA:
    medida = "focos"
    st.sidebar.caption(tr("En un área propia la actividad se mide en **focos**.",
                          "In a custom area, activity is measured in **hotspots**."))
else:
    medida = control(st.sidebar.radio, "medida", ["focos", "dias", "frp"],
                     label=tr("Medir la actividad de fuego por", "Measure fire activity by"), format_func=NOMBRE_MEDIDA.get)
    st.sidebar.caption(EXPLICA_MEDIDA[medida])
anio = control(st.sidebar.selectbox, "anio", anios[::-1], label=tr("Año a analizar", "Year to analyze"))
previos = [a for a in anios if a < anio] or anios
st.session_state.pop(f"ref_{OTRO}", None)
ref_prev = st.session_state.get("ref_val", (previos[0], previos[-1]))
if not (ref_prev[0] in previos and ref_prev[1] in previos):
    ref_prev = (previos[0], previos[-1])
ref_desde, ref_hasta = st.sidebar.select_slider(tr("Período de referencia (lo habitual)", "Reference period (the usual)"),
                                                options=previos, value=ref_prev, key=f"ref_{st.session_state.idioma}")
st.session_state.ref_val = (ref_desde, ref_hasta)
st.sidebar.divider()
st.sidebar.caption(tr(f"Datos de NASA FIRMS hasta el {FECHA_FIN:%d/%m/%Y} (los últimos meses son provisorios). "
                      "Antes de 2012: MODIS armonizado (estimado). Desde 2012: VIIRS.",
                      f"NASA FIRMS data up to {FECHA_FIN:%Y-%m-%d} (latest months are provisional). "
                      "Before 2012: harmonized MODIS (estimated). From 2012: VIIRS."))

# ================================================================ zona elegida
if region == AREA:
    dr, g_zona, K_ZONA, ORIGEN_K, n_focos_area = serie_area(bbox)
    origen_k_txt = (tr("propio del área (calculado con sus focos de 2012–2025)",
                       "specific to the area (computed from its 2012–2025 hotspots)") if ORIGEN_K == "propio" else
                    tr(f"de la región «{zona_txt(region_que_contiene(np.mean(bbox[:2]), np.mean(bbox[2:])))}» "
                       "(el área tiene pocos focos para calcular uno propio)",
                       f"of the «{zona_txt(region_que_contiene(np.mean(bbox[:2]), np.mean(bbox[2:])))}» region "
                       "(the area has too few hotspots for its own)"))
else:
    dr = serie[serie.region == region].copy()
    g_zona = grilla if region == PAIS else grilla[grilla.region == region]
    K_ZONA = FACTOR_PAIS if region == PAIS else FACTORES.get(region, FACTOR_PAIS)
    origen_k_txt = (tr("promedio de Argentina", "Argentina's average") if region == PAIS else
                    tr(f"de la región «{zona_txt(region)}»", f"of the «{zona_txt(region)}» region"))
    n_focos_area = None

col = f"{medida}_equiv"
ref = dr[dr.anio.between(ref_desde, ref_hasta)]
clima = ref.groupby("mes")[col].agg(mediana="median", p10=lambda s: s.quantile(.1), p90=lambda s: s.quantile(.9))
clima = clima.reindex(range(1, 13))
actual = dr[dr.anio == anio].set_index("mes")[col].reindex(range(1, 13))
con_dato = actual.dropna().index

# ================================================================ accesibilidad
cab1, cab2 = st.columns([12, 1])
with cab2:
    with st.popover("👁️", help=tr("Accesibilidad: colores para daltonismo y descripciones de los gráficos",
                                  "Accessibility: colorblind-safe colors and chart descriptions")):
        st.toggle(tr("Modo accesible", "Accessible mode"), key="acc")
        st.caption(tr("Cambia los colores por paletas aptas para daltonismo, agrega marcas y números que no dependen "
                      "del color, y muestra una descripción en texto debajo de cada gráfico.",
                      "Switches to colorblind-safe palettes, adds markers and numbers that do not rely on color, and "
                      "shows a text description under each chart."))
ACC = st.session_state.get("acc", False)

if ACC:
    SEQ, DIV = "Cividis", "PuOr_r"
    C_NORMAL, C_ALTO, C_MODIS, C_VIIRS, C_ARM = "#9DB7D5", "#08306B", "#0072B2", "#E69F00", "#000000"
    C_BANDA = "rgba(0,0,0,0.12)"
    ESCALA_FRANJA = "Cividis"
else:
    SEQ, DIV = "YlOrRd", "RdBu_r"
    C_NORMAL, C_ALTO, C_MODIS, C_VIIRS, C_ARM = AMBAR, BRASA, AGUA, AMBAR, TINTA
    C_BANDA = "rgba(107,123,134,0.18)"
    ESCALA_FRANJA = [[0, "#F4F6F7"], [.35, "#F6D79A"], [.6, AMBAR], [.8, BRASA], [1, "#5B1A10"]]


def describir(texto):
    "Descripción en texto de un gráfico: solo se muestra en modo accesible."
    if ACC:
        st.info("📝 **" + tr("Descripción", "Description") + ":** " + texto)


def colores(escala, valores):
    "Colores de una escala de Plotly para valores entre 0 y 1."
    from plotly.colors import sample_colorscale
    v = np.clip(np.asarray(valores, dtype=float), 0, 1)
    return sample_colorscale(escala, list(v), colortype="rgb") if len(v) else []
SEMAFORO = ({"normal": "#009E73", "elevado": "#E69F00", "excepcional": "#D55E00"} if ACC
            else {"normal": "#4F9D69", "elevado": AMBAR, "excepcional": BRASA})
NIVEL_TXT = {"normal": tr("Dentro de lo habitual", "Within the usual range"),
             "elevado": tr("Actividad elevada", "Elevated activity"),
             "excepcional": tr("Actividad excepcional", "Exceptional activity")}
ICONO = {"normal": "●", "elevado": "▲", "excepcional": "◆"}

st.markdown(f"""
<style>
.semaforo {{ display:flex; gap:1.1rem; align-items:center; background:#FFFFFF; border:1px solid #CBD5DB;
  border-radius:8px; padding:.9rem 1.1rem; margin:.1rem 0 .6rem 0; max-width:940px; }}
.semaforo .luces {{ display:flex; flex-direction:column; gap:5px; background:{TINTA}; padding:6px; border-radius:14px; }}
.semaforo .luz {{ width:16px; height:16px; border-radius:50%; background:#3A4B57; }}
.semaforo .nivel {{ font-weight:800; font-stretch:118%; font-size:1.2rem; line-height:1.2; }}
.semaforo p {{ margin:.25rem 0 0 0; max-width:72ch; line-height:1.5; color:{TINTA}; }}
</style>""", unsafe_allow_html=True)


def fecha_txt(t, con_anio=True):
    if EN:
        return f"{t:%b %d, %Y}" if con_anio else f"{t:%b %d}"
    return f"{t:%d/%m/%Y}" if con_anio else f"{t:%d/%m}"


def ventana_reciente(diario, dias, refs):
    """Focos armonizados de los últimos `dias` días hasta FECHA_FIN y de la misma ventana en cada año de referencia."""
    fin = FECHA_FIN
    ini = fin - pd.Timedelta(days=dias - 1)
    actual = float(diario.equiv.loc[ini:fin].sum())
    filas = []
    for y in refs:
        f_y = fin - pd.DateOffset(years=fin.year - y)          # un 29/02 pasa al 28/02
        i_y = f_y - pd.Timedelta(days=dias - 1)
        if i_y >= diario.index[0]:
            filas.append({"anio": y, "valor": float(diario.equiv.loc[i_y:f_y].sum())})
    return actual, ini, fin, pd.DataFrame(filas, columns=["anio", "valor"])


def nivel_alerta(actual, historicos):
    """Nivel del semáforo: elevado por encima del percentil 75 de los años de referencia, excepcional por encima del 90."""
    h = np.asarray(historicos, dtype=float)
    p75, p90 = np.percentile(h, [75, 90])
    pct = 100 * ((h < actual).sum() + 0.5 * (h == actual).sum()) / len(h)
    if actual > 0 and actual > p90:
        nivel = "excepcional"
    elif actual > 0 and actual > p75:
        nivel = "elevado"
    else:
        nivel = "normal"
    return nivel, pct, p75, p90


def tramo_temporada(y, mes_ini):
    ini = pd.Timestamp(year=y, month=mes_ini, day=1)
    return ini, ini + pd.DateOffset(years=1) - pd.Timedelta(days=1)


def nombre_temporada(y, mes_ini):
    return str(y) if mes_ini == 1 else f"{y}/{str(y + 1)[-2:]}"


def curvas_temporada(diario, mes_ini, temporada, refs):
    """Acumulado diario (365 días) de la temporada elegida y de cada temporada de referencia completa."""
    hist = {}
    for y in refs:
        ini, fin = tramo_temporada(y, mes_ini)
        if y == temporada or ini < diario.index[0] or fin > FECHA_FIN:
            continue
        hist[y] = diario.equiv.loc[ini:fin].cumsum().to_numpy()[:365]
    ini, fin = tramo_temporada(temporada, mes_ini)
    actual = diario.equiv.loc[ini:min(fin, FECHA_FIN)].cumsum().to_numpy()[:365]
    return ini, actual, hist


def adelanto_dias(acumulado, mediana):
    """Días de adelanto (+) o atraso (−) respecto de la temporada mediana. None si ya superó el total habitual."""
    d = len(acumulado) - 1
    v = acumulado[d]
    if v > mediana[-1]:
        return None, None
    j = int(np.argmax(mediana >= v))
    return j - d, j

# ================================================================ cabecera: nombre, estado y franja de fuego
total, esperado = actual.sum(), clima.mediana.reindex(con_dato).sum()
dif = 100 * (total - esperado) / esperado if esperado else np.nan
meses_altos = [MESES_LARGOS[m - 1] for m in con_dato if actual[m] > clima.p90[m]]
pico = int(actual.idxmax()) if actual.notna().any() and actual.max() > 0 else None
pico_normal = int(clima.mediana.idxmax()) if clima.mediana.notna().any() and clima.mediana.max() > 0 else None

with cab1:
    if region == AREA:
        st.title(tr("Área propia", "Custom area"))
        st.markdown(f"<p class='zona-sub'>{bbox[0]:.2f}° {tr('a', 'to')} {bbox[1]:.2f}° lat · "
                    f"{bbox[2]:.2f}° {tr('a', 'to')} {bbox[3]:.2f}° lon</p>", unsafe_allow_html=True)
    else:
        st.title(zona_txt(region))
        st.markdown(f"<p class='zona-sub'>{NOMBRE_MEDIDA[medida]} · {tr('mediana de', 'median of')} "
                    f"{ref_desde}–{ref_hasta}</p>", unsafe_allow_html=True)

if region == AREA and n_focos_area == 0:
    st.warning(tr("No hay focos registrados en esta área. Probá con un área más grande o en otra zona.",
                  "There are no hotspots recorded in this area. Try a larger area or a different place."))
    st.stop()
if focos is not None:
    diario_zona = conteo_diario(region, bbox, K_ZONA)
    refs_hoy = ([y for y in range(ref_desde, ref_hasta + 1) if y < FECHA_FIN.year]
                or [y for y in anios if y < FECHA_FIN.year])
    h1, h2 = st.columns([3, 1.3])
    with h1:
        st.markdown("**" + tr("Situación de los últimos días", "Situation in recent days") + "**")
    with h2:
        dias_v = st.radio(tr("Ventana", "Window"), [7, 15, 30], index=2, horizontal=True, key="ventana_dias",
                          format_func=lambda d: f"{d} {tr('días', 'days')}", label_visibility="collapsed")
    act_v, ini_v, fin_v, hist_v = ventana_reciente(diario_zona, dias_v, refs_hoy)

    if len(hist_v) >= 5:
        nivel, pct_v, p75_v, p90_v = nivel_alerta(act_v, hist_v.valor)
        med_v = float(hist_v.valor.median())
        periodo_ref = f"{min(hist_v.anio)}–{max(hist_v.anio)}"
        if med_v > 0:
            r = act_v / med_v
            if r >= 1.15:
                cmp_txt = tr(f"{miles(r, 1)} veces la mediana", f"{r:.1f} times the median")
            elif r <= 0.87:
                cmp_txt = tr(f"{100 * (1 - r):.0f} % menos que la mediana", f"{100 * (1 - r):.0f}% below the median")
            else:
                cmp_txt = tr("parecido a la mediana", "close to the median")
        else:
            cmp_txt = tr("cuando habitualmente no hay focos", "when there are usually no hotspots")
        mayores = hist_v[hist_v.valor > act_v].sort_values("valor", ascending=False)
        if act_v > 0 and len(mayores) == 0:
            extra = tr(" Es el valor más alto de la serie para estas fechas.", " It is the highest value on record for these dates.")
        elif 0 < len(mayores) <= 3:
            extra = tr(f" Solo {lista([str(a) for a in mayores.anio])} tuvieron más.",
                       f" Only {lista([str(a) for a in mayores.anio])} had more.")
        else:
            extra = ""
        texto_v = (tr(f"Del {fecha_txt(ini_v, False)} al {fecha_txt(fin_v)} hubo <b>{miles(act_v)} focos armonizados</b>, "
                      f"{cmp_txt} de esas mismas fechas en {periodo_ref} ({miles(med_v)}). "
                      f"Supera al {pct_v:.0f} % de los años de referencia.",
                      f"From {fecha_txt(ini_v, False)} to {fecha_txt(fin_v)} there were <b>{miles(act_v)} harmonized hotspots</b>, "
                      f"{cmp_txt} for the same dates in {periodo_ref} ({miles(med_v)}). "
                      f"It is above {pct_v:.0f}% of the reference years.") + extra)
        luces = "".join(
            f"<span class='luz' style='background:{SEMAFORO[n]};box-shadow:0 0 8px {SEMAFORO[n]}'></span>" if n == nivel
            else "<span class='luz'></span>" for n in ["excepcional", "elevado", "normal"])
        st.markdown(f"<div class='semaforo' style='border-left:6px solid {SEMAFORO[nivel]}' role='status'>"
                    f"<div class='luces' aria-hidden='true'>{luces}</div><div>"
                    f"<div class='nivel'>{ICONO[nivel]} {NIVEL_TXT[nivel]}</div><p>{texto_v}</p></div></div>",
                    unsafe_allow_html=True)

        # Dónde cae este período entre los años de referencia
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_v.valor, y=[""] * len(hist_v), mode="markers", name=tr("Años de referencia", "Reference years"),
                                 marker=dict(size=10, color="#9AA5B1", symbol="circle", line=dict(width=1, color="#FFFFFF")),
                                 customdata=hist_v.anio, hovertemplate="%{customdata}: %{x:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[act_v], y=[""], mode="markers", name=tr("Últimos días", "Recent days"),
                                 marker=dict(size=17, color=SEMAFORO[nivel], symbol="diamond", line=dict(width=1.5, color=TINTA)),
                                 hovertemplate=tr("Últimos días", "Recent days") + ": %{x:,.0f}<extra></extra>"))
        fig.add_vline(x=p75_v, line_dash="dot", line_color=CENIZA, annotation_text="p75", annotation_position="top")
        fig.add_vline(x=p90_v, line_dash="dash", line_color=CENIZA, annotation_text="p90", annotation_position="top")
        fig.update_layout(height=150, margin=dict(t=28, b=10, l=10, r=10), showlegend=False,
                          xaxis_title=tr(f"Focos armonizados en {dias_v} días", f"Harmonized hotspots in {dias_v} days"),
                          yaxis=dict(visible=False))
        grafico(fig)
        st.markdown(f"<p class='nota'>{tr('Cada punto gris es un año de referencia en estas mismas fechas; el rombo es el período actual. Elevado: por encima del 75 % de los años; excepcional: por encima del 90 %. Los últimos días son datos provisorios de tiempo casi real y los años anteriores a 2012 son MODIS armonizado (estimado).', 'Each grey dot is a reference year on these same dates; the diamond is the current period. Elevated: above 75% of years; exceptional: above 90%. The latest days are provisional near-real-time data, and years before 2012 are harmonized MODIS (estimated).')}</p>",
                    unsafe_allow_html=True)
        describir(tr(f"Semáforo en «{NIVEL_TXT[nivel]}». ", f"Traffic light at «{NIVEL_TXT[nivel]}». ")
                  + texto_v.replace("<b>", "").replace("</b>", ""))
    else:
        st.info(tr("Hacen falta al menos 5 años de referencia para evaluar los últimos días.",
                   "At least 5 reference years are needed to assess recent days."))
        
if np.isfinite(dif) and len(con_dato):
    periodo = (tr("en todo el año", "for the whole year") if len(con_dato) == 12 else
               tr(f"entre enero y {MESES_LARGOS[int(con_dato.max()) - 1]}", f"from January to {MESES_LARGOS[int(con_dato.max()) - 1]}"))
    sentido = (tr("más", "more") if dif > 0 else tr("menos", "less"))
    st.markdown(f"<p class='estado'>{tr('En', 'In')} {anio} {tr('hubo', 'there were')} <b>{miles(total)} "
                f"{UNIDAD[medida]}</b> {periodo}: <b>{abs(dif):.0f} % {sentido}</b> "
                f"{tr('que lo habitual', 'than usual')}"
                + (tr(f", con {len(meses_altos)} meses por encima del rango normal.",
                      f", with {len(meses_altos)} months above the normal range.") if meses_altos else ".")
                + "</p>", unsafe_allow_html=True)

# Franja de fuego: una raya por mes, de 2003 a hoy
franja = dr.sort_values(["anio", "mes"])
franja = franja[franja[col].notna()]
valores = np.log10(franja[col].values + 1)
fechas = pd.to_datetime(dict(year=franja.anio, month=franja.mes, day=1))
fig = go.Figure(go.Heatmap(z=[valores], x=fechas, y=[""], colorscale=ESCALA_FRANJA, showscale=False,
                           customdata=[[miles(v) for v in franja[col].values]],
                           hovertemplate="%{x|%m/%Y}: %{customdata} " + UNIDAD[medida] + "<extra></extra>",
                           xgap=0.6))
fig.update_layout(height=104, margin=dict(t=4, b=30, l=0, r=0), xaxis=dict(showgrid=False, dtick="M24",
                  tickformat="%Y", linecolor="rgba(0,0,0,0)", tickfont=dict(size=11, color=CENIZA)),
                  yaxis=dict(showgrid=False, visible=False))
grafico(fig)
st.markdown(f"<p class='nota'>{tr('Cada raya es un mes desde 2003: más intenso el color, más actividad de fuego. Pasá el cursor para ver el valor.', 'Each stripe is one month since 2003: the stronger the color, the more fire activity. Hover to see the value.')}</p>",
            unsafe_allow_html=True)
anual_tot = dr[dr.anio <= 2025].groupby("anio")[col].sum()
if len(anual_tot):
    describir(tr(f"Franja de actividad mensual desde 2003. Los años con más actividad fueron "
                 f"{lista([str(a) for a in anual_tot.nlargest(3).index])}.",
                 f"Monthly activity stripe since 2003. The most active years were "
                 f"{lista([str(a) for a in anual_tot.nlargest(3).index])}."))

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{NOMBRE_MEDIDA[medida]} {tr('en', 'in')} {anio}", miles(total),
          f"{dif:+.0f} %" if np.isfinite(dif) else None, delta_color="inverse",
          help=tr("Diferencia respecto de lo habitual (mediana)", "Difference from usual (median)"))
c2.metric(tr("Lo habitual para esos meses", "Usual for those months"), miles(esperado),
          help=tr("Suma de las medianas históricas de esos meses", "Sum of the historical medians of those months"))
c3.metric(tr(f"Mes pico en {anio}", f"Peak month in {anio}"), MESES[pico - 1] if pico else "—",
          f"{tr('habitualmente', 'usually')}: {MESES[pico_normal - 1]}" if pico_normal else None, delta_color="off")
c4.metric(tr("Meses por encima de lo habitual", "Months above usual"), f"{len(meses_altos)} {tr('de', 'of')} {len(con_dato)}",
          help=tr("Meses que superaron al 90 % de los años de referencia", "Months above 90% of the reference years"))
notas = []
if len(con_dato) < 12:
    notas.append(tr(f"{anio} tiene datos hasta {MESES_LARGOS[int(con_dato.max()) - 1]}: se compara solo con esos meses.",
                    f"{anio} has data up to {MESES_LARGOS[int(con_dato.max()) - 1]}: only those months are compared."))
if region == AREA:
    notas.append(tr(f"Área con {miles(n_focos_area)} focos. Factor MODIS→VIIRS: {K_ZONA:.2f}, {origen_k_txt}.",
                    f"Area with {miles(n_focos_area)} hotspots. MODIS→VIIRS factor: {K_ZONA:.2f}, {origen_k_txt}."))
if notas:
    st.markdown(f"<p class='nota'>{' '.join(notas)}</p>", unsafe_allow_html=True)

tab_temporada, *tabs = st.tabs([tr("Temporada en curso", "Current season"),
                                tr("Comparación con lo habitual", "Compared with usual"), tr("Calendario", "Calendar"),
                                tr("Búsqueda por fechas", "Date search"), tr("Mapa", "Map"), tr("Armonización", "Harmonization"),
                                tr("Validación", "Validation"), tr("Datos", "Data"), tr("Cómo lo hicimos", "How we built it")])


# ================================================================ mapas (Leaflet)
CAPAS_BASE = [
    (tr("Claro", "Light"), "CartoDB positron", None),
    (tr("Topografía (curvas de nivel)", "Topography (contour lines)"), "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
     "Map data © OpenStreetMap contributors, SRTM | Map style © OpenTopoMap (CC-BY-SA)"),
    (tr("Relieve submarino (batimetría)", "Seafloor relief (bathymetry)"),
     "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
     "Tiles © Esri — GEBCO, NOAA, National Geographic, Esri"),
    (tr("Satélite", "Satellite"),
     "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
     "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics"),
]


def mapa_base(limites, dibujar=False):
    # Centro y zoom fijos (no fit_bounds): el mapa se crea dentro de una pestaña oculta y Leaflet no conoce su tamaño
    centro = [(limites[0] + limites[1]) / 2, (limites[2] + limites[3]) / 2]
    extension = max(limites[1] - limites[0], (limites[3] - limites[2]) * 1.6)
    zoom = int(np.clip(np.floor(np.log2(360 / extension)) + 1, 4, 13))
    m = folium.Map(location=centro, zoom_start=zoom, tiles=None, prefer_canvas=True, control_scale=True)
    for i, (nombre, url, attr) in enumerate(CAPAS_BASE):
        folium.TileLayer(url, name=nombre, attr=attr, max_zoom=17, show=(i == 0)).add_to(m)
    m.get_root().header.add_child(folium.Element(
        "<style>.leaflet-control-layers{font:13px/1.5 Archivo,system-ui,sans-serif;color:#16232E}"
        ".leaflet-control-layers-separator{margin:4px 0}</style>"))
    if dibujar:
        Draw(draw_options={"polyline": False, "polygon": False, "circle": False, "marker": False,
                           "circlemarker": False, "rectangle": {"shapeOptions": {"color": BRASA, "weight": 2}}},
             edit_options={"edit": False, "remove": True}).add_to(m)
    return m


def capa_ciudades(m):
    fg = folium.FeatureGroup(name=tr("Ciudades principales", "Main cities"), show=False)
    for nombre, la, lo in CIUDADES:
        folium.CircleMarker([la, lo], radius=3.5, color=TINTA, weight=1.5, fill=True, fill_color="#FFFFFF",
                            fill_opacity=1, tooltip=nombre).add_to(fg)
        folium.Marker([la, lo], icon=folium.DivIcon(
            html=f"<div style='font:600 11px Archivo,sans-serif;color:{TINTA};white-space:nowrap;"
                 f"text-shadow:0 0 3px #fff,0 0 3px #fff;transform:translate(8px,-8px)'>{nombre}</div>")).add_to(fg)
    fg.add_to(m)


def capa_regiones(m, solo=None):
    fg = folium.FeatureGroup(name=tr("Regiones analizadas", "Analyzed regions"), show=True)
    contornos = {AREA: solo} if solo else REGIONES
    for nombre, (la0, la1, lo0, lo1) in contornos.items():
        folium.Rectangle([[la0, lo0], [la1, lo1]], color=TINTA, weight=1.6, fill=False, dash_array="5 4",
                         tooltip=zona_txt(nombre)).add_to(fg)
    fg.add_to(m)


def mostrar_mapa(m, clave, alto=600):
    folium.LayerControl(collapsed=False).add_to(m)
    salida = st_folium(m, height=alto, use_container_width=True, key=clave, returned_objects=["last_active_drawing"])
    dib = (salida or {}).get("last_active_drawing")
    if dib and dib.get("geometry", {}).get("type") == "Polygon":
        coords = np.array(dib["geometry"]["coordinates"][0])
        nuevo = (round(float(coords[:, 1].min()), 2), round(float(coords[:, 1].max()), 2),
                 round(float(coords[:, 0].min()), 2), round(float(coords[:, 0].max()), 2))
        if nuevo != st.session_state.get("ultimo_dibujo") and nuevo[0] < nuevo[1] and nuevo[2] < nuevo[3]:
            st.session_state.ultimo_dibujo = nuevo
            st.session_state.bbox_pendiente = (max(nuevo[0], -56.0), min(nuevo[1], -21.0),
                                               max(nuevo[2], -74.0), min(nuevo[3], -53.0))
            st.rerun()


def limites_zona():
    if region == AREA:
        return (bbox[0] - .3, bbox[1] + .3, bbox[2] - .3, bbox[3] + .3)
    if region in REGIONES:
        la0, la1, lo0, lo1 = REGIONES[region]
        return (la0 - .5, la1 + .5, lo0 - .5, lo1 + .5)
    return (-55.5, -21.5, -74.5, -53.5)


# ================================================================ 1. comparación
with tab_temporada:
    if focos is None:
        st.warning(tr("Esta sección necesita los archivos de focos individuales (carpeta `focos`).",
                      "This section needs the individual hotspot files (`focos` folder)."))
    else:
        k1, k2 = st.columns([1, 3])
        with k1:
            mes_ini = st.selectbox(tr("La temporada empieza en", "Season starts in"), [1, 7], key="mes_ini_temporada",
                                   format_func=lambda m: MESES_LARGOS[m - 1].capitalize(),
                                   help=tr("Julio sirve para zonas cuya temporada cruza el cambio de año, como la Patagonia andina.",
                                           "July is useful for areas whose season spans the new year, such as Andean Patagonia."))
        etq = nombre_temporada(anio, mes_ini)
        ini_t, cur, hist = curvas_temporada(diario_zona, mes_ini, anio, range(ref_desde, ref_hasta + 1))
        if ini_t > FECHA_FIN:
            st.info(tr(f"La temporada {etq} todavía no empezó.", f"The {etq} season has not started yet."))
        elif len(hist) < 5:
            st.info(tr("Hacen falta al menos 5 temporadas completas de referencia. Ampliá el período de referencia en la barra lateral.",
                       "At least 5 complete reference seasons are needed. Widen the reference period in the sidebar."))
        else:
            M = np.vstack(list(hist.values()))
            p10, p50, p90 = np.percentile(M, [10, 50, 90], axis=0)
            x = pd.date_range(ini_t, periods=365)
            total_med = p50[-1]
            record = max(hist, key=lambda y: hist[y][-1])
            d = len(cur) - 1
            val = float(cur[d]) if d >= 0 else 0.0

            fig = go.Figure()
            if total_med > 0:                                  # temporada habitual: del 10 % al 90 % de lo acumulado
                j10, j90 = int(np.argmax(p50 >= .1 * total_med)), int(np.argmax(p50 >= .9 * total_med))
                fig.add_vrect(x0=x[j10], x1=x[j90], fillcolor=AMBAR if not ACC else "#BBBBBB", opacity=.10, line_width=0,
                              annotation_text=tr("Temporada habitual", "Usual season"), annotation_position="top left")
            fig.add_trace(go.Scatter(x=x, y=p90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x, y=p10, mode="lines", fill="tonexty", line=dict(width=0), fillcolor=C_BANDA,
                                     name=tr("Rango habitual (10–90 %)", "Usual range (10–90%)"), hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x, y=p50, mode="lines", name=tr("Temporada mediana", "Median season"),
                                     line=dict(color=CENIZA if not ACC else "#4D4D4D", dash="dash"),
                                     hovertemplate="%{y:,.0f}<extra>" + tr("mediana", "median") + "</extra>"))
            fig.add_trace(go.Scatter(x=x, y=hist[record], mode="lines",
                                     name=tr(f"Récord ({nombre_temporada(record, mes_ini)})", f"Record ({nombre_temporada(record, mes_ini)})"),
                                     line=dict(color="#9AA5B1", dash="dot", width=1.5),
                                     hovertemplate="%{y:,.0f}<extra>" + nombre_temporada(record, mes_ini) + "</extra>"))
            if d >= 0:
                fig.add_trace(go.Scatter(x=x[:d + 1], y=cur, mode="lines", name=etq, line=dict(color=C_ALTO, width=3.5),
                                         hovertemplate="%{y:,.0f}<extra>" + etq + "</extra>"))
                fig.add_trace(go.Scatter(x=[x[d]], y=[val], mode="markers", showlegend=False, hoverinfo="skip",
                                         marker=dict(size=10, color=C_ALTO, symbol="diamond", line=dict(width=1.5, color="#FFFFFF"))))
            fig.update_layout(height=440, hovermode="x unified", yaxis_title=tr("Focos armonizados acumulados", "Cumulative harmonized hotspots"),
                              xaxis=dict(tickformat="%d %b" if EN else "%d/%m"))
            grafico(fig)

            if d >= 0:
                med_d = float(p50[d])
                pct_d = 100 * ((M[:, d] < val).sum() + .5 * (M[:, d] == val).sum()) / len(M)
                dif_d = 100 * (val - med_d) / med_d if med_d > 0 else np.nan
                adel, j = adelanto_dias(cur, p50)
                completa = d == 364
                # El adelanto o atraso solo tiene sentido con la temporada en marcha: entre el 10 % y el 90 % de la mediana
                en_curso = total_med > 0 and .1 * total_med <= med_d <= .9 * total_med and val > 0
                if adel is not None and not en_curso:
                    adel = j = None
                    falta = max(total_med - med_d, 0.0)
                else:
                    falta = None
                hoy = tr("al cierre", "at the end") if completa else tr(f"al {fecha_txt(x[d], False)}", f"as of {fecha_txt(x[d], False)}")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric(tr(f"Acumulado {hoy}", f"Cumulative {hoy}"), miles(val),
                          f"{dif_d:+.0f} %" if np.isfinite(dif_d) else None, delta_color="inverse",
                          help=tr("Diferencia con la temporada mediana a la misma fecha", "Difference from the median season on the same date"))
                m2.metric(tr("Mediana a esta fecha", "Median on this date"), miles(med_d))
                m3.metric(tr("Percentil", "Percentile"), f"{pct_d:.0f}",
                          help=tr("Porcentaje de temporadas de referencia con menos focos acumulados a esta fecha",
                                  "Share of reference seasons with fewer cumulative hotspots on this date"))
                supero = val > total_med > 0
                if supero:
                    m4.metric(tr("Ritmo", "Pace"), tr("Sobre el total", "Above total"),
                              help=tr("Ya acumuló más que una temporada mediana completa", "It already exceeds a full median season"))
                elif adel is not None:
                    m4.metric(tr("Ritmo (días)", "Pace (days)"), f"{abs(adel)}",
                              (tr("adelantada", "ahead") if adel > 0 else tr("atrasada", "behind") if adel < 0
                               else tr("al ritmo habitual", "on usual pace")), delta_color="off",
                              help=tr("Cuántos días antes o después la temporada mediana llega a lo acumulado hasta hoy",
                                      "How many days earlier or later the median season reaches today's cumulative value"))
                elif falta is not None and not completa:
                    m4.metric(tr("Resto habitual", "Usual remainder"), miles(falta),
                              help=tr("Focos que una temporada mediana acumula desde esta fecha hasta el cierre",
                                      "Hotspots a median season accumulates from this date to its end"))

                if supero:
                    ritmo = tr(f" Ya superó lo que se acumula en una temporada mediana completa ({miles(total_med)}).",
                               f" It already exceeds a full median season ({miles(total_med)}).")
                elif adel is not None and adel != 0:
                    ritmo = tr(f" Lo acumulado se alcanza habitualmente el {fecha_txt(x[j], False)}: la temporada va "
                               f"{abs(adel)} días {'adelantada' if adel > 0 else 'atrasada'}.",
                               f" The cumulative value is usually reached on {fecha_txt(x[j], False)}: the season is "
                               f"{abs(adel)} days {'ahead' if adel > 0 else 'behind'}.")
                else:
                    ritmo = ""
                if total_med > 0 and not completa:
                    ritmo += tr(f" Habitualmente a esta fecha ya pasó el {100 * med_d / total_med:.0f} % de la temporada.",
                                f" Usually {100 * med_d / total_med:.0f}% of the season has passed by this date.")
                st.markdown(f"<p class='nota'>{tr('Línea gruesa: temporada', 'Thick line: season')} {etq}. "
                            f"{tr('Banda: rango habitual entre el 10 % y el 90 % de las temporadas', 'Band: usual range between 10% and 90% of seasons')} "
                            f"{nombre_temporada(min(hist), mes_ini)}–{nombre_temporada(max(hist), mes_ini)}. "
                            f"{tr('Franja sombreada: temporada habitual, entre el 10 % y el 90 % de lo que acumula la temporada mediana.', 'Shaded strip: usual season, between 10% and 90% of what the median season accumulates.')}"
                            f"{ritmo} {tr('Se usan focos armonizados (escala VIIRS) aunque se elija otra medida.', 'Harmonized hotspots (VIIRS scale) are used whichever measure is selected.')}</p>",
                            unsafe_allow_html=True)
                temp_txt = (tr(f" La temporada habitual va del {fecha_txt(x[j10], False)} al {fecha_txt(x[j90], False)}.",
                               f" The usual season runs from {fecha_txt(x[j10], False)} to {fecha_txt(x[j90], False)}.")
                            if total_med > 0 else "")
                describir(tr(f"Temporada {etq} {hoy}: {miles(val)} focos acumulados, contra una mediana de {miles(med_d)} "
                             f"(percentil {pct_d:.0f}).", f"Season {etq} {hoy}: {miles(val)} cumulative hotspots, versus a median of "
                             f"{miles(med_d)} (percentile {pct_d:.0f}).") + ritmo + temp_txt
                          + tr(f" La temporada récord fue {nombre_temporada(record, mes_ini)} con {miles(hist[record][-1])}.",
                               f" The record season was {nombre_temporada(record, mes_ini)} with {miles(hist[record][-1])}."))
with tabs[0]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=MESES, y=clima.p90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=MESES, y=clima.p10, mode="lines", fill="tonexty", line=dict(width=0), fillcolor=C_BANDA,
                             name=tr("Rango habitual (10–90 %)", "Usual range (10–90%)"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=MESES, y=clima.mediana, name=tr("Mediana histórica", "Historical median"),
                             mode="lines+markers", line=dict(color=CENIZA if not ACC else "#4D4D4D", dash="dash"),
                             marker=dict(symbol="diamond", size=7)))
    alto = (actual > clima.p90).values
    fig.add_trace(go.Bar(x=MESES, y=actual, name=str(anio), marker_color=np.where(alto, C_ALTO, C_NORMAL),
                         marker_pattern_shape=np.where(alto, "/", "") if ACC else None,
                         text=np.where(alto, tr("▲ alto", "▲ high"), "") if ACC else None, textposition="outside"))
    fig.update_layout(height=430, yaxis_title=f"{NOMBRE_MEDIDA[medida]} {tr('por mes', 'per month')}", hovermode="x unified",
                      bargap=.25)
    grafico(fig)
    st.markdown(f"<p class='nota'>{tr('Barras de color brasa' if not ACC else 'Barras oscuras con rayas y «▲ alto»', 'Ember bars' if not ACC else 'Dark striped bars marked «▲ high»')}"
                f"{tr(': meses por encima del rango habitual. Sirve para ver si la temporada viene más intensa o si empezó antes.', ': months above the usual range. Useful to see whether the season is more intense or started earlier.')}</p>",
                unsafe_allow_html=True)
    if pico:
        texto = tr(f"En {anio}, el mes con más actividad fue {MESES_LARGOS[pico - 1]} con {miles(actual[pico])} {UNIDAD[medida]}; "
                   f"habitualmente el pico es en {MESES_LARGOS[pico_normal - 1]}. ",
                   f"In {anio}, the most active month was {MESES_LARGOS[pico - 1]} with {miles(actual[pico])} {UNIDAD[medida]}; "
                   f"the usual peak is in {MESES_LARGOS[pico_normal - 1]}. ")
        texto += (tr(f"Por encima del rango habitual: {lista(meses_altos)}. ", f"Above the usual range: {lista(meses_altos)}. ")
                  if meses_altos else tr("Ningún mes superó el rango habitual. ", "No month exceeded the usual range. "))
        bajos = [MESES_LARGOS[m - 1] for m in con_dato if actual[m] < clima.p10[m]]
        if bajos:
            texto += tr(f"Por debajo del rango habitual: {lista(bajos)}.", f"Below the usual range: {lista(bajos)}.")
        describir(texto)

    anual = dr.groupby("anio")[col].sum(min_count=1).reset_index()
    anual["tipo"] = np.where(anual.anio == anio, tr("Año elegido", "Selected year"),
                             np.where(anual.anio < 2012, tr("MODIS armonizado", "Harmonized MODIS"), "VIIRS"))
    fig2 = px.bar(anual, x="anio", y=col, color="tipo", pattern_shape="tipo" if ACC else None,
                  color_discrete_map={tr("Año elegido", "Selected year"): C_ALTO, "VIIRS": "#9AA5B1",
                                      tr("MODIS armonizado", "Harmonized MODIS"): "#C9D1D9"},
                  labels={"anio": tr("Año", "Year"), col: f"{NOMBRE_MEDIDA[medida]} {tr('por año', 'per year')}", "tipo": ""})
    mediana_anual = ref.groupby("anio")[col].sum().median()
    fig2.add_hline(y=mediana_anual, line_dash="dash", line_color=CENIZA, annotation_text=tr("Mediana de referencia", "Reference median"))
    fig2.update_layout(height=330, bargap=.18)
    grafico(fig2)
    completos = anual[anual.anio <= 2025].dropna()
    if len(completos):
        top = completos.nlargest(3, col)
        valor_anio = anual.set_index("anio")[col].get(anio, np.nan)
        puesto = int((completos[col] > valor_anio).sum()) + 1
        describir(tr(f"Los tres años con más actividad son {lista([f'{a} ({miles(v)})' for a, v in zip(top.anio, top[col])])}. "
                     f"La mediana anual de referencia es {miles(mediana_anual)}. ",
                     f"The three most active years are {lista([f'{a} ({miles(v)})' for a, v in zip(top.anio, top[col])])}. "
                     f"The reference annual median is {miles(mediana_anual)}. ")
                  + (tr(f"{anio} ocupa el puesto {puesto} de {len(completos)}.", f"{anio} ranks {puesto} of {len(completos)}.")
                     if anio <= 2025 else tr(f"{anio} está incompleto.", f"{anio} is incomplete.")))

# ================================================================ 2. calendario
with tabs[1]:
    st.markdown(tr("El **calendario de actividad de fuego**: cada casillero es un mes. Muestra en qué meses se concentra el "
                   "fuego en la zona y qué años fueron excepcionales.",
                   "The **burning activity calendar**: each cell is one month. It shows when fire concentrates in the "
                   "area and which years were exceptional."))
    modo = st.radio(tr("Mostrar", "Show"), ["valores", "anomalia"], horizontal=True,
                    format_func=lambda x: tr("Valores", "Values") if x == "valores" else tr("Comparado con lo habitual del mes", "Compared with the month's usual"))
    tabla = dr.pivot_table(index="anio", columns="mes", values=col).reindex(columns=range(1, 13))
    if modo == "valores":
        z, escala, texto_cb, zmid = np.log10(tabla + 1), (ESCALA_FRANJA if not ACC else SEQ), f"log₁₀({UNIDAD[medida]}+1)", None
        hover = f"%{{y}} %{{x}}: %{{customdata:,.0f}} {UNIDAD[medida]}<extra></extra>"
    else:
        z = np.log2((tabla + 1) / (clima.mediana + 1).values)
        escala, texto_cb, zmid = DIV, tr("veces lo habitual (log₂)", "times usual (log₂)"), 0
        hover = "%{y} %{x}: %{customdata:,.0f} · %{z:+.1f} (log₂)<extra></extra>"
    fig = go.Figure(go.Heatmap(z=z.values, x=MESES, y=tabla.index.astype(str), colorscale=escala, zmid=zmid,
                               customdata=tabla.values, hovertemplate=hover, colorbar=dict(title=texto_cb, thickness=12),
                               xgap=2, ygap=2, text=tabla.map(abreviar).values if ACC else None,
                               texttemplate="%{text}" if ACC else None, textfont=dict(size=9)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    fig.update_layout(height=max(460, 23 * len(tabla)), margin=dict(t=30, b=10))
    grafico(fig)
    if modo == "valores":
        leyenda = tr("Más oscuro = más actividad." if not ACC else "Más claro (amarillo) = más actividad.",
                     "Darker = more activity." if not ACC else "Lighter (yellow) = more activity.")
    else:
        leyenda = tr("Rojo = más que la mediana de ese mes; azul = menos." if not ACC else "Naranja = más que la mediana de ese mes; violeta = menos.",
                     "Red = above the month's median; blue = below." if not ACC else "Orange = above the month's median; purple = below.")
    st.markdown(f"<p class='nota'>{leyenda} {tr('Hasta 2011, MODIS armonizado; desde 2012, VIIRS.', 'Until 2011, harmonized MODIS; from 2012, VIIRS.')}</p>",
                unsafe_allow_html=True)
    temporada = sorted(clima.mediana.sort_values(ascending=False).index[:3])
    extremos = tabla.stack().nlargest(3)
    describir(tr(f"Los meses con más actividad habitual son {lista([MESES_LARGOS[m - 1] for m in temporada])}. Los tres meses más "
                 f"extremos de la serie fueron {lista([f'{MESES_LARGOS[m - 1]} de {a} ({miles(v)})' for (a, m), v in extremos.items()])}. "
                 "Cada casillero muestra su valor (k = miles).",
                 f"The months with the most usual activity are {lista([MESES_LARGOS[m - 1] for m in temporada])}. The three most "
                 f"extreme months were {lista([f'{MESES_LARGOS[m - 1]} {a} ({miles(v)})' for (a, m), v in extremos.items()])}. "
                 "Each cell shows its value (k = thousands)."))

# ================================================================ 3. búsqueda por fechas
with tabs[2]:
    if focos is None:
        st.warning(tr("Esta sección necesita los archivos de focos individuales (carpeta `focos`).",
                      "This section needs the individual hotspot files (`focos` folder)."))
    else:
        st.markdown(tr("Elegí un **rango de fechas** para ver los focos de la zona, compararlos con el **mismo período de los años "
                       "anteriores** y descargarlos.",
                       "Pick a **date range** to see the area's hotspots, compare them with the **same period in previous "
                       "years** and download them."))
        fmin, fmax = dt.date(2003, 1, 1), FECHA_FIN.date()
        b1, _ = st.columns([2, 3])
        with b1:
            rango = st.date_input(tr("Rango de fechas", "Date range"), value=(fmax - dt.timedelta(days=59), fmax),
                                  min_value=fmin, max_value=fmax, format="DD/MM/YYYY" if not EN else "YYYY/MM/DD")
        if not isinstance(rango, (tuple, list)) or len(rango) < 2:
            st.info(tr("Elegí también la fecha final del rango.", "Pick the end date of the range too."))
        else:
            d0, d1 = pd.Timestamp(rango[0]), pd.Timestamp(rango[1])
            if region == AREA:
                pz = focos[focos.lat.between(bbox[0], bbox[1]) & focos.lon.between(bbox[2], bbox[3])]
            elif region == PAIS:
                pz = focos
            else:
                pz = focos[focos.region == region]
            sel = pz[pz.fecha.between(d0, d1)]
            diario = conteo_diario(region, bbox, K_ZONA)
            acum = diario.equiv.cumsum()

            def suma(a, b):
                return float(acum.loc[b] - (acum.loc[a - pd.Timedelta(days=1)] if a > acum.index[0] else 0))

            equiv_sel, largo = suma(d0, d1), (d1 - d0).days
            comp = []
            if largo <= 365:
                for y in anios:
                    try:
                        a = d0.replace(year=y)
                    except ValueError:
                        a = d0.replace(year=y, day=28)
                    b = a + pd.Timedelta(days=largo)
                    if a >= pd.Timestamp("2003-01-01") and b <= FECHA_FIN:
                        comp.append({"inicio": a.date(), "fin": b.date(), "anio": a.year, "focos_armonizados": suma(a, b)})
            comp = pd.DataFrame(comp)
            otros = comp[comp.anio != d0.year].focos_armonizados if len(comp) else pd.Series(dtype=float)
            med = float(otros.median()) if len(otros) else np.nan
            dif_b = 100 * (equiv_sel - med) / med if med else np.nan

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(tr("Focos armonizados", "Harmonized hotspots"), miles(equiv_sel),
                      f"{dif_b:+.0f} % {tr('respecto de la mediana', 'vs. median')}" if np.isfinite(dif_b) else None,
                      delta_color="inverse")
            m2.metric(tr("Mediana del mismo período", "Median of the same period"), miles(med),
                      help=tr("En los demás años con datos", "In the other years with data"))
            m3.metric(tr("Detectados por VIIRS", "Detected by VIIRS"), miles((sel.sensor == "VIIRS").sum()))
            m4.metric(tr("Detectados por MODIS", "Detected by MODIS"), miles((sel.sensor == "MODIS").sum()))

            if len(comp) > 1:
                comp["tipo"] = np.where(comp.anio == d0.year, tr("Período elegido", "Selected period"), tr("Otros años", "Other years"))
                fig = px.bar(comp, x="anio", y="focos_armonizados", color="tipo", pattern_shape="tipo" if ACC else None,
                             color_discrete_map={tr("Período elegido", "Selected period"): C_ALTO, tr("Otros años", "Other years"): "#9AA5B1"},
                             labels={"tipo": "", "anio": tr("Año", "Year"), "focos_armonizados": tr("Focos armonizados", "Harmonized hotspots")},
                             title=tr(f"Mismo período ({d0:%d/%m} – {d1:%d/%m}) en cada año", f"Same period ({d0:%b %d} – {d1:%b %d}) in every year"))
                fig.add_hline(y=med, line_dash="dash", line_color=CENIZA, annotation_text=tr("Mediana", "Median"))
                fig.update_layout(height=340, margin=dict(t=60, b=10), bargap=.18)
                grafico(fig)
                rk = comp.sort_values("focos_armonizados", ascending=False).reset_index(drop=True)
                puesto = int(rk.index[rk.anio == d0.year][0]) + 1 if (rk.anio == d0.year).any() else None
                describir(tr(f"Entre el {d0:%d/%m/%Y} y el {d1:%d/%m/%Y} hubo {miles(equiv_sel)} focos armonizados; la mediana del "
                             f"mismo período es {miles(med)}. " + (f"Es el puesto {puesto} de {len(rk)} años. " if puesto else "") +
                             f"Los años con más focos en ese período: "
                             f"{lista([f'{a} ({miles(v)})' for a, v in zip(rk.anio[:3], rk.focos_armonizados[:3])])}.",
                             f"Between {d0:%Y-%m-%d} and {d1:%Y-%m-%d} there were {miles(equiv_sel)} harmonized hotspots; the median of "
                             f"the same period is {miles(med)}. " + (f"It ranks {puesto} of {len(rk)} years. " if puesto else "") +
                             f"Years with the most hotspots in that period: "
                             f"{lista([f'{a} ({miles(v)})' for a, v in zip(rk.anio[:3], rk.focos_armonizados[:3])])}."))
            elif largo > 365:
                st.markdown(f"<p class='nota'>{tr('La comparación con otros años está disponible para rangos de hasta un año.', 'The comparison with other years is available for ranges of up to one year.')}</p>",
                            unsafe_allow_html=True)

            dd = diario.loc[d0:d1].reset_index().rename(columns={"index": "fecha"})
            fig = go.Figure(go.Bar(x=dd.fecha, y=dd.equiv, marker_color=C_NORMAL, name=tr("Focos armonizados", "Harmonized hotspots"),
                                   hovertemplate="%{x|%d/%m/%Y}: %{y:,.0f}<extra></extra>"))
            fig.update_layout(height=260, title=tr("Focos armonizados por día", "Harmonized hotspots per day"), bargap=.1)
            grafico(fig)
            if len(dd) and dd.equiv.max() > 0:
                dmax = dd.loc[dd.equiv.idxmax()]
                describir(tr(f"El día con más focos fue el {dmax.fecha:%d/%m/%Y}, con {miles(dmax.equiv)}.",
                             f"The day with the most hotspots was {dmax.fecha:%Y-%m-%d}, with {miles(dmax.equiv)}."))

            # Mapa de focos: VIIRS = punto relleno, MODIS = anillo
            if len(sel):
                muestra = sel.sample(min(len(sel), 6000), random_state=0)
                lim = (float(sel.lat.min()) - .3, float(sel.lat.max()) + .3, float(sel.lon.min()) - .3, float(sel.lon.max()) + .3)
                mb = mapa_base(lim)
                for s, nombre in [("VIIRS", "VIIRS"), ("MODIS", "MODIS")]:
                    sub_s = muestra[muestra.sensor == s]
                    if not len(sub_s):
                        continue
                    cols_s = colores(SEQ, np.log10(sub_s.frp.clip(lower=1)) / 3)
                    puntos = {"type": "FeatureCollection", "features": [
                        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(r.lon), float(r.lat)]},
                         "properties": {"c": c, "tip": f"{s} · {r.fecha:%d/%m/%Y} · FRP {r.frp:.1f} MW"}}
                        for r, c in zip(sub_s.itertuples(), cols_s)]}
                    relleno = s == "VIIRS"
                    folium.GeoJson(
                        puntos, name=tr(f"Focos {nombre}", f"{nombre} hotspots"),
                        marker=folium.CircleMarker(radius=3 if relleno else 4.5),
                        style_function=lambda f, relleno=relleno: {
                            "color": f["properties"]["c"], "fillColor": f["properties"]["c"], "weight": 0 if relleno else 2,
                            "fill": relleno, "fillOpacity": .85 if relleno else 0},
                        tooltip=folium.GeoJsonTooltip(fields=["tip"], labels=False)).add_to(mb)
                capa_ciudades(mb)
                if region == AREA:
                    capa_regiones(mb, solo=bbox)
                mostrar_mapa(mb, f"mapa_busqueda_{EN}", alto=520)
                st.markdown(f"<p class='nota'>{tr('Punto relleno = VIIRS; anillo = MODIS. El color indica la potencia del fuego (FRP): más intenso, más potencia.', 'Filled dot = VIIRS; ring = MODIS. Color shows fire radiative power (FRP): stronger color, more power.')}"
                            + (tr(f" Se muestran 6.000 de {miles(len(sel))} focos; la descarga los incluye todos.",
                                  f" Showing 6,000 of {miles(len(sel))} hotspots; the download includes all of them.") if len(sel) > 6000 else "")
                            + "</p>", unsafe_allow_html=True)

            @st.cache_data
            def csv_focos(clave, _df):
                out = _df.rename(columns={"hora_utc": "hora_utc_HHMM", "lat": "latitud", "lon": "longitud", "frp": "frp_MW"})
                out["fecha"] = out["fecha"].dt.strftime("%Y-%m-%d")
                return out.to_csv(index=False).encode("utf-8-sig")

            st.markdown("**" + tr("Descargas de esta búsqueda", "Downloads for this search") + "**")
            e1, e2, e3 = st.columns(3)
            clave = (region, bbox, str(d0.date()), str(d1.date()))
            e1.download_button(tr(f"Focos detectados ({miles(len(sel))}, CSV)", f"Detected hotspots ({miles(len(sel))}, CSV)"),
                               csv_focos(clave, sel), f"focos_{d0:%Y%m%d}_{d1:%Y%m%d}.csv", "text/csv", width="stretch",
                               disabled=len(sel) == 0, icon=":material/download:")
            e2.download_button(tr("Focos por día (CSV)", "Hotspots per day (CSV)"),
                               dd.rename(columns={"MODIS": "focos_MODIS", "VIIRS": "focos_VIIRS", "equiv": "focos_armonizados"})
                               .to_csv(index=False).encode("utf-8-sig"),
                               f"focos_por_dia_{d0:%Y%m%d}_{d1:%Y%m%d}.csv", "text/csv", width="stretch", icon=":material/download:")
            e3.download_button(tr("Comparación con otros años (CSV)", "Comparison with other years (CSV)"),
                               comp.drop(columns="tipo", errors="ignore").to_csv(index=False).encode("utf-8-sig"),
                               f"comparacion_{d0:%m%d}_{d1:%m%d}.csv", "text/csv", width="stretch", disabled=len(comp) == 0,
                               icon=":material/download:")

# ================================================================ 4. mapa
with tabs[3]:
    ca, cb = st.columns([1, 3.2])
    with ca:
        modo_mapa = st.radio(tr("Mostrar", "Show"), ["anio", "mes", "dif"],
                             format_func=lambda x: {"anio": tr("Año completo", "Whole year"), "mes": tr("Un mes", "One month"),
                                                    "dif": tr("Diferencia con lo habitual", "Difference from usual")}[x])
        mes_mapa = (st.selectbox(tr("Mes", "Month"), range(1, 13), format_func=lambda m: MESES_LARGOS[m - 1].capitalize())
                    if modo_mapa == "mes" else None)
    g = g_zona
    paso = 0.05 if region == AREA else 0.25
    if modo_mapa == "dif":
        anual_celda = (g[g.anio.between(ref_desde, ref_hasta)].groupby(["lat_c", "lon_c", "anio"]).focos_equiv.sum()
                       .unstack(fill_value=0).reindex(columns=range(ref_desde, ref_hasta + 1), fill_value=0))
        normal = anual_celda.median(axis=1)
        este = g[g.anio == anio].groupby(["lat_c", "lon_c"]).focos_equiv.sum()
        m = pd.concat([este.rename("actual"), normal.rename("normal")], axis=1).fillna(0).reset_index()
        m["valor"] = m.actual - m.normal
        m = m[m.valor.abs() >= 1]
        lim_c = float(np.nanpercentile(m.valor.abs(), 95)) if len(m) else 1.0
        m["color"] = colores(DIV, .5 + m.valor / (2 * lim_c))
        m["tip"] = [tr(f"Este año: {miles(a)} · Habitual: {miles(n)}", f"This year: {miles(a)} · Usual: {miles(n)}")
                    for a, n in zip(m.actual, m.normal)]
        nota = (tr("Naranja" if ACC else "Rojo", "Orange" if ACC else "Red") + tr(": más focos que lo habitual. ", ": more hotspots than usual. ")
                + tr("Violeta" if ACC else "Azul", "Purple" if ACC else "Blue") + tr(": menos. Lo habitual es la mediana de cada celda.",
                                                                                   ": fewer. Usual is each cell's median."))
    else:
        sel_m = g[g.anio == anio]
        if mes_mapa:
            sel_m = sel_m[sel_m.mes == mes_mapa]
        m = sel_m.groupby(["lat_c", "lon_c"]).focos_equiv.sum().reset_index()
        m["valor"] = m.focos_equiv
        tope = float(np.log10(m.valor.max() + 1)) if len(m) else 1.0
        m["color"] = colores(SEQ, np.log10(m.valor + 1) / max(tope, 1))
        m["tip"] = [f"{miles(v)} {UNIDAD['focos']}" for v in m.valor]
        nota = tr("Focos por celda, en escala VIIRS. ", "Hotspots per cell, on the VIIRS scale. ") + (
            tr("Más claro (amarillo) = más focos." if ACC else "Más oscuro = más focos.",
               "Lighter (yellow) = more hotspots." if ACC else "Darker = more hotspots."))
    mapa = mapa_base(limites_zona(), dibujar=True)
    celdas = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"color": r.color, "tip": r.tip},
         "geometry": {"type": "Polygon", "coordinates": [[[r.lon_c - paso / 2, r.lat_c - paso / 2], [r.lon_c + paso / 2, r.lat_c - paso / 2],
                                                         [r.lon_c + paso / 2, r.lat_c + paso / 2], [r.lon_c - paso / 2, r.lat_c + paso / 2],
                                                         [r.lon_c - paso / 2, r.lat_c - paso / 2]]]}}
        for r in m.itertuples()]}
    if celdas["features"]:
        folium.GeoJson(celdas, name=tr("Actividad de fuego", "Fire activity"),
                       style_function=lambda f: {"fillColor": f["properties"]["color"], "color": f["properties"]["color"],
                                                 "weight": 0.3, "fillOpacity": 0.82},
                       tooltip=folium.GeoJsonTooltip(fields=["tip"], labels=False)).add_to(mapa)
    else:
        with ca:
            st.info(tr("No hay focos para mostrar con esta selección.", "There are no hotspots to show for this selection."))
    capa_regiones(mapa, solo=bbox if region == AREA else None)
    capa_ciudades(mapa)
    with cb:
        mostrar_mapa(mapa, f"mapa_principal_{EN}", alto=640)
    with ca:
        st.markdown(f"<p class='nota'>{nota}</p>", unsafe_allow_html=True)
        st.markdown(tr("**Capas:** arriba a la derecha del mapa podés cambiar el fondo (claro, topografía con curvas de nivel, "
                       "relieve submarino o satélite) y mostrar las **ciudades principales**.",
                       "**Layers:** at the top right of the map you can switch the background (light, topography with contour "
                       "lines, seafloor relief or satellite) and show the **main cities**."))
        st.markdown(tr("**Elegir un área:** usá la herramienta ▭ de la izquierda del mapa para dibujar un rectángulo. La página "
                       "pasa a mostrar esa área en todas las secciones.",
                       "**Pick an area:** use the ▭ tool on the left of the map to draw a rectangle. The whole page then "
                       "shows that area."))
        if len(m):
            if modo_mapa == "dif":
                t = m.sort_values("valor").iloc[-1]
                describir(tr(f"{int((m.valor > 0).sum())} celdas tuvieron más focos que lo habitual y {int((m.valor < 0).sum())} menos. "
                             f"La mayor suba está cerca de {t.lat_c:.2f}°, {t.lon_c:.2f}° ({zona_txt(region_que_contiene(t.lat_c, t.lon_c))}).",
                             f"{int((m.valor > 0).sum())} cells had more hotspots than usual and {int((m.valor < 0).sum())} fewer. "
                             f"The largest increase is near {t.lat_c:.2f}°, {t.lon_c:.2f}° ({zona_txt(region_que_contiene(t.lat_c, t.lon_c))})."))
            else:
                top = m.nlargest(3, "valor")
                describir(tr("Las celdas con más focos están cerca de ", "The cells with the most hotspots are near ") + lista(
                    [f"{r.lat_c:.2f}°, {r.lon_c:.2f}° ({zona_txt(region_que_contiene(r.lat_c, r.lon_c))}, {miles(r.valor)})"
                     for r in top.itertuples()]) + ".")

# ================================================================ 5. armonización
with tabs[4]:
    st.subheader(tr("El problema", "The problem"))
    st.markdown(tr("MODIS (satélites Terra y Aqua, píxel de ~1 km) y VIIRS (satélite Suomi NPP, píxel de 375 m) miden distinto: "
                   "VIIRS divide cada incendio en más píxeles y ve fuegos más chicos. Si se pegan las dos series, parece que el fuego "
                   "se multiplicó en 2012. **Armonizar** es convertir MODIS a la escala de VIIRS para tener una serie continua desde "
                   "2003 en la que ya no importa qué sensor detectó cada foco.",
                   "MODIS (Terra and Aqua satellites, ~1 km pixel) and VIIRS (Suomi NPP satellite, 375 m pixel) measure differently: "
                   "VIIRS splits each fire into more pixels and sees smaller fires. Stitching both series together makes fire seem to "
                   "multiply in 2012. **Harmonizing** means converting MODIS to the VIIRS scale to build one continuous series since "
                   "2003, where it no longer matters which sensor detected each hotspot."))
    an = dr.groupby("anio")[[f"{medida}_modis", f"{medida}_viirs", col]].sum(min_count=1).reset_index()
    an = an[an.anio <= 2025]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=an.anio, y=an[f"{medida}_modis"], name=tr("MODIS (original)", "MODIS (original)"),
                             mode="lines+markers", line=dict(color=C_MODIS, dash="dash"), marker=dict(symbol="circle", size=7)))
    fig.add_trace(go.Scatter(x=an.anio, y=an[f"{medida}_viirs"], name=tr("VIIRS (original)", "VIIRS (original)"),
                             mode="lines+markers", line=dict(color=C_VIIRS, dash="dot"), marker=dict(symbol="square", size=7)))
    fig.add_trace(go.Scatter(x=an.anio, y=an[col], name=tr("Serie armonizada", "Harmonized series"), mode="lines",
                             line=dict(color=C_ARM, width=3.5)))
    fig.add_vline(x=2011.5, line_dash="dot", line_color=CENIZA, annotation_text=tr("empieza VIIRS", "VIIRS starts"))
    fig.update_layout(height=400, yaxis_title=f"{NOMBRE_MEDIDA[medida]} {tr('por año', 'per year')}", hovermode="x unified")
    grafico(fig)
    st.markdown(f"<p class='nota'>{tr('MODIS: línea a rayas con círculos. VIIRS: línea punteada con cuadrados. Serie armonizada: línea gruesa.', 'MODIS: dashed line with circles. VIIRS: dotted line with squares. Harmonized series: thick line.')}</p>",
                unsafe_allow_html=True)
    describir(tr(f"MODIS queda siempre por debajo de VIIRS. En esta zona se multiplica por {K_ZONA:.2f} ({origen_k_txt}) para pasarlo a la "
                 "escala de VIIRS; por eso antes de 2012 la serie armonizada está por encima de MODIS, y desde 2012 coincide con VIIRS.",
                 f"MODIS is always below VIIRS. In this area it is multiplied by {K_ZONA:.2f} ({origen_k_txt}) to reach the VIIRS "
                 "scale; that is why before 2012 the harmonized series is above MODIS, and from 2012 it matches VIIRS."))

    st.subheader(tr("Cómo elegimos el método", "How we chose the method"))
    st.markdown(tr(f"Probamos 4 métodos con los datos de toda Argentina. Cada uno aprendió con **{MET['anios_entrenamiento']}** y se "
                   f"evaluó con **{MET['anios_test']}**, años que no había visto. **Regla:** gana el más simple cuyo error no supere "
                   f"en más de {MET['tolerancia']:.0%} al mejor.",
                   f"We tested 4 methods on all of Argentina. Each learned from **{MET['anios_entrenamiento']}** and was evaluated on "
                   f"**{MET['anios_test']}**, years it had not seen. **Rule:** the simplest method whose error is within "
                   f"{MET['tolerancia']:.0%} of the best wins."))
    res = pd.DataFrame(MET["resultados"][medida]).set_index("método")
    elegido = MET["elegido"][medida]
    mejor = res["MAE"].idxmin()
    if EN:
        st.success(f"Chosen: «{metodo_txt(elegido)}» — MAE {res.loc[elegido, 'MAE']:,.1f} per region and month, total bias "
                   f"{res.loc[elegido, 'Sesgo total %']:+.1f}% and worst year {res.loc[elegido, 'Peor año (país) %']:.1f}% in "
                   f"{MET['anios_test']}. " + ("It has the lowest error." if elegido == mejor else
                                               f"The lowest error is «{metodo_txt(mejor)}», but the difference is under "
                                               f"{MET['tolerancia']:.0%}, so the simpler one is preferred."))
    else:
        st.success(MET["justificacion"][medida])
    cc1, cc2 = st.columns(2)
    with cc1:
        b = res.reset_index()
        b["nombre"] = b["método"].map(metodo_txt)
        fig = go.Figure(go.Bar(x=b["MAE"], y=b["nombre"], orientation="h",
                               marker_color=np.where(b["método"] == elegido, C_ALTO, "#9AA5B1"),
                               marker_pattern_shape=np.where(b["método"] == elegido, "/", "") if ACC else None,
                               text=[("✔ " if mm == elegido else "") + miles(v, 1) for mm, v in zip(b["método"], b["MAE"])],
                               textposition="auto", hovertemplate="%{y}: MAE %{x:,.1f}<extra></extra>"))
        fig.update_layout(height=300, xaxis_title=f"{tr('Error medio por región y mes', 'Mean error per region and month')} ({UNIDAD[medida]})",
                          yaxis=dict(autorange="reversed"), title=tr("Error (MAE): menor es mejor", "Error (MAE): lower is better"))
        grafico(fig)
    with cc2:
        d = pd.DataFrame(MET["anual_test"][medida])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.anio, y=d["Real"], name=tr("VIIRS real", "Actual VIIRS"), line=dict(color=C_ARM, width=4)))
        simbolos = ["circle", "square", "diamond", "triangle-up"]
        for i, c in enumerate([c for c in d.columns if c not in ("anio", "Real")]):
            fig.add_trace(go.Scatter(x=d.anio, y=d[c], name=metodo_txt(c), mode="lines+markers", marker=dict(symbol=simbolos[i % 4], size=8),
                                     line=dict(dash="solid" if c == elegido else "dot", width=3 if c == elegido else 1.5)))
        fig.update_layout(height=300, title=tr("Test: total de Argentina por año", "Test: Argentina's total per year"),
                          xaxis=dict(dtick=1), legend=dict(font=dict(size=10), y=-0.25), hovermode="x unified")
        grafico(fig)
    describir(tr(f"El método elegido es «{elegido}», marcado con ✔. ", f"The chosen method is «{metodo_txt(elegido)}», marked with ✔. ")
              + " ".join(f"{metodo_txt(mm)}: MAE {miles(v, 1)}." for mm, v in zip(res.index, res["MAE"])))
    tabla_res = res.rename(index=metodo_txt)
    tabla_res.index.name = tr("Método", "Method")
    if EN:
        tabla_res = tabla_res.rename(columns={"Sesgo total %": "Total bias %", "Peor año (país) %": "Worst year (country) %"})
    st.dataframe(tabla_res.style.format(precision=2).apply(
        lambda r: ["font-weight: bold; background-color: #FBE3DA" if r.name == metodo_txt(elegido) else ""] * len(r), axis=1),
        width="stretch")
    with st.expander(tr("Ver las fórmulas de cada método", "See each method's formula")):
        st.markdown(tr("**1. Factor fijo por región** — una regla de tres:", "**1. Fixed factor per region** — a rule of three:"))
        st.latex(r"k_r = \frac{\sum V}{\sum M} \qquad \hat V = k_r \cdot M")
        st.markdown(tr("**2. Regresión logarítmica por región y mes** (con corrección de sesgo κ):",
                       "**2. Log regression by region and month** (with bias correction κ):"))
        st.latex(r"\log(1+\hat V) = a + b\,\log(1+M) + c\,\sin\tfrac{2\pi\,m}{12} + d\,\cos\tfrac{2\pi\,m}{12}")
        st.markdown(tr("**3. Mapeo de cuantiles** — se emparejan percentiles:", "**3. Quantile mapping** — percentiles are matched:"))
        st.latex(r"\hat V = F_V^{-1}\big(F_M(M)\big)")
        st.markdown(tr("**4. Random Forest** — promedio de 300 árboles de decisión con MODIS, mes y región.",
                       "**4. Random Forest** — average of 300 decision trees using MODIS, month and region."))
    if medida == "focos":
        fac = pd.Series(FACTORES).sort_values()
        fig = px.bar(x=fac.values, y=[zona_txt(z) for z in fac.index], orientation="h", text=[f"{v:.1f}×" for v in fac.values],
                     labels={"x": "VIIRS / MODIS", "y": ""}, color_discrete_sequence=[C_VIIRS])
        fig.update_layout(height=290, title=tr("Cuántas veces más focos detecta VIIRS que MODIS, por región",
                                               "How many times more hotspots VIIRS detects than MODIS, per region"))
        grafico(fig)

# ================================================================ 6. validación
with tabs[5]:
    st.markdown(tr("Si la armonización es buena, **los años excepcionales tienen que aparecer igual** sin importar cómo se mida el "
                   "fuego. Cada serie (toda Argentina) se expresa como índice: 100 = año promedio.",
                   "If the harmonization works, **exceptional years must show up the same way** however fire is measured. Each "
                   "series (all of Argentina) is shown as an index: 100 = average year."))
    ia = pd.DataFrame(MET["indice_anual"]).set_index("anio")
    nombres_ia = dict(zip(ia.columns, [NOMBRE_MEDIDA["focos"], NOMBRE_MEDIDA["dias"], NOMBRE_MEDIDA["frp"]]))
    fig = go.Figure()
    for c, (color, dash, simb) in zip(ia.columns, [(C_ARM, "solid", "circle"), (C_VIIRS, "dash", "square"), (C_MODIS, "dot", "diamond")]):
        fig.add_trace(go.Scatter(x=ia.index, y=ia[c], name=nombres_ia[c], mode="lines+markers", line=dict(color=color, dash=dash),
                                 marker=dict(symbol=simb, size=7)))
    fig.add_hline(y=100, line_color=CENIZA, line_width=1)
    fig.add_vline(x=2011.5, line_dash="dot", line_color=CENIZA,
                  annotation_text=tr("antes: MODIS armonizado · después: VIIRS", "before: harmonized MODIS · after: VIIRS"))
    fig.update_layout(height=400, yaxis_title=tr("Índice (100 = año promedio)", "Index (100 = average year)"), hovermode="x unified")
    grafico(fig)
    describir(tr("Las tres líneas suben y bajan juntas. Años más activos según cada medida: ", "The three lines rise and fall together. "
                 "Most active years by measure: ") + "; ".join(f"{nombres_ia[c]}: {lista([str(a) for a in ia[c].nlargest(3).index])}" for c in ia) + ".")
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**" + tr("Correlación entre las tres medidas (2003–2025)", "Correlation between the three measures (2003–2025)") + "**")
        corr = pd.DataFrame(MET["correlacion_medidas"]).rename(index=nombres_ia, columns=nombres_ia)
        st.dataframe(corr.style.format("{:.2f}"), width="stretch")
    with v2:
        st.markdown("**" + tr("Los 5 años más activos según cada medida", "Top 5 most active years by measure") + "**")
        st.dataframe(pd.DataFrame({nombres_ia[c]: ia[c].nlargest(5).index.values for c in ia}, index=range(1, 6)), width="stretch")
    ta = pd.Series(MET["terra_sobre_aqua"]).sort_index()
    if len(ta):
        ta.index = ta.index.astype(int)
        fig = px.line(x=ta.index, y=100 * ta.values, markers=True, labels={"x": tr("Año", "Year"), "y": "Terra / Aqua (%)"},
                      color_discrete_sequence=[C_MODIS])
        fig.update_layout(height=290, title=tr("Limitación: el satélite Terra detecta cada vez menos que Aqua",
                                               "Limitation: the Terra satellite detects less and less than Aqua"))
        grafico(fig)
        describir(tr(f"Terra detectaba alrededor del {100 * ta.loc[2012:2020].mean():.0f} % de lo que detecta Aqua hasta 2020 y bajó al "
                     f"{100 * ta.iloc[-1]:.0f} % en {ta.index[-1]}.",
                     f"Terra detected about {100 * ta.loc[2012:2020].mean():.0f}% of what Aqua detects until 2020 and fell to "
                     f"{100 * ta.iloc[-1]:.0f}% in {ta.index[-1]}."))

# ================================================================ 7. datos
with tabs[6]:
    st.markdown(tr("Los datos mensuales que usa la página, para consultarlos o descargarlos. Para los **focos individuales** de un "
                   "período, usá **Búsqueda por fechas**.",
                   "The monthly data behind the page, to browse or download. For the **individual hotspots** of a period, use "
                   "**Date search**."))
    COLUMNAS = {"region": tr("Zona", "Area"), "anio": tr("Año", "Year"), "mes": tr("Mes", "Month"), "fuente": tr("Fuente", "Source"),
                "focos_modis": tr("Focos MODIS", "MODIS hotspots"), "focos_viirs": tr("Focos VIIRS", "VIIRS hotspots"),
                "focos_equiv": tr("Focos armonizados", "Harmonized hotspots"),
                "dias_modis": tr("Días con fuego MODIS", "MODIS fire-days"), "dias_viirs": tr("Días con fuego VIIRS", "VIIRS fire-days"),
                "dias_equiv": tr("Días con fuego armonizados", "Harmonized fire-days"),
                "frp_modis": "FRP MODIS (MW)", "frp_viirs": "FRP VIIRS (MW)", "frp_equiv": tr("FRP armonizada (MW)", "Harmonized FRP (MW)")}
    base = pd.concat([dr, serie], ignore_index=True) if region == AREA else serie
    opciones = ([AREA] if region == AREA else []) + [PAIS] + list(REGIONES) + [RESTO]
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        regs = st.multiselect(tr("Zonas", "Areas"), opciones, default=[region], format_func=zona_txt)
    with f2:
        a_desde, a_hasta = st.select_slider(tr("Años", "Years"), options=anios, value=(anios[0], anios[-1]))
    with f3:
        nivel = st.radio(tr("Detalle", "Detail"), ["mensual", "anual"],
                         format_func=lambda x: tr("Mensual", "Monthly") if x == "mensual" else tr("Anual", "Yearly"))
    filtro = base[base.region.isin(regs) & base.anio.between(a_desde, a_hasta)]
    sel = filtro[[c for c in COLUMNAS if c in filtro.columns]]
    if nivel == "anual":
        num = [c for c in sel.columns if c.endswith(("_modis", "_viirs", "_equiv"))]
        sel = sel.groupby(["region", "anio"], as_index=False).agg({**{c: lambda x: x.sum(min_count=1) for c in num}, "fuente": "first"})
        sel["meses"] = filtro.groupby(["region", "anio"]).focos_equiv.count().values
    sel = sel.assign(region=sel.region.map(zona_txt))
    tabla = sel.rename(columns={**COLUMNAS, "meses": tr("Meses con datos", "Months with data")})
    st.markdown(f"<p class='nota'>{miles(len(tabla))} {tr('filas · celdas vacías = sin dato (VIIRS antes de 2012, o meses que todavía no pasaron).', 'rows · empty cells = no data (VIIRS before 2012, or months not yet elapsed).')}</p>",
                unsafe_allow_html=True)
    st.dataframe(tabla, width="stretch", hide_index=True, height=420,
                 column_config={c: st.column_config.NumberColumn(format="%.0f") for c in tabla.columns
                                if c not in (COLUMNAS["region"], COLUMNAS["fuente"], COLUMNAS["anio"], COLUMNAS["mes"])}
                 | {COLUMNAS["anio"]: st.column_config.NumberColumn(format="%d")})
    a_csv = lambda df: df.to_csv(index=False).encode("utf-8-sig")
    d1_, d2_, d3_, d4_ = st.columns(4)
    d1_.download_button(tr("Esta tabla (CSV)", "This table (CSV)"), a_csv(tabla), "serie_filtrada.csv", "text/csv", width="stretch", icon=":material/download:")
    d2_.download_button(tr("Serie completa (CSV)", "Full series (CSV)"), a_csv(serie), "serie_armonizada.csv", "text/csv", width="stretch", icon=":material/download:")
    d3_.download_button(tr("Grilla del mapa (CSV)", "Map grid (CSV)"), a_csv(grilla), "grilla_mensual.csv", "text/csv", width="stretch", icon=":material/download:")
    d4_.download_button(tr("Resultados de los métodos (JSON)", "Method results (JSON)"), json.dumps(MET, ensure_ascii=False, indent=2).encode("utf-8"),
                        "metricas.json", "application/json", width="stretch", icon=":material/download:")
    st.markdown(tr("**Fuentes originales:** [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/download/) (MODIS C6.1 y VIIRS Suomi NPP C2) · "
                   "[Catálogo de Focos de Calor de la CONAE](https://catalogos5.conae.gov.ar/catalogoFocos/).",
                   "**Original sources:** [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/download/) (MODIS C6.1 and VIIRS Suomi NPP C2) · "
                   "[CONAE Hotspot Catalog](https://catalogos5.conae.gov.ar/catalogoFocos/)."))

# ================================================================ 8. cómo lo hicimos
with tabs[7]:
    res_f = pd.DataFrame(MET["resultados"]["focos"]).set_index("método")
    elegido_f = MET["elegido"]["focos"]
    fac = pd.Series(FACTORES)
    ia = pd.DataFrame(MET["indice_anual"]).set_index("anio")
    corr_min = pd.DataFrame(MET["correlacion_medidas"]).values[np.triu_indices(3, 1)].min()
    tabla_metodos = "\n".join(
        f"| {metodo_txt(mm)}{' ✔' if mm == elegido_f else ''} | {r['MAE']:,.1f} | {r['R² (log)']:.3f} | {r['Sesgo total %']:+.1f} | {r['Peor año (país) %']:.1f} |"
        for mm, r in res_f.iterrows())
    if not EN:
        texto = f"""
<div class='metodo'>

Esta página es un **calendario armonizado de actividad de fuego** para Argentina. Responde la pregunta del desafío de la CONAE:
*¿esta zona, en estas fechas, tiene más o menos fuego que lo habitual?*, usando más de dos décadas de datos de dos sensores
satelitales distintos. Abajo está todo el recorrido, de los datos crudos a la página.

### 1. Los datos
Descargamos de **NASA FIRMS** (*Fire Information for Resource Management System*) todos los focos de calor de Argentina:

| Sensor | Satélites | Píxel | Período | Focos descargados |
|---|---|---|---|---|
| MODIS colección 6.1 | Terra (mañana) y Aqua (tarde) | ~1 km | 2003 – oct. 2026 | 1,6 millones |
| VIIRS 375 m colección 2 | Suomi NPP (tarde) | 375 m | 2012 – oct. 2026 | 2,9 millones |

Cada registro es un **foco de calor**: un píxel con temperatura anómala, con fecha, hora, ubicación, satélite, confianza,
potencia radiativa (FRP) y tipo de fuente. Los datos definitivos (*archive*) se completan con los de tiempo casi real (*NRT*)
de los últimos meses, sin superponerse. Como referencia usamos también el **Catálogo de Focos de Calor de la CONAE**.

**Decisiones:**
- **MODIS desde 2003** y no desde 2000: recién desde 2003 funcionan Terra y Aqua todo el año. Antes, la serie tendría menos focos solo por haber un satélite menos.
- **VIIRS solo de Suomi NPP**: es el único VIIRS que existe desde 2012. Sumar NOAA-20 (2018) y NOAA-21 (2023) haría subir los focos esos años solo por tener más satélites.

### 2. Limpieza
Los mismos criterios para los dos sensores:
- Solo **incendios de vegetación** (`type = 0`): se descartan antorchas de gas de pozos petroleros, industrias, volcanes y fuentes en el mar. En los datos de tiempo casi real, que no traen esa columna, se descartan los focos ubicados donde el archivo definitivo ya registró fuentes fijas.
- Sin detecciones de **baja confianza**: MODIS con confianza ≥ 30 (de 0 a 100); VIIRS sin las marcadas como baja.

Quedan **4,1 millones de focos**.

### 3. Tres formas de medir
Agrupamos los focos por **zona** (país, cinco regiones y resto del país), **año** y **mes**, con tres medidas:
- **Focos**: la cantidad de detecciones.
- **Días con fuego**: pares (celda de 25 km, día) con al menos un foco. Un incendio vale igual aunque el satélite lo divida en muchos píxeles.
- **Energía (FRP)**: la suma de la potencia radiativa, en megavatios.

Las **regiones** son rectángulos aproximados elegidos por su cantidad de focos (Gran Chaco) o por el impacto de sus incendios
(Delta del Paraná, Sierras de Córdoba, Corrientes e Iberá, Patagonia andina).

### 4. El problema, medido
En los años en que funcionan los dos sensores (2012–2025), VIIRS registra en promedio **{miles(FACTOR_PAIS, 1)} veces** más focos
que MODIS. Esa proporción es **estable de un año a otro** (entre 3,0 y 3,8), y es lo que permite convertir un sensor al otro.
Pero **cambia según la región**: va de {miles(fac.min(), 1)} en {zona_txt(fac.idxmin())} a {miles(fac.max(), 1)} en {zona_txt(fac.idxmax())}.
Donde hay incendios grandes, los dos sensores los ven. Donde hay muchos fuegos chicos, VIIRS ve muchos que MODIS no ve.

### 5. Métodos de armonización (machine learning)
Planteamos la armonización como un problema de **aprendizaje supervisado**: a partir de lo que midió MODIS en una región y un
mes, predecir lo que habría medido VIIRS.

- **Entrenamiento:** {MET['anios_entrenamiento']}, años con los dos sensores.
- **Test:** {MET['anios_test']}, años que los modelos **no ven** al aprender. Así medimos si generalizan.
- **Aplicación:** 2003–2011, donde solo hay MODIS. Ahí no se puede testear porque no existe VIIRS: solo se usa el modelo ya validado.

Comparamos cuatro métodos, del más simple al más complejo:
1. **Factor fijo por región**: VIIRS = k × MODIS, con un k por región.
2. **Regresión logarítmica por región y mes**: log(1+VIIRS) = a + b·log(1+MODIS) + estacionalidad. Se corrige el sesgo que aparece al volver de la escala logarítmica.
3. **Mapeo de cuantiles**: se emparejan los percentiles de ambos sensores.
4. **Random Forest**: 300 árboles de decisión con MODIS, mes y región.

### 6. Elección: con resultados, no por gusto
**Regla fijada antes de ver los resultados:** gana el método **más simple** cuyo error no supere en más de
{MET['tolerancia']:.0%} al mejor. Resultados del test para focos:

| Método | MAE (focos por región y mes) | R² (log) | Sesgo total % | Peor año % |
|---|---|---|---|---|
{tabla_metodos}

Ganó el **{metodo_txt(elegido_f).lower()}**, que además tuvo el menor error. El Random Forest no mejoró la precisión: la relación
entre los sensores es tan estable que un modelo complejo no tiene más para aprender, y con pocas filas tiende a memorizar.
Repetimos la comparación para días con fuego (ganó: {metodo_txt(MET['elegido']['dias']).lower()}) y energía (ganó:
{metodo_txt(MET['elegido']['frp']).lower()}).

### 7. Serie armonizada y validación
- **2003–2011:** focos equivalentes = factor de la región × focos de MODIS.
- **2012 en adelante:** el dato real de VIIRS.

Para validar, armamos la serie con las tres medidas por separado. Las tres marcan los mismos años excepcionales
({lista([str(a) for a in ia.iloc[:, 0].nlargest(3).index])}) y se parecen mucho entre sí (correlación mínima {miles(corr_min, 2)}).

### 8. Cómo funciona la página
- **Lo habitual** es la **mediana** histórica de cada mes, que no se deja arrastrar por años extremos, y el rango habitual va del percentil 10 al 90.
- **Área propia:** con los focos individuales, la página calcula el calendario de cualquier rectángulo. Si el área tiene suficientes focos (al menos 300 de MODIS en 2012–2025) calcula su propio factor; si no, usa el de su región.
- **Búsqueda por fechas:** compara cualquier período con el mismo período de todos los años y permite descargar los focos.
- **Mapas** con capas de fondo (topografía con curvas de nivel, relieve submarino, satélite) y ciudades principales.
- **Accesibilidad:** el botón 👁️ activa colores aptos para daltonismo, marcas que no dependen del color y descripciones en texto.

### 9. Limitaciones
- Un foco de calor no es un incendio: un incendio grande genera muchos focos, y las nubes o el humo pueden ocultar fuegos.
- Los valores anteriores a 2012 son estimaciones, con un error de alrededor del ±{res_f.loc[elegido_f, 'Peor año (país) %']:.0f} % en el total anual.
- El satélite Terra está envejeciendo: su órbita se corre y detecta cada vez menos, lo que agrega error en los últimos años.
- Los últimos meses son datos provisorios de tiempo casi real.

### 10. Herramientas, fuentes y uso de IA
Python (pandas, NumPy, scikit-learn), Google Colab, Streamlit, Plotly, Folium/Leaflet y GitHub. Datos: NASA FIRMS; referencia:
CONAE. Mapas: © OpenStreetMap, OpenTopoMap, CARTO y Esri. **Uso de Super Inteligencia (SI):** usamos un asistente de IA
(Claude, de Anthropic) para ayudar a escribir y depurar el código, y para redactar y traducir textos. Todos los resultados
salen de los datos reales de NASA FIRMS, y el equipo revisó los métodos y las decisiones.

</div>"""
    else:
        texto = f"""
<div class='metodo'>

This page is a **harmonized burning activity calendar** for Argentina. It answers the question in CONAE's challenge:
*does this area, on these dates, have more or less fire than usual?*, using more than two decades of data from two different
satellite sensors. Below is the whole journey, from raw data to the page.

### 1. The data
We downloaded every hotspot in Argentina from **NASA FIRMS** (Fire Information for Resource Management System):

| Sensor | Satellites | Pixel | Period | Hotspots downloaded |
|---|---|---|---|---|
| MODIS Collection 6.1 | Terra (morning) and Aqua (afternoon) | ~1 km | 2003 – Oct 2026 | 1.6 million |
| VIIRS 375 m Collection 2 | Suomi NPP (afternoon) | 375 m | 2012 – Oct 2026 | 2.9 million |

Each record is a **hotspot**: a pixel with an anomalous temperature, with date, time, location, satellite, confidence, fire
radiative power (FRP) and source type. Archive data are completed with near-real-time (NRT) data for the latest months, with
no overlap. We also used **CONAE's Hotspot Catalog** as a reference.

**Decisions:**
- **MODIS from 2003**, not 2000: only from 2003 do both Terra and Aqua run all year. Earlier, the series would have fewer hotspots just because one satellite was missing.
- **VIIRS from Suomi NPP only**: it is the only VIIRS since 2012. Adding NOAA-20 (2018) and NOAA-21 (2023) would raise hotspot counts just because of more satellites.

### 2. Cleaning
The same criteria for both sensors:
- **Vegetation fires only** (`type = 0`): gas flares from oil wells, industries, volcanoes and offshore sources are removed. For near-real-time data, which lack that column, hotspots located where the archive already recorded static sources are removed.
- No **low-confidence** detections: MODIS confidence ≥ 30 (0 to 100); VIIRS without those flagged as low.

**4.1 million hotspots** remain.

### 3. Three ways to measure
We grouped hotspots by **area** (country, five regions and rest of the country), **year** and **month**, with three measures:
- **Hotspots**: the number of detections.
- **Fire-days**: (25 km cell, day) pairs with at least one hotspot. A fire counts the same even if the satellite splits it into many pixels.
- **Energy (FRP)**: the sum of fire radiative power, in megawatts.

The **regions** are approximate rectangles chosen for their number of hotspots (Gran Chaco) or the impact of their fires
(Paraná Delta, Córdoba Hills, Corrientes & Iberá, Andean Patagonia).

### 4. The problem, measured
In the years with both sensors (2012–2025), VIIRS records on average **{FACTOR_PAIS:.1f} times** more hotspots than MODIS.
That ratio is **stable from year to year** (between 3.0 and 3.8), which is what makes converting one sensor into the other
possible. But it **changes by region**: from {fac.min():.1f} in {zona_txt(fac.idxmin())} to {fac.max():.1f} in
{zona_txt(fac.idxmax())}. Where fires are large, both sensors see them. Where there are many small fires, VIIRS sees many
that MODIS misses.

### 5. Harmonization methods (machine learning)
We framed harmonization as a **supervised learning** problem: from what MODIS measured in a region and month, predict what
VIIRS would have measured.

- **Training:** {MET['anios_entrenamiento']}, years with both sensors.
- **Test:** {MET['anios_test']}, years the models **never see** while learning, to check that they generalize.
- **Application:** 2003–2011, MODIS only. It cannot be tested there because VIIRS did not exist: the validated model is just applied.

We compared four methods, from simplest to most complex:
1. **Fixed factor per region**: VIIRS = k × MODIS, one k per region.
2. **Log regression by region and month**: log(1+VIIRS) = a + b·log(1+MODIS) + seasonality, correcting the bias that appears when transforming back from the log scale.
3. **Quantile mapping**: percentiles of both sensors are matched.
4. **Random Forest**: 300 decision trees using MODIS, month and region.

### 6. Choosing: by results, not by taste
**Rule set before seeing the results:** the **simplest** method whose error is within {MET['tolerancia']:.0%} of the best
wins. Test results for hotspots:

| Method | MAE (hotspots per region and month) | R² (log) | Total bias % | Worst year % |
|---|---|---|---|---|
{tabla_metodos}

The **{metodo_txt(elegido_f).lower()}** won, and it also had the lowest error. The Random Forest did not improve accuracy:
the relationship between sensors is so stable that a complex model has nothing more to learn, and with few rows it tends to
memorize. We repeated the comparison for fire-days (winner: {metodo_txt(MET['elegido']['dias']).lower()}) and energy (winner:
{metodo_txt(MET['elegido']['frp']).lower()}).

### 7. Harmonized series and validation
- **2003–2011:** equivalent hotspots = region's factor × MODIS hotspots.
- **From 2012:** actual VIIRS data.

To validate, we built the series with the three measures separately. All three flag the same exceptional years
({lista([str(a) for a in ia.iloc[:, 0].nlargest(3).index])}) and closely track each other (minimum correlation {corr_min:.2f}).

### 8. How the page works
- **Usual** is each month's historical **median**, which is not dragged by extreme years; the usual range goes from the 10th to the 90th percentile.
- **Custom area:** using individual hotspots, the page builds the calendar for any rectangle. If the area has enough hotspots (at least 300 MODIS hotspots in 2012–2025) it computes its own factor; otherwise it uses its region's.
- **Date search:** compares any period with the same period in every year and lets you download the hotspots.
- **Maps** with background layers (topography with contour lines, seafloor relief, satellite) and main cities.
- **Accessibility:** the 👁️ button turns on colorblind-safe colors, non-color markers and text descriptions.

### 9. Limitations
- A hotspot is not a fire: a large fire produces many hotspots, and clouds or smoke can hide fires.
- Values before 2012 are estimates, with an error of about ±{res_f.loc[elegido_f, 'Peor año (país) %']:.0f}% in annual totals.
- The Terra satellite is aging: its orbit is drifting and it detects less and less, which adds error in recent years.
- The latest months are provisional near-real-time data.

### 10. Tools, sources and use of AI
Python (pandas, NumPy, scikit-learn), Google Colab, Streamlit, Plotly, Folium/Leaflet and GitHub. Data: NASA FIRMS; reference:
CONAE. Maps: © OpenStreetMap, OpenTopoMap, CARTO and Esri. **Use of Super Intelligence (SI):** we used an AI assistant
(Claude, by Anthropic) to help write and debug the code, and to draft and translate texts. All results come from the real NASA
FIRMS data, and the team reviewed the methods and decisions.

</div>"""
    st.markdown(texto, unsafe_allow_html=True)
