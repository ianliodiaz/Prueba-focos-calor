"""
FIREN'T — Calendario armonizado de actividad de fuego (MODIS + VIIRS) para Argentina
NASA Space Apps Challenge 2026 · Desafío CONAE "Harmonization of MODIS and VIIRS Hot Spots"

Para correrla en la computadora:  streamlit run app.py
"""
import datetime as dt
import json
import urllib.parse
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
import streamlit.components.v1 as components

VERSION_APP = "v8.1"

# ---- Video de YouTube: peguen acá el link cuando lo suban (por ejemplo "https://youtu.be/XXXXXXXX"). Vacío = no se muestra.
VIDEO_URL = ""
# ---- Comentarios: formulario de Google (link prellenado) y planilla publicada con los comentarios aprobados
FORM_URL = ("https://docs.google.com/forms/d/e/1FAIpQLSeXcpDScTTaLzCYbw7vCMWxWpWZGVu9SXVJoxgt4RP_XLK7tw/viewform"
            "?usp=pp_url&entry.1762721848=")
COMENTARIOS_CSV = ("https://docs.google.com/spreadsheets/d/e/2PACX-1vR0NBQVGJJwolEzoN7LYbJ2SU4B2kBryosau9gmWg8CN_ldV28rt"
                   "FO7HOrLDcmHjsWvq_fvPsqxVcog/pub?gid=2138309698&single=true&output=csv")
RAIZ = Path(__file__).parent
_encontrados = sorted(RAIZ.rglob("serie_armonizada.csv"), key=lambda p: len(p.parts))
CARPETA = _encontrados[0].parent if _encontrados else RAIZ / "datos"
ARCHIVOS_FOCOS = sorted(RAIZ.rglob("focos_*.parquet"))

PAIS, AREA, RESTO = "Argentina (todo el país)", "Área propia", "Resto del país"
INICIO_VIIRS = pd.Timestamp("2012-01-20")

st.set_page_config(page_title="FIREN'T · Focos de calor en Argentina", page_icon="🔥", layout="wide")

# ================================================================ idioma
IDIOMAS = ["ES", "EN", "PT", "GN"]
NOMBRE_IDIOMA = {"ES": "Español", "EN": "English", "PT": "Português", "GN": "Guaraní"}
VOZ = {"ES": "es-AR", "EN": "en-US", "PT": "pt-BR", "GN": "es-AR"}   # el guaraní no tiene voz: se lee en español

# ---- Link compartido: los parámetros de la dirección se aplican una sola vez, al abrir la página
if "_params_ok" not in st.session_state:
    st.session_state._params_ok = True
    qp = st.query_params
    try:
        if qp.get("lang") in IDIOMAS:
            st.session_state.idioma = qp["lang"]
        if qp.get("zona") == "area" and qp.get("area"):
            la0, la1, lo0, lo1 = [float(x) for x in qp["area"].split(",")]
            st.session_state.update({"tipo_zona_val": "area", "lat_s": la0, "lat_n": la1, "lon_o": lo0, "lon_e": lo1})
        elif qp.get("zona"):
            st.session_state.update({"tipo_zona_val": "region", "region_val": qp["zona"]})
        if qp.get("medida") in ("focos", "dias", "frp"):
            st.session_state.medida_val = qp["medida"]
        if qp.get("anio"):
            st.session_state.anio_val = int(qp["anio"])
        if qp.get("ref"):
            r0, r1 = [int(x) for x in qp["ref"].split("-")]
            st.session_state.ref_val = (r0, r1)
        st.session_state.acc = qp.get("acc") == "1"
        st.session_state.grande = qp.get("grande") == "1"
        if any(k in qp for k in ("lang", "zona", "anio")):
            st.session_state.bienvenida_hecha = True      # quien abre un link compartido no ve la bienvenida
    except (ValueError, KeyError):
        pass
if "idioma" not in st.session_state:
    st.session_state.idioma = "ES"
IDIOMA = st.session_state.idioma
EN, PT, GN = IDIOMA == "EN", IDIOMA == "PT", IDIOMA == "GN"

# Guaraní: traducción básica (prototipo). Solo palabras y frases cortas verificadas; el resto queda en español.
GN_TXT = {
    "Idioma": "Ñe'ẽ (Idioma)",
    "Calendario": "Arapapaha (Calendario)",
    "Mapa": "Mapa",
    "Año a analizar": "Ary (Año a analizar)",
    "Sí": "Heẽ (Sí)",
    "No": "Nahániri (No)",
    "Gracias": "Aguyje (Gracias)",
}


def tr(es, en, pt=None, gn=None):
    "Devuelve el texto en el idioma elegido (portugués y guaraní vuelven al español si falta la traducción)."
    if EN:
        return en
    if PT:
        return pt if pt is not None else es
    if GN:
        return gn if gn is not None else GN_TXT.get(es, es)
    return es


MESES = tr("Ene Feb Mar Abr May Jun Jul Ago Sep Oct Nov Dic", "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec", pt="Jan Fev Mar Abr Mai Jun Jul Ago Set Out Nov Dez").split()
MESES_LARGOS = tr("enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre",
                  "January February March April May June July August September October November December", pt="janeiro fevereiro março abril maio junho julho agosto setembro outubro novembro dezembro").split()
NOMBRE_ZONA_EN = {PAIS: "Argentina (whole country)", "Gran Chaco": "Gran Chaco", "Delta del Paraná": "Paraná Delta",
                  "Sierras de Córdoba": "Córdoba Hills", "Corrientes e Iberá": "Corrientes & Iberá",
                  "Patagonia andina": "Andean Patagonia", RESTO: "Rest of the country", AREA: "Custom area"}
NOMBRE_MEDIDA = {"focos": tr("Focos", "Hotspots", pt="Focos"), "dias": tr("Días con fuego", "Fire-days", pt="Dias com fogo"),
                 "frp": tr("Energía (FRP, MW)", "Energy (FRP, MW)", pt="Energia (FRP, MW)")}
UNIDAD = {"focos": tr("focos", "hotspots", pt="focos"), "dias": tr("días-celda", "cell-days", pt="dias-célula"), "frp": "MW"}
EXPLICA_MEDIDA = {
    "focos": tr("Cantidad de detecciones satelitales (píxeles con fuego).",
                "Number of satellite detections (pixels with fire).", pt="Quantidade de detecções por satélite (pixels com fogo)."),
    "dias": tr("Pares (celda de 25 km, día) con al menos un foco: un incendio cuenta igual aunque el satélite lo "
               "divida en muchos píxeles.",
               "Pairs (25 km cell, day) with at least one hotspot: a fire counts the same even if the satellite "
               "splits it into many pixels.", pt="Pares (célula de 25 km, dia) com pelo menos um foco: um incêndio conta igual mesmo que o satélite o divida em muitos pixels."),
    "frp": tr("Suma de la potencia radiativa del fuego, en megavatios: mide energía, no cantidad de píxeles.",
              "Sum of fire radiative power, in megawatts: it measures energy, not number of pixels.", pt="Soma da potência radiativa do fogo, em megawatts: mede energia, não quantidade de pixels."),
}
NOMBRE_METODO_EN = {"Factor fijo por región": "Fixed factor per region",
                    "Regresión log por región y mes": "Log regression by region and month",
                    "Mapeo de cuantiles": "Quantile mapping", "Random Forest": "Random Forest"}


NOMBRE_ZONA_PT = {PAIS: "Argentina (todo o país)", "Gran Chaco": "Gran Chaco", "Delta del Paraná": "Delta do Paraná",
                  "Sierras de Córdoba": "Serras de Córdoba", "Corrientes e Iberá": "Corrientes e Iberá",
                  "Patagonia andina": "Patagônia andina", RESTO: "Resto do país", AREA: "Área personalizada"}
NOMBRE_METODO_PT = {"Factor fijo por región": "Fator fixo por região",
                    "Regresión log por región y mes": "Regressão log por região e mês",
                    "Mapeo de cuantiles": "Mapeamento de quantis", "Random Forest": "Random Forest"}


def zona_txt(z):
    if EN:
        return NOMBRE_ZONA_EN.get(z, z)
    return NOMBRE_ZONA_PT.get(z, z) if PT else z


def metodo_txt(m):
    if EN:
        return NOMBRE_METODO_EN.get(m, m)
    return NOMBRE_METODO_PT.get(m, m) if PT else m


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
    return ", ".join(items[:-1]) + tr(" y ", " and ", pt=" e ") + items[-1]


# ================================================================ estilo
TINTA, PAPEL, BRASA, AMBAR, AGUA, CENIZA = "#16232E", "#EEF2F4", "#E4572E", "#F2A541", "#2C7DA0", "#6B7B86"
GRANDE = st.session_state.get("grande", False)          # letra grande y alto contraste
if GRANDE:
    TINTA, PAPEL, CENIZA, BRASA = "#000000", "#FFFFFF", "#26343D", "#C2361A"
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
[data-testid="stSidebar"] [data-testid="stPopover"] button, [data-testid="stSidebar"] [data-testid="stLinkButton"] a {{
  background: #24394A !important; border: 1px solid #5B7385 !important; }}
[data-testid="stSidebar"] [data-testid="stPopover"] button:hover, [data-testid="stSidebar"] [data-testid="stLinkButton"] a:hover {{
  border-color: {BRASA} !important; }}
[data-testid="stSidebar"] [data-testid="stAlert"] * {{ color: {TINTA} !important; }}
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
div[data-testid="stDialog"] button {{ min-height: 3.6rem; font-size: 1.2rem !important; font-weight: 700; }}
div[data-testid="stDialog"] button p {{ font-size: 1.2rem !important; }}
</style>""", unsafe_allow_html=True)
if GRANDE:
    st.markdown(f"""<style>
html {{ font-size: 21px; }}
[data-testid="stMain"] .stMarkdown p, [data-testid="stMain"] .stMarkdown li, [data-testid="stMain"] label,
[data-testid="stMain"] .stRadio label p, [data-testid="stMain"] button p, [data-testid="stMain"] [data-testid="stCaptionContainer"] {{
  font-size: 1.12rem !important; color: {TINTA} !important; }}
[data-testid="stSidebar"] label p, [data-testid="stSidebar"] .stRadio label p, [data-testid="stSidebar"] .stMarkdown p {{
  font-size: 1.1rem !important; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ font-size: 1rem !important; }}
[data-testid="stSidebar"] {{ background: #000000; }}
[data-testid="stSidebar"] * {{ color: #FFFFFF !important; }}
[data-testid="stSidebar"] [data-baseweb="select"] *, [data-testid="stSidebar"] input {{ color: #000000 !important; }}
.estado {{ font-size: 1.55rem; }}
.nota {{ color: {CENIZA}; font-size: 1.08rem; }}
[data-testid="stMetricValue"] {{ font-size: 2.3rem; }}
[data-testid="stMetricLabel"] p {{ color: {TINTA} !important; font-size: 1.05rem; }}
.stTabs [data-baseweb="tab"] p {{ font-size: 1.15rem !important; color: {TINTA}; }}
:focus-visible {{ outline: 4px solid #0050FF !important; outline-offset: 3px; }}
a {{ text-decoration: underline; }}
</style>""", unsafe_allow_html=True)

pio.templates["firent"] = go.layout.Template(layout=dict(
    font=dict(family="Archivo, system-ui, sans-serif", color=TINTA, size=17 if GRANDE else 13),
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


# ================================================================ bienvenida: idioma, letra grande y colores
def _cerrar_bienvenida():
    st.session_state.bienvenida_hecha = True
    st.session_state.paso_bv = 0


@st.dialog("🔥 FIREN'T", width="large", on_dismiss=_cerrar_bienvenida)
def bienvenida():
    paso = st.session_state.get("paso_bv", 0)
    if paso == 0:
        st.markdown("### Elegí tu idioma · Choose your language · Escolha seu idioma · Eiporavo ne ñe'ẽ")
        b = st.columns(2)
        for i, l_ in enumerate(IDIOMAS):
            etiqueta = NOMBRE_IDIOMA[l_] + (" (prototipo)" if l_ == "GN" else "")
            if b[i % 2].button(etiqueta, key=f"bv_{l_}", width="stretch"):
                st.session_state.idioma, st.session_state.paso_bv = l_, 1
                st.rerun()
    elif paso == 1:
        st.markdown("### " + tr("¿Preferís ver la página con letra más grande y más contraste?",
                                "Would you like the page with larger text and higher contrast?", pt="Prefere ver a página com letra maior e mais contraste?",
                                gn="Mba'éichapa! ¿Preferís ver la página con letra más grande y más contraste?"))
        st.markdown(tr("Podés cambiarlo cuando quieras con el botón 👁️, arriba a la derecha.",
                       "You can change it anytime with the 👁️ button, top right.", pt="Você pode mudar isso quando quiser com o botão 👁️, no canto superior direito."))
        if st.button("🔍 " + tr("Sí, letra grande", "Yes, larger text", pt="Sim, letra grande", gn="Heẽ (Sí), letra grande"),
                     key="bv_grande_si", type="primary", width="stretch"):
            st.session_state.grande, st.session_state.paso_bv = True, 2
            st.rerun()
        if st.button(tr("No, tamaño estándar", "No, standard size", pt="Não, tamanho padrão", gn="Nahániri (No), tamaño estándar"),
                     key="bv_grande_no", width="stretch"):
            st.session_state.grande, st.session_state.paso_bv = False, 2
            st.rerun()
    else:
        st.markdown("### " + tr("¿Te cuesta distinguir algunos colores, como el rojo y el verde?",
                                "Do you find it hard to tell some colors apart, such as red and green?", pt="Você tem dificuldade para distinguir algumas cores, como vermelho e verde?"))
        st.markdown(tr("Si elegís «Sí», los gráficos usan colores adaptados, marcas que no dependen del color y "
                       "descripciones en texto.",
                       "If you choose «Yes», charts use adapted colors, markers that do not depend on color and "
                       "text descriptions.", pt="Se escolher «Sim», os gráficos usam cores adaptadas, marcas que não dependem da cor e descrições em texto."))
        if st.button("🎨 " + tr("Sí, usar colores adaptados", "Yes, use adapted colors", pt="Sim, usar cores adaptadas", gn="Heẽ (Sí), colores adaptados"),
                     key="bv_acc_si", type="primary", width="stretch"):
            st.session_state.acc = True
            _cerrar_bienvenida()
            st.rerun()
        if st.button(tr("No, colores estándar", "No, standard colors", pt="Não, cores padrão", gn="Nahániri (No), colores estándar"),
                     key="bv_acc_no", width="stretch"):
            st.session_state.acc = False
            _cerrar_bienvenida()
            st.rerun()


if not st.session_state.get("bienvenida_hecha"):
    bienvenida()


def boton_voz(texto, etiqueta):
    "Botón que lee un texto en voz alta con la voz del navegador (sin servicios externos)."
    html = f"""
<button id="b" aria-label="{etiqueta}" style="font:600 {16 if GRANDE else 14}px Archivo,system-ui,sans-serif;
 padding:.45rem .9rem;border:1.5px solid {TINTA};border-radius:6px;background:#fff;color:{TINTA};cursor:pointer">🔊 {etiqueta}</button>
<script>
const texto = {json.dumps(texto)}, lang = "{VOZ[IDIOMA]}";
document.getElementById("b").onclick = () => {{
  const s = window.speechSynthesis;
  if (!s) {{ document.getElementById("b").textContent = "🔇"; return; }}
  if (s.speaking) {{ s.cancel(); return; }}
  const u = new SpeechSynthesisUtterance(texto); u.lang = lang; u.rate = 0.95;
  const v = s.getVoices().find(v => (v.lang || "").toLowerCase().startsWith(lang.slice(0, 2)));
  if (v) u.voice = v;
  s.speak(u);
}};
</script>"""
    alto = 52 if GRANDE else 46
    if hasattr(st, "iframe"):
        st.iframe(html, height=alto)
    else:
        components.html(html, height=alto)


# ================================================================ datos
@st.cache_data
def cargar():
    serie = pd.read_csv(CARPETA / "serie_armonizada.csv")
    grilla = pd.read_csv(CARPETA / "grilla_mensual.csv")
    regiones = json.loads((CARPETA / "regiones.json").read_text(encoding="utf-8"))
    metricas = json.loads((CARPETA / "metricas.json").read_text(encoding="utf-8"))
    huecos = (pd.read_csv(CARPETA / "dias_sin_dato.csv", parse_dates=["fecha"]) if (CARPETA / "dias_sin_dato.csv").exists()
              else pd.DataFrame(columns=["fecha", "satelite", "focos_observados", "focos_esperados", "relleno_con"]))
    return serie, grilla, regiones, metricas, huecos


@st.cache_resource(show_spinner=False)
def cargar_focos():
    """Focos individuales. Se guardan una sola vez en memoria y se comparten (cache_resource no hace copias):
    con 4 millones de filas, copiarlos en cada interacción superaba la memoria de Streamlit Cloud. No modificar in situ."""
    if not ARCHIVOS_FOCOS:
        return None
    f = pd.concat([pd.read_parquet(a) for a in ARCHIVOS_FOCOS], ignore_index=True)
    f["fecha"] = pd.to_datetime(f["fecha"])
    for c in ["sensor", "satelite", "confianza", "region"]:
        if c in f and f[c].dtype != "category":
            f[c] = f[c].astype("category")
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

serie, grilla, REGIONES, MET, HUECOS = cargar()
HUECOS_VIIRS = HUECOS[HUECOS.satelite == "VIIRS S-NPP"]
DIAS_HUECO_VIIRS = set(pd.to_datetime(HUECOS_VIIRS.fecha))
FECHAS_SENSOR = MET.get("fechas_sensor", {})
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
# Rótulos geográficos sin punto de ciudad
ROTULOS = [("Islas Malvinas", -51.75, -59.6)]


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
    for l_ in IDIOMAS:
        st.session_state.pop(f"tipo_zona_{l_}", None)
for k_, v_ in {"lat_n": -30.5, "lat_s": -32.5, "lon_o": -65.2, "lon_e": -64.0}.items():
    st.session_state.setdefault(k_, v_)

# ================================================================ barra lateral
# Los controles cuyo texto cambia con el idioma guardan su valor aparte, para no perderlo al cambiar de idioma
OTROS = [l_ for l_ in IDIOMAS if l_ != IDIOMA]


def control(func, nombre, opciones, **kw):
    for l_ in OTROS:
        st.session_state.pop(f"{nombre}_{l_}", None)
    previo = st.session_state.get(f"{nombre}_val", opciones[0])
    v = func(options=opciones, index=opciones.index(previo) if previo in opciones else 0,
             key=f"{nombre}_{st.session_state.idioma}", **kw)
    st.session_state[f"{nombre}_val"] = v
    return v

st.sidebar.markdown("<div class='marca'>FIREN<span>'</span>T</div>", unsafe_allow_html=True)
st.sidebar.caption(tr("Calendario armonizado de focos de calor · Argentina", "Harmonized hotspot calendar · Argentina", pt="Calendário harmonizado de focos de calor · Argentina")
                   + f" · {VERSION_APP}")
st.sidebar.selectbox("🌐 " + tr("Idioma", "Language", "Idioma") + " / Language", IDIOMAS, key="idioma",
                     format_func=lambda l_: NOMBRE_IDIOMA[l_] + (" (prototipo)" if l_ == "GN" else ""))
if GN:
    st.sidebar.info("**Guaraní (prototipo).** Esta traducción es básica y está en revisión: solo algunas palabras están en "
                    "guaraní y el resto en español. Si hablás guaraní, tu ayuda es bienvenida: dejanos tus correcciones y "
                    "sugerencias en la pestaña **💬 Comentarios**. ¡Aguyje (gracias)!")
if PT:
    st.sidebar.caption("Tradução feita com apoio de inteligência artificial. Sugestões são bem-vindas na aba 💬 Comentários.")

opciones_zona = ["region", "area"] if focos is not None else ["region"]
tipo_zona = control(st.sidebar.radio, "tipo_zona", opciones_zona, label=tr("Zona", "Area", pt="Área"), horizontal=True,
                    format_func=lambda x: tr("Región", "Region", pt="Região") if x == "region" else tr("Área propia", "Custom area", pt="Área personalizada"))
bbox = None
if tipo_zona == "region":
    region = control(st.sidebar.selectbox, "region", [PAIS] + list(REGIONES) + [RESTO], label=tr("Región", "Region", pt="Região"),
                     format_func=zona_txt)
else:
    region = AREA
    st.sidebar.caption(tr("Dibujá un rectángulo en el **Mapa** o escribí los límites en grados decimales "
                          "(sur y oeste son negativos).",
                          "Draw a rectangle on the **Map** or type the limits in decimal degrees "
                          "(south and west are negative).", pt="Desenhe um retângulo no **Mapa** ou digite os limites em graus decimais (sul e oeste são negativos)."))
    s1, s2 = st.sidebar.columns(2)
    s1.number_input(tr("Latitud norte", "North latitude", pt="Latitude norte"), -56.0, -21.0, step=0.1, format="%.2f", key="lat_n")
    s2.number_input(tr("Latitud sur", "South latitude", pt="Latitude sul"), -56.0, -21.0, step=0.1, format="%.2f", key="lat_s")
    s1.number_input(tr("Longitud oeste", "West longitude", pt="Longitude oeste"), -74.0, -53.0, step=0.1, format="%.2f", key="lon_o")
    s2.number_input(tr("Longitud este", "East longitude", pt="Longitude leste"), -74.0, -53.0, step=0.1, format="%.2f", key="lon_e")
    ss = st.session_state
    bbox = (min(ss.lat_n, ss.lat_s), max(ss.lat_n, ss.lat_s), min(ss.lon_o, ss.lon_e), max(ss.lon_o, ss.lon_e))
    if bbox[0] == bbox[1] or bbox[2] == bbox[3]:
        st.sidebar.error(tr("El área no puede tener ancho o alto cero.", "The area cannot have zero width or height.", pt="A área não pode ter largura ou altura zero."))
        st.stop()

if region == AREA:
    medida = "focos"
    st.sidebar.caption(tr("En un área propia la actividad se mide en **focos**.",
                          "In a custom area, activity is measured in **hotspots**.", pt="Numa área personalizada a atividade é medida em **focos**."))
else:
    medida = control(st.sidebar.radio, "medida", ["focos", "dias", "frp"],
                     label=tr("Medir la actividad de fuego por", "Measure fire activity by", pt="Medir a atividade de fogo por"), format_func=NOMBRE_MEDIDA.get)
    st.sidebar.caption(EXPLICA_MEDIDA[medida])
anio = control(st.sidebar.selectbox, "anio", anios[::-1], label=tr("Año a analizar", "Year to analyze", pt="Ano a analisar"))
previos = [a for a in anios if a < anio] or anios
for l_ in OTROS:
    st.session_state.pop(f"ref_{l_}", None)
ref_prev = st.session_state.get("ref_val", (previos[0], previos[-1]))
if not (ref_prev[0] in previos and ref_prev[1] in previos):
    ref_prev = (previos[0], previos[-1])
ref_desde, ref_hasta = st.sidebar.select_slider(tr("Período de referencia (lo habitual)", "Reference period (the usual)", pt="Período de referência (o habitual)"),
                                                options=previos, value=ref_prev, key=f"ref_{st.session_state.idioma}")
st.session_state.ref_val = (ref_desde, ref_hasta)
# ---- Link para compartir: la dirección de la página guarda la vista actual
params = {"lang": IDIOMA, "anio": str(anio), "ref": f"{ref_desde}-{ref_hasta}"}
if region == AREA:
    params.update(zona="area", area=",".join(f"{x:.2f}" for x in bbox))
else:
    params.update(zona=region, medida=medida)
if st.session_state.get("acc"):
    params["acc"] = "1"
if GRANDE:
    params["grande"] = "1"
st.query_params.from_dict(params)
try:
    BASE_URL = (st.context.url or "").split("?")[0]
except Exception:
    BASE_URL = ""
URL_COMPARTIR = (BASE_URL or "https://<dirección-de-la-página>") + "?" + urllib.parse.urlencode(params)

st.sidebar.divider()
with st.sidebar.popover("🔗 " + tr("Compartir esta vista", "Share this view", pt="Compartilhar esta visualização"), width="stretch"):
    st.markdown(tr("Copiá este link (botón de la derecha). Quien lo abra ve la misma zona, año, medida, idioma y "
                   "configuración de accesibilidad.",
                   "Copy this link (button on the right). Whoever opens it sees the same area, year, measure, language "
                   "and accessibility settings.", pt="Copie este link (botão à direita). Quem abrir verá a mesma área, ano, medida, idioma e configuração de acessibilidade."))
    st.code(URL_COMPARTIR, language=None, wrap_lines=True)
contexto_form = (f"zona: {region if region != AREA else 'area ' + params['area']}; año: {anio}; medida: {medida}; "
                 f"idioma: {IDIOMA}; link: {URL_COMPARTIR}")
st.sidebar.link_button("💬 " + tr("Comentar o reportar un dato raro", "Comment or report odd data", pt="Comentar ou relatar um dado estranho"),
                       FORM_URL + urllib.parse.quote(contexto_form), width="stretch")
st.sidebar.divider()
fs = FECHAS_SENSOR
st.sidebar.caption(
    tr(f"Datos de NASA FIRMS hasta el {FECHA_FIN:%d/%m/%Y} (los últimos meses son provisorios). "
       "Antes de 2012: MODIS armonizado (estimado). Desde 2012: VIIRS S-NPP; los días en que no midió se completan con "
       "VIIRS NOAA-20.",
       f"NASA FIRMS data up to {FECHA_FIN:%Y-%m-%d} (latest months are provisional). "
       "Before 2012: harmonized MODIS (estimated). From 2012: VIIRS S-NPP; days when it did not measure are filled with "
       "VIIRS NOAA-20.", pt=f"Dados do NASA FIRMS até {FECHA_FIN:%d/%m/%Y} (os últimos meses são provisórios). Antes de 2012: MODIS harmonizado (estimado). Desde 2012: VIIRS S-NPP; os dias em que ele não mediu são completados com VIIRS NOAA-20.")
    + (" " + tr("Datos de cada satélite", "Data per satellite", pt="Dados de cada satélite") + ": "
       + " · ".join(f"{n} {v['desde'][:4]}–{v['hasta']}" for n, v in
                    zip(["MODIS", "S-NPP", "NOAA-20"], [fs.get("modis"), fs.get("viirs"), fs.get("n20")]) if v)
       if fs else ""))

# ================================================================ zona elegida
if region == AREA:
    dr, g_zona, K_ZONA, ORIGEN_K, n_focos_area = serie_area(bbox)
    origen_k_txt = (tr("propio del área (calculado con sus focos de 2012–2025)",
                       "specific to the area (computed from its 2012–2025 hotspots)", pt="próprio da área (calculado com seus focos de 2012–2025)") if ORIGEN_K == "propio" else
                    tr(f"de la región «{zona_txt(region_que_contiene(np.mean(bbox[:2]), np.mean(bbox[2:])))}» "
                       "(el área tiene pocos focos para calcular uno propio)",
                       f"of the «{zona_txt(region_que_contiene(np.mean(bbox[:2]), np.mean(bbox[2:])))}» region "
                       "(the area has too few hotspots for its own)", pt=f"da região «{zona_txt(region_que_contiene(np.mean(bbox[:2]), np.mean(bbox[2:])))}» (a área tem poucos focos para calcular um próprio)"))
else:
    dr = serie[serie.region == region].copy()
    g_zona = grilla if region == PAIS else grilla[grilla.region == region]
    K_ZONA = FACTOR_PAIS if region == PAIS else FACTORES.get(region, FACTOR_PAIS)
    origen_k_txt = (tr("promedio de Argentina", "Argentina's average", pt="média da Argentina") if region == PAIS else
                    tr(f"de la región «{zona_txt(region)}»", f"of the «{zona_txt(region)}» region", pt=f"da região «{zona_txt(region)}»"))
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
    with st.popover("👁️", help=tr("Accesibilidad: letra grande, alto contraste, colores para daltonismo y descripciones",
                                  "Accessibility: larger text, high contrast, colorblind-safe colors and descriptions", pt="Acessibilidade: letra grande, alto contraste, cores para daltonismo e descrições")):
        st.toggle(tr("Letra grande y alto contraste", "Larger text and high contrast", pt="Letra grande e alto contraste"), key="grande")
        st.toggle(tr("Colores para daltonismo y descripciones", "Colorblind-safe colors and descriptions", pt="Cores para daltonismo e descrições"), key="acc")
        st.caption(tr("Colores para daltonismo: cambia los colores por paletas aptas para daltonismo, agrega marcas y números "
                      "que no dependen del color, y muestra una descripción en texto debajo de cada gráfico.",
                      "Colorblind-safe colors: switches to colorblind-safe palettes, adds markers and numbers that do not "
                      "rely on color, and shows a text description under each chart.", pt="Cores para daltonismo: troca as cores por paletas adequadas para daltonismo, acrescenta marcas e números que não dependem da cor e mostra uma descrição em texto abaixo de cada gráfico."))
        if st.button(tr("Volver a ver la bienvenida", "Show the welcome again", pt="Ver a boas-vindas de novo"), width="stretch"):
            st.session_state.bienvenida_hecha, st.session_state.paso_bv = False, 0
            st.rerun()
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
        st.info("📝 **" + tr("Descripción", "Description", pt="Descrição") + ":** " + texto)


def colores(escala, valores):
    "Colores de una escala de Plotly para valores entre 0 y 1."
    from plotly.colors import sample_colorscale
    v = np.clip(np.asarray(valores, dtype=float), 0, 1)
    return sample_colorscale(escala, list(v), colortype="rgb") if len(v) else []


# ================================================================ cabecera: nombre, estado y franja de fuego
total, esperado = actual.sum(), clima.mediana.reindex(con_dato).sum()
dif = 100 * (total - esperado) / esperado if esperado else np.nan
meses_altos = [MESES_LARGOS[m - 1] for m in con_dato if actual[m] > clima.p90[m]]
pico = int(actual.idxmax()) if actual.notna().any() and actual.max() > 0 else None
pico_normal = int(clima.mediana.idxmax()) if clima.mediana.notna().any() and clima.mediana.max() > 0 else None

with cab1:
    if region == AREA:
        st.title(tr("Área propia", "Custom area", pt="Área personalizada"))
        st.markdown(f"<p class='zona-sub'>{bbox[0]:.2f}° {tr('a', 'to', pt='a')} {bbox[1]:.2f}° lat · "
                    f"{bbox[2]:.2f}° {tr('a', 'to', pt='a')} {bbox[3]:.2f}° lon</p>", unsafe_allow_html=True)
    else:
        st.title(zona_txt(region))
        st.markdown(f"<p class='zona-sub'>{NOMBRE_MEDIDA[medida]} · {tr('mediana de', 'median of', pt='mediana de')} "
                    f"{ref_desde}–{ref_hasta}</p>", unsafe_allow_html=True)

if region == AREA and n_focos_area == 0:
    st.warning(tr("No hay focos registrados en esta área. Probá con un área más grande o en otra zona.",
                  "There are no hotspots recorded in this area. Try a larger area or a different place.", pt="Não há focos registrados nesta área. Tente uma área maior ou outro lugar."))
    st.stop()

if np.isfinite(dif) and len(con_dato):
    periodo = (tr("en todo el año", "for the whole year", pt="no ano todo") if len(con_dato) == 12 else
               tr(f"entre enero y {MESES_LARGOS[int(con_dato.max()) - 1]}", f"from January to {MESES_LARGOS[int(con_dato.max()) - 1]}", pt=f"entre janeiro e {MESES_LARGOS[int(con_dato.max()) - 1]}"))
    sentido = (tr("más", "more", pt="mais") if dif > 0 else tr("menos", "less", pt="menos"))
    frase_estado = (f"{tr('En', 'In', pt='Em')} {anio} {tr('hubo', 'there were', pt='houve')} <b>{miles(total)} "
                    f"{UNIDAD[medida]}</b> {periodo}: <b>{abs(dif):.0f} % {sentido}</b> "
                    f"{tr('que lo habitual', 'than usual', pt='do que o habitual')}"
                    + (tr(f", con {len(meses_altos)} meses por encima del rango normal.",
                          f", with {len(meses_altos)} months above the normal range.", pt=f", com {len(meses_altos)} meses acima da faixa normal.") if meses_altos else "."))
    st.markdown(f"<p class='estado'>{frase_estado}</p>", unsafe_allow_html=True)
else:
    frase_estado = ""

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
st.markdown(f"<p class='nota'>{tr('Cada raya es un mes desde 2003: más intenso el color, más actividad de fuego. Pasá el cursor para ver el valor.', 'Each stripe is one month since 2003: the stronger the color, the more fire activity. Hover to see the value.', pt='Cada faixa é um mês desde 2003: quanto mais intensa a cor, mais atividade de fogo. Passe o cursor para ver o valor.')}</p>",
            unsafe_allow_html=True)
anual_tot = dr[dr.anio <= 2025].groupby("anio")[col].sum()
if len(anual_tot):
    describir(tr(f"Franja de actividad mensual desde 2003. Los años con más actividad fueron "
                 f"{lista([str(a) for a in anual_tot.nlargest(3).index])}.",
                 f"Monthly activity stripe since 2003. The most active years were "
                 f"{lista([str(a) for a in anual_tot.nlargest(3).index])}.", pt=f"Faixa de atividade mensal desde 2003. Os anos com mais atividade foram {lista([str(a) for a in anual_tot.nlargest(3).index])}."))

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{NOMBRE_MEDIDA[medida]} {tr('en', 'in', pt='em')} {anio}", miles(total),
          f"{dif:+.0f} %" if np.isfinite(dif) else None, delta_color="inverse",
          help=tr("Diferencia respecto de lo habitual (mediana)", "Difference from usual (median)", pt="Diferença em relação ao habitual (mediana)"))
c2.metric(tr("Lo habitual para esos meses", "Usual for those months", pt="O habitual para esses meses"), miles(esperado),
          help=tr("Suma de las medianas históricas de esos meses", "Sum of the historical medians of those months", pt="Soma das medianas históricas desses meses"))
c3.metric(tr(f"Mes pico en {anio}", f"Peak month in {anio}", pt=f"Mês de pico em {anio}"), MESES[pico - 1] if pico else "—",
          f"{tr('habitualmente', 'usually', pt='normalmente')}: {MESES[pico_normal - 1]}" if pico_normal else None, delta_color="off")
c4.metric(tr("Meses por encima de lo habitual", "Months above usual", pt="Meses acima do habitual"), f"{len(meses_altos)} {tr('de', 'of', pt='de')} {len(con_dato)}",
          help=tr("Meses que superaron al 90 % de los años de referencia", "Months above 90% of the reference years", pt="Meses que superaram 90 % dos anos de referência"))
notas = []
if len(con_dato) < 12:
    notas.append(tr(f"{anio} tiene datos hasta {MESES_LARGOS[int(con_dato.max()) - 1]}: se compara solo con esos meses.",
                    f"{anio} has data up to {MESES_LARGOS[int(con_dato.max()) - 1]}: only those months are compared.", pt=f"{anio} tem dados até {MESES_LARGOS[int(con_dato.max()) - 1]}: compara-se só com esses meses."))
if region == AREA:
    notas.append(tr(f"Área con {miles(n_focos_area)} focos. Factor MODIS→VIIRS: {K_ZONA:.2f}, {origen_k_txt}.",
                    f"Area with {miles(n_focos_area)} hotspots. MODIS→VIIRS factor: {K_ZONA:.2f}, {origen_k_txt}.", pt=f"Área com {miles(n_focos_area)} focos. Fator MODIS→VIIRS: {K_ZONA:.2f}, {origen_k_txt}."))
if notas:
    st.markdown(f"<p class='nota'>{' '.join(notas)}</p>", unsafe_allow_html=True)

# Leer en voz alta el resumen (y botón del video, si ya está publicado)
nombre_zona = tr("Área propia", "Custom area", pt="Área personalizada") if region == AREA else zona_txt(region)
resumen_voz = (f"{nombre_zona}. " + frase_estado.replace("<b>", "").replace("</b>", "") + " "
               + tr(f"Lo habitual para esos meses es {miles(esperado)}.", f"The usual for those months is {miles(esperado)}.", pt=f"O habitual para esses meses é {miles(esperado)}.")
               + (" " + tr(f"El mes con más actividad fue {MESES_LARGOS[pico - 1]}.",
                           f"The most active month was {MESES_LARGOS[pico - 1]}.", pt=f"O mês com mais atividade foi {MESES_LARGOS[pico - 1]}.") if pico else ""))
v1_, v2_ = st.columns([1, 3]) if VIDEO_URL else (st.container(), None)
with v1_:
    boton_voz(resumen_voz, tr("Escuchar el resumen", "Listen to the summary", pt="Ouvir o resumo"))
if VIDEO_URL:
    with v2_:
        with st.popover("▶ " + tr("Ver el video del proyecto", "Watch the project video", pt="Ver o vídeo do projeto")):
            st.video(VIDEO_URL)

tabs = st.tabs([tr("Comparación con lo habitual", "Compared with usual", pt="Comparação com o habitual"), tr("Calendario", "Calendar", pt="Calendário"),
                tr("Búsqueda por fechas", "Date search", pt="Busca por datas"), tr("Mapa", "Map", pt="Mapa"), tr("Armonización", "Harmonization", pt="Harmonização"),
                tr("Validación", "Validation", pt="Validação"), tr("Datos", "Data", pt="Dados"), tr("Cómo lo hicimos", "How we built it", pt="Como fizemos"),
                "💬 " + tr("Comentarios", "Comments", pt="Comentários")])


# ================================================================ mapas (Leaflet)
CAPAS_BASE = [
    (tr("Claro", "Light", pt="Claro"), "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
     "Tiles © Esri — Esri, HERE, Garmin, © OpenStreetMap contributors"),
    (tr("Calles (OpenStreetMap)", "Streets (OpenStreetMap)", pt="Ruas (OpenStreetMap)"), "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
     "© OpenStreetMap contributors"),
    (tr("Topografía (curvas de nivel)", "Topography (contour lines)", pt="Topografia (curvas de nível)"), "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
     "Map data © OpenStreetMap contributors, SRTM | Map style © OpenTopoMap (CC-BY-SA)"),
    (tr("Relieve submarino (batimetría)", "Seafloor relief (bathymetry)", pt="Relevo submarino (batimetria)"),
     "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
     "Tiles © Esri — GEBCO, NOAA, National Geographic, Esri"),
    (tr("Satélite", "Satellite", pt="Satélite"),
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
    fg = folium.FeatureGroup(name=tr("Ciudades principales", "Main cities", pt="Cidades principais"), show=False)
    for nombre, la, lo in CIUDADES:
        folium.CircleMarker([la, lo], radius=3.5, color=TINTA, weight=1.5, fill=True, fill_color="#FFFFFF",
                            fill_opacity=1, tooltip=nombre).add_to(fg)
        folium.Marker([la, lo], icon=folium.DivIcon(
            html=f"<div style='font:600 11px Archivo,sans-serif;color:{TINTA};white-space:nowrap;"
                 f"text-shadow:0 0 3px #fff,0 0 3px #fff;transform:translate(8px,-8px)'>{nombre}</div>")).add_to(fg)
    for nombre, la, lo in ROTULOS:
        folium.Marker([la, lo], icon=folium.DivIcon(
            html=f"<div style='font:italic 600 12px Archivo,sans-serif;color:{TINTA};white-space:nowrap;"
                 f"text-shadow:0 0 3px #fff,0 0 3px #fff;transform:translate(-50%,-50%)'>{nombre}</div>")).add_to(fg)
    fg.add_to(m)


def capa_regiones(m, solo=None):
    fg = folium.FeatureGroup(name=tr("Regiones analizadas", "Analyzed regions", pt="Regiões analisadas"), show=True)
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
            recorte = (max(nuevo[0], -56.0), min(nuevo[1], -21.0), max(nuevo[2], -74.0), min(nuevo[3], -53.0))
            if recorte[0] < recorte[1] and recorte[2] < recorte[3]:
                st.session_state.bbox_pendiente = recorte
                st.rerun()
            else:
                st.warning(tr("El rectángulo quedó fuera de Argentina. Dibujalo sobre el país.",
                              "The rectangle is outside Argentina. Draw it over the country.",
                              pt="O retângulo ficou fora da Argentina. Desenhe-o sobre o país."))


def capa_puntos(m, df):
    "Focos individuales: VIIRS = punto relleno, MODIS = anillo; el color indica la potencia (FRP)."
    for s_ in ["VIIRS", "MODIS"]:
        sub_s = df[df.sensor == s_]
        if not len(sub_s):
            continue
        cols_s = colores(SEQ, np.log10(sub_s.frp.clip(lower=1)) / 3)
        puntos = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(r.lon), float(r.lat)]},
             "properties": {"c": c, "tip": f"{s_} · {r.fecha:%d/%m/%Y} · FRP {r.frp:.1f} MW"}}
            for r, c in zip(sub_s.itertuples(), cols_s)]}
        relleno = s_ == "VIIRS"
        folium.GeoJson(puntos, name=tr(f"Focos {s_}", f"{s_} hotspots", pt=f"Focos {s_}"),
                       marker=folium.CircleMarker(radius=3 if relleno else 4.5),
                       style_function=lambda f, relleno=relleno: {
                           "color": f["properties"]["c"], "fillColor": f["properties"]["c"], "weight": 0 if relleno else 2,
                           "fill": relleno, "fillOpacity": .85 if relleno else 0},
                       tooltip=folium.GeoJsonTooltip(fields=["tip"], labels=False)).add_to(m)


def focos_de_zona():
    if focos is None:
        return None
    if region == AREA:
        return focos[focos.lat.between(bbox[0], bbox[1]) & focos.lon.between(bbox[2], bbox[3])]
    if region == PAIS:
        return focos
    return focos[focos.region == region]


def limites_zona():
    if region == AREA:
        return (bbox[0] - .3, bbox[1] + .3, bbox[2] - .3, bbox[3] + .3)
    if region in REGIONES:
        la0, la1, lo0, lo1 = REGIONES[region]
        return (la0 - .5, la1 + .5, lo0 - .5, lo1 + .5)
    return (-55.5, -21.5, -74.5, -53.5)


# ================================================================ 1. comparación
with tabs[0]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=MESES, y=clima.p90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=MESES, y=clima.p10, mode="lines", fill="tonexty", line=dict(width=0), fillcolor=C_BANDA,
                             name=tr("Rango habitual (10–90 %)", "Usual range (10–90%)", pt="Faixa habitual (10–90 %)"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=MESES, y=clima.mediana, name=tr("Mediana histórica", "Historical median", pt="Mediana histórica"),
                             mode="lines+markers", line=dict(color=CENIZA if not ACC else "#4D4D4D", dash="dash"),
                             marker=dict(symbol="diamond", size=7)))
    alto = (actual > clima.p90).values
    fig.add_trace(go.Bar(x=MESES, y=actual, name=str(anio), marker_color=np.where(alto, C_ALTO, C_NORMAL),
                         marker_pattern_shape=np.where(alto, "/", "") if ACC else None,
                         text=np.where(alto, tr("▲ alto", "▲ high", pt="▲ alto"), "") if ACC else None, textposition="outside"))
    fig.update_layout(height=430, yaxis_title=f"{NOMBRE_MEDIDA[medida]} {tr('por mes', 'per month', pt='por mês')}", hovermode="x unified",
                      bargap=.25)
    grafico(fig)
    st.markdown(f"<p class='nota'>{tr('Barras de color brasa' if not ACC else 'Barras oscuras con rayas y «▲ alto»', 'Ember bars' if not ACC else 'Dark striped bars marked «▲ high»', pt='Barras cor de brasa' if not ACC else 'Barras escuras com listras e «▲ alto»')}"
                f"{tr(': meses por encima del rango habitual. Sirve para ver si la temporada viene más intensa o si empezó antes.', ': months above the usual range. Useful to see whether the season is more intense or started earlier.', pt=': meses acima da faixa habitual. Serve para ver se a temporada está mais intensa ou se começou antes.')}</p>",
                unsafe_allow_html=True)
    if pico:
        texto = tr(f"En {anio}, el mes con más actividad fue {MESES_LARGOS[pico - 1]} con {miles(actual[pico])} {UNIDAD[medida]}. ",
                   f"In {anio}, the most active month was {MESES_LARGOS[pico - 1]} with {miles(actual[pico])} {UNIDAD[medida]}. ", pt=f"Em {anio}, o mês com mais atividade foi {MESES_LARGOS[pico - 1]} com {miles(actual[pico])} {UNIDAD[medida]}. ")
        if pico_normal:
            texto += tr(f"Habitualmente el pico es en {MESES_LARGOS[pico_normal - 1]}. ",
                        f"The usual peak is in {MESES_LARGOS[pico_normal - 1]}. ", pt=f"Normalmente o pico é em {MESES_LARGOS[pico_normal - 1]}. ")
        texto += (tr(f"Por encima del rango habitual: {lista(meses_altos)}. ", f"Above the usual range: {lista(meses_altos)}. ", pt=f"Acima da faixa habitual: {lista(meses_altos)}. ")
                  if meses_altos else tr("Ningún mes superó el rango habitual. ", "No month exceeded the usual range. ", pt="Nenhum mês superou a faixa habitual. "))
        bajos = [MESES_LARGOS[m - 1] for m in con_dato if actual[m] < clima.p10[m]]
        if bajos:
            texto += tr(f"Por debajo del rango habitual: {lista(bajos)}.", f"Below the usual range: {lista(bajos)}.", pt=f"Abaixo da faixa habitual: {lista(bajos)}.")
        describir(texto)

    anual = dr.groupby("anio")[col].sum(min_count=1).reset_index()
    anual["tipo"] = np.where(anual.anio == anio, tr("Año elegido", "Selected year", pt="Ano escolhido"),
                             np.where(anual.anio < 2012, tr("MODIS armonizado", "Harmonized MODIS", pt="MODIS harmonizado"), "VIIRS"))
    fig2 = px.bar(anual, x="anio", y=col, color="tipo", pattern_shape="tipo" if ACC else None,
                  color_discrete_map={tr("Año elegido", "Selected year", pt="Ano escolhido"): C_ALTO, "VIIRS": "#9AA5B1",
                                      tr("MODIS armonizado", "Harmonized MODIS", pt="MODIS harmonizado"): "#C9D1D9"},
                  labels={"anio": tr("Año", "Year", pt="Ano"), col: f"{NOMBRE_MEDIDA[medida]} {tr('por año', 'per year', pt='por ano')}", "tipo": ""})
    mediana_anual = ref.groupby("anio")[col].sum().median()
    fig2.add_hline(y=mediana_anual, line_dash="dash", line_color=CENIZA, annotation_text=tr("Mediana de referencia", "Reference median", pt="Mediana de referência"))
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
                     f"The reference annual median is {miles(mediana_anual)}. ", pt=f"Os três anos com mais atividade são {lista([f'{a} ({miles(v)})' for a, v in zip(top.anio, top[col])])}. A mediana anual de referência é {miles(mediana_anual)}. ")
                  + (tr(f"{anio} ocupa el puesto {puesto} de {len(completos)}.", f"{anio} ranks {puesto} of {len(completos)}.", pt=f"{anio} ocupa a posição {puesto} de {len(completos)}.")
                     if anio <= 2025 else tr(f"{anio} está incompleto.", f"{anio} is incomplete.", pt=f"{anio} está incompleto.")))

# ================================================================ 2. calendario
with tabs[1]:
    st.markdown(tr("El **calendario de actividad de fuego**: cada casillero es un mes. Muestra en qué meses se concentra el "
                   "fuego en la zona y qué años fueron excepcionales.",
                   "The **burning activity calendar**: each cell is one month. It shows when fire concentrates in the "
                   "area and which years were exceptional.", pt="O **calendário da atividade de fogo**: cada quadrado é um mês. Mostra em quais meses o fogo se concentra na área e quais anos foram excepcionais."))
    modo = st.radio(tr("Mostrar", "Show", pt="Mostrar"), ["valores", "anomalia"], horizontal=True,
                    format_func=lambda x: tr("Valores", "Values", pt="Valores") if x == "valores" else tr("Comparado con lo habitual del mes", "Compared with the month's usual", pt="Comparado com o habitual do mês"))
    tabla = dr.pivot_table(index="anio", columns="mes", values=col).reindex(columns=range(1, 13))
    if modo == "valores":
        z, escala, texto_cb, zmid = np.log10(tabla + 1), (ESCALA_FRANJA if not ACC else SEQ), f"log₁₀({UNIDAD[medida]}+1)", None
        hover = f"%{{y}} %{{x}}: %{{customdata:,.0f}} {UNIDAD[medida]}<extra></extra>"
    else:
        z = np.log2((tabla + 1) / (clima.mediana + 1).values)
        escala, texto_cb, zmid = DIV, tr("veces lo habitual (log₂)", "times usual (log₂)", pt="vezes o habitual (log₂)"), 0
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
                     "Darker = more activity." if not ACC else "Lighter (yellow) = more activity.", pt="Mais escuro = mais atividade." if not ACC else "Mais claro (amarelo) = mais atividade.")
    else:
        leyenda = tr("Rojo = más que la mediana de ese mes; azul = menos." if not ACC else "Naranja = más que la mediana de ese mes; violeta = menos.",
                     "Red = above the month's median; blue = below." if not ACC else "Orange = above the month's median; purple = below.", pt="Vermelho = acima da mediana desse mês; azul = abaixo." if not ACC else "Laranja = acima da mediana desse mês; roxo = abaixo.")
    st.markdown(f"<p class='nota'>{leyenda} {tr('Hasta 2011, MODIS armonizado; desde 2012, VIIRS.', 'Until 2011, harmonized MODIS; from 2012, VIIRS.', pt='Até 2011, MODIS harmonizado; desde 2012, VIIRS.')}</p>",
                unsafe_allow_html=True)
    temporada = sorted(clima.mediana.sort_values(ascending=False).index[:3])
    extremos = tabla.stack().nlargest(3)
    describir(tr(f"Los meses con más actividad habitual son {lista([MESES_LARGOS[m - 1] for m in temporada])}. Los tres meses más "
                 f"extremos de la serie fueron {lista([f'{MESES_LARGOS[m - 1]} de {a} ({miles(v)})' for (a, m), v in extremos.items()])}. "
                 "Cada casillero muestra su valor (k = miles).",
                 f"The months with the most usual activity are {lista([MESES_LARGOS[m - 1] for m in temporada])}. The three most "
                 f"extreme months were {lista([f'{MESES_LARGOS[m - 1]} {a} ({miles(v)})' for (a, m), v in extremos.items()])}. "
                 "Each cell shows its value (k = thousands).", pt=f"Os meses com mais atividade habitual são {lista([MESES_LARGOS[m - 1] for m in temporada])}. Os três meses mais extremos da série foram {lista([f'{MESES_LARGOS[m - 1]} de {a} ({miles(v)})' for (a, m), v in extremos.items()])}. Cada quadrado mostra seu valor (k = milhares)."))

# ================================================================ 3. búsqueda por fechas
with tabs[2]:
    if focos is None:
        st.warning(tr("Esta sección necesita los archivos de focos individuales (carpeta `focos`).",
                      "This section needs the individual hotspot files (`focos` folder).", pt="Esta seção precisa dos arquivos de focos individuais (pasta `focos`)."))
    else:
        st.markdown(tr("Elegí un **rango de fechas** para ver los focos de la zona, compararlos con el **mismo período de los años "
                       "anteriores** y descargarlos.",
                       "Pick a **date range** to see the area's hotspots, compare them with the **same period in previous "
                       "years** and download them.", pt="Escolha um **intervalo de datas** para ver os focos da área, compará-los com o **mesmo período dos anos anteriores** e baixá-los."))
        fmin, fmax = dt.date(2003, 1, 1), FECHA_FIN.date()
        b1, _ = st.columns([2, 3])
        with b1:
            rango = st.date_input(tr("Rango de fechas", "Date range", pt="Intervalo de datas"), value=(fmax - dt.timedelta(days=59), fmax),
                                  min_value=fmin, max_value=fmax, format="DD/MM/YYYY" if not EN else "YYYY/MM/DD")
        if not isinstance(rango, (tuple, list)) or len(rango) < 2:
            st.info(tr("Elegí también la fecha final del rango.", "Pick the end date of the range too.", pt="Escolha também a data final do intervalo."))
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
            m1.metric(tr("Focos armonizados", "Harmonized hotspots", pt="Focos harmonizados"), miles(equiv_sel),
                      f"{dif_b:+.0f} % {tr('respecto de la mediana', 'vs. median', pt='em relação à mediana')}" if np.isfinite(dif_b) else None,
                      delta_color="inverse")
            m2.metric(tr("Mediana del mismo período", "Median of the same period", pt="Mediana do mesmo período"), miles(med),
                      help=tr("En los demás años con datos", "In the other years with data", pt="Nos demais anos com dados"))
            m3.metric(tr("Detectados por VIIRS", "Detected by VIIRS", pt="Detectados pelo VIIRS"), miles((sel.sensor == "VIIRS").sum()))
            m4.metric(tr("Detectados por MODIS", "Detected by MODIS", pt="Detectados pelo MODIS"), miles((sel.sensor == "MODIS").sum()))

            if len(comp) > 1:
                comp["tipo"] = np.where(comp.anio == d0.year, tr("Período elegido", "Selected period", pt="Período escolhido"), tr("Otros años", "Other years", pt="Outros anos"))
                fig = px.bar(comp, x="anio", y="focos_armonizados", color="tipo", pattern_shape="tipo" if ACC else None,
                             color_discrete_map={tr("Período elegido", "Selected period", pt="Período escolhido"): C_ALTO, tr("Otros años", "Other years", pt="Outros anos"): "#9AA5B1"},
                             labels={"tipo": "", "anio": tr("Año", "Year", pt="Ano"), "focos_armonizados": tr("Focos armonizados", "Harmonized hotspots", pt="Focos harmonizados")},
                             title=tr(f"Mismo período ({d0:%d/%m} – {d1:%d/%m}) en cada año", f"Same period ({d0:%b %d} – {d1:%b %d}) in every year", pt=f"Mesmo período ({d0:%d/%m} – {d1:%d/%m}) em cada ano"))
                fig.add_hline(y=med, line_dash="dash", line_color=CENIZA, annotation_text=tr("Mediana", "Median", pt="Mediana"))
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
                             f"{lista([f'{a} ({miles(v)})' for a, v in zip(rk.anio[:3], rk.focos_armonizados[:3])])}.", pt=f"Entre {d0:%d/%m/%Y} e {d1:%d/%m/%Y} houve {miles(equiv_sel)} focos harmonizados; a mediana do mesmo período é {miles(med)}. " + (f"É a posição {puesto} de {len(rk)} anos. " if puesto else "") + f"Os anos com mais focos nesse período: {lista([f'{a} ({miles(v)})' for a, v in zip(rk.anio[:3], rk.focos_armonizados[:3])])}."))
            elif largo > 365:
                st.markdown(f"<p class='nota'>{tr('La comparación con otros años está disponible para rangos de hasta un año.', 'The comparison with other years is available for ranges of up to one year.', pt='A comparação com outros anos está disponível para intervalos de até um ano.')}</p>",
                            unsafe_allow_html=True)

            dd = diario.loc[d0:d1].reset_index().rename(columns={"index": "fecha"})
            dd["relleno"] = dd.fecha.isin(DIAS_HUECO_VIIRS)
            fig = go.Figure(go.Bar(x=dd.fecha, y=dd.equiv, marker_color=np.where(dd.relleno, C_VIIRS if ACC else AGUA, C_NORMAL),
                                   marker_pattern_shape=np.where(dd.relleno, "x", ""),
                                   name=tr("Focos armonizados", "Harmonized hotspots", pt="Focos harmonizados"),
                                   hovertemplate="%{x|%d/%m/%Y}: %{y:,.0f}<extra></extra>"))
            fig.update_layout(height=260, title=tr("Focos armonizados por día", "Harmonized hotspots per day", pt="Focos harmonizados por dia"), bargap=.1)
            grafico(fig)
            n_rell = int(dd.relleno.sum())
            if n_rell:
                st.markdown(f"<p class='nota'>" + tr(
                    f"Barras cuadriculadas: {n_rell} días en que el satélite VIIRS S-NPP no midió. Esos días se muestran "
                    "los focos del satélite VIIRS NOAA-20, que lleva el mismo instrumento.",
                    f"Cross-hatched bars: {n_rell} days when the VIIRS S-NPP satellite did not measure. On those days the "
                    "page shows hotspots from the VIIRS NOAA-20 satellite, which carries the same instrument.", pt=f"Barras quadriculadas: {n_rell} dias em que o satélite VIIRS S-NPP não mediu. Nesses dias são mostrados os focos do satélite VIIRS NOAA-20, que leva o mesmo instrumento.") + "</p>",
                    unsafe_allow_html=True)
            if len(dd) and dd.equiv.max() > 0:
                dmax = dd.loc[dd.equiv.idxmax()]
                describir(tr(f"El día con más focos fue el {dmax.fecha:%d/%m/%Y}, con {miles(dmax.equiv)}.",
                             f"The day with the most hotspots was {dmax.fecha:%Y-%m-%d}, with {miles(dmax.equiv)}.", pt=f"O dia com mais focos foi {dmax.fecha:%d/%m/%Y}, com {miles(dmax.equiv)}."))

            # Mapa de focos: VIIRS = punto relleno, MODIS = anillo
            if len(sel):
                muestra = sel.sample(min(len(sel), 6000), random_state=0)
                lim = (float(sel.lat.min()) - .3, float(sel.lat.max()) + .3, float(sel.lon.min()) - .3, float(sel.lon.max()) + .3)
                mb = mapa_base(lim)
                capa_puntos(mb, muestra)
                capa_ciudades(mb)
                if region == AREA:
                    capa_regiones(mb, solo=bbox)
                mostrar_mapa(mb, f"mapa_busqueda_{IDIOMA}", alto=520)
                st.markdown(f"<p class='nota'>{tr('Punto relleno = VIIRS; anillo = MODIS. El color indica la potencia del fuego (FRP): más intenso, más potencia.', 'Filled dot = VIIRS; ring = MODIS. Color shows fire radiative power (FRP): stronger color, more power.', pt='Ponto cheio = VIIRS; anel = MODIS. A cor indica a potência do fogo (FRP): mais intensa, mais potência.')}"
                            + (tr(f" Se muestran 6.000 de {miles(len(sel))} focos; la descarga los incluye todos.",
                                  f" Showing 6,000 of {miles(len(sel))} hotspots; the download includes all of them.", pt=f" São mostrados 6.000 de {miles(len(sel))} focos; o download inclui todos.") if len(sel) > 6000 else "")
                            + "</p>", unsafe_allow_html=True)

            @st.cache_data
            def csv_focos(clave, _df):
                out = _df.rename(columns={"hora_utc": "hora_utc_HHMM", "lat": "latitud", "lon": "longitud", "frp": "frp_MW"})
                out["fecha"] = out["fecha"].dt.strftime("%Y-%m-%d")
                return out.to_csv(index=False).encode("utf-8-sig")

            st.markdown("**" + tr("Descargas de esta búsqueda", "Downloads for this search", pt="Downloads desta busca") + "**")
            e1, e2, e3 = st.columns(3)
            clave = (region, bbox, str(d0.date()), str(d1.date()))
            e1.download_button(tr(f"Focos detectados ({miles(len(sel))}, CSV)", f"Detected hotspots ({miles(len(sel))}, CSV)", pt=f"Focos detectados ({miles(len(sel))}, CSV)"),
                               csv_focos(clave, sel), f"focos_{d0:%Y%m%d}_{d1:%Y%m%d}.csv", "text/csv", width="stretch",
                               disabled=len(sel) == 0, icon=":material/download:")
            e2.download_button(tr("Focos por día (CSV)", "Hotspots per day (CSV)", pt="Focos por dia (CSV)"),
                               dd.rename(columns={"MODIS": "focos_MODIS", "VIIRS": "focos_VIIRS", "equiv": "focos_armonizados"})
                               .to_csv(index=False).encode("utf-8-sig"),
                               f"focos_por_dia_{d0:%Y%m%d}_{d1:%Y%m%d}.csv", "text/csv", width="stretch", icon=":material/download:")
            e3.download_button(tr("Comparación con otros años (CSV)", "Comparison with other years (CSV)", pt="Comparação com outros anos (CSV)"),
                               comp.drop(columns="tipo", errors="ignore").to_csv(index=False).encode("utf-8-sig"),
                               f"comparacion_{d0:%m%d}_{d1:%m%d}.csv", "text/csv", width="stretch", disabled=len(comp) == 0,
                               icon=":material/download:")

# ================================================================ 4. mapa
with tabs[3]:
    ca, cb = st.columns([1, 3.2])
    with ca:
        modo_mapa = st.radio(tr("Mostrar", "Show", pt="Mostrar"), ["anio", "mes", "dif"],
                             format_func=lambda x: {"anio": tr("Año completo", "Whole year", pt="Ano completo"), "mes": tr("Un mes", "One month", pt="Um mês"),
                                                    "dif": tr("Diferencia con lo habitual", "Difference from usual", pt="Diferença em relação ao habitual")}[x])
        mes_mapa = (st.selectbox(tr("Mes", "Month", pt="Mês"), range(1, 13), format_func=lambda m: MESES_LARGOS[m - 1].capitalize())
                    if modo_mapa == "mes" else None)
        forma = "celdas"
        if modo_mapa != "dif" and focos is not None:
            forma = st.radio(tr("Dibujar como", "Draw as", pt="Desenhar como"), ["celdas", "puntos"],
                             format_func=lambda x: tr("Celdas (resumen)", "Cells (summary)", pt="Células (resumo)") if x == "celdas"
                             else tr("Focos individuales", "Individual hotspots", pt="Focos individuais"))
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
        m["tip"] = [tr(f"Este año: {miles(a)} · Habitual: {miles(n)}", f"This year: {miles(a)} · Usual: {miles(n)}", pt=f"Este ano: {miles(a)} · Habitual: {miles(n)}")
                    for a, n in zip(m.actual, m.normal)]
        nota = (tr("Naranja" if ACC else "Rojo", "Orange" if ACC else "Red", pt="Laranja" if ACC else "Vermelho") + tr(": más focos que lo habitual. ", ": more hotspots than usual. ", pt=": mais focos do que o habitual. ")
                + tr("Violeta" if ACC else "Azul", "Purple" if ACC else "Blue", pt="Roxo" if ACC else "Azul") + tr(": menos. Lo habitual es la mediana de cada celda.",
                                                                                   ": fewer. Usual is each cell's median.", pt=": menos. O habitual é a mediana de cada célula."))
    else:
        sel_m = g[g.anio == anio]
        if mes_mapa:
            sel_m = sel_m[sel_m.mes == mes_mapa]
        m = sel_m.groupby(["lat_c", "lon_c"]).focos_equiv.sum().reset_index()
        m["valor"] = m.focos_equiv
        tope = float(np.log10(m.valor.max() + 1)) if len(m) else 1.0
        m["color"] = colores(SEQ, np.log10(m.valor + 1) / max(tope, 1))
        m["tip"] = [f"{miles(v)} {UNIDAD['focos']}" for v in m.valor]
        nota = tr("Focos por celda, en escala VIIRS. ", "Hotspots per cell, on the VIIRS scale. ", pt="Focos por célula, na escala do VIIRS. ") + (
            tr("Más claro (amarillo) = más focos." if ACC else "Más oscuro = más focos.",
               "Lighter (yellow) = more hotspots." if ACC else "Darker = more hotspots.", pt="Mais claro (amarelo) = mais focos." if ACC else "Mais escuro = mais focos."))
    mapa = mapa_base(limites_zona(), dibujar=True)
    celdas = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"color": r.color, "tip": r.tip},
         "geometry": {"type": "Polygon", "coordinates": [[[r.lon_c - paso / 2, r.lat_c - paso / 2], [r.lon_c + paso / 2, r.lat_c - paso / 2],
                                                         [r.lon_c + paso / 2, r.lat_c + paso / 2], [r.lon_c - paso / 2, r.lat_c + paso / 2],
                                                         [r.lon_c - paso / 2, r.lat_c - paso / 2]]]}}
        for r in m.itertuples()]}
    if celdas["features"]:
        folium.GeoJson(celdas, name=tr("Actividad de fuego", "Fire activity", pt="Atividade de fogo"),
                       style_function=lambda f: {"fillColor": f["properties"]["color"], "color": f["properties"]["color"],
                                                 "weight": 0.3, "fillOpacity": 0.82},
                       tooltip=folium.GeoJsonTooltip(fields=["tip"], labels=False)).add_to(mapa)
    else:
        with ca:
            st.info(tr("No hay focos para mostrar con esta selección.", "There are no hotspots to show for this selection.", pt="Não há focos para mostrar com esta seleção."))
    if forma == "puntos":
        pz = focos_de_zona()
        pts = pz[pz.fecha.dt.year == anio]
        if mes_mapa:
            pts = pts[pts.fecha.dt.month == mes_mapa]
        mapa = mapa_base(limites_zona(), dibujar=True)
        if len(pts):
            capa_puntos(mapa, pts.sample(min(len(pts), 6000), random_state=0))
        nota = (tr("Cada punto es un foco detectado: punto relleno = VIIRS; anillo = MODIS. El color indica la potencia del fuego (FRP). ",
                   "Each dot is one detected hotspot: filled dot = VIIRS; ring = MODIS. Color shows fire radiative power (FRP). ", pt="Cada ponto é um foco detectado: ponto cheio = VIIRS; anel = MODIS. A cor indica a potência do fogo (FRP). ")
                + (tr(f"Se muestran 6.000 de {miles(len(pts))} focos (elegí un mes o un área más chica para verlos todos).",
                      f"Showing 6,000 of {miles(len(pts))} hotspots (pick a month or a smaller area to see them all).", pt=f"São mostrados 6.000 de {miles(len(pts))} focos (escolha um mês ou uma área menor para ver todos).")
                   if len(pts) > 6000 else tr(f"{miles(len(pts))} focos.", f"{miles(len(pts))} hotspots.", pt=f"{miles(len(pts))} focos.")))
        m = pd.DataFrame(columns=["lat_c", "lon_c", "valor"])
        with ca:
            describir(nota)
    capa_regiones(mapa, solo=bbox if region == AREA else None)
    capa_ciudades(mapa)
    with cb:
        mostrar_mapa(mapa, f"mapa_principal_{IDIOMA}", alto=640)
    with ca:
        st.markdown(f"<p class='nota'>{nota}</p>", unsafe_allow_html=True)
        st.markdown(tr("**Capas:** arriba a la derecha del mapa podés cambiar el fondo (claro, topografía con curvas de nivel, "
                       "relieve submarino o satélite) y mostrar las **ciudades principales**.",
                       "**Layers:** at the top right of the map you can switch the background (light, topography with contour "
                       "lines, seafloor relief or satellite) and show the **main cities**.", pt="**Camadas:** no canto superior direito do mapa você pode trocar o fundo (claro, topografia com curvas de nível, relevo submarino ou satélite) e mostrar as **cidades principais**."))
        st.markdown(tr("**Elegir un área:** usá la herramienta ▭ de la izquierda del mapa para dibujar un rectángulo. La página "
                       "pasa a mostrar esa área en todas las secciones.",
                       "**Pick an area:** use the ▭ tool on the left of the map to draw a rectangle. The whole page then "
                       "shows that area.", pt="**Escolher uma área:** use a ferramenta ▭ à esquerda do mapa para desenhar um retângulo. A página passa a mostrar essa área em todas as seções."))
        if len(m):
            if modo_mapa == "dif":
                t = m.sort_values("valor").iloc[-1]
                describir(tr(f"{int((m.valor > 0).sum())} celdas tuvieron más focos que lo habitual y {int((m.valor < 0).sum())} menos. "
                             f"La mayor suba está cerca de {t.lat_c:.2f}°, {t.lon_c:.2f}° ({zona_txt(region_que_contiene(t.lat_c, t.lon_c))}).",
                             f"{int((m.valor > 0).sum())} cells had more hotspots than usual and {int((m.valor < 0).sum())} fewer. "
                             f"The largest increase is near {t.lat_c:.2f}°, {t.lon_c:.2f}° ({zona_txt(region_que_contiene(t.lat_c, t.lon_c))}).", pt=f"{int((m.valor > 0).sum())} células tiveram mais focos do que o habitual e {int((m.valor < 0).sum())} menos. O maior aumento está perto de {t.lat_c:.2f}°, {t.lon_c:.2f}° ({zona_txt(region_que_contiene(t.lat_c, t.lon_c))})."))
            else:
                top = m.nlargest(3, "valor")
                describir(tr("Las celdas con más focos están cerca de ", "The cells with the most hotspots are near ", pt="As células com mais focos estão perto de ") + lista(
                    [f"{r.lat_c:.2f}°, {r.lon_c:.2f}° ({zona_txt(region_que_contiene(r.lat_c, r.lon_c))}, {miles(r.valor)})"
                     for r in top.itertuples()]) + ".")

# ================================================================ 5. armonización
with tabs[4]:
    st.subheader(tr("El problema", "The problem", pt="O problema"))
    st.markdown(tr("MODIS (satélites Terra y Aqua, píxel de ~1 km) y VIIRS (satélite Suomi NPP, píxel de 375 m) miden distinto: "
                   "VIIRS divide cada incendio en más píxeles y ve fuegos más chicos. Si se pegan las dos series, parece que el fuego "
                   "se multiplicó en 2012. **Armonizar** es convertir MODIS a la escala de VIIRS para tener una serie continua desde "
                   "2003 en la que ya no importa qué sensor detectó cada foco.",
                   "MODIS (Terra and Aqua satellites, ~1 km pixel) and VIIRS (Suomi NPP satellite, 375 m pixel) measure differently: "
                   "VIIRS splits each fire into more pixels and sees smaller fires. Stitching both series together makes fire seem to "
                   "multiply in 2012. **Harmonizing** means converting MODIS to the VIIRS scale to build one continuous series since "
                   "2003, where it no longer matters which sensor detected each hotspot.", pt="MODIS (satélites Terra e Aqua, pixel de ~1 km) e VIIRS (satélite Suomi NPP, pixel de 375 m) medem de forma diferente: o VIIRS divide cada incêndio em mais pixels e vê fogos menores. Se as duas séries forem coladas, parece que o fogo se multiplicou em 2012. **Harmonizar** é converter o MODIS para a escala do VIIRS para ter uma série contínua desde 2003 em que já não importa qual sensor detectou cada foco."))
    an = dr.groupby("anio")[[f"{medida}_modis", f"{medida}_viirs", col]].sum(min_count=1).reset_index()
    an = an[an.anio <= 2025]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=an.anio, y=an[f"{medida}_modis"], name=tr("MODIS (original)", "MODIS (original)", pt="MODIS (original)"),
                             mode="lines+markers", line=dict(color=C_MODIS, dash="dash"), marker=dict(symbol="circle", size=7)))
    fig.add_trace(go.Scatter(x=an.anio, y=an[f"{medida}_viirs"], name=tr("VIIRS (original)", "VIIRS (original)", pt="VIIRS (original)"),
                             mode="lines+markers", line=dict(color=C_VIIRS, dash="dot"), marker=dict(symbol="square", size=7)))
    fig.add_trace(go.Scatter(x=an.anio, y=an[col], name=tr("Serie armonizada", "Harmonized series", pt="Série harmonizada"), mode="lines",
                             line=dict(color=C_ARM, width=3.5)))
    fig.add_vline(x=2011.5, line_dash="dot", line_color=CENIZA, annotation_text=tr("empieza VIIRS", "VIIRS starts", pt="começa o VIIRS"))
    fig.update_layout(height=400, yaxis_title=f"{NOMBRE_MEDIDA[medida]} {tr('por año', 'per year', pt='por ano')}", hovermode="x unified")
    grafico(fig)
    st.markdown(f"<p class='nota'>{tr('MODIS: línea a rayas con círculos. VIIRS: línea punteada con cuadrados. Serie armonizada: línea gruesa.', 'MODIS: dashed line with circles. VIIRS: dotted line with squares. Harmonized series: thick line.', pt='MODIS: linha tracejada com círculos. VIIRS: linha pontilhada com quadrados. Série harmonizada: linha grossa.')}</p>",
                unsafe_allow_html=True)
    describir(tr(f"MODIS queda siempre por debajo de VIIRS: en esta zona VIIRS detecta en promedio {K_ZONA:.2f} veces más focos "
                 f"({origen_k_txt}). Antes de 2012 la serie armonizada convierte MODIS a la escala de VIIRS con el método elegido; "
                 "desde 2012 es VIIRS, con los días sin dato completados con NOAA-20.",
                 f"MODIS is always below VIIRS: in this area VIIRS detects on average {K_ZONA:.2f} times more hotspots "
                 f"({origen_k_txt}). Before 2012 the harmonized series converts MODIS to the VIIRS scale with the chosen method; "
                 "from 2012 it is VIIRS, with no-data days filled with NOAA-20.", pt=f"O MODIS fica sempre abaixo do VIIRS: nesta área o VIIRS detecta em média {K_ZONA:.2f} vezes mais focos ({origen_k_txt}). Antes de 2012 a série harmonizada converte o MODIS para a escala do VIIRS com o método escolhido; desde 2012 é VIIRS, com os dias sem dado completados com o NOAA-20."))

    st.subheader(tr("Cómo elegimos el método", "How we chose the method", pt="Como escolhemos o método"))
    st.markdown(tr(f"Probamos 4 métodos con los datos de toda Argentina. Cada uno aprendió con **{MET['anios_entrenamiento']}** y se "
                   f"evaluó con **{MET['anios_test']}**, años que no había visto. Solo se usaron días en que los dos satélites "
                   f"funcionaron. **Regla, fijada antes de ver los resultados:** gana el método más simple que cumpla tres "
                   f"condiciones: error (MAE) a no más de {MET['tolerancia']:.0%} del mejor, sesgo total entre −10 % y +10 %, "
                   "y error del peor año menor al 25 %.",
                   f"We tested 4 methods on all of Argentina. Each learned from **{MET['anios_entrenamiento']}** and was evaluated on "
                   f"**{MET['anios_test']}**, years it had not seen. Only days when both satellites worked were used. **Rule, set "
                   f"before seeing the results:** the simplest method that meets three conditions wins: error (MAE) within "
                   f"{MET['tolerancia']:.0%} of the best, total bias between −10% and +10%, and worst-year error below 25%.", pt=f"Testamos 4 métodos com os dados de toda a Argentina. Cada um aprendeu com **{MET['anios_entrenamiento']}** e foi avaliado com **{MET['anios_test']}**, anos que não tinha visto. Só foram usados dias em que os dois satélites funcionaram. **Regra, fixada antes de ver os resultados:** vence o método mais simples que cumpra três condições: erro (MAE) no máximo {MET['tolerancia']:.0%} acima do melhor, viés total entre −10 % e +10 % e erro do pior ano menor que 25 %."))
    res = pd.DataFrame(MET["resultados"][medida]).set_index("método")
    elegido = MET["elegido"][medida]
    mejor = res["MAE"].idxmin()
    ic = MET.get("intervalos_mae", {}).get(medida, {})
    e_ = res.loc[elegido]
    simple = res.index[0]
    texto_eleccion = tr(
        f"Se eligió «{metodo_txt(elegido)}»: error medio (MAE) {miles(e_['MAE'], 1)} por región y mes, sesgo total "
        f"{e_['Sesgo total %']:+.1f} % y peor año {e_['Peor año (país) %']:.1f} % en {MET['anios_test']}. ",
        f"Chosen: «{metodo_txt(elegido)}»: mean error (MAE) {e_['MAE']:,.1f} per region and month, total bias "
        f"{e_['Sesgo total %']:+.1f}% and worst year {e_['Peor año (país) %']:.1f}% in {MET['anios_test']}. ", pt=f"Foi escolhido «{metodo_txt(elegido)}»: erro médio (MAE) {miles(e_['MAE'], 1)} por região e mês, viés total {e_['Sesgo total %']:+.1f} % e pior ano {e_['Peor año (país) %']:.1f} % em {MET['anios_test']}. ")
    if elegido == mejor and elegido != simple:
        texto_eleccion += tr(
            f"Es el de menor error. El método más simple («{metodo_txt(simple)}») tuvo un error "
            f"{100 * (res.loc[simple, 'MAE'] / e_['MAE'] - 1):.1f} % mayor, por encima de la tolerancia. Esa diferencia no es "
            "estadísticamente significativa: los dos métodos dan resultados muy parecidos, y eso muestra que la serie no depende "
            "del método.",
            f"It has the lowest error. The simplest method («{metodo_txt(simple)}») had a "
            f"{100 * (res.loc[simple, 'MAE'] / e_['MAE'] - 1):.1f}% higher error, above the tolerance. That difference is not "
            "statistically significant: both methods give very similar results, which shows the series does not depend on "
            "the method.", pt=f"É o de menor erro. O método mais simples («{metodo_txt(simple)}») teve um erro {100 * (res.loc[simple, 'MAE'] / e_['MAE'] - 1):.1f} % maior, acima da tolerância. Essa diferença não é estatisticamente significativa: os dois métodos dão resultados muito parecidos, o que mostra que a série não depende do método.")
    elif elegido == mejor:
        texto_eleccion += tr("Es el de menor error.", "It has the lowest error.", pt="É o de menor erro.")
    else:
        texto_eleccion += tr(f"El de menor error es «{metodo_txt(mejor)}», pero la diferencia está dentro de la tolerancia: "
                             "se prefiere el más simple y explicable.",
                             f"The lowest error is «{metodo_txt(mejor)}», but the difference is within the tolerance: the "
                             "simpler, more explainable one is preferred.", pt=f"O de menor erro é «{metodo_txt(mejor)}», mas a diferença está dentro da tolerância: prefere-se o mais simples e explicável.")
    st.success(texto_eleccion)
    cc1, cc2 = st.columns(2)
    with cc1:
        b = res.reset_index()
        b["nombre"] = b["método"].map(metodo_txt)
        err = dict(type="data", symmetric=False, color=CENIZA, thickness=1.5,
                   array=[ic[mm][1] - v for mm, v in zip(b["método"], b["MAE"])],
                   arrayminus=[v - ic[mm][0] for mm, v in zip(b["método"], b["MAE"])]) if ic else None
        fig = go.Figure(go.Bar(x=b["MAE"], y=b["nombre"], orientation="h", error_x=err,
                               marker_color=np.where(b["método"] == elegido, C_ALTO, "#9AA5B1"),
                               marker_pattern_shape=np.where(b["método"] == elegido, "/", "") if ACC else None,
                               text=[("✔ " if mm == elegido else "") + miles(v, 1) for mm, v in zip(b["método"], b["MAE"])],
                               textposition="auto", hovertemplate="%{y}: MAE %{x:,.1f}<extra></extra>"))
        fig.update_layout(height=300, xaxis_title=f"{tr('Error medio por región y mes', 'Mean error per region and month', pt='Erro médio por região e mês')} ({UNIDAD[medida]})",
                          yaxis=dict(autorange="reversed"), title=tr("Error (MAE): menor es mejor", "Error (MAE): lower is better", pt="Erro (MAE): menor é melhor"))
        grafico(fig)
        if ic:
            st.markdown(f"<p class='nota'>{tr('La línea fina sobre cada barra es el intervalo de confianza del 95 %: si dos intervalos se superponen mucho, los métodos son prácticamente equivalentes.', 'The thin line on each bar is the 95% confidence interval: if two intervals overlap a lot, the methods are practically equivalent.', pt='A linha fina sobre cada barra é o intervalo de confiança de 95 %: se dois intervalos se sobrepõem muito, os métodos são praticamente equivalentes.')}</p>",
                        unsafe_allow_html=True)
    with cc2:
        d = pd.DataFrame(MET["anual_test"][medida])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.anio, y=d["Real"], name=tr("VIIRS real", "Actual VIIRS", pt="VIIRS real"), line=dict(color=C_ARM, width=4)))
        simbolos = ["circle", "square", "diamond", "triangle-up"]
        for i, c in enumerate([c for c in d.columns if c not in ("anio", "Real")]):
            fig.add_trace(go.Scatter(x=d.anio, y=d[c], name=metodo_txt(c), mode="lines+markers", marker=dict(symbol=simbolos[i % 4], size=8),
                                     line=dict(dash="solid" if c == elegido else "dot", width=3 if c == elegido else 1.5)))
        fig.update_layout(height=300, title=tr("Test: total de Argentina por año", "Test: Argentina's total per year", pt="Teste: total da Argentina por ano"),
                          xaxis=dict(dtick=1), legend=dict(font=dict(size=10), y=-0.25), hovermode="x unified")
        grafico(fig)
    describir(tr(f"El método elegido es «{elegido}», marcado con ✔. ", f"The chosen method is «{metodo_txt(elegido)}», marked with ✔. ", pt=f"O método escolhido é «{metodo_txt(elegido)}», marcado com ✔. ")
              + " ".join(f"{metodo_txt(mm)}: MAE {miles(v, 1)}." for mm, v in zip(res.index, res["MAE"])))
    tabla_res = res.rename(index=metodo_txt)
    tabla_res.index.name = tr("Método", "Method", pt="Método")
    tabla_res = tabla_res.rename(columns={"Sesgo total %": tr("Sesgo total %", "Total bias %", pt="Viés total %"),
                                          "Peor año (país) %": tr("Peor año (país) %", "Worst year (country) %", pt="Pior ano (país) %")})
    st.dataframe(tabla_res.style.format(precision=2).apply(
        lambda r: ["font-weight: bold; background-color: #FBE3DA" if r.name == metodo_txt(elegido) else ""] * len(r), axis=1),
        width="stretch")
    with st.expander(tr("Ver las fórmulas de cada método", "See each method's formula", pt="Ver as fórmulas de cada método")):
        st.markdown(tr("**1. Factor fijo por región** — una regla de tres:", "**1. Fixed factor per region** — a rule of three:", pt="**1. Fator fixo por região** — uma regra de três:"))
        st.latex(r"k_r = \frac{\sum V}{\sum M} \qquad \hat V = k_r \cdot M")
        st.markdown(tr("**2. Regresión logarítmica por región y mes** (con corrección de sesgo κ):",
                       "**2. Log regression by region and month** (with bias correction κ):", pt="**2. Regressão logarítmica por região e mês** (com correção de viés κ):"))
        st.latex(r"\log(1+\hat V) = a + b\,\log(1+M) + c\,\sin\tfrac{2\pi\,m}{12} + d\,\cos\tfrac{2\pi\,m}{12}")
        st.markdown(tr("**3. Mapeo de cuantiles** — se emparejan percentiles:", "**3. Quantile mapping** — percentiles are matched:", pt="**3. Mapeamento de quantis** — os percentis são emparelhados:"))
        st.latex(r"\hat V = F_V^{-1}\big(F_M(M)\big)")
        st.markdown(tr("**4. Random Forest** — promedio de 300 árboles de decisión con MODIS, mes y región.",
                       "**4. Random Forest** — average of 300 decision trees using MODIS, month and region.", pt="**4. Random Forest** — média de 300 árvores de decisão com MODIS, mês e região."))
    if medida == "focos":
        fac = pd.Series(FACTORES).sort_values()
        fig = px.bar(x=fac.values, y=[zona_txt(z) for z in fac.index], orientation="h", text=[f"{v:.1f}×" for v in fac.values],
                     labels={"x": "VIIRS / MODIS", "y": ""}, color_discrete_sequence=[C_VIIRS])
        fig.update_layout(height=290, title=tr("Cuántas veces más focos detecta VIIRS que MODIS, por región",
                                               "How many times more hotspots VIIRS detects than MODIS, per region", pt="Quantas vezes mais focos o VIIRS detecta do que o MODIS, por região"))
        grafico(fig)

# ================================================================ 6. validación
with tabs[5]:
    st.markdown(tr("Si la armonización es buena, **los años excepcionales tienen que aparecer igual** sin importar cómo se mida el "
                   "fuego. Cada serie (toda Argentina) se expresa como índice: 100 = año promedio.",
                   "If the harmonization works, **exceptional years must show up the same way** however fire is measured. Each "
                   "series (all of Argentina) is shown as an index: 100 = average year.", pt="Se a harmonização for boa, **os anos excepcionais têm que aparecer igual** não importa como o fogo seja medido. Cada série (toda a Argentina) é expressa como índice: 100 = ano médio."))
    ia = pd.DataFrame(MET["indice_anual"]).set_index("anio")
    nombres_ia = dict(zip(ia.columns, [NOMBRE_MEDIDA["focos"], NOMBRE_MEDIDA["dias"], NOMBRE_MEDIDA["frp"]]))
    fig = go.Figure()
    for c, (color, dash, simb) in zip(ia.columns, [(C_ARM, "solid", "circle"), (C_VIIRS, "dash", "square"), (C_MODIS, "dot", "diamond")]):
        fig.add_trace(go.Scatter(x=ia.index, y=ia[c], name=nombres_ia[c], mode="lines+markers", line=dict(color=color, dash=dash),
                                 marker=dict(symbol=simb, size=7)))
    fig.add_hline(y=100, line_color=CENIZA, line_width=1)
    fig.add_vline(x=2011.5, line_dash="dot", line_color=CENIZA,
                  annotation_text=tr("antes: MODIS armonizado · después: VIIRS", "before: harmonized MODIS · after: VIIRS", pt="antes: MODIS harmonizado · depois: VIIRS"))
    fig.update_layout(height=400, yaxis_title=tr("Índice (100 = año promedio)", "Index (100 = average year)", pt="Índice (100 = ano médio)"), hovermode="x unified")
    grafico(fig)
    describir(tr("Las tres líneas suben y bajan juntas. Años más activos según cada medida: ", "The three lines rise and fall together. "
                 "Most active years by measure: ", pt="As três linhas sobem e descem juntas. Anos mais ativos segundo cada medida: ") + "; ".join(f"{nombres_ia[c]}: {lista([str(a) for a in ia[c].nlargest(3).index])}" for c in ia) + ".")
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**" + tr("Correlación entre las tres medidas (2003–2025)", "Correlation between the three measures (2003–2025)", pt="Correlação entre as três medidas (2003–2025)") + "**")
        corr = pd.DataFrame(MET["correlacion_medidas"]).rename(index=nombres_ia, columns=nombres_ia)
        st.dataframe(corr.style.format("{:.2f}"), width="stretch")
    with v2:
        st.markdown("**" + tr("Los 5 años más activos según cada medida", "Top 5 most active years by measure", pt="Os 5 anos mais ativos segundo cada medida") + "**")
        st.dataframe(pd.DataFrame({nombres_ia[c]: ia[c].nlargest(5).index.values for c in ia}, index=range(1, 6)), width="stretch")
    ta = pd.Series(MET["terra_sobre_aqua"]).sort_index()
    if len(ta):
        ta.index = ta.index.astype(int)
        fig = px.line(x=ta.index, y=100 * ta.values, markers=True, labels={"x": tr("Año", "Year", pt="Ano"), "y": "Terra / Aqua (%)"},
                      color_discrete_sequence=[C_MODIS])
        fig.update_layout(height=290, title=tr("Limitación: el satélite Terra detecta cada vez menos que Aqua",
                                               "Limitation: the Terra satellite detects less and less than Aqua", pt="Limitação: o satélite Terra detecta cada vez menos que o Aqua"))
        grafico(fig)
        describir(tr(f"Terra detectaba alrededor del {100 * ta.loc[2012:2020].mean():.0f} % de lo que detecta Aqua hasta 2020 y bajó al "
                     f"{100 * ta.iloc[-1]:.0f} % en {ta.index[-1]}.",
                     f"Terra detected about {100 * ta.loc[2012:2020].mean():.0f}% of what Aqua detects until 2020 and fell to "
                     f"{100 * ta.iloc[-1]:.0f}% in {ta.index[-1]}.", pt=f"O Terra detectava cerca de {100 * ta.loc[2012:2020].mean():.0f} % do que o Aqua detecta até 2020 e caiu para {100 * ta.iloc[-1]:.0f} % em {ta.index[-1]}."))

    # Días sin dato y relleno con NOAA-20
    st.subheader(tr("Días en que el satélite no midió", "Days when the satellite did not measure", pt="Dias em que o satélite não mediu"))
    st.markdown(tr(
        "Un día sin focos puede ser «no hubo fuego» o «el satélite no midió». Si se cuenta como cero, la serie baja de forma "
        "falsa. Por eso comparamos los satélites entre sí, día por día: si uno detectó menos del 10 % de lo esperado según "
        "los otros, ese día se marca **sin dato**. En la serie publicada, los días sin dato de VIIRS S-NPP se completan con "
        "**VIIRS NOAA-20**, que lleva el mismo instrumento, calibrado contra S-NPP. Los modelos nunca se entrenan con "
        "datos completados.",
        "A day with no hotspots can mean «there was no fire» or «the satellite did not measure». Counting it as zero makes "
        "the series drop falsely. So we compare the satellites with each other, day by day: if one detected less than 10% "
        "of what the others suggest, that day is flagged as **no data**. In the published series, VIIRS S-NPP no-data days "
        "are filled with **VIIRS NOAA-20**, which carries the same instrument, calibrated against S-NPP. Models are never "
        "trained on filled data.", pt="Um dia sem focos pode ser «não houve fogo» ou «o satélite não mediu». Se for contado como zero, a série cai de forma falsa. Por isso comparamos os satélites entre si, dia a dia: se um detectou menos de 10 % do esperado segundo os outros, esse dia é marcado como **sem dado**. Na série publicada, os dias sem dado do VIIRS S-NPP são completados com o **VIIRS NOAA-20**, que leva o mesmo instrumento, calibrado contra o S-NPP. Os modelos nunca são treinados com dados completados."))
    h1, h2 = st.columns([1.1, 1])
    with h1:
        if len(HUECOS_VIIRS):
            hv = HUECOS_VIIRS.sort_values("fecha")
            grupo = (hv.fecha.diff() != pd.Timedelta("1D")).cumsum()
            per = hv.groupby(grupo).agg(desde=("fecha", "min"), hasta=("fecha", "max"), dias=("fecha", "size"),
                                         con=("relleno_con", "first"))
            per = per[per.dias >= 3].sort_values("desde")
            per["desde"], per["hasta"] = per.desde.dt.strftime("%d/%m/%Y"), per.hasta.dt.strftime("%d/%m/%Y")
            st.markdown("**" + tr(f"VIIRS S-NPP: {len(hv)} días sin dato. Períodos de 3 días o más:",
                                  f"VIIRS S-NPP: {len(hv)} no-data days. Periods of 3 days or more:", pt=f"VIIRS S-NPP: {len(hv)} dias sem dado. Períodos de 3 dias ou mais:") + "**")
            st.dataframe(per.rename(columns={"desde": tr("Desde", "From", pt="De"), "hasta": tr("Hasta", "To", pt="Até"),
                                             "dias": tr("Días", "Days", pt="Dias"), "con": tr("Completado con", "Filled with", pt="Completado com")}),
                         hide_index=True, width="stretch")
    with h2:
        val = MET.get("n20_validacion")
        if val:
            v = pd.DataFrame(val).set_index("medida")
            v.index = [NOMBRE_MEDIDA.get({"Focos": "focos", "Días con fuego": "dias"}.get(i, "frp"), i) for i in v.index]
            v = v.rename(columns={"S-NPP / NOAA-20 (país)": "S-NPP / NOAA-20",
                                  "error mensual medio %": tr("Error mensual medio %", "Mean monthly error %", pt="Erro mensal médio %"),
                                  "error total 2022–2025 %": tr("Error total 2022–2025 %", "Total error 2022–2025 %", pt="Erro total 2022–2025 %"),
                                  "correlación diaria": tr("Correlación diaria", "Daily correlation", pt="Correlação diária")})
            st.markdown("**" + tr("Qué tan bien reemplaza NOAA-20 a S-NPP (calibrado con 2018–2021, probado con 2022–2025)",
                                  "How well NOAA-20 replaces S-NPP (calibrated on 2018–2021, tested on 2022–2025)", pt="Quão bem o NOAA-20 substitui o S-NPP (calibrado com 2018–2021, testado com 2022–2025)") + "**")
            st.dataframe(v.style.format("{:.2f}"), width="stretch")
            describir(tr(f"NOAA-20 y S-NPP detectan casi lo mismo: la relación es {v.iloc[0, 0]:.2f} y el error mensual medio "
                         f"en focos es {v.iloc[0, 1]:.1f} %.",
                         f"NOAA-20 and S-NPP detect almost the same: the ratio is {v.iloc[0, 0]:.2f} and the mean monthly error "
                         f"for hotspots is {v.iloc[0, 1]:.1f}%.", pt=f"O NOAA-20 e o S-NPP detectam quase o mesmo: a relação é {v.iloc[0, 0]:.2f} e o erro mensal médio em focos é {v.iloc[0, 1]:.1f} %."))

# ================================================================ 7. datos
with tabs[6]:
    st.markdown(tr("Los datos mensuales que usa la página, para consultarlos o descargarlos. Para los **focos individuales** de un "
                   "período, usá **Búsqueda por fechas**.",
                   "The monthly data behind the page, to browse or download. For the **individual hotspots** of a period, use "
                   "**Date search**.", pt="Os dados mensais que a página usa, para consultar ou baixar. Para os **focos individuais** de um período, use **Busca por datas**."))
    COLUMNAS = {"region": tr("Zona", "Area", pt="Área"), "anio": tr("Año", "Year", pt="Ano"), "mes": tr("Mes", "Month", pt="Mês"), "fuente": tr("Fuente", "Source", pt="Fonte"),
                "focos_modis": tr("Focos MODIS", "MODIS hotspots", pt="Focos MODIS"), "focos_viirs": tr("Focos VIIRS", "VIIRS hotspots", pt="Focos VIIRS"),
                "focos_equiv": tr("Focos armonizados", "Harmonized hotspots", pt="Focos harmonizados"),
                "dias_modis": tr("Días con fuego MODIS", "MODIS fire-days", pt="Dias com fogo MODIS"), "dias_viirs": tr("Días con fuego VIIRS", "VIIRS fire-days", pt="Dias com fogo VIIRS"),
                "dias_equiv": tr("Días con fuego armonizados", "Harmonized fire-days", pt="Dias com fogo harmonizados"),
                "frp_modis": "FRP MODIS (MW)", "frp_viirs": "FRP VIIRS (MW)", "frp_equiv": tr("FRP armonizada (MW)", "Harmonized FRP (MW)", pt="FRP harmonizada (MW)"),
                "focos_relleno": tr("Focos completados (NOAA-20)", "Filled hotspots (NOAA-20)", pt="Focos completados (NOAA-20)"),
                "dias_sin_dato": tr("Días sin dato de S-NPP", "S-NPP no-data days", pt="Dias sem dado do S-NPP")}
    base = pd.concat([dr, serie], ignore_index=True) if region == AREA else serie
    opciones = ([AREA] if region == AREA else []) + [PAIS] + list(REGIONES) + [RESTO]
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        regs = st.multiselect(tr("Zonas", "Areas", pt="Áreas"), opciones, default=[region], format_func=zona_txt)
    with f2:
        a_desde, a_hasta = st.select_slider(tr("Años", "Years", pt="Anos"), options=anios, value=(anios[0], anios[-1]))
    with f3:
        nivel = st.radio(tr("Detalle", "Detail", pt="Detalhe"), ["mensual", "anual"],
                         format_func=lambda x: tr("Mensual", "Monthly", pt="Mensal") if x == "mensual" else tr("Anual", "Yearly", pt="Anual"))
    filtro = base[base.region.isin(regs) & base.anio.between(a_desde, a_hasta)]
    sel = filtro[[c for c in COLUMNAS if c in filtro.columns]]
    if nivel == "anual":
        num = [c for c in sel.columns if c.endswith(("_modis", "_viirs", "_equiv", "_relleno")) or c == "dias_sin_dato"]
        sel = sel.groupby(["region", "anio"], as_index=False).agg({**{c: lambda x: x.sum(min_count=1) for c in num}, "fuente": "first"})
        sel["meses"] = filtro.groupby(["region", "anio"]).focos_equiv.count().values
    sel = sel.assign(region=sel.region.map(zona_txt))
    tabla = sel.rename(columns={**COLUMNAS, "meses": tr("Meses con datos", "Months with data", pt="Meses com dados")})
    st.markdown(f"<p class='nota'>{miles(len(tabla))} {tr('filas · celdas vacías = sin dato (VIIRS antes de 2012, o meses que todavía no pasaron). «Focos VIIRS» es lo observado por S-NPP; «Focos completados» es lo que se agregó en los días en que S-NPP no midió; «Focos armonizados» es la suma.', 'rows · empty cells = no data (VIIRS before 2012, or months not yet elapsed). «VIIRS hotspots» is what S-NPP observed; «Filled hotspots» is what was added on days when S-NPP did not measure; «Harmonized hotspots» is the sum.', pt='linhas · células vazias = sem dado (VIIRS antes de 2012, ou meses que ainda não passaram). «Focos VIIRS» é o observado pelo S-NPP; «Focos completados» é o que foi acrescentado nos dias em que o S-NPP não mediu; «Focos harmonizados» é a soma.')}</p>",
                unsafe_allow_html=True)
    st.dataframe(tabla, width="stretch", hide_index=True, height=420,
                 column_config={c: st.column_config.NumberColumn(format="%.0f") for c in tabla.columns
                                if c not in (COLUMNAS["region"], COLUMNAS["fuente"], COLUMNAS["anio"], COLUMNAS["mes"])}
                 | {COLUMNAS["anio"]: st.column_config.NumberColumn(format="%d")})
    a_csv = lambda df: df.to_csv(index=False).encode("utf-8-sig")
    d1_, d2_, d3_, d4_ = st.columns(4)
    d1_.download_button(tr("Esta tabla (CSV)", "This table (CSV)", pt="Esta tabela (CSV)"), a_csv(tabla), "serie_filtrada.csv", "text/csv", width="stretch", icon=":material/download:")
    d2_.download_button(tr("Serie completa (CSV)", "Full series (CSV)", pt="Série completa (CSV)"), a_csv(serie), "serie_armonizada.csv", "text/csv", width="stretch", icon=":material/download:")
    d3_.download_button(tr("Grilla del mapa (CSV)", "Map grid (CSV)", pt="Grade do mapa (CSV)"), a_csv(grilla), "grilla_mensual.csv", "text/csv", width="stretch", icon=":material/download:")
    d4_.download_button(tr("Resultados de los métodos (JSON)", "Method results (JSON)", pt="Resultados dos métodos (JSON)"), json.dumps(MET, ensure_ascii=False, indent=2).encode("utf-8"),
                        "metricas.json", "application/json", width="stretch", icon=":material/download:")
    st.markdown(tr("**Fuentes originales:** [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/download/) (MODIS C6.1 y VIIRS Suomi NPP y NOAA-20 C2) · "
                   "[Catálogo de Focos de Calor de la CONAE](https://catalogos5.conae.gov.ar/catalogoFocos/).",
                   "**Original sources:** [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/download/) (MODIS C6.1 and VIIRS Suomi NPP and NOAA-20 C2) · "
                   "[CONAE Hotspot Catalog](https://catalogos5.conae.gov.ar/catalogoFocos/).", pt="**Fontes originais:** [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/download/) (MODIS C6.1 e VIIRS Suomi NPP e NOAA-20 C2) · [Catálogo de Focos de Calor da CONAE](https://catalogos5.conae.gov.ar/catalogoFocos/)."))

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
    n_huecos = MET.get("dias_sin_dato", {}).get("VIIRS S-NPP", 0)
    v20 = (MET.get("n20_validacion") or [{}])[0]
    r20, e20 = v20.get("S-NPP / NOAA-20 (país)", float("nan")), v20.get("error mensual medio %", float("nan"))
    top3 = lista([str(a) for a in ia.iloc[:, 0].nlargest(3).index])
    peor = res_f.loc[elegido_f, "Peor año (país) %"]
    gano_txt = {m: metodo_txt(MET["elegido"][m]).lower() for m in ("focos", "dias", "frp")}
    if VIDEO_URL:
        st.video(VIDEO_URL)
    if not EN and not PT:
        texto = f"""
<div class='metodo'>

Esta página es un **calendario armonizado de actividad de fuego** para Argentina. Responde la pregunta del desafío de la CONAE:
*¿esta zona, en estas fechas, tiene más o menos fuego que lo habitual?*, usando más de dos décadas de datos de satélites
distintos. Abajo está todo el recorrido, de los datos crudos a la página.

### 1. Los datos
Descargamos de **NASA FIRMS** (*Fire Information for Resource Management System*) todos los focos de calor de Argentina:

| Sensor | Satélites | Píxel | Período | Focos descargados | Uso |
|---|---|---|---|---|---|
| MODIS colección 6.1 | Terra (mañana) y Aqua (tarde) | ~1 km | 2003 – oct. 2026 | 1,6 millones | serie 2003–2011 y entrenamiento |
| VIIRS 375 m colección 2 | Suomi NPP (tarde) | 375 m | 2012 – oct. 2026 | 3,0 millones | base de la serie desde 2012 |
| VIIRS 375 m colección 2 | NOAA-20 (tarde) | 375 m | abr. 2018 – nov. 2026 | 1,8 millones | solo para completar días sin dato |

Cada registro es un **foco de calor**: un píxel con temperatura anómala, con fecha, hora, ubicación, satélite, confianza,
potencia radiativa (FRP) y tipo de fuente. Los datos definitivos (*archive*) se completan con los de tiempo casi real (*NRT*)
de los últimos meses, sin superponerse. Como referencia usamos también el **Catálogo de Focos de Calor de la CONAE**.

**Decisiones:**
- **MODIS desde 2003** y no desde 2000: recién desde 2003 funcionan Terra y Aqua todo el año.
- **Un solo VIIRS como base (Suomi NPP)**: es el único que existe desde 2012. Sumar satélites contaría el mismo fuego varias
  veces y haría subir los focos en 2018 y 2023 solo por haber más satélites. NOAA-20 se usa únicamente para los días en que
  S-NPP no midió.

### 2. Revisión y limpieza
Antes de armonizar revisamos los datos (análisis exploratorio): valores faltantes, valores imposibles, duplicados, puntos fuera
de Argentina y valores atípicos. No hubo faltantes ni valores imposibles, y ningún punto cae fuera del país. Los valores
extremos (por ejemplo, el récord de focos en el Gran Chaco en 2013 y 2020) son **eventos reales**: se conservan.
Criterios de limpieza, iguales para todos los satélites:
- Solo **incendios de vegetación** (`type = 0`): se descartan antorchas de gas, industrias, volcanes y fuentes en el mar. En
  los datos de tiempo casi real, que no traen esa columna, se descartan los focos donde el archivo definitivo ya registró
  fuentes fijas.
- Sin detecciones de **baja confianza**: MODIS con confianza ≥ 30 (de 0 a 100); VIIRS sin las marcadas como baja.

Quedan **4,1 millones de focos** de MODIS y VIIRS S-NPP.

### 3. Días en que el satélite no midió
Un día sin focos puede ser «no hubo fuego» o «el satélite no midió». Comparando los satélites día por día encontramos
**{n_huecos} días sin dato de VIIRS S-NPP**; el corte más largo fue del 26/07 al 10/08/2022. Si se cuentan como cero, la serie
baja de forma falsa. Por eso:
- **para entrenar y evaluar** los modelos solo se usan los días en que MODIS y S-NPP funcionaron los dos;
- **en la serie publicada** esos días se completan con **NOAA-20**, calibrado contra S-NPP (relación {miles(r20, 2)}, error
  mensual medio {miles(e20, 1)} % en años de prueba). Cada dato completado queda marcado en los archivos.

### 4. Tres formas de medir
Agrupamos los focos por **zona**, **año** y **mes**, con tres medidas:
- **Focos**: la cantidad de detecciones.
- **Días con fuego**: pares (celda de 25 km, día) con al menos un foco. Un incendio vale igual aunque el satélite lo divida en muchos píxeles.
- **Energía (FRP)**: la suma de la potencia radiativa, en megavatios.

Las **zonas** cubren todo el país: cinco regiones (rectángulos aproximados elegidos por su cantidad de focos o por el impacto
de sus incendios: Gran Chaco, Delta del Paraná, Sierras de Córdoba, Corrientes e Iberá, Patagonia andina) y el **resto del
país**. Argentina es la suma de las seis.

### 5. El problema, medido
En los años en que funcionan los dos sensores (2012–2025), VIIRS registra en promedio **{miles(FACTOR_PAIS, 1)} veces** más focos
que MODIS. Esa proporción es **estable de un año a otro** (entre 3,1 y 3,9), y es lo que permite convertir un sensor al otro.
Pero **cambia según la región**: va de {miles(fac.min(), 1)} en {zona_txt(fac.idxmin())} a {miles(fac.max(), 1)} en {zona_txt(fac.idxmax())}.

### 6. Métodos de armonización (machine learning)
Planteamos la armonización como un problema de **aprendizaje supervisado**: a partir de lo que midió MODIS en una región y un
mes, predecir lo que habría medido VIIRS.
- **Entrenamiento:** {MET['anios_entrenamiento']}. **Prueba:** {MET['anios_test']}, años que los modelos no ven al aprender.
- **Aplicación:** 2003–2011, donde solo hay MODIS.

Comparamos cuatro métodos, del más simple al más complejo: (1) **factor fijo por región**, VIIRS = k × MODIS; (2) **regresión
logarítmica por región y mes**, con estacionalidad y corrección del sesgo de volver de la escala logarítmica; (3) **mapeo de
cuantiles**; (4) **Random Forest** de 300 árboles.

### 7. Elección: con resultados, no por gusto
**Regla fijada antes de ver los resultados:** gana el método **más simple** que cumpla tres condiciones: error (MAE) a no más
de {MET['tolerancia']:.0%} del mejor, sesgo total entre −10 % y +10 %, y error del peor año menor al 25 %. Además medimos la
incertidumbre de cada error con **bootstrap** (intervalos de confianza del 95 %). Resultados de la prueba para focos:

| Método | MAE (focos por región y mes) | R² (log) | Sesgo total % | Peor año % |
|---|---|---|---|---|
{tabla_metodos}

Para focos ganó la **{gano_txt['focos']}**. El factor fijo quedó apenas por encima de la tolerancia, y la diferencia entre los
dos no es estadísticamente significativa: **la serie no depende del método elegido**. El Random Forest no justificó su
complejidad: la relación entre sensores es tan estable que un modelo complejo no tiene más para aprender. Para días con fuego
ganó el {gano_txt['dias']} y para energía, la {gano_txt['frp']}.

### 8. Serie armonizada y validación
- **2003–2011:** MODIS convertido a la escala de VIIRS con el método elegido para cada medida. En el **mapa** y en las
  **áreas propias** se usa el factor de la región, porque en una celda o en un área chica no hay datos suficientes para
  ajustar una regresión.
- **2012 en adelante:** el dato real de VIIRS S-NPP, con los días sin dato completados con NOAA-20.

Para validar, armamos la serie con las tres medidas por separado. Las tres marcan los mismos años excepcionales ({top3}) y se
parecen mucho entre sí (correlación mínima {miles(corr_min, 2)}).

### 9. Cómo funciona la página
- **Lo habitual** es la **mediana** histórica de cada mes, y el rango habitual va del percentil 10 al 90.
- **Área propia:** la página calcula el calendario de cualquier rectángulo con los focos individuales.
- **Búsqueda por fechas:** compara cualquier período con el mismo período de todos los años y permite descargar los focos.
- **Mapas** con capas de fondo (topografía con curvas de nivel, relieve submarino, satélite) y ciudades principales.
- **Accesibilidad:** al entrar, la página pregunta el idioma y si se prefiere letra grande o colores para daltonismo; todo se
  cambia con el botón 👁️. El resumen se puede escuchar en voz alta. Hay versión en español, inglés, portugués y un prototipo
  en guaraní.
- **Compartir y comentar:** el link de la página guarda la vista elegida, y cualquier persona puede dejar un comentario o
  reportar un dato raro.

### 10. Limitaciones
- Un foco de calor no es un incendio: un incendio grande genera muchos focos, y las nubes o el humo pueden ocultar fuegos.
- Los valores anteriores a 2012 son estimaciones, con un error de alrededor del ±{peor:.0f} % en el total anual.
- Antes de 2012 no se pueden detectar los días en que MODIS no midió, porque no hay otro satélite con qué compararlo.
- El satélite Terra está envejeciendo: detecta cada vez menos, lo que agrega error en los últimos años.
- «Resto del país» es una zona grande y variada que usa un solo modelo; dividirla en más regiones es trabajo futuro.
- Los últimos meses son datos provisorios de tiempo casi real.

### 11. Herramientas, fuentes y uso de Super Inteligencia (SI)
Python (pandas, NumPy, scikit-learn), Google Colab, Streamlit, Plotly, Folium/Leaflet y GitHub. Datos: NASA FIRMS; referencia:
CONAE. Mapas: © OpenStreetMap, OpenTopoMap y Esri. **Uso de Super Inteligencia (SI):** usamos Claude (Anthropic), una
herramienta de SI, para ayudar a escribir y depurar el código, y para redactar y traducir textos. Todos los resultados salen de
los datos reales de NASA FIRMS, y el equipo definió el alcance, revisó los métodos y tomó las decisiones.

</div>"""
    elif EN:
        texto = f"""
<div class='metodo'>

This page is a **harmonized burning activity calendar** for Argentina. It answers the question in CONAE's challenge:
*does this area, on these dates, have more or less fire than usual?*, using more than two decades of data from different
satellites. Below is the whole journey, from raw data to the page.

### 1. The data
We downloaded every hotspot in Argentina from **NASA FIRMS** (Fire Information for Resource Management System):

| Sensor | Satellites | Pixel | Period | Hotspots downloaded | Use |
|---|---|---|---|---|---|
| MODIS Collection 6.1 | Terra (morning) and Aqua (afternoon) | ~1 km | 2003 – Oct 2026 | 1.6 million | 2003–2011 series and training |
| VIIRS 375 m Collection 2 | Suomi NPP (afternoon) | 375 m | 2012 – Oct 2026 | 3.0 million | base of the series from 2012 |
| VIIRS 375 m Collection 2 | NOAA-20 (afternoon) | 375 m | Apr 2018 – Nov 2026 | 1.8 million | only to fill no-data days |

Each record is a **hotspot**: a pixel with an anomalous temperature, with date, time, location, satellite, confidence, fire
radiative power (FRP) and source type. Archive data are completed with near-real-time (NRT) data for the latest months, with
no overlap. We also used **CONAE's Hotspot Catalog** as a reference.

**Decisions:**
- **MODIS from 2003**, not 2000: only from 2003 do both Terra and Aqua run all year.
- **A single VIIRS as the base (Suomi NPP)**: it is the only one since 2012. Adding satellites would count the same fire
  several times and raise hotspot counts in 2018 and 2023 just because of more satellites. NOAA-20 is used only on days when
  S-NPP did not measure.

### 2. Review and cleaning
Before harmonizing we reviewed the data (exploratory analysis): missing values, impossible values, duplicates, points outside
Argentina and outliers. There were no missing or impossible values, and no point falls outside the country. Extreme values
(for example, the hotspot record in the Gran Chaco in 2013 and 2020) are **real events**: they are kept.
Cleaning criteria, the same for all satellites:
- **Vegetation fires only** (`type = 0`): gas flares, industries, volcanoes and offshore sources are removed. For
  near-real-time data, which lack that column, hotspots located where the archive already recorded static sources are removed.
- No **low-confidence** detections: MODIS confidence ≥ 30 (0 to 100); VIIRS without those flagged as low.

**4.1 million MODIS and VIIRS S-NPP hotspots** remain.

### 3. Days when the satellite did not measure
A day with no hotspots can mean «there was no fire» or «the satellite did not measure». Comparing satellites day by day we
found **{n_huecos} no-data days for VIIRS S-NPP**; the longest outage was from 26 July to 10 August 2022. Counting them as zero
makes the series drop falsely. So:
- **to train and evaluate** the models, only days when both MODIS and S-NPP worked are used;
- **in the published series** those days are filled with **NOAA-20**, calibrated against S-NPP (ratio {r20:.2f}, mean monthly
  error {e20:.1f}% in test years). Every filled value is flagged in the files.

### 4. Three ways to measure
We grouped hotspots by **area**, **year** and **month**, with three measures:
- **Hotspots**: the number of detections.
- **Fire-days**: (25 km cell, day) pairs with at least one hotspot. A fire counts the same even if the satellite splits it into many pixels.
- **Energy (FRP)**: the sum of fire radiative power, in megawatts.

The **areas** cover the whole country: five regions (approximate rectangles chosen for their number of hotspots or the
impact of their fires: Gran Chaco, Paraná Delta, Córdoba Hills, Corrientes & Iberá, Andean Patagonia) and the **rest of the
country**. Argentina is the sum of the six.

### 5. The problem, measured
In the years with both sensors (2012–2025), VIIRS records on average **{FACTOR_PAIS:.1f} times** more hotspots than MODIS.
That ratio is **stable from year to year** (between 3.1 and 3.9), which makes converting one sensor into the other possible.
But it **changes by region**: from {fac.min():.1f} in {zona_txt(fac.idxmin())} to {fac.max():.1f} in {zona_txt(fac.idxmax())}.

### 6. Harmonization methods (machine learning)
We framed harmonization as a **supervised learning** problem: from what MODIS measured in a region and month, predict what
VIIRS would have measured.
- **Training:** {MET['anios_entrenamiento']}. **Test:** {MET['anios_test']}, years the models never see while learning.
- **Application:** 2003–2011, MODIS only.

We compared four methods, from simplest to most complex: (1) **fixed factor per region**, VIIRS = k × MODIS; (2) **log
regression by region and month**, with seasonality and correction of the back-transformation bias; (3) **quantile mapping**;
(4) **Random Forest** with 300 trees.

### 7. Choosing: by results, not by taste
**Rule set before seeing the results:** the **simplest** method that meets three conditions wins: error (MAE) within
{MET['tolerancia']:.0%} of the best, total bias between −10% and +10%, and worst-year error below 25%. We also measured the
uncertainty of each error with a **bootstrap** (95% confidence intervals). Test results for hotspots:

| Method | MAE (hotspots per region and month) | R² (log) | Total bias % | Worst year % |
|---|---|---|---|---|
{tabla_metodos}

For hotspots the **{gano_txt['focos']}** won. The fixed factor was just above the tolerance, and the difference between the
two is not statistically significant: **the series does not depend on the chosen method**. The Random Forest did not justify
its complexity: the relationship between sensors is so stable that a complex model has nothing more to learn. For fire-days
the {gano_txt['dias']} won, and for energy, the {gano_txt['frp']}.

### 8. Harmonized series and validation
- **2003–2011:** MODIS converted to the VIIRS scale with the method chosen for each measure. The **map** and **custom areas**
  use the region's factor, because a single cell or a small area does not have enough data to fit a regression.
- **From 2012:** actual VIIRS S-NPP data, with no-data days filled with NOAA-20.

To validate, we built the series with the three measures separately. All three flag the same exceptional years ({top3}) and
closely track each other (minimum correlation {corr_min:.2f}).

### 9. How the page works
- **Usual** is each month's historical **median**; the usual range goes from the 10th to the 90th percentile.
- **Custom area:** the page builds the calendar for any rectangle from individual hotspots.
- **Date search:** compares any period with the same period in every year and lets you download the hotspots.
- **Maps** with background layers (topography with contour lines, seafloor relief, satellite) and main cities.
- **Accessibility:** on arrival, the page asks for the language and whether larger text or colorblind-safe colors are
  preferred; everything can be changed with the 👁️ button. The summary can be read aloud. The page is available in Spanish,
  English, Portuguese and a Guaraní prototype.
- **Share and comment:** the page link keeps the chosen view, and anyone can leave a comment or report odd data.

### 10. Limitations
- A hotspot is not a fire: a large fire produces many hotspots, and clouds or smoke can hide fires.
- Values before 2012 are estimates, with an error of about ±{peor:.0f}% in annual totals.
- Before 2012, days when MODIS did not measure cannot be detected, because there is no other satellite to compare with.
- The Terra satellite is aging: it detects less and less, which adds error in recent years.
- «Rest of the country» is a large, varied area that uses a single model; splitting it into more regions is future work.
- The latest months are provisional near-real-time data.

### 11. Tools, sources and use of Super Intelligence (SI)
Python (pandas, NumPy, scikit-learn), Google Colab, Streamlit, Plotly, Folium/Leaflet and GitHub. Data: NASA FIRMS; reference:
CONAE. Maps: © OpenStreetMap, OpenTopoMap and Esri. **Use of Super Intelligence (SI):** we used Claude (Anthropic), an SI tool,
to help write and debug the code, and to draft and translate texts. All results come from the real NASA FIRMS data, and the
team defined the scope, reviewed the methods and made the decisions.

</div>"""
    else:
        texto = f"""
<div class='metodo'>

Esta página é um **calendário harmonizado da atividade de fogo** para a Argentina. Ela responde à pergunta do desafio da CONAE:
*esta área, nestas datas, tem mais ou menos fogo do que o habitual?*, usando mais de duas décadas de dados de satélites
diferentes. Abaixo está todo o percurso, dos dados brutos até a página.

### 1. Os dados
Baixamos do **NASA FIRMS** (*Fire Information for Resource Management System*) todos os focos de calor da Argentina:

| Sensor | Satélites | Pixel | Período | Focos baixados | Uso |
|---|---|---|---|---|---|
| MODIS coleção 6.1 | Terra (manhã) e Aqua (tarde) | ~1 km | 2003 – out. 2026 | 1,6 milhão | série 2003–2011 e treinamento |
| VIIRS 375 m coleção 2 | Suomi NPP (tarde) | 375 m | 2012 – out. 2026 | 3,0 milhões | base da série desde 2012 |
| VIIRS 375 m coleção 2 | NOAA-20 (tarde) | 375 m | abr. 2018 – nov. 2026 | 1,8 milhão | só para completar dias sem dado |

Cada registro é um **foco de calor**: um pixel com temperatura anômala, com data, hora, localização, satélite, confiança,
potência radiativa (FRP) e tipo de fonte. Os dados definitivos (*archive*) são completados com os de tempo quase real (*NRT*)
dos últimos meses, sem sobreposição. Como referência, usamos também o **Catálogo de Focos de Calor da CONAE**.

**Decisões:**
- **MODIS desde 2003**, e não desde 2000: só a partir de 2003 Terra e Aqua funcionam o ano todo.
- **Um único VIIRS como base (Suomi NPP)**: é o único que existe desde 2012. Somar satélites contaria o mesmo fogo várias
  vezes e faria os focos subirem em 2018 e 2023 só por haver mais satélites. O NOAA-20 é usado apenas nos dias em que o
  S-NPP não mediu.

### 2. Revisão e limpeza
Antes de harmonizar, revisamos os dados (análise exploratória): valores faltantes, valores impossíveis, duplicados, pontos fora
da Argentina e valores atípicos. Não houve faltantes nem valores impossíveis, e nenhum ponto cai fora do país. Os valores
extremos (por exemplo, o recorde de focos no Gran Chaco em 2013 e 2020) são **eventos reais**: são mantidos.
Critérios de limpeza, iguais para todos os satélites:
- Só **incêndios de vegetação** (`type = 0`): são descartados queimadores de gás, indústrias, vulcões e fontes no mar. Nos
  dados de tempo quase real, que não trazem essa coluna, são descartados os focos onde o arquivo definitivo já registrou
  fontes fixas.
- Sem detecções de **baixa confiança**: MODIS com confiança ≥ 30 (de 0 a 100); VIIRS sem as marcadas como baixa.

Restam **4,1 milhões de focos** de MODIS e VIIRS S-NPP.

### 3. Dias em que o satélite não mediu
Um dia sem focos pode ser «não houve fogo» ou «o satélite não mediu». Comparando os satélites dia a dia, encontramos
**{n_huecos} dias sem dado do VIIRS S-NPP**; a interrupção mais longa foi de 26/07 a 10/08/2022. Se forem contados como zero, a
série cai de forma falsa. Por isso:
- **para treinar e avaliar** os modelos, usam-se só os dias em que MODIS e S-NPP funcionaram;
- **na série publicada**, esses dias são completados com o **NOAA-20**, calibrado contra o S-NPP (relação {miles(r20, 2)}, erro
  mensal médio de {miles(e20, 1)} % nos anos de teste). Cada dado completado fica marcado nos arquivos.

### 4. Três formas de medir
Agrupamos os focos por **área**, **ano** e **mês**, com três medidas:
- **Focos**: a quantidade de detecções.
- **Dias com fogo**: pares (célula de 25 km, dia) com pelo menos um foco. Um incêndio vale o mesmo mesmo que o satélite o divida em muitos pixels.
- **Energia (FRP)**: a soma da potência radiativa, em megawatts.

As **áreas** cobrem todo o país: cinco regiões (retângulos aproximados escolhidos pela quantidade de focos ou pelo impacto de
seus incêndios: Gran Chaco, Delta do Paraná, Serras de Córdoba, Corrientes e Iberá, Patagônia andina) e o **resto do país**.
A Argentina é a soma das seis.

### 5. O problema, medido
Nos anos em que os dois sensores funcionam (2012–2025), o VIIRS registra em média **{miles(FACTOR_PAIS, 1)} vezes** mais focos
que o MODIS. Essa proporção é **estável de um ano para outro** (entre 3,1 e 3,9), e é isso que permite converter um sensor no
outro. Mas ela **muda conforme a região**: vai de {miles(fac.min(), 1)} em {zona_txt(fac.idxmin())} a {miles(fac.max(), 1)} em {zona_txt(fac.idxmax())}.

### 6. Métodos de harmonização (machine learning)
Tratamos a harmonização como um problema de **aprendizado supervisionado**: a partir do que o MODIS mediu numa região e num
mês, prever o que o VIIRS teria medido.
- **Treinamento:** {MET['anios_entrenamiento']}. **Teste:** {MET['anios_test']}, anos que os modelos não veem ao aprender.
- **Aplicação:** 2003–2011, quando só existe MODIS.

Comparamos quatro métodos, do mais simples ao mais complexo: (1) **fator fixo por região**, VIIRS = k × MODIS; (2) **regressão
logarítmica por região e mês**, com sazonalidade e correção do viés de voltar da escala logarítmica; (3) **mapeamento de
quantis**; (4) **Random Forest** com 300 árvores.

### 7. Escolha: com resultados, não por gosto
**Regra fixada antes de ver os resultados:** vence o método **mais simples** que cumpra três condições: erro (MAE) no máximo
{MET['tolerancia']:.0%} acima do melhor, viés total entre −10 % e +10 % e erro do pior ano menor que 25 %. Também medimos a
incerteza de cada erro com **bootstrap** (intervalos de confiança de 95 %). Resultados do teste para focos:

| Método | MAE (focos por região e mês) | R² (log) | Viés total % | Pior ano % |
|---|---|---|---|---|
{tabla_metodos}

Para focos venceu a **{gano_txt['focos']}**. O fator fixo ficou logo acima da tolerância, e a diferença entre os dois não é
estatisticamente significativa: **a série não depende do método escolhido**. O Random Forest não justificou sua
complexidade: a relação entre os sensores é tão estável que um modelo complexo não tem mais o que aprender. Para dias com fogo
venceu o {gano_txt['dias']} e, para energia, a {gano_txt['frp']}.

### 8. Série harmonizada e validação
- **2003–2011:** MODIS convertido para a escala do VIIRS com o método escolhido para cada medida. No **mapa** e nas **áreas
  personalizadas** usa-se o fator da região, porque uma célula ou uma área pequena não tem dados suficientes para ajustar
  uma regressão.
- **De 2012 em diante:** o dado real do VIIRS S-NPP, com os dias sem dado completados com o NOAA-20.

Para validar, montamos a série com as três medidas separadamente. As três marcam os mesmos anos excepcionais ({top3}) e são
muito parecidas entre si (correlação mínima de {miles(corr_min, 2)}).

### 9. Como a página funciona
- **O habitual** é a **mediana** histórica de cada mês, e a faixa habitual vai do percentil 10 ao 90.
- **Área personalizada:** a página calcula o calendário de qualquer retângulo com os focos individuais.
- **Busca por datas:** compara qualquer período com o mesmo período de todos os anos e permite baixar os focos.
- **Mapas** com camadas de fundo (topografia com curvas de nível, relevo submarino, satélite) e cidades principais.
- **Acessibilidade:** ao entrar, a página pergunta o idioma e se a pessoa prefere letra grande ou cores para daltonismo; tudo
  pode ser mudado com o botão 👁️. O resumo pode ser ouvido em voz alta. A página está em espanhol, inglês, português e tem um
  protótipo em guarani.
- **Compartilhar e comentar:** o link da página guarda a visualização escolhida, e qualquer pessoa pode deixar um comentário
  ou relatar um dado estranho.

### 10. Limitações
- Um foco de calor não é um incêndio: um incêndio grande gera muitos focos, e nuvens ou fumaça podem esconder fogos.
- Os valores anteriores a 2012 são estimativas, com um erro de cerca de ±{peor:.0f} % no total anual.
- Antes de 2012 não é possível detectar os dias em que o MODIS não mediu, porque não há outro satélite para comparar.
- O satélite Terra está envelhecendo: detecta cada vez menos, o que acrescenta erro nos últimos anos.
- «Resto do país» é uma área grande e variada que usa um único modelo; dividi-la em mais regiões é trabalho futuro.
- Os últimos meses são dados provisórios de tempo quase real.

### 11. Ferramentas, fontes e uso de Superinteligência (SI)
Python (pandas, NumPy, scikit-learn), Google Colab, Streamlit, Plotly, Folium/Leaflet e GitHub. Dados: NASA FIRMS; referência:
CONAE. Mapas: © OpenStreetMap, OpenTopoMap e Esri. **Uso de Superinteligência (SI):** usamos o Claude (Anthropic), uma
ferramenta de SI, para ajudar a escrever e depurar o código, e para redigir e traduzir textos. Todos os resultados vêm dos
dados reais do NASA FIRMS, e a equipe definiu o escopo, revisou os métodos e tomou as decisões.

</div>"""
    st.markdown(texto, unsafe_allow_html=True)

# ================================================================ 9. comentarios
with tabs[8]:
    st.subheader("💬 " + tr("Comentarios de la comunidad", "Community comments", pt="Comentários da comunidade"))
    st.markdown(tr(
        "¿Viste algo raro en los datos, encontraste un error o tenés una sugerencia? Dejanos un comentario. El formulario se "
        "abre con la zona, el año y la medida que estás mirando, así podemos revisar exactamente lo mismo. Los comentarios se "
        "publican acá después de que el equipo los revisa.",
        "Did you see something odd in the data, find an error or have a suggestion? Leave us a comment. The form opens with "
        "the area, year and measure you are looking at, so we can check exactly the same thing. Comments are published here "
        "after the team reviews them.", pt="Viu algo estranho nos dados, encontrou um erro ou tem uma sugestão? Deixe um comentário. O formulário abre com a área, o ano e a medida que você está olhando, para podermos revisar exatamente o mesmo. Os comentários são publicados aqui depois que a equipe os revisa."))
    st.link_button("💬 " + tr("Comentar o reportar un dato raro", "Comment or report odd data", pt="Comentar ou relatar um dado estranho"),
                   FORM_URL + urllib.parse.quote(contexto_form), type="primary")
    st.caption(tr("No incluyas datos personales (teléfono, dirección, email): tu comentario puede publicarse.",
                  "Do not include personal data (phone, address, email): your comment may be published.", pt="Não inclua dados pessoais (telefone, endereço, e-mail): seu comentário pode ser publicado."))

    @st.cache_data(ttl=300, show_spinner=False)
    def leer_comentarios():
        try:
            return pd.read_csv(COMENTARIOS_CSV)
        except Exception:
            return None

    com = leer_comentarios()
    if com is None:
        st.info(tr("No se pudieron cargar los comentarios en este momento. Probá de nuevo en unos minutos.",
                   "Comments could not be loaded right now. Please try again in a few minutes.", pt="Não foi possível carregar os comentários agora. Tente de novo em alguns minutos."))
    elif com.dropna(how="all").empty:
        st.info(tr("Todavía no hay comentarios publicados. ¡Podés ser la primera persona!",
                   "There are no published comments yet. You can be the first!", pt="Ainda não há comentários publicados. Você pode ser a primeira pessoa!"))
    else:
        com = com.dropna(how="all").iloc[::-1]          # los más nuevos primero
        for _, r in com.iterrows():
            v = [("" if pd.isna(x) else str(x)) for x in r.tolist()] + [""] * 6
            fecha_c, tipo_c, texto_c, nombre_c, contexto_c, resp_c = v[:6]
            with st.container(border=True):
                st.markdown(f"**{tipo_c}** · <span class='nota'>{fecha_c}</span>", unsafe_allow_html=True)
                st.markdown(texto_c)
                if nombre_c:
                    st.caption("— " + nombre_c)
                if resp_c:
                    st.markdown("**" + tr("Respuesta del equipo", "Team's reply", pt="Resposta da equipe") + ":** " + resp_c)
    st.success(tr("¡Muchas gracias por su aporte! Recordemos que la ciencia se hace y se descubre juntos, colectivamente.",
                  "Thank you very much for your contribution! Let's remember that science is built and discovered together, "
                  "collectively.", pt="Muito obrigado pela sua contribuição! Lembremos que a ciência se faz e se descobre juntos, coletivamente.", gn="¡Aguyje! ¡Muchas gracias por su aporte! Recordemos que la ciencia se hace y se descubre "
                  "juntos, colectivamente."))
