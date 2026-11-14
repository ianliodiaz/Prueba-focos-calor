"""
Focos de calor en Argentina — NASA Space Apps Challenge 2026 (desafío CONAE)

Calendario armonizado de actividad de fuego (MODIS + VIIRS).
Para correrla en la computadora:  streamlit run app.py
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

VERSION_APP = "v5"
RAIZ = Path(__file__).parent
# Los resultados se buscan en todo el repositorio, por si quedaron dentro de otra carpeta
_encontrados = sorted(RAIZ.rglob("serie_armonizada.csv"), key=lambda p: len(p.parts))
CARPETA = _encontrados[0].parent if _encontrados else RAIZ / "datos"
ARCHIVOS_FOCOS = sorted(RAIZ.rglob("focos_*.parquet"))
# Contornos de países para el mapa: desde la propia app si están; si no, desde el sitio de Plotly
TOPOJSON_LOCAL = (RAIZ / "static" / "topojson" / "world_50m.json").exists()
CONFIG_MAPA = {"topojsonURL": "app/static/topojson/"} if TOPOJSON_LOCAL else {}

PAIS = "Argentina (todo el país)"
AREA = "Área propia"
INICIO_VIIRS = pd.Timestamp("2012-01-20")
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
MESES_LARGOS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
                "octubre", "noviembre", "diciembre"]
UNIDAD = {"focos": "focos", "dias": "días-celda", "frp": "MW"}
EXPLICA_MEDIDA = {
    "focos": "Cantidad de detecciones satelitales (píxeles con fuego).",
    "dias": "Cantidad de pares (celda de 25 km, día) con al menos un foco. Un incendio cuenta igual aunque "
            "el satélite lo divida en muchos píxeles.",
    "frp": "Suma de la potencia radiativa del fuego, en megavatios. Mide energía, no cantidad de píxeles.",
}

st.set_page_config(page_title="Focos de calor en Argentina", page_icon="🔥", layout="wide")


def miles(x, dec=0):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def abreviar(x):
    if not np.isfinite(x):
        return ""
    return f"{x / 1000:.0f}k" if x >= 10000 else (f"{x / 1000:.1f}k".replace(".", ",") if x >= 1000 else f"{x:.0f}")


def lista(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1] if items else ""


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
NOMBRES = MET["nombres"]
FECHA_FIN = pd.Timestamp(MET["fecha_fin"])
FACTORES = MET["factores_focos"]
FACTOR_PAIS = MET.get("factor_pais_focos", 3.35)
anios = sorted(serie.anio.unique())


def region_que_contiene(lat, lon):
    for nombre, (la0, la1, lo0, lo1) in REGIONES.items():
        if la0 <= lat <= la1 and lo0 <= lon <= lo1:
            return nombre
    return "Resto del país"


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
        k, origen = comun.VIIRS.sum() / comun.MODIS.sum(), "propio del área (calculado con sus focos de 2012–2025)"
    else:
        reg = region_que_contiene((la0 + la1) / 2, (lo0 + lo1) / 2)
        k, origen = FACTORES.get(reg, FACTOR_PAIS), f"de la región «{reg}» (el área tiene pocos focos para calcular uno propio)"
    d = m.reset_index().rename(columns={"MODIS": "focos_modis", "VIIRS": "focos_viirs"})
    d["focos_viirs"] = np.where(d.anio >= 2012, d.focos_viirs, np.nan)
    d["focos_equiv"] = np.where(d.anio >= 2012, d.focos_viirs, d.focos_modis * k)
    futuro = d.anio * 12 + d.mes > FECHA_FIN.year * 12 + FECHA_FIN.month
    d.loc[futuro, ["focos_modis", "focos_viirs", "focos_equiv"]] = np.nan
    d["region"], d["fuente"] = AREA, np.where(d.anio >= 2012, "VIIRS", "MODIS armonizado")
    # Grilla fina (~5 km) para el mapa del área
    c = 0.05
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


# ================================================================ barra lateral
st.sidebar.title("🔥 Focos de calor")
st.sidebar.caption(f"Calendario armonizado MODIS + VIIRS · NASA FIRMS · Argentina · {VERSION_APP}")

opciones_zona = ["Región", "Área propia (coordenadas)"] if focos is not None else ["Región"]
tipo_zona = st.sidebar.radio("Zona", opciones_zona, horizontal=True)
bbox = None
if tipo_zona == "Región":
    region = st.sidebar.selectbox("Región", [PAIS] + list(REGIONES) + ["Resto del país"])
else:
    region = AREA
    st.sidebar.caption("Ingresá los límites del área en grados decimales (sur y oeste son negativos). "
                       "Podés copiarlos de Google Maps con clic derecho sobre el mapa.")
    s1, s2 = st.sidebar.columns(2)
    lat_n = s1.number_input("Latitud norte", -56.0, -21.0, -30.5, 0.1, format="%.2f")
    lat_s = s2.number_input("Latitud sur", -56.0, -21.0, -32.5, 0.1, format="%.2f")
    lon_o = s1.number_input("Longitud oeste", -74.0, -53.0, -65.2, 0.1, format="%.2f")
    lon_e = s2.number_input("Longitud este", -74.0, -53.0, -64.0, 0.1, format="%.2f")
    bbox = (min(lat_n, lat_s), max(lat_n, lat_s), min(lon_o, lon_e), max(lon_o, lon_e))
    if bbox[0] == bbox[1] or bbox[2] == bbox[3]:
        st.sidebar.error("El área no puede tener ancho o alto cero.")
        st.stop()

if region == AREA:
    medida = "focos"
    st.sidebar.caption("En un área propia la actividad se mide en **focos**.")
else:
    nombre_medida = st.sidebar.radio("Medir la actividad de fuego por", list(NOMBRES.values()))
    medida = {v: k for k, v in NOMBRES.items()}[nombre_medida]
    st.sidebar.caption(EXPLICA_MEDIDA[medida])
anio = st.sidebar.selectbox("Año a analizar", anios[::-1])
previos = [a for a in anios if a < anio] or anios
ref_desde, ref_hasta = st.sidebar.select_slider("Período de referencia (lo habitual)", options=previos,
                                                value=(previos[0], previos[-1]))
st.sidebar.divider()
st.sidebar.caption(f"Datos hasta el {FECHA_FIN:%d/%m/%Y} (los últimos meses son provisorios). "
                   "Antes de 2012 los valores son MODIS armonizado (estimados); desde 2012, VIIRS.")

# ================================================================ zona elegida
if region == AREA:
    dr, g_zona, K_ZONA, ORIGEN_K, n_focos_area = serie_area(bbox)
else:
    dr = serie[serie.region == region].copy()
    g_zona = grilla if region == PAIS else grilla[grilla.region == region]
    K_ZONA = FACTOR_PAIS if region == PAIS else FACTORES.get(region, FACTOR_PAIS)
    ORIGEN_K = "promedio de Argentina" if region == PAIS else f"de la región «{region}»"

col = f"{medida}_equiv"
ref = dr[dr.anio.between(ref_desde, ref_hasta)]
clima = ref.groupby("mes")[col].agg(mediana="median", p10=lambda s: s.quantile(.1), p90=lambda s: s.quantile(.9))
clima = clima.reindex(range(1, 13))
actual = dr[dr.anio == anio].set_index("mes")[col].reindex(range(1, 13))
con_dato = actual.dropna().index

# ================================================================ cabecera + accesibilidad
cab1, cab2 = st.columns([10, 1.4])
with cab2:
    with st.popover("👁️", help="Accesibilidad: colores para daltonismo y descripciones de los gráficos"):
        st.toggle("Modo accesible", key="acc")
        st.caption("Cambia los colores por paletas aptas para daltonismo, agrega marcas y números que no "
                   "dependen del color, y muestra una descripción en texto debajo de cada gráfico.")
ACC = st.session_state.get("acc", False)

# Paletas: normales, y aptas para daltonismo (Okabe-Ito, Cividis, Púrpura-Naranja)
if ACC:
    SEQ, DIV = "Cividis", "PuOr_r"
    C_NORMAL, C_ALTO, C_MODIS, C_VIIRS, C_ARM = "#9DB7D5", "#08306B", "#0072B2", "#E69F00", "#000000"
    C_BANDA = "rgba(0,0,0,0.12)"
else:
    SEQ, DIV = "YlOrRd", "RdBu_r"
    C_NORMAL, C_ALTO, C_MODIS, C_VIIRS, C_ARM = "#E8833A", "#C0392B", "#2E6DB4", "#E8833A", "#1F2933"
    C_BANDA = "rgba(93,109,126,0.18)"


def describir(texto):
    "Descripción en texto de un gráfico: solo se muestra en modo accesible."
    if ACC:
        st.info("📝 **Descripción:** " + texto)


with cab1:
    st.title(region if region != AREA else
             f"Área propia ({bbox[0]:.2f}° a {bbox[1]:.2f}° lat · {bbox[2]:.2f}° a {bbox[3]:.2f}° lon)")
    st.caption(f"{NOMBRES[medida]} · año {anio} comparado con la mediana de {ref_desde}–{ref_hasta}")

if region == AREA and n_focos_area == 0:
    st.warning("No hay focos registrados en esta área. Prueben con un área más grande o en otra zona.")
    st.stop()

total, esperado = actual.sum(), clima.mediana.reindex(con_dato).sum()
dif = 100 * (total - esperado) / esperado if esperado else np.nan
meses_altos = [MESES_LARGOS[m - 1] for m in con_dato if actual[m] > clima.p90[m]]
pico = int(actual.idxmax()) if actual.notna().any() and actual.max() > 0 else None
pico_normal = int(clima.mediana.idxmax()) if clima.mediana.notna().any() and clima.mediana.max() > 0 else None

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{NOMBRES[medida]} en {anio}", miles(total),
          f"{dif:+.0f} % respecto de lo habitual" if np.isfinite(dif) else None, delta_color="inverse")
c2.metric("Lo habitual para esos meses", miles(esperado), help="Suma de las medianas históricas de esos meses")
c3.metric(f"Mes pico en {anio}", MESES[pico - 1] if pico else "—",
          f"habitualmente: {MESES[pico_normal - 1]}" if pico_normal else None, delta_color="off")
c4.metric("Meses por encima de lo habitual", f"{len(meses_altos)} de {len(con_dato)}",
          help="Meses que superaron al 90 % de los años del período de referencia")
if len(con_dato) < 12:
    st.info(f"{anio} tiene datos hasta {MESES_LARGOS[int(con_dato.max()) - 1]}: se compara solo con esos meses.")
if region == AREA:
    st.caption(f"Área con {miles(n_focos_area)} focos. Para convertir MODIS a la escala de VIIRS se usó un factor "
               f"de {K_ZONA:.2f}, {ORIGEN_K}.")

tabs = st.tabs(["📊 Comparación con lo habitual", "🗓️ Calendario", "🔎 Búsqueda por fechas", "🗺️ Mapa",
                "🛰️ Armonización", "✅ Validación", "📥 Datos", "ℹ️ Acerca de"])

# ================================================================ 1. comparación
with tabs[0]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=MESES, y=clima.p90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=MESES, y=clima.p10, mode="lines", fill="tonexty", line=dict(width=0),
                             fillcolor=C_BANDA, name="Rango habitual (10–90 %)"))
    fig.add_trace(go.Scatter(x=MESES, y=clima.mediana, name="Mediana histórica", mode="lines+markers",
                             line=dict(color="#4D4D4D", dash="dash"), marker=dict(symbol="diamond", size=7)))
    alto = (actual > clima.p90).values
    fig.add_trace(go.Bar(x=MESES, y=actual, name=str(anio), marker_color=np.where(alto, C_ALTO, C_NORMAL),
                         marker_pattern_shape=np.where(alto, "/", "") if ACC else None, opacity=.92,
                         text=np.where(alto, "▲ alto", "") if ACC else None, textposition="outside"))
    fig.update_layout(height=430, yaxis_title=f"{NOMBRES[medida]} por mes", legend=dict(orientation="h", y=1.12),
                      margin=dict(t=30, b=10), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption(("Barras oscuras con rayas y «▲ alto»" if ACC else "Barras rojas") +
               ": meses por encima del rango habitual. Sirve para ver si la temporada viene más intensa o si empezó antes.")
    if pico:
        texto = (f"En {anio}, el mes con más actividad fue {MESES_LARGOS[pico - 1]} con {miles(actual[pico])} "
                 f"{UNIDAD[medida]}; habitualmente el pico es en {MESES_LARGOS[pico_normal - 1]}. ")
        texto += (f"Estuvieron por encima del rango habitual: {lista(meses_altos)}. " if meses_altos
                  else "Ningún mes superó el rango habitual. ")
        bajos = [MESES_LARGOS[m - 1] for m in con_dato if actual[m] < clima.p10[m]]
        if bajos:
            texto += f"Por debajo del rango habitual: {lista(bajos)}. "
        texto += (f"En total, el año va {abs(dif):.0f} % {'por encima' if dif > 0 else 'por debajo'} de lo habitual."
                  if np.isfinite(dif) else "")
        describir(texto)

    anual = dr.groupby("anio")[col].sum(min_count=1).reset_index()
    anual["tipo"] = np.where(anual.anio == anio, "Año elegido", np.where(anual.anio < 2012, "MODIS armonizado", "VIIRS"))
    fig2 = px.bar(anual, x="anio", y=col, color="tipo", pattern_shape="tipo" if ACC else None,
                  color_discrete_map={"Año elegido": C_ALTO, "VIIRS": "#9AA5B1", "MODIS armonizado": "#C9D1D9"},
                  labels={"anio": "Año", col: f"{NOMBRES[medida]} por año", "tipo": ""})
    mediana_anual = ref.groupby("anio")[col].sum().median()
    fig2.add_hline(y=mediana_anual, line_dash="dash", annotation_text="Mediana de referencia")
    fig2.update_layout(height=320, margin=dict(t=20, b=10), legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig2, width="stretch")
    completos = anual[anual.anio <= 2025].dropna()
    if len(completos):
        top = completos.nlargest(3, col)
        puesto = int((completos[col] > anual.set_index("anio")[col].get(anio, np.nan)).sum()) + 1
        describir(f"Los tres años con más actividad son {lista([f'{a} ({miles(v)})' for a, v in zip(top.anio, top[col])])}. "
                  f"La mediana anual de referencia es {miles(mediana_anual)}. "
                  + (f"{anio} ocupa el puesto {puesto} de {len(completos)} años completos." if anio <= 2025 else
                     f"{anio} está incompleto, por eso su barra es más baja."))

# ================================================================ 2. calendario
with tabs[1]:
    st.markdown("Cada casillero es un mes. **Más intenso el color = más actividad.** Muestra en qué meses se "
                "concentra el fuego en la zona y qué años fueron excepcionales.")
    modo = st.radio("Mostrar", ["Valores", "Comparado con lo habitual del mes"], horizontal=True)
    tabla = dr.pivot_table(index="anio", columns="mes", values=col).reindex(columns=range(1, 13))
    if modo == "Valores":
        z, escala, texto_cb, zmid = np.log10(tabla + 1), SEQ, f"log₁₀({UNIDAD[medida]}+1)", None
        hover = f"%{{y}} %{{x}}: %{{customdata:,.0f}} {UNIDAD[medida]}<extra></extra>"
    else:
        z = np.log2((tabla + 1) / (clima.mediana + 1).values)
        escala, texto_cb, zmid = DIV, "veces lo habitual (log₂)", 0
        hover = "%{y} %{x}: %{customdata:,.0f} · %{z:+.1f} (log₂ respecto de la mediana)<extra></extra>"
    textos = tabla.map(abreviar).values if ACC else None
    fig = go.Figure(go.Heatmap(z=z.values, x=MESES, y=tabla.index.astype(str), colorscale=escala, zmid=zmid,
                               customdata=tabla.values, hovertemplate=hover, colorbar=dict(title=texto_cb),
                               text=textos, texttemplate="%{text}" if ACC else None, textfont=dict(size=9)))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=max(440, 22 * len(tabla)), margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    if modo == "Valores":
        st.caption(("Más claro (amarillo) = más actividad; azul oscuro = poca." if ACC else "Más oscuro = más actividad.")
                   + " Hasta 2011 los valores son MODIS armonizado; desde 2012, VIIRS.")
    else:
        st.caption(("Naranja = más que la mediana de ese mes; violeta = menos." if ACC else
                    "Rojo = más que la mediana de ese mes; azul = menos.") + " Hasta 2011, MODIS armonizado.")
    orden = clima.mediana.sort_values(ascending=False)
    temporada = sorted(orden.index[:3])
    extremos = tabla.stack().nlargest(3)
    describir(f"Los meses con más actividad habitual son {lista([MESES_LARGOS[m - 1] for m in temporada])}. "
              f"Los tres meses más extremos de toda la serie fueron "
              f"{lista([f'{MESES_LARGOS[m - 1]} de {a} ({miles(v)})' for (a, m), v in extremos.items()])}. "
              "Con el modo accesible, cada casillero muestra su valor (k = miles).")

# ================================================================ 3. búsqueda por fechas
with tabs[2]:
    if focos is None:
        st.warning("Esta sección necesita los archivos de focos individuales (carpeta `focos`). "
                   "Súbanlos al repositorio para activarla.")
    else:
        st.markdown("Elegí un **rango de fechas** para ver los focos detectados en la zona, compararlos con el "
                    "**mismo período de los años anteriores** y descargarlos.")
        fmin, fmax = dt.date(2003, 1, 1), FECHA_FIN.date()
        b1, b2 = st.columns([2, 3])
        with b1:
            rango = st.date_input("Rango de fechas", value=(fmax - dt.timedelta(days=59), fmax),
                                  min_value=fmin, max_value=fmax, format="DD/MM/YYYY")
        if not isinstance(rango, (tuple, list)) or len(rango) < 2:
            st.info("Elegí también la fecha final del rango.")
        else:
            d0, d1 = pd.Timestamp(rango[0]), pd.Timestamp(rango[1])
            if region == AREA:
                la0, la1, lo0, lo1 = bbox
                pz = focos[focos.lat.between(la0, la1) & focos.lon.between(lo0, lo1)]
            elif region == PAIS:
                pz = focos
            else:
                pz = focos[focos.region == region]
            sel = pz[pz.fecha.between(d0, d1)]

            # Focos por día y sensor, y su versión armonizada (antes del inicio de VIIRS: MODIS × factor)
            diario = pz.groupby(["fecha", "sensor"], observed=True).size().unstack("sensor")
            diario = diario.reindex(pd.date_range("2003-01-01", FECHA_FIN), fill_value=0).fillna(0)
            for s in ["MODIS", "VIIRS"]:
                if s not in diario:
                    diario[s] = 0
            diario["equiv"] = np.where(diario.index >= INICIO_VIIRS, diario.VIIRS, diario.MODIS * K_ZONA)
            acum = diario.equiv.cumsum()

            def suma(a, b):
                return float(acum.loc[b] - (acum.loc[a - pd.Timedelta(days=1)] if a > acum.index[0] else 0))

            equiv_sel = suma(d0, d1)
            largo = (d1 - d0).days
            comp = []
            if largo <= 365:
                for y in anios:
                    try:
                        a = d0.replace(year=y)
                    except ValueError:                       # 29 de febrero
                        a = d0.replace(year=y, day=28)
                    b = a + pd.Timedelta(days=largo)
                    if a >= pd.Timestamp("2003-01-01") and b <= FECHA_FIN:
                        comp.append({"Inicio": a.date(), "Fin": b.date(), "Año": a.year, "Focos armonizados": suma(a, b)})
            comp = pd.DataFrame(comp)
            otros = comp[comp["Año"] != d0.year]["Focos armonizados"] if len(comp) else pd.Series(dtype=float)
            med = float(otros.median()) if len(otros) else np.nan
            dif_b = 100 * (equiv_sel - med) / med if med else np.nan

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Focos armonizados", miles(equiv_sel),
                      f"{dif_b:+.0f} % respecto de la mediana" if np.isfinite(dif_b) else None, delta_color="inverse")
            m2.metric("Mediana del mismo período", miles(med), help="En los demás años con datos")
            m3.metric("Detectados por VIIRS", miles((sel.sensor == "VIIRS").sum()))
            m4.metric("Detectados por MODIS", miles((sel.sensor == "MODIS").sum()))

            if len(comp) > 1:
                comp["tipo"] = np.where(comp["Año"] == d0.year, "Período elegido", "Otros años")
                fig = px.bar(comp, x="Año", y="Focos armonizados", color="tipo",
                             pattern_shape="tipo" if ACC else None,
                             color_discrete_map={"Período elegido": C_ALTO, "Otros años": "#9AA5B1"},
                             labels={"tipo": ""},
                             title=f"Mismo período ({d0:%d/%m} – {d1:%d/%m}) en cada año")
                fig.add_hline(y=med, line_dash="dash", annotation_text="Mediana")
                fig.update_layout(height=340, margin=dict(t=50, b=10), legend=dict(orientation="h", y=1.12))
                st.plotly_chart(fig, width="stretch")
                rk = comp.sort_values("Focos armonizados", ascending=False).reset_index(drop=True)
                puesto = int(rk.index[rk["Año"] == d0.year][0]) + 1 if (rk["Año"] == d0.year).any() else None
                describir(f"Entre el {d0:%d/%m/%Y} y el {d1:%d/%m/%Y} hubo {miles(equiv_sel)} focos armonizados. "
                          f"La mediana del mismo período en los otros años es {miles(med)}. "
                          + (f"Es el puesto {puesto} de {len(rk)} años. " if puesto else "")
                          + f"Los años con más focos en ese período fueron "
                            f"{lista([f'{a} ({miles(v)})' for a, v in zip(rk['Año'][:3], rk['Focos armonizados'][:3])])}.")
            elif largo > 365:
                st.caption("La comparación con otros años está disponible para rangos de hasta un año.")

            dd = diario.loc[d0:d1].reset_index().rename(columns={"index": "Fecha"})
            fig = go.Figure(go.Bar(x=dd.Fecha, y=dd.equiv, marker_color=C_NORMAL, name="Focos armonizados"))
            fig.update_layout(height=260, margin=dict(t=40, b=10), title="Focos armonizados por día",
                              yaxis_title="Focos")
            st.plotly_chart(fig, width="stretch")
            if len(dd) and dd.equiv.max() > 0:
                dmax = dd.loc[dd.equiv.idxmax()]
                describir(f"El día con más focos del período fue el {dmax.Fecha:%d/%m/%Y}, con {miles(dmax.equiv)}.")

            # Mapa de los focos del período (VIIRS: círculo; MODIS: diamante)
            muestra = sel.sample(min(len(sel), 30000), random_state=0) if len(sel) else sel
            if len(muestra):
                fig = go.Figure()
                for s, simbolo in [("VIIRS", "circle"), ("MODIS", "diamond")]:
                    p = muestra[muestra.sensor == s]
                    if len(p):
                        fig.add_trace(go.Scattergeo(
                            lat=p.lat, lon=p.lon, mode="markers", name=s,
                            customdata=np.c_[p.fecha.dt.strftime("%d/%m/%Y"), p.frp],
                            hovertemplate=s + " · %{customdata[0]} · FRP %{customdata[1]:.1f} MW<extra></extra>",
                            marker=dict(symbol=simbolo, size=5 if s == "VIIRS" else 7, opacity=.75,
                                        color=np.log10(p.frp.clip(lower=1)), colorscale=SEQ, cmin=0, cmax=3,
                                        colorbar=dict(title="FRP (MW)", tickvals=[0, 1, 2, 3],
                                                      ticktext=["1", "10", "100", "1000"], x=1.0) if s == "VIIRS" else None,
                                        line=dict(width=.5, color="#333") if s == "MODIS" else dict(width=0))))
                la_r = [sel.lat.min() - .5, sel.lat.max() + .5]
                lo_r = [sel.lon.min() - .5, sel.lon.max() + .5]
                fig.update_geos(projection_type="mercator", lataxis_range=la_r, lonaxis_range=lo_r,
                                showcountries=True, countrycolor="#9AA5B1", showsubunits=True, subunitcolor="#D5DBE0",
                                showland=True, landcolor="#F4F6F7", showocean=True, oceancolor="#EAF2F8", resolution=50)
                fig.update_layout(height=520, margin=dict(t=10, b=0, l=0, r=0), legend=dict(orientation="h", y=1.05))
                st.plotly_chart(fig, width="stretch", config=CONFIG_MAPA)
                st.caption("Cada punto es un foco: círculo = VIIRS, diamante = MODIS. El color indica la potencia "
                           "del fuego (FRP)." + (" Se muestra una muestra de 30.000 focos." if len(sel) > 30000 else ""))

            @st.cache_data
            def csv_focos(clave, _df):
                out = _df.rename(columns={"fecha": "fecha", "hora_utc": "hora_utc_HHMM", "lat": "latitud",
                                         "lon": "longitud", "frp": "frp_MW"})
                out["fecha"] = out["fecha"].dt.strftime("%Y-%m-%d")
                return out.to_csv(index=False).encode("utf-8-sig")

            st.markdown("**Descargas de esta búsqueda**")
            e1, e2, e3 = st.columns(3)
            clave = (region, bbox, str(d0.date()), str(d1.date()))
            e1.download_button(f"⬇️ Focos detectados ({miles(len(sel))}, CSV)", csv_focos(clave, sel),
                               f"focos_{d0:%Y%m%d}_{d1:%Y%m%d}.csv", "text/csv", width="stretch",
                               disabled=len(sel) == 0)
            e2.download_button("⬇️ Focos por día (CSV)",
                               dd.rename(columns={"MODIS": "focos_MODIS", "VIIRS": "focos_VIIRS",
                                                  "equiv": "focos_armonizados"}).to_csv(index=False).encode("utf-8-sig"),
                               f"focos_por_dia_{d0:%Y%m%d}_{d1:%Y%m%d}.csv", "text/csv", width="stretch")
            e3.download_button("⬇️ Comparación con otros años (CSV)",
                               comp.drop(columns="tipo", errors="ignore").to_csv(index=False).encode("utf-8-sig"),
                               f"comparacion_{d0:%m%d}_{d1:%m%d}.csv", "text/csv", width="stretch",
                               disabled=len(comp) == 0)
            st.caption("Los focos detectados incluyen fecha, hora (UTC), latitud, longitud, sensor, satélite, "
                       "confianza, potencia (FRP) y región. Solo incendios de vegetación, sin detecciones de baja confianza.")

# ================================================================ 4. mapa
with tabs[3]:
    ca, cb = st.columns([1, 3])
    with ca:
        modo_mapa = st.radio("Mostrar", ["Año completo", "Un mes", "Diferencia con lo habitual"])
        mes_mapa = (st.selectbox("Mes", range(1, 13), format_func=lambda m: MESES[m - 1])
                    if modo_mapa == "Un mes" else None)
    g = g_zona
    if modo_mapa == "Diferencia con lo habitual":
        anual_celda = (g[g.anio.between(ref_desde, ref_hasta)].groupby(["lat_c", "lon_c", "anio"]).focos_equiv.sum()
                       .unstack(fill_value=0).reindex(columns=range(ref_desde, ref_hasta + 1), fill_value=0))
        normal = anual_celda.median(axis=1)
        este = g[g.anio == anio].groupby(["lat_c", "lon_c"]).focos_equiv.sum()
        m = pd.concat([este.rename("actual"), normal.rename("normal")], axis=1).fillna(0).reset_index()
        m["valor"] = m.actual - m.normal
        m = m[m.valor.abs() >= 1].sort_values("valor", key=abs)
        lim = float(np.nanpercentile(m.valor.abs(), 95)) if len(m) else 1.0
        color = dict(color=m.valor, colorscale=DIV, cmin=-lim, cmax=lim, colorbar=dict(title="Diferencia<br>(focos)"))
        hover = "Este año: %{customdata[0]:,.0f}<br>Habitual (mediana): %{customdata[1]:,.0f}<extra></extra>"
        custom = m[["actual", "normal"]].values
        nota = (("Naranja" if ACC else "Rojo") + ": más focos que lo habitual. " + ("Violeta" if ACC else "Azul") +
                ": menos. Lo habitual es la mediana de cada celda en el período de referencia.")
    else:
        sel_m = g[g.anio == anio]
        if mes_mapa:
            sel_m = sel_m[sel_m.mes == mes_mapa]
        m = sel_m.groupby(["lat_c", "lon_c"]).focos_equiv.sum().reset_index().sort_values("focos_equiv")
        m["valor"] = m.focos_equiv
        tope = float(np.log10(m.valor.max() + 1)) if len(m) else 1.0
        color = dict(color=np.log10(m.valor + 1), colorscale=SEQ, cmin=0, cmax=max(tope, 1),
                     colorbar=dict(title="Focos", tickvals=list(range(int(tope) + 1)),
                                   ticktext=[miles(10 ** k) for k in range(int(tope) + 1)]))
        hover = "%{customdata[0]:,.0f} focos<extra></extra>"
        custom = m[["valor"]].values
        nota = ("Focos por celda (en escala VIIRS). Escala de colores logarítmica: " +
                ("más claro (amarillo) = más focos." if ACC else "más oscuro = más focos."))
    if region == AREA:
        la0, la1, lo0, lo1 = bbox
        lat_rango, lon_rango = [la0 - .3, la1 + .3], [lo0 - .3, lo1 + .3]
        tam = 7 if max(la1 - la0, lo1 - lo0) < 3 else 4
    elif region in REGIONES:
        la0, la1, lo0, lo1 = REGIONES[region]
        lat_rango, lon_rango, tam = [la0 - 1, la1 + 1], [lo0 - 1, lo1 + 1], 9
    else:
        lat_rango, lon_rango, tam = [-56, -21], [-75, -53], 4.5
    fig = go.Figure(go.Scattergeo(lat=m.lat_c, lon=m.lon_c, mode="markers", customdata=custom, hovertemplate=hover,
                                  showlegend=False, marker=dict(symbol="square", size=tam, opacity=.9,
                                                                line=dict(width=0), **color)))
    contornos = dict(REGIONES)
    if region == AREA:
        contornos = {AREA: bbox}
    for nombre, (la0, la1, lo0, lo1) in contornos.items():
        fig.add_trace(go.Scattergeo(lat=[la0, la0, la1, la1, la0], lon=[lo0, lo1, lo1, lo0, lo0], mode="lines",
                                    line=dict(width=1.5, color="#333"), name=nombre, hoverinfo="name", showlegend=False))
    fig.update_geos(projection_type="mercator", lataxis_range=lat_rango, lonaxis_range=lon_rango,
                    showcountries=True, countrycolor="#9AA5B1", showsubunits=True, subunitcolor="#D5DBE0",
                    showland=True, landcolor="#F4F6F7", showocean=True, oceancolor="#EAF2F8",
                    showcoastlines=True, coastlinecolor="#9AA5B1", resolution=50)
    fig.update_layout(height=680, margin=dict(t=0, b=0, l=0, r=0))
    with cb:
        st.plotly_chart(fig, width="stretch", config=CONFIG_MAPA)
    with ca:
        st.caption(nota)
        st.caption("Los rectángulos marcan las regiones analizadas." if region != AREA else "El rectángulo marca el área elegida.")
        if len(m):
            if modo_mapa == "Diferencia con lo habitual":
                suben, bajan = int((m.valor > 0).sum()), int((m.valor < 0).sum())
                t = m.sort_values("valor").iloc[-1]
                describir(f"{suben} celdas tuvieron más focos que lo habitual y {bajan} tuvieron menos. La mayor suba "
                          f"está cerca de {t.lat_c:.2f}° de latitud y {t.lon_c:.2f}° de longitud "
                          f"({region_que_contiene(t.lat_c, t.lon_c)}), con {miles(t.valor)} focos más que lo habitual.")
            else:
                top = m.nlargest(3, "valor")
                describir("Las celdas con más focos están cerca de " + lista(
                    [f"{r.lat_c:.2f}°, {r.lon_c:.2f}° ({region_que_contiene(r.lat_c, r.lon_c)}, {miles(r.valor)} focos)"
                     for r in top.itertuples()]) + ".")

# ================================================================ 5. armonización
with tabs[4]:
    st.subheader("El problema")
    st.markdown("MODIS (satélites Terra y Aqua, píxel de ~1 km) y VIIRS (satélite Suomi NPP, píxel de 375 m) miden "
                "distinto: VIIRS divide cada incendio en más píxeles y ve fuegos más chicos. Si se pegan las dos "
                "series, parece que el fuego se multiplicó en 2012. **Armonizar** es convertir MODIS a la escala "
                "de VIIRS para tener una serie continua desde 2003, en la que ya no importa qué sensor detectó cada foco.")
    an = dr.groupby("anio")[[f"{medida}_modis", f"{medida}_viirs", col]].sum(min_count=1).reset_index()
    an = an[an.anio <= 2025]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=an.anio, y=an[f"{medida}_modis"], name="MODIS (original)", mode="lines+markers",
                             line=dict(color=C_MODIS, dash="dash"), marker=dict(symbol="circle", size=7)))
    fig.add_trace(go.Scatter(x=an.anio, y=an[f"{medida}_viirs"], name="VIIRS (original)", mode="lines+markers",
                             line=dict(color=C_VIIRS, dash="dot"), marker=dict(symbol="square", size=7)))
    fig.add_trace(go.Scatter(x=an.anio, y=an[col], name="Serie armonizada", mode="lines",
                             line=dict(color=C_ARM, width=3.5)))
    fig.add_vline(x=2011.5, line_dash="dot", annotation_text="empieza VIIRS")
    fig.update_layout(height=400, yaxis_title=f"{NOMBRES[medida]} por año", legend=dict(orientation="h", y=1.12),
                      margin=dict(t=30, b=10), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption("MODIS: línea a rayas con círculos. VIIRS: línea punteada con cuadrados. Serie armonizada: línea gruesa continua.")
    describir(f"La línea de MODIS queda siempre por debajo de la de VIIRS. Para pasar MODIS a la escala de VIIRS en esta "
              f"zona se multiplica por {K_ZONA:.2f} ({ORIGEN_K}). Por eso, antes de 2012 la serie armonizada está por "
              "encima de MODIS, y desde 2012 coincide con VIIRS.")

    st.subheader("Cómo elegimos el método")
    st.markdown(f"Probamos 4 métodos con los datos de toda Argentina. Cada uno aprendió con **{MET['anios_entrenamiento']}** "
                f"y se evaluó con **{MET['anios_test']}**, años que no había visto. **Regla:** gana el método más simple "
                f"cuyo error no supere en más de {MET['tolerancia']:.0%} al mejor.")
    res = pd.DataFrame(MET["resultados"][medida]).set_index("método")
    elegido = MET["elegido"][medida]
    st.success(MET["justificacion"][medida])
    cc1, cc2 = st.columns(2)
    with cc1:
        b = res.reset_index()
        fig = go.Figure(go.Bar(x=b["MAE"], y=b["método"], orientation="h",
                               marker_color=np.where(b["método"] == elegido, C_ALTO, "#9AA5B1"),
                               marker_pattern_shape=np.where(b["método"] == elegido, "/", "") if ACC else None,
                               text=[("✔ " if m == elegido else "") + miles(v, 1) for m, v in zip(b["método"], b["MAE"])],
                               textposition="auto"))
        fig.update_layout(height=300, xaxis_title=f"Error medio por región y mes ({UNIDAD[medida]})",
                          yaxis=dict(autorange="reversed"), margin=dict(t=30, b=10), title="Error (MAE): menor es mejor")
        st.plotly_chart(fig, width="stretch")
    with cc2:
        d = pd.DataFrame(MET["anual_test"][medida])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.anio, y=d["Real"], name="VIIRS real", line=dict(color=C_ARM, width=4)))
        simbolos = ["circle", "square", "diamond", "triangle-up"]
        for i, c in enumerate([c for c in d.columns if c not in ("anio", "Real")]):
            fig.add_trace(go.Scatter(x=d.anio, y=d[c], name=c, mode="lines+markers",
                                     marker=dict(symbol=simbolos[i % 4], size=8),
                                     line=dict(dash="solid" if c == elegido else "dot", width=3 if c == elegido else 1.5)))
        fig.update_layout(height=300, title="Test: total de Argentina por año", margin=dict(t=30, b=10),
                          xaxis=dict(dtick=1), legend=dict(font=dict(size=10)))
        st.plotly_chart(fig, width="stretch")
    describir(f"El método elegido es «{elegido}», marcado con ✔. "
              + " ".join(f"{m}: error medio {miles(v, 1)}." for m, v in zip(res.index, res["MAE"])))
    st.dataframe(res.style.format({"MAE": "{:,.1f}", "R² (log)": "{:.3f}", "Sesgo total %": "{:+.1f}",
                                   "Peor año (país) %": "{:.1f}"})
                 .apply(lambda r: ["background-color: #FDEDEC; font-weight: bold" if r.name == elegido else ""] * len(r),
                        axis=1), width="stretch")
    st.caption("MAE: error promedio por región y mes. R² (log): qué parte de la variación explica (1 = perfecto). "
               "Sesgo total: si se pasa (+) o se queda corto (−) en el total de 2022–2025. "
               "Peor año: el mayor error en el total anual de Argentina. La fila del método elegido va en negrita.")
    with st.expander("Ver las fórmulas de cada método"):
        st.markdown("**1. Factor fijo por región** — una regla de tres:")
        st.latex(r"k_r = \frac{\sum V}{\sum M} \qquad \hat V = k_r \cdot M")
        st.markdown("**2. Regresión logarítmica por región y mes** (con corrección de sesgo κ):")
        st.latex(r"\log(1+\hat V) = a + b\,\log(1+M) + c\,\sin\tfrac{2\pi\,mes}{12} + d\,\cos\tfrac{2\pi\,mes}{12}")
        st.markdown("**3. Mapeo de cuantiles** — se emparejan percentiles:")
        st.latex(r"\hat V = F_V^{-1}\big(F_M(M)\big)")
        st.markdown("**4. Random Forest** — promedio de 300 árboles de decisión con MODIS, mes y región.")
    if medida == "focos":
        fac = pd.Series(FACTORES).sort_values()
        fig = px.bar(x=fac.values, y=fac.index, orientation="h", text=[f"{v:.1f}×" for v in fac.values],
                     labels={"x": "VIIRS / MODIS", "y": ""}, color_discrete_sequence=[C_VIIRS])
        fig.update_layout(height=280, margin=dict(t=40, b=10), title="Cuántas veces más focos detecta VIIRS que MODIS, por región")
        st.plotly_chart(fig, width="stretch")
        st.caption("Donde hay incendios grandes (Chaco) los dos sensores los ven y la diferencia es menor. "
                   "Donde hay muchos fuegos chicos (Delta, Patagonia), VIIRS detecta muchos que MODIS no ve.")

# ================================================================ 6. validación
with tabs[5]:
    st.markdown("Si la armonización es buena, **los años excepcionales tienen que aparecer igual** sin importar "
                "cómo se mida el fuego. Cada serie (toda Argentina) se expresa como índice: 100 = año promedio.")
    ia = pd.DataFrame(MET["indice_anual"]).set_index("anio")
    fig = go.Figure()
    estilos = [(C_ARM, "solid", "circle"), (C_VIIRS, "dash", "square"), (C_MODIS, "dot", "diamond")]
    for c, (color, dash, simb) in zip(ia.columns, estilos):
        fig.add_trace(go.Scatter(x=ia.index, y=ia[c], name=c, mode="lines+markers", line=dict(color=color, dash=dash),
                                 marker=dict(symbol=simb, size=7)))
    fig.add_hline(y=100, line_color="gray", line_width=1)
    fig.add_vline(x=2011.5, line_dash="dot", annotation_text="antes: MODIS armonizado · después: VIIRS")
    fig.update_layout(height=400, yaxis_title="Índice (100 = año promedio)", legend=dict(orientation="h", y=1.12),
                      margin=dict(t=30, b=10), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    tops = {c: list(ia[c].nlargest(3).index) for c in ia}
    describir("Las tres líneas suben y bajan juntas. Los tres años más activos según cada medida: " +
              "; ".join(f"{c}: {lista([str(a) for a in v])}" for c, v in tops.items()) + ".")
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Correlación entre las tres medidas** (2003–2025)")
        st.dataframe(pd.DataFrame(MET["correlacion_medidas"]).style.format("{:.2f}"), width="stretch")
    with v2:
        st.markdown("**Los 5 años más activos según cada medida**")
        st.dataframe(pd.DataFrame({c: ia[c].nlargest(5).index.values for c in ia}, index=range(1, 6)), width="stretch")
    ta = pd.Series(MET["terra_sobre_aqua"]).sort_index()
    if len(ta):
        ta.index = ta.index.astype(int)
        fig = px.line(x=ta.index, y=100 * ta.values, markers=True, labels={"x": "Año", "y": "Terra / Aqua (%)"},
                      color_discrete_sequence=[C_MODIS])
        fig.update_layout(height=280, margin=dict(t=40, b=10), title="Limitación: el satélite Terra detecta cada vez menos que Aqua")
        st.plotly_chart(fig, width="stretch")
        st.caption("Coincide con que la órbita de Terra se fue corriendo y pasa cada vez más temprano, cuando hay "
                   "menos fuego. Explica parte del error de los modelos en los últimos años.")
        describir(f"Terra detectaba alrededor del {100 * ta.loc[2012:2020].mean():.0f} % de lo que detecta Aqua hasta 2020, "
                  f"y bajó al {100 * ta.iloc[-1]:.0f} % en {ta.index[-1]}.")

# ================================================================ 7. datos
with tabs[6]:
    st.markdown("Los datos mensuales que usa esta página, para consultarlos o descargarlos. Para descargar los "
                "**focos individuales** de un período, usá la pestaña **🔎 Búsqueda por fechas**.")
    COLUMNAS = {
        "region": "Región", "anio": "Año", "mes": "Mes", "fuente": "Fuente",
        "focos_modis": "Focos MODIS", "focos_viirs": "Focos VIIRS", "focos_equiv": "Focos armonizados",
        "dias_modis": "Días con fuego MODIS", "dias_viirs": "Días con fuego VIIRS", "dias_equiv": "Días con fuego armonizados",
        "frp_modis": "FRP MODIS (MW)", "frp_viirs": "FRP VIIRS (MW)", "frp_equiv": "FRP armonizada (MW)",
    }
    base = pd.concat([dr, serie], ignore_index=True) if region == AREA else serie
    opciones = ([AREA] if region == AREA else []) + [PAIS] + list(REGIONES) + ["Resto del país"]
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        regs = st.multiselect("Zonas", opciones, default=[region])
    with f2:
        a_desde, a_hasta = st.select_slider("Años", options=anios, value=(anios[0], anios[-1]))
    with f3:
        nivel = st.radio("Detalle", ["Mensual", "Anual"])
    filtro = base[base.region.isin(regs) & base.anio.between(a_desde, a_hasta)]
    sel = filtro[[c for c in COLUMNAS if c in filtro.columns]]
    if nivel == "Anual":
        num = [c for c in sel.columns if c.endswith(("_modis", "_viirs", "_equiv"))]
        sel = sel.groupby(["region", "anio"], as_index=False).agg(
            {**{c: lambda x: x.sum(min_count=1) for c in num}, "fuente": "first"})
        sel["meses con datos"] = filtro.groupby(["region", "anio"]).focos_equiv.count().values
    tabla = sel.rename(columns=COLUMNAS)
    st.caption(f"{miles(len(tabla))} filas · «None» = sin dato (VIIRS antes de 2012, o meses que todavía no pasaron).")
    st.dataframe(tabla, width="stretch", hide_index=True, height=420,
                 column_config={c: st.column_config.NumberColumn(format="%.0f") for c in tabla.columns
                                if c not in ("Región", "Fuente", "Año", "Mes", "meses con datos")}
                 | {"Año": st.column_config.NumberColumn(format="%d")})
    a_csv = lambda df: df.to_csv(index=False).encode("utf-8-sig")
    d1_, d2_, d3_, d4_ = st.columns(4)
    d1_.download_button("⬇️ Esta tabla (CSV)", a_csv(tabla), "serie_filtrada.csv", "text/csv", width="stretch")
    d2_.download_button("⬇️ Serie completa (CSV)", a_csv(serie), "serie_armonizada.csv", "text/csv", width="stretch")
    d3_.download_button("⬇️ Grilla del mapa (CSV)", a_csv(grilla), "grilla_mensual.csv", "text/csv", width="stretch")
    d4_.download_button("⬇️ Resultados de los métodos (JSON)", json.dumps(MET, ensure_ascii=False, indent=2).encode("utf-8"),
                        "metricas.json", "application/json", width="stretch")
    with st.expander("Qué significa cada columna"):
        st.markdown("""
| Columna | Qué es |
|---|---|
| Región, Año, Mes | Dónde y cuándo |
| Fuente | `VIIRS` (medición real, desde 2012) o `MODIS armonizado` (estimado, 2003–2011) |
| Focos MODIS / VIIRS | Cantidad de detecciones de cada sensor (solo vegetación, sin baja confianza) |
| Focos armonizados | La serie continua en escala VIIRS: la que usan los gráficos |
| Días con fuego | Pares (celda de ~25 km, día) con al menos un foco |
| FRP | Suma de la potencia radiativa del fuego, en megavatios |
| meses con datos | En la vista anual: cuántos meses tiene el año (2026 está incompleto) |
""")
    st.markdown("**Fuentes originales:** [NASA FIRMS — descarga de datos](https://firms.modaps.eosdis.nasa.gov/download/) "
                "(MODIS C6.1 y VIIRS Suomi NPP C2) · [Catálogo de Focos de Calor de la CONAE]"
                "(https://catalogos5.conae.gov.ar/catalogoFocos/). El código para reproducir todo el procesamiento "
                "está en el repositorio del proyecto.")

# ================================================================ 8. acerca de
with tabs[7]:
    st.markdown("""
### Sobre este proyecto
Calendario armonizado de actividad de fuego desarrollado para el **NASA Space Apps Challenge 2026**, desafío
*Harmonization of MODIS and VIIRS Hot Spots*, propuesto por la **CONAE**.

**Datos:** NASA FIRMS. MODIS colección 6.1 (Terra + Aqua, desde 2003) y VIIRS 375 m colección 2 (Suomi NPP,
desde 2012), para Argentina. Se usan solo incendios de vegetación (sin antorchas de gas, industrias ni volcanes)
y se descartan las detecciones de baja confianza.

**Lo habitual** se define con la **mediana** histórica de cada mes (y el rango entre los percentiles 10 y 90),
que no se deja arrastrar por años extremos.

**Accesibilidad:** el botón 👁️ arriba a la derecha activa colores aptos para daltonismo (paletas Cividis,
Púrpura-Naranja y Okabe-Ito), marcas que no dependen del color y descripciones en texto de cada gráfico.

**Limitaciones**
- Un foco de calor no es un incendio: un incendio grande genera muchos focos, y las nubes o el humo pueden ocultar fuegos.
- Los valores anteriores a 2012 son estimaciones, no mediciones de VIIRS.
- Los datos de los últimos meses son de "tiempo casi real" (provisorios).
- Terra está envejeciendo y detecta menos fuego en los últimos años.

**Para quién sirve:** brigadas, organismos de emergencia, científicos y administradores de tierras que necesitan
saber si una temporada viene peor que lo habitual, cuándo y dónde.
""")
