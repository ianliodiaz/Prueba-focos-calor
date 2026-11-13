"""
Focos de calor en Argentina — NASA Space Apps Challenge 2026 (desafío CONAE)

Página web interactiva con la serie armonizada MODIS–VIIRS.
Para correrla en la computadora:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).parent
# Los resultados pueden estar en la carpeta "datos" o sueltos al lado de app.py: se usa la que los tenga
# (los busca en todo el repositorio, por si quedaron dentro de otra carpeta)
_encontrados = sorted(RAIZ.rglob("serie_armonizada.csv"), key=lambda p: len(p.parts))
CARPETA = _encontrados[0].parent if _encontrados else RAIZ / "datos"
# Contornos de países para el mapa: desde la propia app si están; si no, desde el sitio de Plotly
TOPOJSON_LOCAL = (RAIZ / "static" / "topojson" / "world_50m.json").exists()
CONFIG_MAPA = {"topojsonURL": "app/static/topojson/"} if TOPOJSON_LOCAL else {}
PAIS = "Argentina (todo el país)"
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
COLOR_MODIS, COLOR_VIIRS, COLOR_ARM, COLOR_ALERTA = "#2E6DB4", "#E8833A", "#1F2933", "#C0392B"
UNIDAD = {"focos": "focos", "dias": "días-celda", "frp": "MW"}
EXPLICA_MEDIDA = {
    "focos": "Cantidad de detecciones satelitales (píxeles con fuego).",
    "dias": "Cantidad de pares (celda de 25 km, día) con al menos un foco. Un incendio cuenta igual aunque "
            "el satélite lo divida en muchos píxeles.",
    "frp": "Suma de la potencia radiativa del fuego, en megavatios. Mide energía, no cantidad de píxeles.",
}

st.set_page_config(page_title="Focos de calor en Argentina", page_icon="🔥", layout="wide")
VERSION_APP = "v3"


def miles(x, dec=0):
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data
def cargar():
    serie = pd.read_csv(CARPETA / "serie_armonizada.csv")
    grilla = pd.read_csv(CARPETA / "grilla_mensual.csv")
    regiones = json.loads((CARPETA / "regiones.json").read_text(encoding="utf-8"))
    metricas = json.loads((CARPETA / "metricas.json").read_text(encoding="utf-8"))
    return serie, grilla, regiones, metricas


NECESARIOS = ["serie_armonizada.csv", "grilla_mensual.csv", "regiones.json", "metricas.json"]
faltan = [f for f in NECESARIOS if not (CARPETA / f).exists()]
if faltan:
    st.caption(f"Versión de la página: {VERSION_APP}")
    hay = sorted(p.relative_to(RAIZ).as_posix() for p in RAIZ.rglob("*") if p.is_file() and ".git" not in p.parts)
    st.error("No encuentro estos archivos de resultados: " + ", ".join(faltan) + ". "
             "Súbanlos al repositorio de GitHub (los 4 archivos de la carpeta `salida` de Drive), "
             "todos juntos en la misma carpeta.")
    st.write("Archivos que sí encuentro en el repositorio:", hay[:50])
    st.stop()

serie, grilla, REGIONES, MET = cargar()
NOMBRES = MET["nombres"]
anios = sorted(serie.anio.unique())

# ---------------------------------------------------------------- barra lateral
st.sidebar.title("🔥 Focos de calor")
st.sidebar.caption(f"Serie armonizada MODIS + VIIRS · NASA FIRMS · Argentina · {VERSION_APP}")
region = st.sidebar.selectbox("Región", [PAIS] + list(REGIONES) + ["Resto del país"])
nombre_medida = st.sidebar.radio("Medir la actividad de fuego por", list(NOMBRES.values()))
medida = {v: k for k, v in NOMBRES.items()}[nombre_medida]
st.sidebar.caption(EXPLICA_MEDIDA[medida])
anio = st.sidebar.selectbox("Año a analizar", anios[::-1])
previos = [a for a in anios if a < anio] or anios
ref_desde, ref_hasta = st.sidebar.select_slider("Período de referencia (lo \"normal\")", options=previos,
                                                value=(previos[0], previos[-1]))
st.sidebar.divider()
st.sidebar.caption(f"Datos hasta el {pd.Timestamp(MET['fecha_fin']):%d/%m/%Y}. "
                   "Antes de 2012 los valores son MODIS armonizado (estimados); desde 2012, VIIRS.")

col = f"{medida}_equiv"
dr = serie[serie.region == region].copy()
ref = dr[dr.anio.between(ref_desde, ref_hasta)]
clima = ref.groupby("mes")[col].agg(promedio="mean", p10=lambda s: s.quantile(.1), p90=lambda s: s.quantile(.9))
clima = clima.reindex(range(1, 13))
actual = dr[dr.anio == anio].set_index("mes")[col].reindex(range(1, 13))
con_dato = actual.dropna().index

# ---------------------------------------------------------------- cabecera
st.title(f"{region}")
st.caption(f"{NOMBRES[medida]} · año {anio} comparado con {ref_desde}–{ref_hasta}")

total, esperado = actual.sum(), clima.promedio.reindex(con_dato).sum()
dif = 100 * (total - esperado) / esperado if esperado else np.nan
altos = int((actual.reindex(con_dato) > clima.p90.reindex(con_dato)).sum())
pico = MESES[int(actual.idxmax()) - 1] if actual.notna().any() and actual.max() > 0 else "—"
mes_normal_pico = MESES[int(clima.promedio.idxmax()) - 1] if clima.promedio.notna().any() else "—"

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{NOMBRES[medida]} en {anio}", miles(total),
          f"{dif:+.0f} % respecto de lo normal" if np.isfinite(dif) else None, delta_color="inverse")
c2.metric("Lo esperado para esos meses", miles(esperado))
c3.metric(f"Mes pico en {anio}", pico, f"normalmente: {mes_normal_pico}", delta_color="off")
c4.metric("Meses por encima de lo habitual", f"{altos} de {len(con_dato)}",
          help="Meses que superaron al 90 % de los años del período de referencia")
if len(con_dato) < 12:
    st.info(f"{anio} tiene datos hasta {MESES[int(con_dato.max()) - 1]}: se compara solo con esos meses.")

tabs = st.tabs(["📊 Comparación con lo normal", "🗓️ Calendario", "🗺️ Mapa", "🛰️ Armonización",
                "✅ Validación", "ℹ️ Acerca de"])

# ---------------------------------------------------------------- 1. comparación
with tabs[0]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=MESES, y=clima.p90, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=MESES, y=clima.p10, mode="lines", fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(93,109,126,0.18)", name="Rango habitual (10–90 %)"))
    fig.add_trace(go.Scatter(x=MESES, y=clima.promedio, name="Promedio histórico",
                             line=dict(color="#5D6D7E", dash="dash")))
    colores = np.where(actual > clima.p90, COLOR_ALERTA, COLOR_VIIRS)
    fig.add_trace(go.Bar(x=MESES, y=actual, name=str(anio), marker_color=colores, opacity=.9))
    fig.update_layout(height=420, yaxis_title=f"{NOMBRES[medida]} por mes", legend=dict(orientation="h", y=1.12),
                      margin=dict(t=30, b=10), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption("Barras rojas: meses por encima del rango habitual. Sirve para ver si la temporada viene "
               "más intensa de lo normal o si empezó antes.")

    anual = dr.groupby("anio")[col].sum(min_count=1).reset_index()
    anual["tipo"] = np.where(anual.anio == anio, "Año elegido",
                             np.where(anual.anio < 2012, "MODIS armonizado", "VIIRS"))
    fig2 = px.bar(anual, x="anio", y=col, color="tipo",
                  color_discrete_map={"Año elegido": COLOR_ALERTA, "VIIRS": "#9AA5B1", "MODIS armonizado": "#C9D1D9"},
                  labels={"anio": "Año", col: f"{NOMBRES[medida]} por año", "tipo": ""})
    fig2.add_hline(y=ref.groupby("anio")[col].sum().mean(), line_dash="dash", annotation_text="Promedio de referencia")
    fig2.update_layout(height=320, margin=dict(t=20, b=10), legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig2, width="stretch")

# ---------------------------------------------------------------- 2. calendario
with tabs[1]:
    st.markdown("Cada casillero es un mes. **Más oscuro = más actividad.** Permite ver cuándo es la temporada "
                "de incendios en la región y qué años fueron excepcionales.")
    modo = st.radio("Mostrar", ["Valores", "Comparado con lo normal del mes"], horizontal=True)
    tabla = dr.pivot_table(index="anio", columns="mes", values=col).reindex(columns=range(1, 13))
    if modo == "Valores":
        z, escala, texto_cb = np.log10(tabla + 1), "YlOrRd", f"log₁₀({UNIDAD[medida]}+1)"
        hover = f"%{{y}} %{{x}}: %{{customdata:,.0f}} {UNIDAD[medida]}<extra></extra>"
        zmid = None
    else:
        z = np.log2((tabla + 1) / (clima.promedio + 1).values)
        escala, texto_cb, zmid = "RdBu_r", "veces lo normal (log₂)", 0
        hover = "%{y} %{x}: %{customdata:,.0f} · " + "%{z:+.1f} (log₂ vs. normal)<extra></extra>"
    fig = go.Figure(go.Heatmap(z=z.values, x=MESES, y=tabla.index.astype(str), colorscale=escala, zmid=zmid,
                               customdata=tabla.values, hovertemplate=hover, colorbar=dict(title=texto_cb)))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=max(420, 20 * len(tabla)), margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("En «comparado con lo normal», rojo = más que el promedio de ese mes; azul = menos. "
               "Hasta 2011 los valores son MODIS armonizado; desde 2012, VIIRS.")

# ---------------------------------------------------------------- 3. mapa
with tabs[2]:
    ca, cb = st.columns([1, 3])
    with ca:
        modo_mapa = st.radio("Mostrar", ["Año completo", "Un mes", "Diferencia con lo normal"])
        mes_mapa = (st.selectbox("Mes", range(1, 13), format_func=lambda m: MESES[m - 1])
                    if modo_mapa == "Un mes" else None)
    g = grilla if region == PAIS else grilla[grilla.region == region]
    if modo_mapa == "Diferencia con lo normal":
        normal = (g[g.anio.between(ref_desde, ref_hasta)].groupby(["lat_c", "lon_c"]).focos_equiv.sum()
                  / (ref_hasta - ref_desde + 1))
        este = g[g.anio == anio].groupby(["lat_c", "lon_c"]).focos_equiv.sum()
        m = pd.concat([este.rename("actual"), normal.rename("normal")], axis=1).fillna(0).reset_index()
        m["valor"] = m.actual - m.normal
        m = m[m.valor.abs() >= 1].sort_values("valor", key=abs)
        lim = float(np.nanpercentile(m.valor.abs(), 95)) if len(m) else 1.0
        color = dict(color=m.valor, colorscale="RdBu_r", cmin=-lim, cmax=lim,
                     colorbar=dict(title="Diferencia<br>(focos)"))
        hover = "Este año: %{customdata[0]:,.0f}<br>Normal: %{customdata[1]:,.0f}<extra></extra>"
        custom = m[["actual", "normal"]].values
        nota = "Rojo: más focos que lo normal. Azul: menos. Cada cuadrado es una celda de ~25 km."
    else:
        sel = g[g.anio == anio]
        if mes_mapa:
            sel = sel[sel.mes == mes_mapa]
        m = sel.groupby(["lat_c", "lon_c"]).focos_equiv.sum().reset_index().sort_values("focos_equiv")
        m["valor"] = m.focos_equiv
        tope = float(np.log10(m.valor.max() + 1)) if len(m) else 1.0
        color = dict(color=np.log10(m.valor + 1), colorscale="YlOrRd", cmin=0, cmax=tope,
                     colorbar=dict(title="Focos", tickvals=list(range(int(tope) + 1)),
                                   ticktext=[miles(10 ** k) for k in range(int(tope) + 1)]))
        hover = "%{customdata[0]:,.0f} focos<extra></extra>"
        custom = m[["valor"]].values
        nota = "Focos por celda de ~25 km (en escala VIIRS). Escala de colores logarítmica."
    zoom_region = region in REGIONES
    fig = go.Figure(go.Scattergeo(lat=m.lat_c, lon=m.lon_c, mode="markers", customdata=custom,
                                  hovertemplate=hover, showlegend=False,
                                  marker=dict(symbol="square", size=9 if zoom_region else 4.5, opacity=.9,
                                              line=dict(width=0), **color)))
    for nombre, (la0, la1, lo0, lo1) in REGIONES.items():
        fig.add_trace(go.Scattergeo(lat=[la0, la0, la1, la1, la0], lon=[lo0, lo1, lo1, lo0, lo0], mode="lines",
                                    line=dict(width=1.3, color="#333"), name=nombre, hoverinfo="name",
                                    showlegend=False))
    if zoom_region:
        la0, la1, lo0, lo1 = REGIONES[region]
        lat_rango, lon_rango = [la0 - 1, la1 + 1], [lo0 - 1, lo1 + 1]
    else:
        lat_rango, lon_rango = [-56, -21], [-75, -53]
    fig.update_geos(projection_type="mercator", lataxis_range=lat_rango, lonaxis_range=lon_rango,
                    showcountries=True, countrycolor="#9AA5B1", showsubunits=True, subunitcolor="#D5DBE0",
                    showland=True, landcolor="#F4F6F7", showocean=True, oceancolor="#EAF2F8",
                    showcoastlines=True, coastlinecolor="#9AA5B1", resolution=50)
    fig.update_layout(height=680, margin=dict(t=0, b=0, l=0, r=0))
    with cb:
        # El contorno de los países se sirve desde la propia app (carpeta static/), así el mapa funciona sin depender de otro sitio
        st.plotly_chart(fig, width="stretch", config=CONFIG_MAPA)
    with ca:
        st.caption(nota)
        st.caption("Los rectángulos son las regiones analizadas. El mapa siempre muestra focos.")

# ---------------------------------------------------------------- 4. armonización
with tabs[3]:
    st.subheader("El problema")
    st.markdown("MODIS (satélites Terra y Aqua, píxel de ~1 km) y VIIRS (satélite Suomi NPP, píxel de 375 m) miden "
                "distinto: VIIRS divide cada incendio en más píxeles y ve fuegos más chicos. Si se pegan las dos "
                "series, parece que el fuego se multiplicó en 2012. **Armonizar** es convertir MODIS a la escala "
                "de VIIRS para tener una serie continua desde 2003.")
    an = dr.groupby("anio")[[f"{medida}_modis", f"{medida}_viirs", col]].sum(min_count=1).reset_index()
    an = an[an.anio <= 2025]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=an.anio, y=an[f"{medida}_modis"], name="MODIS (original)", mode="lines+markers",
                             line=dict(color=COLOR_MODIS)))
    fig.add_trace(go.Scatter(x=an.anio, y=an[f"{medida}_viirs"], name="VIIRS (original)", mode="lines+markers",
                             line=dict(color=COLOR_VIIRS)))
    fig.add_trace(go.Scatter(x=an.anio, y=an[col], name="Serie armonizada", line=dict(color=COLOR_ARM, width=3.5)))
    fig.add_vline(x=2011.5, line_dash="dot", annotation_text="empieza VIIRS")
    fig.update_layout(height=400, yaxis_title=f"{NOMBRES[medida]} por año", legend=dict(orientation="h", y=1.12),
                      margin=dict(t=30, b=10), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Cómo elegimos el método")
    st.markdown(f"Probamos 4 métodos. Cada uno aprendió con **{MET['anios_entrenamiento']}** y se evaluó con "
                f"**{MET['anios_test']}**, años que no había visto. **Regla:** gana el método más simple cuyo error "
                f"no supere en más de {MET['tolerancia']:.0%} al mejor.")
    res = pd.DataFrame(MET["resultados"][medida]).set_index("método")
    elegido = MET["elegido"][medida]
    st.success(MET["justificacion"][medida])

    cc1, cc2 = st.columns([1, 1])
    with cc1:
        b = res.reset_index()
        b["color"] = np.where(b["método"] == elegido, COLOR_ALERTA, "#9AA5B1")
        fig = go.Figure(go.Bar(x=b["MAE"], y=b["método"], orientation="h", marker_color=b["color"],
                               text=[miles(v, 1) for v in b["MAE"]], textposition="auto"))
        fig.update_layout(height=300, xaxis_title=f"Error medio por región y mes ({UNIDAD[medida]})",
                          yaxis=dict(autorange="reversed"), margin=dict(t=30, b=10), title="Error (MAE): menor es mejor")
        st.plotly_chart(fig, width="stretch")
    with cc2:
        d = pd.DataFrame(MET["anual_test"][medida])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.anio, y=d["Real"], name="VIIRS real", line=dict(color=COLOR_ARM, width=4)))
        for c in [c for c in d.columns if c not in ("anio", "Real")]:
            fig.add_trace(go.Scatter(x=d.anio, y=d[c], name=c, mode="lines+markers",
                                     line=dict(dash="solid" if c == elegido else "dot",
                                               width=3 if c == elegido else 1.5)))
        fig.update_layout(height=300, title="Test: total de Argentina por año", margin=dict(t=30, b=10),
                          xaxis=dict(dtick=1), legend=dict(font=dict(size=10)))
        st.plotly_chart(fig, width="stretch")

    st.dataframe(res.style.format({"MAE": "{:,.1f}", "R² (log)": "{:.3f}", "Sesgo total %": "{:+.1f}",
                                   "Peor año (país) %": "{:.1f}"})
                 .apply(lambda r: ["background-color: #FDEDEC" if r.name == elegido else ""] * len(r), axis=1),
                 width="stretch")
    st.caption("MAE: error promedio por región y mes. R² (log): qué parte de la variación explica (1 = perfecto). "
               "Sesgo total: si se pasa (+) o se queda corto (−) en el total de 2022–2025. "
               "Peor año: el mayor error en el total anual de Argentina.")

    with st.expander("Ver las fórmulas de cada método"):
        st.markdown("**1. Factor fijo por región** — una regla de tres:")
        st.latex(r"k_r = \frac{\sum V}{\sum M} \qquad \hat V = k_r \cdot M")
        st.markdown("**2. Regresión logarítmica por región y mes** (con corrección de sesgo κ):")
        st.latex(r"\log(1+\hat V) = a + b\,\log(1+M) + c\,\sin\tfrac{2\pi\,mes}{12} + d\,\cos\tfrac{2\pi\,mes}{12}")
        st.markdown("**3. Mapeo de cuantiles** — se emparejan percentiles:")
        st.latex(r"\hat V = F_V^{-1}\big(F_M(M)\big)")
        st.markdown("**4. Random Forest** — promedio de 300 árboles de decisión con MODIS, mes y región.")

    if medida == "focos":
        fac = pd.Series(MET["factores_focos"]).sort_values()
        fig = px.bar(x=fac.values, y=fac.index, orientation="h", text=[f"{v:.1f}×" for v in fac.values],
                     labels={"x": "VIIRS / MODIS", "y": ""}, color_discrete_sequence=[COLOR_VIIRS])
        fig.update_layout(height=280, margin=dict(t=40, b=10),
                          title="Cuántas veces más focos detecta VIIRS que MODIS, por región")
        st.plotly_chart(fig, width="stretch")
        st.caption("Donde hay incendios grandes (Chaco) los dos sensores los ven y la diferencia es menor. "
                   "Donde hay muchos fuegos chicos (Delta, Patagonia), VIIRS detecta muchos que MODIS no ve.")

# ---------------------------------------------------------------- 5. validación
with tabs[4]:
    st.markdown("Si la armonización es buena, **los años excepcionales tienen que aparecer igual** sin importar "
                "cómo se mida el fuego. Cada serie se expresa como índice: 100 = año promedio.")
    ia = pd.DataFrame(MET["indice_anual"]).set_index("anio")
    fig = go.Figure()
    for c, color in zip(ia.columns, [COLOR_ARM, COLOR_VIIRS, COLOR_MODIS]):
        fig.add_trace(go.Scatter(x=ia.index, y=ia[c], name=c, mode="lines+markers", line=dict(color=color)))
    fig.add_hline(y=100, line_color="gray", line_width=1)
    fig.add_vline(x=2011.5, line_dash="dot", annotation_text="antes: MODIS armonizado · después: VIIRS")
    fig.update_layout(height=400, yaxis_title="Índice (100 = año promedio)", legend=dict(orientation="h", y=1.12),
                      margin=dict(t=30, b=10), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Correlación entre las tres medidas** (2003–2025)")
        st.dataframe(pd.DataFrame(MET["correlacion_medidas"]).style.format("{:.2f}"), width="stretch")
    with v2:
        st.markdown("**Los 5 años más activos según cada medida**")
        st.dataframe(pd.DataFrame({c: ia[c].nlargest(5).index.values for c in ia}, index=range(1, 6)),
                     width="stretch")

    ta = pd.Series(MET["terra_sobre_aqua"]).sort_index()
    if len(ta):
        ta.index = ta.index.astype(int)
        fig = px.line(x=ta.index, y=100 * ta.values, markers=True,
                      labels={"x": "Año", "y": "Terra / Aqua (%)"}, color_discrete_sequence=[COLOR_MODIS])
        fig.update_layout(height=280, margin=dict(t=40, b=10),
                          title="Limitación: el satélite Terra detecta cada vez menos que Aqua")
        st.plotly_chart(fig, width="stretch")
        st.caption("Coincide con que la órbita de Terra se fue corriendo y pasa cada vez más temprano, cuando hay "
                   "menos fuego. Explica parte del error de los modelos en los últimos años.")

# ---------------------------------------------------------------- 6. acerca de
with tabs[5]:
    st.markdown("""
### Sobre este proyecto
Desarrollado para el **NASA Space Apps Challenge 2026**, desafío *Harmonization of MODIS and VIIRS Hot Spots*,
propuesto por la **CONAE**.

**Datos:** NASA FIRMS. MODIS colección 6.1 (Terra + Aqua, desde 2003) y VIIRS 375 m colección 2 (Suomi NPP,
desde 2012), para Argentina. Se usan solo incendios de vegetación (sin antorchas de gas, industrias ni volcanes)
y se descartan las detecciones de baja confianza.

**Regiones:** rectángulos aproximados elegidos por su cantidad de focos o por el impacto de sus incendios.

**Limitaciones**
- Un foco de calor no es un incendio: un incendio grande genera muchos focos, y las nubes o el humo pueden ocultar fuegos.
- Los valores anteriores a 2012 son estimaciones del modelo, no mediciones de VIIRS.
- Los datos de los últimos meses son de "tiempo casi real" (provisorios).
- Terra está envejeciendo y detecta menos fuego en los últimos años.

**Para quién sirve:** brigadas, organismos de emergencia e investigadores que necesitan saber si una temporada
viene peor que lo normal, cuándo y dónde.
""")
