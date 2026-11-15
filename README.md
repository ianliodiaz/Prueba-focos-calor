# FIREN'T — Harmonized Burning Activity Calendar for Argentina

**NASA Space Apps Challenge 2026 · Challenge: Harmonization of MODIS and VIIRS Hot Spots (proposed by CONAE, Argentina)**

FIREN'T merges two decades of NASA satellite fire detections from two different sensors — **MODIS** (Terra and Aqua, since 2003)
and **VIIRS** (Suomi NPP, since 2012) — into one comparable record for Argentina, and turns it into an interactive
**burning activity calendar**. Emergency teams, scientists and land managers can pick a region (or draw their own area), choose
any date range, compare it with the historical median and download the hotspots and analytics.

- **Live web app:** [add the Streamlit link here]
- **Languages:** English, Spanish, Portuguese and a basic Guaraní prototype (selector at the top of the left panel).
- **Anomaly levels without normality assumptions:** each month is flagged as *anomalous* (above 90% of reference years) or *record* (above all of them). We use percentiles instead of Z-scores because fire data are strongly skewed (82% of region-month combinations have skewness > 1), so a single extreme year would inflate the standard deviation and hide later events.
- **Featured cases:** one-click buttons load well-known events (Paraná Delta 2020, Corrientes 2022, Córdoba Hills 2020) and a low-fire year (2026); a mini-map in the sidebar shows where the selected area is and where it burned that year.
- **Inclusion:** on arrival the app asks for the language and offers larger text with high contrast and colorblind-safe colors
  (changeable with the 👁️ button), plus an on-screen magnifier that follows the cursor; the summary can be read aloud; a presentation video opens after the welcome and can be closed; every view can be shared with a link; and anyone can leave a
  comment or report odd data through a moderated form.

## The problem

VIIRS has 375 m pixels; MODIS has ~1 km pixels. VIIRS splits each fire into more pixels and detects smaller fires, so it records
about **3.3 times more hotspots** than MODIS. Stitching both records together creates a false jump in fire activity in 2012. As
MODIS reaches the end of its life, ~10 years of fire history would become incomparable without harmonization.

## How it works

1. **Data:** NASA FIRMS active fire data for Argentina — MODIS Collection 6.1 (2003–2026), VIIRS 375 m Collection 2 Suomi NPP
   (2012–2026) as the base of the series, and VIIRS NOAA-20 (2018–2026) used only to fill days when Suomi NPP did not measure.
   Archive plus near-real-time, downloaded from <https://firms.modaps.eosdis.nasa.gov/download/> (Country: Argentina · CSV).
2. **Exploratory analysis and cleaning:** missing and impossible values, duplicates, points outside Argentina and outliers are
   checked (extreme values are real events and are kept); vegetation fires only (gas flares, industrial sources and volcanoes
   removed) and low-confidence detections removed, the same criteria for all satellites → 4.1 million hotspots.
3. **No-data days:** a day with no hotspots can mean "no fire" or "the satellite did not measure". Comparing satellites day by
   day, 62 Suomi NPP no-data days were found (the longest outage: 26 Jul – 10 Aug 2022). Models are trained only on days when
   both sensors worked; in the published series those days are filled with NOAA-20 calibrated against Suomi NPP (ratio 1.03,
   mean monthly error ~4% on unseen years) and flagged.
4. **Aggregation:** by region, year and month, with three measures: hotspot counts, fire-days per 25 km cell, and fire radiative power (FRP).
5. **Harmonization as supervised learning:** four methods compared — fixed factor per region, log-linear regression with
   seasonality, quantile mapping and Random Forest. Trained on 2012–2021, tested on 2022–2025 (unseen years).
6. **Selection rule (set before seeing results):** the simplest method with error (MAE) within 5% of the best, total bias within
   ±10% and worst-year error below 25%; uncertainty measured with a block bootstrap (95% confidence intervals). For hotspot
   counts the **log regression by region and month** won (bias −3.5%, worst year 13.5%); the fixed regional factor was 5.6%
   worse, a difference that is not statistically significant, so the series does not depend on the method. Quantile mapping
   won for fire-days and log regression for FRP. Random Forest did not justify its complexity. VIIRS records ≈ 3.4 times more
   hotspots than MODIS, from 3.1 in Gran Chaco to 4.6 in the Paraná Delta.
7. **Validation:** the three measures flag the same exceptional years (2020, 2003, 2008), correlation 0.92–0.98.

The full methodology is in the app's **"How we built it"** tab.

## Repository contents

| File / folder | What it is |
|---|---|
| `app.py` | Streamlit web app |
| `requirements.txt` | Python libraries for the app |
| `datos/` | Harmonized outputs produced by the notebook (monthly series, map grid, method results, regions, no-data days, library versions) |
| `datos/focos/` | Individual cleaned hotspots, one Parquet file per year (used for date searches, custom areas and downloads) |
| `static/`, `.streamlit/` | Map outlines and app configuration |
| `notebook_armonizacion.ipynb` | Processing and harmonization notebook (run it in Google Colab) |
| `notebook_armonizacion_resultados.ipynb` | The same notebook, already executed, showing every result |

The notebooks are commented in Spanish; every step is summarized above and in the app.

## Reproduce

1. Download the three FIRMS zip files (MODIS, VIIRS S-NPP and VIIRS NOAA-20) (see step 1 above) and put them in Google Drive under `My Drive/Challenge NASA 2026/datos`.
2. Open `notebook_armonizacion.ipynb` in Google Colab and run all cells. Outputs are saved to `Challenge NASA 2026/salida`.
3. Copy the outputs into `datos/` and run the app: `pip install -r requirements.txt` and `streamlit run app.py`.

## Data sources and credits

- **NASA FIRMS** — MODIS C6.1 and VIIRS 375 m C2 (Suomi NPP and NOAA-20) active fire products.
- **CONAE** (Argentina) — Hotspot and Burned Area Catalog, used as reference.
- Map tiles: © OpenStreetMap contributors, OpenTopoMap (CC-BY-SA), Esri (GEBCO, NOAA, Maxar). Country outlines: Natural Earth via Plotly.
- Tools: Python (pandas, NumPy, scikit-learn), Google Colab, Streamlit, Plotly, Folium/Leaflet.

## Use of Super Intelligence (SI)

Code and texts in this repository were developed with the help of an AI assistant (Claude, by Anthropic), which was used to
write and debug the Python code (notebook and web app), discuss harmonization methods, and draft and translate texts
(including the Portuguese version and the basic Guaraní prototype, which is flagged in the app as under review). Every code
cell of the notebook and the notebook metadata state this. All results
are computed from the real NASA FIRMS data, and the team reviewed the methods, decisions and outputs. No AI-generated images,
audio or video are included.

## Team

[Team members' names]
