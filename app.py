import streamlit as st
import yfinance as yf
from hmmlearn import hmm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Terminal Pro - Villa Constitución", layout="wide")

st.title("📊 Tablero Maestro de Inversiones 360°")
st.markdown("---")

# --- PARÁMETROS DE CARTERA ---
CAPITAL_TOTAL = 30000000  # Referencia a tus $30M ARS
RIESGO_POR_ACTIVO = 0.01 

# Lista oficial de seguimiento
tickers_default = "SPY, GGAL.BA, COST, VIST, META, MSFT, YPFD.BA, UBER, QQQ, AAPL"

st.sidebar.header("Configuración de Análisis")
tickers_input = st.sidebar.text_area("Lista de Activos:", tickers_default)
tickers = [t.strip().upper() for t in tickers_input.split(",")]

btn_ejecutar = st.sidebar.button("🚀 ACTUALIZAR INFORME")

if btn_ejecutar:
    with st.spinner('Sincronizando mercados y calculando riesgos...'):
        # CÁLCULO DE TC FINANCIERO (CCL)
        try:
            ccl_ref = yf.download(["GGAL", "GGAL.BA"], period="1d", progress=False)['Close']
            TC = (ccl_ref['GGAL.BA'].iloc[-1] / ccl_ref['GGAL'].iloc[-1]) * 10
            if isinstance(TC, pd.Series): TC = TC.item()
        except:
            TC = 1265

        resultados = []

        for ticker in tickers:
            try:
                asset = yf.Ticker(ticker)
                data = asset.history(period="10y")
                if data.empty or len(data) < 200: continue
                
                info = asset.info
                precio = float(data['Close'].iloc[-1])
                es_merval = ticker.endswith(".BA")

                # --- 1. MODELO MARKOV Y AGOTAMIENTO ---
                rets = np.log(data['Close'] / data['Close'].shift(1)).dropna().values.reshape(-1, 1)
                model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=3000, random_state=42)
                model.fit(rets)
                orden = np.argsort(model.means_.flatten())
                mapa = {orden[0]: "Bajista", orden[1]: "Lateral", orden[2]: "Alcista"}
                estados = model.predict(rets)
                regimen = mapa[estados[-1]]

                cont = 0
                for i in range(len(estados)-1, -1, -1):
                    if estados[i] == estados[-1]: cont += 1
                    else: break
                p_quedarse = model.transmat_[estados[-1], estados[-1]]
                agotamiento = "ALTO" if cont > (1/(1-p_quedarse)) else "Normal"

                # --- 2. VALUACIÓN MODERNA Y MONEDA (FIX YPF/GGAL) ---
                fv_analistas = info.get('targetMeanPrice')
                # Si el target está en USD pero el precio en ARS (Escala < 1/100)
                if fv_analistas and es_merval and fv_analistas < (precio / 100):
                    fv_analistas *= TC
                
                fwd_pe = info.get('forwardPE', 25) # COST suele tener PE alto
                eps = info.get('trailingEps')
                fv_pe = (eps * fwd_pe) if eps else None

                valid_fv = [v for v in [fv_analistas, fv_pe] if v is not None and v > 0]
                fv_final = np.mean(valid_fv) if valid_fv else precio
                upside = (fv_final / precio) - 1

                # --- 3. RIESGO Y STOP LOSS ---
                sigma = np.sqrt(model.covars_[estados[-1]][0][0])
                var = abs(norm.ppf(0.05, model.means_[estados[-1]][0], sigma))
                stop_loss = precio * (1 - var)
                monto_sug = (CAPITAL_TOTAL * RIESGO_POR_ACTIVO) / var
                cant_sug = int(monto_sug / precio)

                # --- 4. ACCIÓN Y ESTRATEGIA ---
                if regimen == "Alcista":
                    if upside > 0.05:
                        accion, deriva = "COMPRA DIRECTA", "Bull Call Spreads"
                    else:
                        accion, deriva = "MANTENER (Caro)", "Lanzamiento Cubierto"
                elif regimen == "Lateral":
                    accion, deriva = "ESPERAR (Rango)", "Iron Condors / Tasa"
                else:
                    accion, deriva = "VENTA / LIQUIDEZ", "Puts Protectoras"

                # LAS 12 COLUMNAS
                resultados.append({
                    "Ticker": ticker,
                    "Régimen": regimen,
                    "Días": cont,
                    "Agotamiento": agotamiento,
                    "Precio Actual": round(precio, 2),
                    "Fair Value": round(fv_final, 2),
                    "Upside (%)": round(upside * 100, 2),
                    "Acción": accion,
                    "Estrategia Opciones": deriva,
                    "Stop Loss": round(stop_loss, 2),
                    "Monto Sug. (ARS)": round(monto_sug, 2),
                    "Cant. Acciones": cant_sug
                })
            except: continue

        # --- SALIDA EN LA APP ---
        df = pd.DataFrame(resultados)
        
        # Alertas Visuales
        for _, r in df.iterrows():
            if r['Precio Actual'] <= r['Stop Loss'] or r['Acción'] == "VENTA / LIQUIDEZ":
                st.error(f"🚨 ALERTA CRÍTICA: {r['Ticker']} rompió Stop Loss o entró en VENTA.")

        st.write("### Informe Estratégico Detallado")
        st.dataframe(df.style.background_gradient(subset=['Upside (%)'], cmap='RdYlGn'))
        
        # Gráfico interactivo
        st.subheader("Visualización: Upside vs. Riesgo")
        st.scatter_chart(df, x="Upside (%)", y="Stop Loss", color="Régimen")
