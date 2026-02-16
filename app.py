import streamlit as st
import yfinance as yf
from hmmlearn import hmm
import pandas as pd
import numpy as np
from scipy.stats import norm

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="Inversiones Pro - Villa Constitución", layout="wide")

st.title("📊 Mi Terminal de Inversiones 360°")
st.markdown("---")

# --- SIDEBAR (PANEL LATERAL) ---
st.sidebar.header("Configuración")
capital = st.sidebar.number_input("Capital Total (ARS)", value=30000000)
riesgo_pct = st.sidebar.slider("Riesgo por activo (%)", 0.5, 3.0, 1.0) / 100

# Tickers solicitados
default_tickers = "SPY, GGAL.BA, COST, VIST, META, MSFT, YPFD.BA, UBER, QQQ, AAPL"
lista_input = st.sidebar.text_area("Lista de Activos:", default_tickers)
tickers = [t.strip().upper() for t in lista_input.split(",")]

btn_ejecutar = st.sidebar.button("🚀 ACTUALIZAR ANÁLISIS")

if btn_ejecutar:
    with st.spinner('Analizando regímenes y valuaciones...'):
        # 1. CÁLCULO DE TC (CCL)
        try:
            ccl_ref = yf.download(["GGAL", "GGAL.BA"], period="1d", progress=False)['Close']
            tc_ccl = (ccl_ref['GGAL.BA'].iloc[-1] / ccl_ref['GGAL'].iloc[-1]) * 10
            if isinstance(tc_ccl, pd.Series): tc_ccl = tc_ccl.item()
        except:
            tc_ccl = 1260

        resultados = []

        for ticker in tickers:
            try:
                asset = yf.Ticker(ticker)
                data = asset.history(period="10y")
                if data.empty: continue
                
                info = asset.info
                precio = float(data['Close'].iloc[-1])
                es_merval = ticker.endswith(".BA")

                # --- MARKOV Y AGOTAMIENTO ---
                rets = np.log(data['Close'] / data['Close'].shift(1)).dropna().values.reshape(-1, 1)
                model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=3000, random_state=42)
                model.fit(rets)
                orden = np.argsort(model.means_.flatten())
                mapa = {orden[0]: "Bajista", orden[1]: "Lateral", orden[2]: "Alcista"}
                estados = model.predict(rets)
                regimen = mapa[estados[-1]]

                # Días y Agotamiento
                cont = 0
                for i in range(len(estados)-1, -1, -1):
                    if estados[i] == estados[-1]: cont += 1
                    else: break
                agotamiento = "ALTO" if cont > (1/(1-model.transmat_[estados[-1], estados[-1]])) else "Normal"

                # --- FAIR VALUE MODERNO (Ajuste COST y YPF) ---
                fv_analistas = info.get('targetMeanPrice')
                # Ajuste de escala moneda
                if fv_analistas and es_merval and fv_analistas < (precio / 100):
                    fv_analistas *= tc_ccl
                
                # Múltiplo P/E dinámico (Forward P/E)
                fwd_pe = info.get('forwardPE', 15)
                eps = info.get('trailingEps')
                fv_pe = (eps * fwd_pe) if eps else None

                valid_fv = [v for v in [fv_analistas, fv_pe] if v is not None and v > 0]
                fv_final = np.mean(valid_fv) if valid_fv else precio
                upside = (fv_final / precio) - 1

                # --- RIESGO Y STOP LOSS ---
                sigma = np.sqrt(model.covars_[estados[-1]][0][0])
                var = abs(norm.ppf(0.05, model.means_[estados[-1]][0], sigma))
                stop_loss = precio * (1 - var)
                monto_sug = (capital * riesgo_pct) / var
                cant_sug = int(monto_sug / precio)

                # --- ACCIÓN Y ESTRATEGIA ---
                if regimen == "Alcista":
                    if upside > 0.05:
                        accion, deriva = "COMPRA", "Bull Call Spreads"
                    else:
                        accion, deriva = "MANTENER (Caro)", "Lanzamiento Cubierto"
                elif regimen == "Lateral":
                    accion, deriva = "ESPERAR (Rango)", "Iron Condors"
                else:
                    accion, deriva = "VENTA / LIQUIDEZ", "Puts Protectoras"

                resultados.append({
                    "Ticker": ticker, "Régimen": regimen, "Días": cont, "Agotamiento": agotamiento,
                    "Precio Act.": round(precio, 2), "Fair Value": round(fv_final, 2), 
                    "Upside (%)": round(upside * 100, 2), "Acción": accion, "Opciones": deriva, 
                    "Stop Loss": round(stop_loss, 2), "Monto Sug. (ARS)": round(monto_sug, 2), "Cant.": cant_sug
                })
            except: continue

        # --- MOSTRAR RESULTADOS ---
        df = pd.DataFrame(resultados)
        
        # Alertas críticas en pantalla
        for _, r in df.iterrows():
            if r['Precio Act.'] <= r['Stop Loss'] or r['Acción'] == "VENTA / LIQUIDEZ":
                st.error(f"🚨 ALERTA: {r['Ticker']} rompió Stop Loss o entró en VENTA.")

        st.dataframe(df.style.highlight_max(axis=0, subset=['Upside (%)'], color='lightgreen'))
        
        # Gráfico interactivo
        st.subheader("Gráfico Riesgo vs Retorno")
        st.scatter_chart(df, x="VaR (%)", y="Upside (%)", color="Régimen")
