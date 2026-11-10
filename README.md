# 📊 Análise Técnica de Ativos com Streamlit

Aplicação web (Streamlit) para análise técnica de ações e criptomoedas, com dados do Yahoo Finance.

## 🚀 Funcionalidades
- **Ativos:** lista (BR/EUA/Cripto) ou ticker manual (`TSLA`, `PETR4.SA`, `BTC-USD`).
- **Períodos:** `7d` a `10y`. **Intervalos:** `15m`, `30m`, `1h`, `4h` (resample de 1h), `1d`, `1wk`, `1mo`.
  Combinações não suportadas pelo Yahoo (15m/30m ≤ 60 dias; 1h/4h ≤ ~700 dias) são ajustadas com aviso.
- **Gráfico interativo (Plotly):** candles ou linha, zoom/hover, fins de semana e horas fora do pregão ocultos.
- **Indicadores:** MA21/MA200, EMA17/72/305 (PhiCube), RSI(14, suavização de Wilder), MACD(12,26,9),
  Bollinger(20,2), Estocástico(14,3), ATR(14), VWAP de sessão (intraday), Volume e Volume Financeiro.
- **Fibonacci** sobre máxima/mínima, na direção da tendência do período, e **suporte/resistência** por pivôs.
- **Diagnóstico** (RSI, MACD, MA200, alinhamento das EMAs, Bollinger, volume, volatilidade, divergência de RSI)
  e **eventos/alertas** por cruzamentos, com destaque para o último candle.
- **Backtest** long-only de 4 regras (com custo por operação), curva de capital, taxa de acerto e drawdown.
- **Comparação de ativos** (base 100) e **fundamentos** básicos (P/L, P/VP, valor de mercado…).
- **Fontes e validação:** links de RI, CVM/SEC e provedores; exportação dos dados em CSV.

> ⚠️ Ferramenta educacional. Não é recomendação de investimento; o backtest não inclui impostos nem slippage.

## 🗂️ Estrutura
```
app_analise_final.py   # interface Streamlit
analise/
  config.py            # constantes e regras período × intervalo
  data.py              # download (retentativas), normalização, resample
  indicators.py        # indicadores, Fibonacci, suporte/resistência
  diagnostico.py       # diagnóstico textual e eventos
  backtest.py          # estratégias e métricas
  charts.py            # gráficos Plotly
  fundamentos.py       # Ticker.info
  sources.py + fontes.json   # fontes oficiais e links
tests/                 # pytest
```

## ▶️ Rodar localmente
```
pip install -r requirements.txt
streamlit run app_analise_final.py
```

## 🧪 Testes
```
pip install -r requirements-dev.txt
pytest                # unitários (sem internet)
pytest -m network     # confere se os links de fontes.json respondem
```
O workflow de CI (`.github/workflows/ci.yml`) roda os testes a cada push e confere os links toda segunda-feira.

## ☁️ Deploy (Streamlit Community Cloud)
1. Suba este repositório no GitHub.
2. Em https://streamlit.io/cloud → **New app** → escolha o repositório.
3. **Main file path:** `app_analise_final.py` → **Deploy**.

## ℹ️ Observações
- O Yahoo Finance **não fornece** “número de negócios”; o app mostra Volume e Volume Financeiro (Volume×Close).
- 4h é obtido por resample de 1h, alinhado ao início do pregão (ações) ou à grade UTC (cripto).
- EMA305/MA200 só aparecem com candles suficientes; o app informa quando faltam.
