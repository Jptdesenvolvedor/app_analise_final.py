# 📊 Análise Técnica de Ativos com Streamlit

Aplicação web (Streamlit) para análise técnica de ações e criptomoedas, com dados do Yahoo Finance.

## 🚀 Funcionalidades
- **Painel de decisão** no topo: selo grande **COMPRAR · MANTER · AGUARDAR · REDUZIR · VENDER**, pontuação de −100 a +100
  com barra de zonas, nível de confiança, **tendência de curto, médio e longo prazo** (alta/baixa/lateral e força) e a
  **tendência do mercado geral** (S&P 500 para ações dos EUA, Ibovespa para ações do Brasil, Bitcoin para cripto),
  além de stop e alvo sugeridos e a explicação regra a regra ("Por que este sinal?").
  É um sinal **automático** baseado nas regras validadas (regime MA200, modelo de tendência, momentum e timing), calibrado
  para candles diários — não é recomendação de investimento.
- **Ativos:** lista (BR/EUA/Cripto) ou ticker manual (`TSLA`, `PETR4.SA`, `BTC-USD`).
- **Períodos:** `7d` a `10y`. **Intervalos:** `15m`, `30m`, `1h`, `4h` (resample de 1h), `1d`, `1wk`, `1mo`.
  Combinações não suportadas pelo Yahoo (15m/30m ≤ 60 dias; 1h/4h ≤ ~700 dias) são ajustadas com aviso.
- **Gráfico interativo (Plotly):** candles ou linha, zoom/hover, fins de semana e horas fora do pregão ocultos.
- **Indicadores:** MA21/MA200, EMA17/72/305 (PhiCube), RSI(14, suavização de Wilder), MACD(12,26,9),
  Bollinger(20,2), Estocástico(14,3), ATR(14), VWAP de sessão (intraday), Volume e Volume Financeiro.
- **Fibonacci** em dois modos: sobre máxima/mínima do período inteiro ou sobre a **perna de alta recente**
  (últimos 120 candles, sem olhar o futuro) com zona de ouro (38,2%–61,8%) e alvos de extensão (127,2% / 161,8%);
  e **suporte/resistência** por pivôs.
- **Diagnóstico** (RSI, MACD, MA200, alinhamento das EMAs, Bollinger, volume, volatilidade, divergência de RSI)
  e **eventos/alertas** por cruzamentos, com destaque para o último candle.
- **Backtest** long-only de 6 regras (com custo por operação), curva de capital, taxa de acerto, drawdown,
  **CAGR e Sharpe** lado a lado com o buy & hold, **stop móvel por ATR** opcional e **treino × teste**
  (a mesma regra em duas janelas seguidas, para não se enganar com a sorte de um período).
- **Regime e risco** (aba 🧭): núcleo defensivo (MA200 com banda de 2%), modelo de tendência
  (Minervini/O'Neil, 6 critérios), momentum 12-1 e **calculadora de tamanho de posição** por risco fixo
  (stop em N×ATR, teto por ativo; frações de unidade em cripto).
- **Comparação de ativos** (base 100) e **fundamentos** básicos (P/L, P/VP, valor de mercado…).
- **Fontes e validação:** links de RI, CVM/SEC e provedores; exportação dos dados em CSV.

> ⚠️ Ferramenta educacional. Não é recomendação de investimento; o backtest não inclui impostos nem slippage.

## 🗂️ Estrutura
```
app_analise_final.py   # interface Streamlit
analise/
  config.py            # constantes e regras período × intervalo
  data.py              # download (retentativas), normalização, resample
  indicators.py        # indicadores, Fibonacci (período e perna recente), suporte/resistência
  regime.py            # regime defensivo, modelo de tendência, momentum, tamanho de posição
  sinal.py             # sinal de decisão, tendência por prazo e do mercado geral
  apresentacao.py      # componentes visuais (HTML/CSS) do painel de decisão
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

## 🔬 De onde vêm as regras da aba "Regime e risco"
Foram avaliadas em backtests longos (2016–2026), com ~4.000 ações **incluindo as que saíram da bolsa**
(sem viés de sobrevivência), custos descontados e **treino 2017–22 × teste 2023–26**:
- Núcleo defensivo no SPY: ~15,8% a.a. com carteira 70/30 contra 15,0% do buy & hold, e queda máxima de −23% contra −34%.
- Regras clássicas isoladas (MACD, cruzamento de EMAs, RSI) e Fibonacci como gatilho de compra **não** bateram o mercado.
- Tendência + momentum foi o único sinal de seleção consistente nos dois períodos, mas com quedas profundas
  (use como parte pequena da carteira).

Isto é evidência histórica, não garantia de resultado futuro.

## ℹ️ Observações
- O Yahoo Finance **não fornece** “número de negócios”; o app mostra Volume e Volume Financeiro (Volume×Close).
- 4h é obtido por resample de 1h, alinhado ao início do pregão (ações) ou à grade UTC (cripto).
- EMA305/MA200 só aparecem com candles suficientes; o app informa quando faltam.
