# 📊 Análise Técnica de Ativos com Streamlit

Aplicação web (Streamlit) para análise técnica de ações e criptomoedas com **MA/EMA (PhiCube)**, **RSI**, **MACD**, **Fibonacci** e **Volume Financeiro**.

## 🚀 Funcionalidades
- Seleção de ativos por lista (BR/USA/Cripto) ou ticker manual (ex.: `TSLA`, `PETR4.SA`, `BTC-USD`).
- Períodos: `7d`, `1mo`, `3mo`, `6mo`, `1y`, `5y`, `10y`.
- Intervalos: `15m`, `30m`, `1h`, **`4h (resample)`**, `1d`.
- Ajuste automático quando a combinação Período × Intervalo não é suportada pelo Yahoo Finance (com aviso).
- Indicadores: **MA21/MA200**, **EMA17/72/305 (PhiCube)**, **RSI(14)**, **MACD(12,26,9)**, **Volume** e **Volume Financeiro**.
- Linhas de **Fibonacci** calculadas automaticamente sobre o período carregado.

## 📦 Requisitos
```
pip install -r requirements.txt
```

## ▶️ Rodar localmente
```
streamlit run app_analise_final.py
```

## ☁️ Deploy (Streamlit Community Cloud)
1. Faça um fork/clone deste repositório no GitHub com estes arquivos:
   - `app_analise_final.py`
   - `requirements.txt`
   - `README.md`
2. Em https://streamlit.io/cloud → **New app** → selecione o repositório
3. **Main file path**: `app_analise_final.py` → **Deploy**

## ℹ️ Observações
- Yahoo Finance **não fornece** “número de negócios”. O app exibe **Volume** e **Volume Financeiro (Volume×Close)**.
- O intervalo **4h** é obtido por **resample** de dados **1h**.
- Se a combinação escolhida não existir, o app ajusta período/intervalo automaticamente e informa no topo.