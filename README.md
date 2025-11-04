# 🧠 Forecast de Vendas Semanais — Machine Learning com RandomForest e XGBoost

---


## 📘 Visão Geral

Este projeto tem como objetivo **prever o volume semanal de vendas** com base em variáveis econômicas e sazonais, aplicando algoritmos de aprendizado de máquina supervisionados.
O foco principal é comparar o desempenho entre **Random Forest Regressor** e **XGBoost Regressor**, dois modelos robustos para problemas de previsão contínua.

---



## 🎯 Objetivos

- Desenvolver um modelo de **previsão iterativa** capaz de projetar vendas semana a semana, sem vazamento de dados.
- Explorar técnicas de **feature engineering** para capturar padrões temporais e sazonais.
- Comparar a precisão e a estabilidade dos modelos com métricas estatísticas e **gráficos de erro**.

---


## 🧩 Etapas do Projeto

### 1️⃣ Pré-processamento e preparação dos dados

- Conversão de datas e ordenação temporal.
- Ajuste de escala das variáveis (`Fuel_Price`, `Unemployment`).
- Divisão temporal em **treino (até 2011)** e **teste (2012)** para simular previsões reais.

### 2️⃣ Feature Engineering

Para capturar dependências temporais e padrões sazonais, foram criadas novas variáveis:

| Tipo                            | Features criadas                                                                |
| ------------------------------- | ------------------------------------------------------------------------------- |
| **Lags**                  | `Weekly_Sales_lag_1`, `lag_2`, `lag_3`, `lag_4`, `lag_12`, `lag_52` |
| **Médias móveis**       | `Weekly_Sales_roll_4`, `Weekly_Sales_roll_12`                               |
| **Calendário**           | `month`, `weekofyear`, `quarter`, `is_month_start`, `is_month_end`    |
| **Sazonalidade cíclica** | `month_sin`, `month_cos`                                                    |

Essas features permitem ao modelo aprender padrões de **curto, médio e longo prazo**, além de ciclos anuais.

---

## ⚙️ Modelos Aplicados

### 🌲 Random Forest Regressor

- Ensemble de árvores com **bagging** e amostragem aleatória.
- Hiperparâmetros ajustados: `n_estimators=700`, `max_depth=14`, `min_samples_leaf=2`.
- Ótimo para capturar **relações não lineares** e **interações entre variáveis**.

### 🚀 XGBoost Regressor

- Modelo baseado em **boosting**, com ajustes progressivos para corrigir erros das previsões anteriores.
- Hiperparâmetros principais:
  - `n_estimators=800`
  - `learning_rate=0.05`
  - `max_depth=6`
  - `subsample=0.8`
  - `colsample_bytree=0.8`
- Tende a gerar previsões mais **suaves e precisas**, especialmente em séries com ruído.

---

## 🔁 Forecast Iterativo

O modelo prevê **sem conhecer o futuro real**, atualizando o valor de `Weekly_Sales_lag_1` a cada iteração com a **última previsão gerada**.
Assim, ele simula o comportamento real de produção, onde apenas o passado é conhecido.

---

## 📈 Métricas de Avaliação

| Métrica                                        | Significado                                      | Ideal                |
| ----------------------------------------------- | ------------------------------------------------ | -------------------- |
| **RMSE (Root Mean Squared Error)**        | Mede o erro médio absoluto em unidades de venda | Quanto menor, melhor |
| **R² (Coeficiente de Determinação)**   | Mede quanto da variação o modelo explica       | Próximo de 1        |
| **MAPE (Mean Absolute Percentage Error)** | Erro percentual médio                           | Quanto menor, melhor |

---

## 🔍 Análise dos Erros

Para entender **onde o modelo acerta e onde erra**, foram gerados gráficos diagnósticos:

### 📊 Erro ao longo do tempo

Mostra a diferença entre valores reais e previstos semana a semana.

- Linhas próximas de **zero** → modelo estável.
- Erros repetitivos → indicam sazonalidade não capturada.
- Erros positivos → modelo subestimou vendas.
- Erros negativos → modelo superestimou.

### 📦 Boxplot de erro por mês

Mostra a **distribuição dos erros mensais**, permitindo identificar viés sazonal.

- Boxes pequenos → previsões consistentes.
- Boxes deslocados → tendência sistemática (por exemplo, subestimar em dezembro).

Esses gráficos ajudam a identificar **padrões temporais de erro**, permitindo ajustes futuros nas features.

---

### ⚙️ Instalação e Dependências

Clone o repositório:

```bash
git clone https://github.com/helgg/FIAP_Forecast_Vendas.git
cd Fiap_forecast
```

Crie e ative um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

Instale as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```
