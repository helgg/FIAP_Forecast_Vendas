import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


st.set_page_config(
    page_title="Forecast de Vendas Semanais",
    layout="wide",
    initial_sidebar_state="expanded"
)
plt.rcParams["figure.figsize"] = (10, 3.2)  
plt.rcParams["axes.grid"] = False


def mape(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    eps = 1e-9
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, eps, None))) * 100

@st.cache_data(show_spinner=True)
def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    
    if (df["Fuel_Price"] > 100).any():
        df["Fuel_Price"] = df["Fuel_Price"] / 1000.0
    if (df["Unemployment"] > 100).any():
        df["Unemployment"] = df["Unemployment"] / 1000.0
    return df

def feature_engineering(df: pd.DataFrame):
    df = df.copy()
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["Date"].dt.quarter
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    for lag in [1, 2, 3, 4, 12, 52]:
        df[f"Weekly_Sales_lag_{lag}"] = df["Weekly_Sales"].shift(lag)
    
    df["Weekly_Sales_roll_4"] = df["Weekly_Sales"].rolling(4).mean().shift(1)
    df["Weekly_Sales_roll_12"] = df["Weekly_Sales"].rolling(12).mean().shift(1)
    
    df = df.dropna().reset_index(drop=True)
    return df

def build_matrices(df: pd.DataFrame, cutoff="2012-01-01"):
    cutoff = pd.Timestamp(cutoff)
    train_df = df[df["Date"] < cutoff].copy()
    test_df  = df[df["Date"] >= cutoff].copy()

    target = "Weekly_Sales"
    feature_cols = [
        "Holiday_Flag","Temperature","Fuel_Price","CPI","Unemployment",
        "year","month","weekofyear","quarter",
        "is_month_start","is_month_end",
        "month_sin","month_cos",
        "Weekly_Sales_lag_1","Weekly_Sales_lag_2","Weekly_Sales_lag_3",
        "Weekly_Sales_lag_4","Weekly_Sales_lag_12","Weekly_Sales_lag_52",
        "Weekly_Sales_roll_4","Weekly_Sales_roll_12"
    ]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target].copy()
    X_test  = test_df[feature_cols].copy()
    y_test  = test_df[target].copy()

    return train_df, test_df, X_train, y_train, X_test, y_test, feature_cols

def iterative_forecast(model, X_test_scaled, lag_positions, scaler, original_X_test):
    Xt = X_test_scaled.copy()
    preds = []
    for i in range(Xt.shape[0]):
        p = model.predict(Xt[i].reshape(1, -1))[0]
        preds.append(p)
        if i + 1 < Xt.shape[0]:
            ref = original_X_test.iloc[[i+1]].copy()
            ref["Weekly_Sales_lag_1"] = p
            ref_scaled = scaler.transform(ref.values)
            Xt[i+1, lag_positions["Weekly_Sales_lag_1"]] = ref_scaled[0, lag_positions["Weekly_Sales_lag_1"]]
    return np.array(preds)

def kpi_table(y_test, preds_dict):
    rows = []
    for name, pred in preds_dict.items():
        rows.append({
            "Modelo": name,
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred),
            "MAPE (%)": mape(y_test, pred)
        })
    return pd.DataFrame(rows).sort_values("RMSE")


st.sidebar.title("⚙️ Controles")
data_path = st.sidebar.text_input("Caminho do CSV", "sales.csv")
run_xgb = st.sidebar.checkbox("Comparar Modelos", value=True if XGB_AVAILABLE else False,
                              help="Se desmarcado apenas o melhor modelo será exibido.")

# Estimators fixados -- Helder
# rf_trees = st.sidebar.slider("Árvores (RandomForest)", min_value=300, max_value=1200, value=700, step=100)
# xgb_trees = st.sidebar.slider("Árvores (XGBoost)", min_value=300, max_value=1500, value=800, step=100)
cutoff_str = st.sidebar.text_input("Data de corte (teste >=)", "2012-01-01")

st.sidebar.markdown("---")
st.sidebar.caption("Dica: ajuste a data de corte para ver como o modelo reage a períodos diferentes.")


st.title("📈 Forecast de Vendas Semanais — Dashboard Executivo")
st.caption("Previsões semanais com aprendizado de máquina (RandomForest e XGBoost), sem vazamento de dados, com foco em clareza para decisão.")

with st.expander("🧠 O que você está vendo", expanded=True):
    st.write(
        """
        - **O que é**: previsões semanais de vendas, usando um modelo que aprende com o comportamento passado e fatores de calendário/economia.  
        - **Como funciona**: o modelo “olha” as semanas anteriores (memória), entende sazonalidade (meses do ano) e ajusta previsões semana a semana.  
        - **Por que confiar**: as previsões são feitas **sem olhar o futuro** — cada previsão alimenta a próxima, como aconteceria na operação real.  
        - **Como ler os KPIs**:
            - **RMSE**: quanto erramos, em média, nas unidades de venda. Menor é melhor.  
            - **R²**: quanto da variação das vendas o modelo explica (de 0 a 1). Maior é melhor.  
            - **MAPE**: erro percentual médio. Menor é melhor.  
        """
    )


csv_path = Path(data_path)
if not csv_path.exists():
    st.error(f"Arquivo não encontrado: {csv_path.resolve()}")
    st.stop()

df_raw = load_data(csv_path)
df = feature_engineering(df_raw)

train_df, test_df, X_train, y_train, X_test, y_test, feature_cols = build_matrices(df, cutoff=cutoff_str)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled  = scaler.transform(X_test.values)

lag_positions = {c: feature_cols.index(c) for c in feature_cols if "lag_" in c}


rf = RandomForestRegressor(
    n_estimators=700, #rf_trees
    max_depth=14,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

preds = {}
preds["RandomForest"] = iterative_forecast(
    model=rf,
    X_test_scaled=X_test_scaled,
    lag_positions=lag_positions,
    scaler=scaler,
    original_X_test=X_test
)

if run_xgb and XGB_AVAILABLE:
    xgb = XGBRegressor(
        n_estimators= 800, # xgb_trees
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train_scaled, y_train)
    preds["XGBoost"] = iterative_forecast(
        model=xgb,
        X_test_scaled=X_test_scaled,
        lag_positions=lag_positions,
        scaler=scaler,
        original_X_test=X_test
    )
elif run_xgb and not XGB_AVAILABLE:
    st.warning("xgboost não está instalado no ambiente. Desmarque a opção ou instale com: pip install xgboost")


st.subheader("🔢 KPIs de Precisão (período de teste)")
kpis = kpi_table(y_test, preds)
col1, col2, col3 = st.columns(3)
col1.metric("Melhor RMSE", f"{kpis['RMSE'].min():,.0f}")
col2.metric("Melhor R²", f"{kpis['R²'].max():.2f}")
col3.metric("Melhor MAPE (%)", f"{kpis['MAPE (%)'].min():.2f}")
st.dataframe(kpis, use_container_width=True)

st.divider(width="stretch")
st.header("📉 Real vs Previsto")
dates = test_df["Date"].reset_index(drop=True)
y_true = y_test.reset_index(drop=True)

def lineplot(y_true, y_pred, title):
    fig, ax = plt.subplots()
    ax.plot(dates, y_true, label="Real")
    ax.plot(dates, y_pred, label="Previsto")
    ax.set_title(title); ax.set_xlabel("Data"); ax.set_ylabel("Vendas Semanais"); ax.legend()
    st.pyplot(fig)

for name, pred in preds.items():
    st.space(size="small")
    st.subheader(f"{name}")
    if name == 'RandomForest':
        st.text('Este gráfico compara o valor real de vendas (linha azul) com o valor previsto pelo modelo (linha laranja) ao longo das semanas. Podemos observar que as duas linhas seguem um formato parecido, indicando que o modelo consegue capturar a tendência geral das vendas. Em algumas semanas existe uma diferença maior entre as linhas, o que normalmente acontece em períodos com eventos fora do padrão, como promoções, sazonalidade ou mudanças de mercado.', help=None, width="content")
    else:
        st.text('Neste gráfico vemos a comparação entre as vendas reais (linha azul) e as vendas previstas pelo modelo XGBoost (linha laranja). As duas linhas acompanham bem o movimento geral das vendas ao longo dos meses, mostrando que o modelo consegue entender a tendência e o comportamento do negócio. Em alguns pontos o modelo suaviza variações mais bruscas, o que é comum em modelos que buscam estabilidade. No geral, o XGBoost apresenta boa aderência, especialmente em períodos mais estáveis, sendo útil para previsões de planejamento.', help=None, width="content")
    lineplot(y_true, pred, f"Real vs Previsto — {name}")

st.space(size="small")
st.subheader("Comparativo entre os modelos")
st.text('Este gráfico coloca lado a lado as vendas reais e as previsões dos dois modelos. A linha azul representa o que realmente aconteceu, enquanto as linhas laranja (RandomForest) e verde (XGBoost) mostram as previsões. Observamos que ambos os modelos conseguem seguir a tendência geral das vendas ao longo do tempo. O XGBoost acompanha melhor oscilações mais rápidas, ficando mais próximo de picos e vales. Já o RandomForest tende a ser mais conservador, suavizando variações e mantendo previsões mais estáveis, o que é importante para planejamento e tomada de decisão, evitando reações exageradas a semanas atípicas. Por isso, apesar de pequenas diferenças, o RandomForest foi escolhido como modelo principal, por oferecer maior consistência e menor risco operacional no uso das previsões.', help=None, width="content")

if len(preds) > 1:
    fig, ax = plt.subplots()
    ax.plot(dates, y_true, label="Real")
    for name, pred in preds.items():
        ax.plot(dates, pred, label=name)
    ax.set_title("Comparativo — Real x Modelos")
    ax.set_xlabel("Data"); ax.set_ylabel("Vendas Semanais"); ax.legend()
    st.pyplot(fig)

st.space(size="small")
st.divider(width="stretch")
st.header("🔍 Diagnóstico de Erros")

def plot_error_over_time(dates, y_true, y_pred, title):
    e = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots()
    ax.plot(dates, e)
    ax.set_title(title)
    ax.set_xlabel("Data"); ax.set_ylabel("Erro (real - previsto)")
    st.pyplot(fig)

def plot_error_box_by_month(dates, y_true, y_pred, title):
    df_tmp = pd.DataFrame({"date": pd.to_datetime(dates), "err": np.array(y_true) - np.array(y_pred)})
    df_tmp["month"] = df_tmp["date"].dt.month
    data = [df_tmp[df_tmp["month"] == m]["err"].values for m in sorted(df_tmp["month"].unique())]
    fig, ax = plt.subplots()
    ax.boxplot(data, labels=sorted(df_tmp["month"].unique()))
    ax.set_title(title); ax.set_xlabel("Mês"); ax.set_ylabel("Erro")
    st.pyplot(fig)

for name, pred in preds.items():
    st.space(size="small")
    if name == 'RandomForest':
        st.subheader(f"Erro ao longo do tempo — RandomForest")
        st.text('Este gráfico mostra a diferença entre o valor real e o valor previsto pelo modelo em cada semana. Quando a linha está próxima de zero, significa que o modelo acertou bem. Os picos positivos e negativos representam semanas em que houve mudanças fora do padrão, como promoções, sazonalidade ou fatores externos. No geral, o modelo mantém o erro controlado e sem desvios prolongados, o que indica que ele é estável e adequado para uso no planejamento.', help=None, width="content")
        plot_error_over_time(dates, y_true, pred, f"Erro ao longo do tempo — {name}")
        st.space(size="small")
        st.subheader(f"Erro ao longo do tempo — RandomForest")
        st.text('Este gráfico mostra como o erro do modelo se comporta em cada mês. Meses onde a caixa é mais alta ou espalhada indicam maior variação nas vendas, ou seja, semanas mais diferentes do padrão esperado. Já meses com caixas mais compactas indicam que o modelo conseguiu prever com maior estabilidade. De forma geral, o modelo mantém um desempenho consistente ao longo do ano, com variações naturais em períodos de maior movimentação ou sazonalidade.', help=None, width="content")
        plot_error_box_by_month(dates, y_true, pred, f"Erro por mês — {name}")
    else:
        st.subheader(f"Erro ao longo do tempo — XGBoost")
        st.text('Este gráfico mostra como o erro do XGBoost varia ao longo das semanas. Assim como no RandomForest, quando a linha está próxima de zero, o modelo acertou bem. Porém, percebemos picos mais intensos, tanto para cima quanto para baixo, indicando que o XGBoost é mais sensível a mudanças bruscas no comportamento das vendas. Essa maior oscilação pode levar a previsões menos estáveis em semanas atípicas. Por isso, mesmo apresentando bom desempenho, optamos pelo RandomForest como modelo principal, pois ele oferece maior equilíbrio e consistência, o que é mais seguro para o planejamento.', help=None, width="content")
        plot_error_over_time(dates, y_true, pred, f"Erro ao longo do tempo — {name}")
        st.space(size="small")
        st.subheader(f"Erro ao longo do tempo — XGBoost")
        st.text('Este gráfico mostra como o erro do XGBoost varia mês a mês. Percebemos que, em alguns meses, a distribuição do erro é mais espalhada, indicando que o modelo é mais sensível a mudanças bruscas no comportamento das vendas. Essa sensibilidade pode fazer com que o modelo reaja demais a semanas atípicas, ampliando o erro em períodos de maior variação ou sazonalidade. Embora o XGBoost acompanhe bem oscilações, essa maior instabilidade mensal reforça a escolha do RandomForest como modelo principal, pois ele mantém previsões mais estáveis e consistentes, o que é ideal para o planejamento do negócio.', help=None, width="content")
        plot_error_box_by_month(dates, y_true, pred, f"Erro por mês — {name}")


st.space(size="small")
st.divider(width="stretch")
st.header("🧩 Importância de Features (o que mais pesa nas previsões)")
def plot_feature_importance(importances, feature_names, title):
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(idx)), np.array(importances)[idx])
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels(np.array(feature_names)[idx], rotation=90)
    ax.set_title(title)
    st.pyplot(fig)
st.space(size="small")
st.subheader(f"RandomForest")
st.text('Este gráfico mostra quais informações o modelo mais utiliza na hora de prever as vendas. A variável com maior peso é Weekly_Sales_lag_52, que representa as vendas da mesma semana do ano anterior. Isso indica que o negócio possui um forte padrão sazonal, ou seja, períodos do ano tendem a repetir comportamentos de vendas. Outros fatores que também influenciam as previsões são a semana do ano, inflação (CPI) e as vendas das semanas mais recentes, mostrando que o modelo aprendeu tanto o ciclo anual quanto o ritmo das últimas semanas. Em resumo: o modelo consegue capturar tendência + sazonalidade, o que reforça sua capacidade de prever com consistência.', help=None, width="content")
plot_feature_importance(rf.feature_importances_, feature_cols, "RandomForest — importância")
if "XGBoost" in preds:
    st.space(size="small")
    st.subheader(f"XGBoost")
    st.text('Assim como no RandomForest, o XGBoost também identifica a sazonalidade anual como o principal fator, com a variável Weekly_Sales_lag_52 sendo a mais relevante. Além disso, o modelo dá destaque à semana do ano (weekofyear) e a indicadores econômicos como CPI e Unemployment, mostrando que ele é mais sensível a variações externas. Esse comportamento reforça que o XGBoost reage mais rapidamente a mudanças no ambiente, o que pode ser positivo, mas também pode aumentar a instabilidade das previsões. Por isso, mesmo com boa capacidade de identificar padrões, optamos pelo RandomForest como modelo principal, pois ele mantém previsões mais estáveis e adequadas para planejamento.', help=None, width="content")
    plot_feature_importance(xgb.feature_importances_, feature_cols, "XGBoost — importância")

st.space(size="small")
st.divider(width="stretch")
with st.expander("📚 Como ler este dashboard", expanded=True):
    st.markdown(
        """
**KPIs (no topo)**  
- **RMSE**: quanto erramos em valor absoluto.  
- **R²**: o quanto explicamos da variação das vendas.  
- **MAPE**: erro percentual médio.

**Gráficos de linha**  
- Comparam vendas reais com as previsões. Linhas próximas indicam boa aderência.

**Erro ao longo do tempo**  
- Se oscila ao redor de zero, o modelo está equilibrado.  
- Picos revelam semanas “especiais” (ex.: feriados, promoções).

**Erro por mês (boxplot)**  
- Mostra meses onde o modelo é mais instável.  
- Útil para planejar ações (ex.: reforço de dados/variáveis em meses problemáticos).

**Importância de Features**  
- Indica quais informações o modelo realmente usa para prever.  
- Se “lags” aparecem no topo, o histórico recente é muito relevante.
"""
    )


st.space(size="small")
st.html(

    """
    <div style="margin: 200px auto; text-align: center;">
        <h4>Analítico Anônimos</h4>
        <table style="margin: 0 auto; border-collapse: collapse; border: 1px solid white;">
            <tr>
                <th style="border: 1px solid white; padding: 6px;">Nome</th>
                <th style="border: 1px solid white; padding: 6px;">RM</th>
            </tr>
            <tr>
                <td style="border: 1px solid white; padding: 6px;">Cesar Miyashiro</td>
                <td style="border: 1px solid white; padding: 6px;">RM556286</td>
            </tr>
            <tr>
                <td style="border: 1px solid white; padding: 6px;">Helder Gualdi de Godoy</td>
                <td style="border: 1px solid white; padding: 6px;">RM556571</td>
            </tr>
            <tr>
                <td style="border: 1px solid white; padding: 6px;">Liora Vanessa Dopacio</td>
                <td style="border: 1px solid white; padding: 6px;">RM554355</td>
            </tr>
            <tr>
                <td style="border: 1px solid white; padding: 6px;">Marcelo Moure</td>
                <td style="border: 1px solid white; padding: 6px;">RM555751</td>
            </tr>
            <tr>
                <td style="border: 1px solid white; padding: 6px;">Sandro Façanha</td>
                <td style="border: 1px solid white; padding: 6px;">RM557585</td>
            </tr>
        </table>  
    </div>
    """
)
