import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
import pickle
import streamlit as st

df = pd.read_csv('corporate_rating.csv')

list_order = df.iloc[:,5:].columns.tolist()

rating_dict = {'AAA':'Risco Muito Baixo',
               'AA':'Risco Baixo',
               'A':'Risco Baixo',
               'BBB':'Risco Médio',
               'BB':'Risco Alto',
               'B':'Risco Alto',
               'CCC':'Risco Muito Alto',
               'CC':'Risco Muito Alto',
               'C':'Risco Muito Alto',
               'D':'In Default'}

df.Rating = df.Rating.map(rating_dict)

le = preprocessing.LabelEncoder()
le.fit(df.Sector)

## Definir as opções de setor
sector_list = df.Sector.unique()
sector_list = tuple(sector_list)

## Dentro de cada setor, pega a média dos indicadores financeiros
grouped_df = df.iloc[:,5:].groupby('Sector').mean()


with open('RF_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Avaliação de Empresa - Risco")

selected_sector = st.selectbox('Selecione o setor.', sector_list)
adj_sector = le.transform([selected_sector])

# Suponha que grouped_df seja o seu DataFrame com os dados agrupados

# Dividindo o DataFrame em diferentes grupos
liquidity_df = pd.DataFrame(grouped_df.loc[selected_sector,['currentRatio', 'quickRatio', 'cashRatio']]).T
profitability_df = pd.DataFrame(grouped_df.loc[selected_sector,['grossProfitMargin', 'operatingProfitMargin', 'pretaxProfitMargin', 'netProfitMargin', 'effectiveTaxRate', 'returnOnAssets', 'returnOnEquity', 'returnOnCapitalEmployed']]).T
debt_ratios_df = pd.DataFrame(grouped_df.loc[selected_sector, ['debtRatio', 'debtEquityRatio']]).T
operating_performance_df = pd.DataFrame(grouped_df.loc[selected_sector, ['assetTurnover']]).T
cash_flow_indicators_df = pd.DataFrame(grouped_df.loc[selected_sector,['operatingCashFlowPerShare', 'freeCashFlowPerShare', 'cashPerShare', 'operatingCashFlowSalesRatio', 'freeCashFlowOperatingCashFlowRatio']]).T

colunas_para_manter = [coluna for coluna in grouped_df.columns if
                       coluna not in liquidity_df.columns and
                       coluna not in profitability_df.columns and
                       coluna not in debt_ratios_df.columns and
                       coluna not in operating_performance_df.columns and
                       coluna not in cash_flow_indicators_df]

outros_df = pd.DataFrame(grouped_df.loc[selected_sector,colunas_para_manter]).T



# Visualização
st.title('Análise de Indicadores Financeiros')

# Liquidez
st.header('Indicadores de Liquidez')
input_liquidity = st.data_editor(liquidity_df)

# Rentabilidade
st.header('Indicadores de Rentabilidade')
input_profit = st.data_editor(profitability_df)

# Índices de Endividamento
st.header('Índices de Endividamento')
input_debt = st.data_editor(debt_ratios_df)

# Índices de Desempenho Operacional
st.header('Índices de Desempenho Operacional')
input_operating = st.data_editor(operating_performance_df)

# Indicadores de Fluxo de Caixa
st.header('Indicadores de Fluxo de Caixa')
input_cashflow = st.data_editor(cash_flow_indicators_df)

st.header('Outros Indicadores')
input_outros = st.data_editor(outros_df)

df_consolidada = pd.concat([input_liquidity.T, input_profit.T,input_debt.T,input_operating.T,input_cashflow.T,input_outros.T]).T
df_consolidada['Sector'] = adj_sector[0]

df_consolidada = df_consolidada[list_order]

#inputs = st.data_editor(df_consolidada)
#inputs['Sector'] = adj_sector[0]
#input_data = np.array([inputs])


prediction = model.predict(df_consolidada)

results = df.Rating[prediction].values

st.write(f'A classificação de risco da empresa é: {results[0]}')
