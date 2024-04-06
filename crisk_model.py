import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
import pickle
import streamlit as st

df = pd.read_csv('corporate_rating.csv')

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


inputs = st.data_editor(grouped_df.loc[selected_sector])
inputs['Sector'] = adj_sector[0]


input_data = np.array([inputs])

prediction = model.predict(input_data)

results = df.Rating[prediction].values


st.write(f'A classificação de risco da empresa é: {results[0]}')
