import pandas as pd
import numpy as np
from sklearn import preprocessing

corporate_data = pd.read_csv('corporate_rating.csv', encoding='UTF-8')
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

corporate_data.Rating = corporate_data.Rating.map(rating_dict)

corporate_data = corporate_data[corporate_data['Rating']!='Lowest Risk'] # filter Lowest Risk
corporate_data = corporate_data[corporate_data['Rating']!='In Default']  # filter In Default
corporate_data.reset_index(inplace = True, drop=True) # reset index


min_max_scaler = preprocessing.MinMaxScaler()

for c in corporate_data.columns[6:31]:

    corporate_data[[c]] = min_max_scaler.fit_transform(corporate_data[[c]].to_numpy())*1000
    corporate_data[[c]] = corporate_data[[c]].apply(lambda x: np.log10(x+0.01))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

le = preprocessing.LabelEncoder()
le.fit(corporate_data.Sector)
corporate_data.Sector = le.transform(corporate_data.Sector) # Transformar o setor para uma variavel utilizavel
le.fit(corporate_data.Rating)
corporate_data.Rating = le.transform(corporate_data.Rating) # Transformar o Rating obtido para uma variavel utilizavel

df_train, df_test = train_test_split(corporate_data, test_size=0.2, random_state = 1234)
X_train, y_train = df_train.iloc[:,5:31], df_train.iloc[:,0]
X_test, y_test = df_test.iloc[:,5:31], df_test.iloc[:,0]

RF_model = RandomForestClassifier(random_state=1234)
RF_model.fit(X_train,y_train)
y_pred_RF = RF_model.predict(X_test)
Accuracy_RF = metrics.accuracy_score(y_test, y_pred_RF)
print("RF Accuracy:",Accuracy_RF)

