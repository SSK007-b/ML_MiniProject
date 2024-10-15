import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

df = pd.read_csv(r"D:\ML\dataset\pistachio.csv")
X = df.iloc[: , :-1].values
y = df.iloc[: , -1].values

le = LabelEncoder()
y = le.fit_transform(y)

joblib.dump(le , 'Encode.pkl')

classifier = XGBClassifier()
classifier.fit(X , y)
# print(le.inverse_transform(classifier.predict([[60955,999.789,386.9247,209.1255,0.8414,278.5863,0.9465,64400,0.7263,1.8502,0.7663,0.72,0.0063,0.0034,0.5184,0.9591]])))
joblib.dump(classifier , 'Qpistachio.pkl')