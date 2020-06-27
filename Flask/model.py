import pandas as pd
import numpy as np

import pickle

train = pd.read_csv('train.csv')

train.drop(['User_ID', 'Product_ID', 'Gender', 'City_Category', 
            'Marital_Status', 'Product_Category_3'], axis = 1, inplace = True)
    
train['Product_Category_2'].fillna(train['Product_Category_2'].median(), inplace = True)

train['Product_Category_2'] = train['Product_Category_2'].astype('int')

train['Stay_In_Current_City_Years'] = train['Stay_In_Current_City_Years'].apply(lambda x : str(x).replace('4+', '4'))

train['Stay_In_Current_City_Years'] = train['Stay_In_Current_City_Years'].astype('int')

train['Age'] = train['Age'].map(
                            {'0-17' : 1,
                             '18-25' : 2,
                             '26-35' : 3,
                             '36-45' : 4,
                             '46-50' : 5,
                             '51-55' : 6,
                             '55+' : 7
                             })

X = train.drop('Purchase', axis = 1)
Y = train['Purchase']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = scaler.fit_transform(X)  

from xgboost import XGBRegressor

xgb = XGBRegressor(learning_rate = 0.1, max_depth = 8, min_child_weight = 56, verbosity = 0, random_state = 42)

xgb.fit(X, Y)

pickle.dump(xgb, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

prediction = model.predict(scaler.transform(np.array([[1, 10, 2, 3, 9]])))



