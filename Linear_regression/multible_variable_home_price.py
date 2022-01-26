import pandas as pd
import numpy as np
from sklearn import linear_model

df=pd.read_csv("home_price_multi.csv")        #read data csv

import math       #to use floor func.
x=math.floor(df.bedrooms.median())        
df["bedrooms"].fillna(x,inplace=True)         #fill nul in 'bedrooms'

reg=linear_model.LinearRegression()           #build model
reg.fit(df[['area','bedrooms','age']],df.price) #train the model

#coefficients
reg.coef_
reg.intercept_

reg.predict([[4000,5,8]])     #predict the model
