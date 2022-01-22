#lesson single variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv("home_price_linear_regression.csv")
print(df)

%matplotlib inline
plt.xlabel("Area")
plt.ylabel("Prices")
plt.scatter (df.area, df.price,color='red',marker='+')   #scatter data

reg=linear_model.LinearRegression()  #using linear model
reg.fit(df[['area']],df.price)       #using the model to fit on data

x=[[5000],[2000]]                    #test for two variables
reg.predict(x)

reg.coef_                           #slop(m) of model equation
reg.intercept_                      #c which is y=mx+c

d=pd.read_csv("areas.csv")          #data for using in prediction

y=reg.predict(d)
d['predicted_prices']=y            #create column "predicted_prices" 

d.to_csv("predicted_data.csv",index=False)          #extract new data to csv file withput index

%matplotlib inline
plt.xlabel("Area")
plt.ylabel("Prices")
plt.scatter (df.area, df.price,color='red',marker='+')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')        #graph linear and scatter for real and predicted data
