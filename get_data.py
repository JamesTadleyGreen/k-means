import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


#Set the data_path to be the location of get_data.py then in the folder data.
data_path = os.path.dirname(__file__) + "\data\\"

#Load in the data
industry_df = pd.read_csv(data_path + "demo_industry_comp.csv")
comp_df = pd.read_csv(data_path + "data_tool_export.csv")

#Drop the blank rows and reset the index
comp_df = comp_df[comp_df['City'].notna()].reset_index(drop = True)

#18-29 table
young_columns = ["City"] + [col for col in comp_df.columns if 'Population Aged 18-29' in col]
young_df = comp_df[young_columns].fillna(0)

years = []
i = 1991
while i <= 2018:
    years.append(i)
    i += 1

for i in range(40):
    Y = young_df.iloc[i,1:].values.reshape(-1, 1)  # first row without first column
    X = np.reshape(years, (-1, 1))  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = make_pipeline(PolynomialFeatures(3), LinearRegression())  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions


    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.title(young_df.iloc[i,0])
    plt.show()