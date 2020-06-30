import pandas as pd
import numpy as np
import os
import time
import re #regex
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


#Set the data_path to be the location of get_data.py then in the folder data.
data_path = os.path.dirname(__file__) + "\data\\"

#Load in the data
industry_df = pd.read_csv(data_path + "demo_industry_comp.csv")
comp_df = pd.read_csv(data_path + "data_tool_export.csv")

#Drop Belfast
comp_df = comp_df[comp_df.City != "Belfast"]

#Drop the blank rows and reset the index
comp_df = comp_df[comp_df['City'].notna()].reset_index(drop = True)

#Working age population
for i in range(1991, 2019):
    comp_df[f'WA Population {i}  (%)'] = 100 - (comp_df[f'Population Aged 0-17 {i}  (%)'] + comp_df[f'Population Aged 67+ {i}  (%)'])

#Old age dependency
for i in range(1991, 2019):
    comp_df[f'Old Age Dependency {i}  (%)'] = comp_df[f'Population Aged 67+ {i}  (%)'] / comp_df[f'WA Population {i}  (%)']

#Aging
for i in range(1992, 2019):
    comp_df[f'Aging Rate {i}'] = comp_df[f'Population Aged 67+ {i}  (%)'] / comp_df[f'Population Aged 67+ {(i-1)}  (%)']

#Grad gain / loss
for i in range(2005, 2019):
    comp_df[f'Graduate Retention {i}'] = comp_df[f'Working Age Population with a Qualification at NVQ4 or Above {i}  (%)'] / comp_df[f'Working Age Population with a Qualification at NVQ4 or Above {(i-1)}  (%)']

#Service to product ratio
for i in [2014,2017]:
    comp_df[f'Service Product Ratio {i}'] = comp_df[f'Services exports per job {i}  (Â£)'] / comp_df[f'Goods exports per job {i}  (Â£)']

# for col in comp_df.columns:
#     print(col)

def grouping_regression(col_name, regression_order):
    # todo this fuction assumes that there are no graps in the years given. This is not always the case, needs fix.
    columns = ["City"] + [col for col in comp_df.columns if col_name in col]
    restricted_df = comp_df[columns].fillna(0)

    years = []
    i = int(re.search(r"(\d+)(?!.*\d)",columns[1]).groups()[0]) #get the year from the second column
    while i <= int(re.search(r"(\d+)(?!.*\d)",columns[len(columns)-1]).groups()[0]): #get the year from the lest column
        years.append(i)
        i += 1

    for i in range(5):
        Y = restricted_df.iloc[i,1:].values.reshape(-1, 1)  # first row without first column
        X = np.reshape(years, (-1, 1))  # -1 means that calculate the dimension of rows, but have 1 column
        linear_regressor = make_pipeline(PolynomialFeatures(regression_order), LinearRegression())  # create object for the class
        linear_regressor.fit(X, Y)  # perform linear regression
        Y_pred = linear_regressor.predict(X)  # make predictions


        plt.scatter(X, Y)
        plt.plot(X, Y_pred, color='red')
        plt.title(restricted_df.iloc[i,0])
        plt.show()
    
grouping_regression("Service Product", 2)