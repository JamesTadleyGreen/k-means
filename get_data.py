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
data_path = os.path.dirname(__file__) + r"\data\\"

#Load in the data
industry_df = pd.read_csv(data_path + "demo_industry_comp.csv")
comp_df = pd.read_csv(data_path + "data_tool_export.csv")

#Drop Belfast
comp_df = comp_df[comp_df.City != "Belfast"]

#Drop the blank rows and reset the index
comp_df = comp_df[comp_df['City'].notna()].reset_index(drop = True)

# ! Create new fields
#Working age population and Old age dependency
for i in range(1991, 2019):
    comp_df[f'WA Population {i}  (%)'] = 100 - (comp_df[f'Population Aged 0-17 {i}  (%)'] + comp_df[f'Population Aged 67+ {i}  (%)'])
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

#PKItoMFC
for i in range(2010, 2019):
    comp_df[f'PKItoMFC {i}'] = comp_df[f'Private Knowledge Intensive Business Services {i}  (%)'] / comp_df[f'Manufacturing {i}  (%)']

#Business start up rate
for i in range(2004, 2019):
    comp_df[f'Business Start-ups Rate {i}  (%)'] = comp_df[f'Business Start-ups {i}  (per 10,000 population)'] / 10000

#Business stock
for i in range(2004, 2019):
    comp_df[f'Business Stock Rate {i}  (%)'] = comp_df[f'Business stock {i}  (per 10,000 population)'] / 10000

#Business stock
for i in range(2015, 2019):
    comp_df[f'Patent Applications Rate {i}  (%)'] = comp_df[f'Patent Applications {i}  (per 100,000 of population)'] / 100000

for col in comp_df.columns:
    print(col)


def grouping_regression(col_name, regression_order):
    # todo this fuction assumes that there are no graps in the years given. This is not always the case, needs fix.
    columns = ["City"] + [col for col in comp_df.columns if col_name in col]
    restricted_df = comp_df[columns].fillna(0)

    years = []
    i = int(re.search(r"(\d+)(?!.*\d)",columns[1]).groups()[0]) #get the year from the second column
    while i <= int(re.search(r"(\d+)(?!.*\d)",columns[len(columns)-1]).groups()[0]): #get the year from the lest column
        years.append(i)
        i += 1
    
    future_years = []
    i = int(re.search(r"(\d+)(?!.*\d)",columns[len(columns)-1]).groups()[0]) + 1 #get the year from the last column
    while i <= 2030: #prediction year
        future_years.append(i)
        i += 1

    for i in range(5):
        Y = restricted_df.iloc[i,1:].values.reshape(-1, 1)  # first row without first column
        X = np.reshape(years, (-1, 1))  # -1 means that calculate the dimension of rows, but have 1 column
        X_new = np.reshape(future_years, (-1, 1))
        linear_regressor = make_pipeline(PolynomialFeatures(regression_order), LinearRegression())  # create object for the class
        linear_regressor.fit(X, Y)  # perform linear regression
        Y_pred = linear_regressor.predict(X)  # make predictions
        Y_new = linear_regressor.predict(X_new)
        X_comb = np.concatenate((X, X_new))
        Y_comb = np.concatenate((Y_pred, Y_new))

        plt.scatter(X, Y)
        plt.scatter(X_new, Y_new, color='red')
        #plt.plot(X_comb, Y_comb, color='red')
        plt.title(restricted_df.iloc[i,0])
        plt.xlim(right=2030)
        plt.show()

#print([col for col in comp_df.columns if ("Business Start-ups" in col) & ("(%)" in col)])
#grouping_regression("Employment Rate", 1)

regression_dict = {
    "WA Population":1, 
    "Old Age Dependency":2, 
    "Population Aged 18-29":3,
    "Working Age Population with No Formal Qualifications":1,
    "Working Age Population with a Qualification at NVQ4 or Above":1,
    "Aging Rate":1,
    "Graduate Retention":1,
    "Employment Rate":1,
    "Population":1, #This is broken
    "Service Product Ratio":0, #Two data points
    "PKI":1,
    "Patent Applications Rate":0,
    "Business Start-ups Rate":2,
    "Business Stock Rate":1,
    }

def prediction(col_name):
    # todo this fuction assumes that there are no graps in the years given. This is not always the case, needs fix.
    columns = ["City"] + [col for col in comp_df.columns if col_name in col]
    restricted_df = comp_df[columns].fillna(0)

    years = []
    i = int(re.search(r"(\d+)(?!.*\d)",columns[1]).groups()[0]) #get the year from the second column
    while i <= int(re.search(r"(\d+)(?!.*\d)",columns[len(columns)-1]).groups()[0]): #get the year from the lest column
        years.append(i)
        i += 1

    pred = []
    for i in range(5):
        Y = restricted_df.iloc[i,1:].values.reshape(-1, 1)  # first row without first column
        X = np.reshape(years, (-1, 1))  # -1 means that calculate the dimension of rows, but have 1 column
        X_predict = np.reshape([2030], (-1, 1))
        linear_regressor = make_pipeline(PolynomialFeatures(regression_dict[col_name]), LinearRegression())  # create object for the class
        linear_regressor.fit(X, Y)  # perform linear regression
        Y_predict = linear_regressor.predict(X_predict)
        pred.append([restricted_df.iloc[i,0],Y_predict])
    return(pred)

print(prediction("Employment Rate"))