
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#######

#this imports a dataset of S&P 500 companies (ticker symbols and full names) as well as their industry

import datapackage
data_url = 'https://datahub.io/core/s-and-p-500-companies/datapackage.json'

# to load Data Package into storage
package = datapackage.Package(data_url)

# to load only tabular data
resources = package.resources
for resource in resources:
    if resource.tabular:
        data = pd.read_csv(resource.descriptor['path'])
        print(data)
        
                
list_names = list(data['Symbol']) # creating list with names of S&P 500 companies
list_sector = list(data['Sector']) # creating list with names of S&P 500 companies corresponding industries

# data.Sector.value_counts() # counts number of unique industries

data.drop(['Name'], axis=1, inplace=True) # drop extra column with full name of S&P 500 companies
data = data.set_index('Symbol') # set ticker column as index

# importing income statement data for S&P 500 companies
from yahoofinancials import YahooFinancials

yahoo_financials = YahooFinancials(list_names)

# pullling operating income data for most recent year reported
minilist_opinc = yahoo_financials.get_operating_income()
minilist_opinc_df = pd.Series(minilist_opinc).to_frame('op_inc')
print(minilist_opinc_df)

# pulling operating expense data for most recent year reported
minilist_opex = yahoo_financials.get_total_operating_expense()
minilist_opex_df = pd.Series(minilist_opex).to_frame('op_ex')
print(minilist_opex_df)

# pulling total revenue data for most recent year reported
minilist_rev = yahoo_financials.get_total_revenue()
minilist_rev_df = pd.Series(minilist_rev).to_frame('rev')
print(minilist_rev_df)

mergeddf = minilist_opinc_df.join(minilist_opex_df) # merging operating income dataframe and operating expense dataframe
mergeddf = mergeddf.join(minilist_rev_df) # joining revenue dataframe

mergeddf['MTC'] = mergeddf['op_inc']/mergeddf['op_ex'] # generating col for markup on total cost
mergeddf['OM'] = mergeddf['op_inc']/mergeddf['rev'] # generating col for operating margin


mergeddf.drop(['op_inc', 'op_ex'], axis=1, inplace=True) # dropping op_inc and op_ex column to reduce dimentionality
print(mergeddf)

total = mergeddf.join(data) # merge financial dataframe with 
total = pd.get_dummies(total, columns=['Sector']) # generating dummy var cols (one hot encoding) for industry sector
print(total)

from sklearn import preprocessing
# scaling column values to be between 0 and 1 using min-max scaler
scaler = preprocessing.MinMaxScaler()
robust_scaled_df = total.dropna()
robust_scaled_df['rev'] = scaler.fit_transform(robust_scaled_df['rev'].values.reshape(-1,1))
robust_scaled_df['MTC'] = scaler.fit_transform(robust_scaled_df['MTC'].values.reshape(-1,1))
robust_scaled_df['OM'] = scaler.fit_transform(robust_scaled_df['OM'].values.reshape(-1,1))


### creating client-feature rows

# example, client 1: Rnd heavy

Rnd_tech = robust_scaled_df.loc[robust_scaled_df['Sector_Information Technology'] == 1]
Rnd_tech = Rnd_tech.loc[Rnd_tech['rev'] >= 0.15]
Rnd_add = robust_scaled_df.loc[robust_scaled_df['rev'] >= 0.3]
Rnd_tech = Rnd_tech.append(Rnd_add)
Rnd_tech = Rnd_tech.reset_index().drop_duplicates(subset='index',
                                       keep='first').set_index('index')
print(Rnd_tech)

Rnd_tech = Rnd_tech.append(Rnd_tech.mean().rename('client1')).assign(mean=lambda Rnd_tech: Rnd_tech.mean(1))

n = len(Rnd_tech)-1

Rnd_tech.drop(Rnd_tech.head(n).index,inplace=True)

# example, client 2: manufacturing heavy
manuf = robust_scaled_df.loc[robust_scaled_df['Sector_Industrials'] == 1]
manuf2 = robust_scaled_df.loc[robust_scaled_df['Sector_Materials'] == 1]
manuf = manuf.append(manuf2)
manuf = manuf.sample(n=6)
manuf = manuf.append(manuf.mean().rename('client2')).assign(mean=lambda manuf: manuf.mean(1))
manuf.drop(manuf.head(n).index,inplace=True)
print(manuf)

# example, client 3: services 
gna = robust_scaled_df.loc[robust_scaled_df['MTC'] >= 0.05]
gna = gna.loc[gna['MTC'] <= 0.09]
gna2 = robust_scaled_df.loc[robust_scaled_df['Sector_Financials'] == 1]
gna2 = gna2.sample(n=2)
gna = gna.append(gna2)
gna = gna.sample(n=6)
gna = gna.append(gna.mean().rename('client3')).assign(mean=lambda gna: gna.mean(1))
gna.drop(gna.head(n).index, inplace = True)
print(gna)

## generating client-feature matrix
client_feature_matrix = Rnd_tech.append(manuf)
client_feature_matrix = client_feature_matrix.append(gna)
client_feature_matrix = client_feature_matrix.drop(['mean'], axis =1)
print(client_feature_matrix)

# importing cosine similarity as a method to determine similarity (distance) between client and comparable companies
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(client_feature_matrix, robust_scaled_df)

indices = pd.Series(client_feature_matrix.index)

#  function which returns the top 10 recommended comparable companies
def recommendations(client, cosine_sim = cosine_sim):
    
    recommended_comparables = []
    
    idx = indices[indices == client].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar comparable companies
    top_10_indexes = list(score_series.iloc[0:10].index)
    
    for i in top_10_indexes:
        recommended_comparables.append(list(robust_scaled_df.index)[i])
        
    return recommended_comparables

## example of how to generate recommendations for client1
recommendations('client1')












