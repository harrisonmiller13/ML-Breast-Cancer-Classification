#%%
import pandas as pd
import matplotlib as plt
import plotly.express as px
import numpy as np
import tpot as tp

#%%
df = pd.read_csv('wdbc.data', header = None)
data = pd.read_csv('breast-cancer-wisconsin.data',names=['id_number','Clump Thickness', 'Uniformity of Cell Size',
                 'Uniformity of Cell Shape', 'Marginal Adhesion', 
                 'Single Epithelial Cell Size','Bare Nuclei',
                 'Bland Chromatin','Normal Nucleoli','Mitosis','Class'])


#%%
df.head()

#%%
fig = px.bar(df, x =df[1].count, color = df[1])
fig.show()

#%%
