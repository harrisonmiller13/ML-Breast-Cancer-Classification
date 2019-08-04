#%%
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import tpot as tp
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
#%%
# nuclei_measurements = pd.read_csv('wdbc.data', names = ['ID','Diagnosis', 'radius',
#                                         'texture','perimeter','area',
#                                         'smoothness','compactness',
#                                         'concavity','concave_points',
#                                         'symmetry','fractal_dimension'

#  ])
cell_data = pd.read_csv('breast-cancer-wisconsin.data',names=['id_number','Clump Thickness', 'Uniformity of Cell Size',
                 'Uniformity of Cell Shape', 'Marginal Adhesion', 
                 'Single Epithelial Cell Size','Bare Nuclei',
                 'Bland Chromatin','Normal Nucleoli','Mitosis','Class'])


#%%
# nuclei_measurements.head()
cell_data.head()

#%%
fig = px.bar(cell_data, x = 'Class', color = 'Class')
fig.show()

#%%
cell_data.isnull().count

#%%
target = cell_data['Class']
cell_data.drop('Class', axis = 1, inplace = True)


#%%
target.head()

#%%
cell_data.dtypes

#%%
cell_data.head()

#%%
cell_dummies = pd.get_dummies(cell_data)
cell_dummies.head()

#%%
cell_dummies_train, cell_dummies_test, target_train, target_test = train_test_split(cell_dummies,target,test_size = 0.25)


#%%
tree_clf = DecisionTreeClassifier(max_depth = 5)
tree_clf.fit(cell_dummies_train,target_train)

#%%
tree_clf.feature_importances_

#%%
def plot_feature_importances(model):
    n_features = cell_dummies_train.shape[1]
    plt.figure(figsize=(8,15))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), cell_dummies_train.columns.values) 
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")



#%%
plot_feature_importances(tree_clf)

#%%
pred = tree_clf.predict(cell_dummies_test)
print(confusion_matrix(target_test,pred))
print(classification_report(target_test,pred))

#%%
print("Testing Accuracy for Decision Tree Classifier: {:.4}%".format(accuracy_score(target_test, pred) * 100))

#%%
#TPOT
from tpot import TPOTClassifier
tpot = TPOTClassifier(generations= 100, population_size= 100, verbosity= 2,n_jobs= 1)
tpot.fit(cell_dummies_train,target_train)

#%%
