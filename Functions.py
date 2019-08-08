



def plot_ROC_curve(x_test,y_test,model):
    """
    enter your features test set, target test set, and the model you are using
    
    plots an ROC curve and AUC value as the legend
    
    """
    import seaborn as sns
    sns.set()
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    
    probs = model.predict_proba(x_test)
    probs = probs[:,1]
    fpr, tpr, thresholds = roc_curve(y_test,probs, pos_label= 4)
    auc = roc_auc_score(y_test, probs)
    plt.figure(figsize=(12,8))
    plt.title("ROC",fontsize = 14)
    plt.xlabel("FPR", fontsize = 12)
    plt.ylabel("TPR", fontsize = 12)
    plt.plot([0,1],[0,1], linestyle='--')
    plt.plot(fpr, tpr, marker = '.', linewidth = 4)
    plt.legend(["AUC=%.3f"%auc],loc = 'lower right',prop={'size': 30})
    plt.show()
    
def plot_precision_recall_curve(x_test,x_train,y_test,model):
    
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import f1_score
    from sklearn.metrics import auc
    from sklearn.metrics import average_precision_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    
    probs = model.predict_proba(x_test)
    probs = probs[:,1]
    yhat = model.predict(x_train)
    precision, recall, thresholds = precision_recall_curve(y_test,probs, pos_label= 4)
    plt.figure(figsize=(12,8))
    plt.title("Precision-Recall Curve",fontsize = 14)
    plt.xlabel("Recall", fontsize = 12)
    plt.ylabel("Precision", fontsize = 12)
    plt.plot([0,1],[0,1], linestyle='--')
    plt.plot(recall,precision, marker ='.', linewidth = 4)
    plt.show()

def plot_feature_importances(x_train,model):
    import matplotlib.pyplot as plt
    import numpy as np
    n_features = x_train.shape[1]
    plt.figure(figsize=(8,15))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), x_train.columns.values) 
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

def plot_confusion_matrix(y_test,model_pred):
    from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
    import seaborn as sns
    sns.set()
    import matplotlib.pyplot as plt
    
    conf_mat = confusion_matrix(y_true=y_test, y_pred=model_pred)
    ax= plt.subplot()
    sns.heatmap(conf_mat, annot=True, ax = ax, fmt = 'g', cmap = 'Greens'); 
    ax.set_xlabel('Predicted');ax.set_ylabel('Expected'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['Benign', 'Malignant']);
    ax.yaxis.set_ticklabels(['Benign', 'Malignant']);
    print(classification_report(y_test,model_pred))
    
    
def plot_2d_space(X, y, label='Classes'):   
    import numpy as np
    import matplotlib.pyplot as plt
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
    
def get_target_counts(dataframe):
    target_count = dataframe.Class.value_counts()
    print('Benign:', target_count[2])
    print('Malignant:', target_count[4])
    print('Proportion:', round(target_count[2] / target_count[4], 2), ': 1')
    target_count.plot(kind='bar', title='Count (target)');
    

    
def prepare_data(dataframe):
    import pandas as pd
    bc_data_with_dummies = pd.get_dummies(dataframe)
    bc_data_with_dummies.set_index('id_number', inplace = True)
    bc_data_with_dummies.drop(['id_number.1',
                               'Diagnosis_B',
                               'Diagnosis_M'], axis = 1, inplace = True)
    bc_data_full = bc_data_with_dummies.interpolate(method = 'linear', axis = 1)
    return bc_data_full
   
    
# def train_model(dataframe):
#     from imblearn.over_sampling import SMOTE
#     from sklearn.ensemble import ExtraTreesClassifier
#     from sklearn.model_selection import train_test_split
    
#     features = bc_data_full
#     X = bc_data_full
#     y = target
#     features_train,features_test,target_train,target_test = train_test_split(X,
#                                                                              y,
#                                                                              test_size = 0.25,
#                                                                              random_state = 1)