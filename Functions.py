



def plot_ROC_curve(x_test,y_test,model):
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
    
def plot_feature_importances(model):
    n_features = features_train.shape[1]
    plt.figure(figsize=(8,15))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), features_train.columns.values) 
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

def plot_confusion_matrix(y_test,model_pred):
    conf_mat = confusion_matrix(y_true=y_test, y_pred=model_pred)
    ax= plt.subplot()
    sns.heatmap(conf_mat, annot=True, ax = ax, fmt = 'g', cmap = 'Greens'); 
    ax.set_xlabel('Predicted');ax.set_ylabel('Expected'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['Benign', 'Malignant']);
    ax.yaxis.set_ticklabels(['Benign', 'Malignant']);
    print(classification_report(y_test,model_pred))
    
    
def plot_2d_space(X, y, label='Classes'):   
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
    
    
