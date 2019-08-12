# ML-Breast-Cancer-Classification
The goal of this project is to be able to classify a fine needle aspiration sample from a breast lesion as benign or malignant.  The sample data comes from the Wisconsin Breast Cancer Data data set.  The samples were stained to highlight cellular nuclei and then scanned using a digital camera.  The scanned images were then run through a program called Xcyt to measure the nuclear features of the cells.


## Features
Class (2 = benign, 4 = Malignant)
Radius
Texture
Perimeter
Area
Smoothness( Local variation in radius lengths)
Compactness(perimeter^2/area -1)
Concavity (severity of concave portions of the contour)
Concave points(number of concave portions of the contour)
Symmetry
Fractal Dimension
Clump Thickness
Uniformity of cell size
Uniformity of cell shape
Marginal Adhesion
Single Epithelial cell size
Bare nuclei
Bland chromatin 
Normal Nucleoli
Mitosis

## 

## Methods and Models to be used
I will first be using PCA to reduce the amount of features and then use the chosen principal components to run through the AutoML TPOT to find the optimal classification model to use for the data.
I went two different routes in this project.  One route used PCA for dimensionaltiy reduction before classification and the other just used the raw features.  In both cases the class imbalance was dealt with using SMOTE oversampling.  In the case of the PCA version, logistic regression was used after a gridsearch was done to find the best hyperparameters.  In the non-PCA version, the AutoML TPOT was used to find the optimal classification model and hyperparameters for classification which ended up being the extra trees classifier.




## Results
The baseline accuracy of the data was 66% by the zero rule algorithm.  The PCA/linear regression model resulted in 96% accuracy and an AUC of .997.  The extratrees model resulted in a 97% accuracy and an AUC of .998.  The PCA model however, ended up with a higher recall which would be preferred in this case due to wanting to catch more cases of cancer.  For a next step it would be interesting to look at the cases that were misclassified in the PCA model and see why they were classified that way.  