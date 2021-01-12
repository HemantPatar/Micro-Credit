# Micro-Credit
https://nbviewer.jupyter.org/github/HemantPatar/Micro-Credit/blob/main/Micro_Credit_Defaulter.ipynb
Micro credit defaulter prediction classification project (internship)

Data set size 209593 rows × 36 columns.

Data cleaning & data visualization for model training.

Seprating input and output from dataset.

Feature importance using ExtraTreesClassifier.
plot graph of feature importances.

Handling Outliers using KNNImputer.

Normalise the skewed data.

Standardisation of input is done using StandardScaler.

Train test split.

Function that runs the requested algorithm and returns the accuracy metrics.

Use (LogisticRegression,KNeighborsClassifier,GaussianNB,DecisionTreeClassifier,RandomForestClassifier,GradientBoostingClassifier)

Logistic Regression find best parameters (penaalty,class_weight,C,intercept_scaling) using RandomizedSearchCV 
find max r2 score for Logistic Regression.
Find mean validation score.
Find accuracy,classification report.
plot ROC curve.

KNN Regression find best parameters (n_neighbors,n_jobs) using RandomizedSearchCV 
find max r2 score for KNN Regression.
Find mean validation score.
Find accuracy,classification report.
plot ROC curve.

Gaussian Naive Bayes Regression 
find max r2 score for Gaussian Naive Bayes Regression.
Find mean validation score.
Find accuracy,classification report.
plot ROC curve.

SGDClassifier find best parameters (n_jobs) using RandomizedSearchCV 
find max r2 score for Linear SGDClassifier.
Find mean validation score.
Find accuracy,classification report.

Decision Tree Classifier
find max r2 score for Decision Tree Classifier.
Find mean validation score.
Find accuracy,classification report.
plot ROC curve.

Random Forest Classifier find best parameters (max_depth,max_features,min_samples_split,min_samples_leaf,bootstrap,criterion) using RandomizedSearchCV
find max r2 score for Random Forest Classifier.
Find mean validation score.
Find accuracy,classification report.
plot ROC curve.

GradientBoostingClassifier
find max r2 score for GradientBoostingClassifier.
Find mean validation score.
Find accuracy,classification report.
plot ROC curve.

Compare accuracy score of all the models.
Compare ROC curve of all the above models by ploting.

Use(Random Forest Classifier)
Imbalance is countered using SMOTE.

Get prediction, confusion matrix, f1 score, classification report, AUC ROC score.

Save model (pickle).

