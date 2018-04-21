#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages',
                 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi']  # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df[features_list]
df['poi'] = df['poi'].astype('uint8')
df = df.replace("NaN", np.nan, regex=True)


# df = df.apply(pd.to_numeric)

def remove_outlier(df_in):
    q1 = df_in.quantile(0.25)
    q3 = df_in.quantile(0.75)
    iqr = q3 - q1
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    df_out = df_in[(df_in > fence_low) & (df_in < fence_high)]
    return df_out


d_poi = df['poi']

df = df.drop(['poi'], axis=1)
df = df.apply(remove_outlier, axis=0)
df['poi'] = d_poi
df = df.fillna(0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = df.to_dict(orient='index')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# print (features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

nb_Clf = Pipeline([('scaling', StandardScaler()),
                   ('pca', PCA(n_components=2, whiten=True)),
                   ('clf', GaussianNB())])

l_Clf = Pipeline([('scaling', StandardScaler()),
                  ('pca', PCA(n_components=2, whiten=True)),
                  ('clf', LogisticRegression())])

svm_Clf = Pipeline([('scaling', StandardScaler()),
                    ('pca', PCA(n_components=2, whiten=True)),
                    ('clf', SVC())])

knn_Clf = Pipeline([('scaling', StandardScaler()),
                    ('pca', PCA(n_components=2, whiten=True)),
                    ('clf', KNeighborsClassifier())])

lda_Clf = Pipeline([('scaling', StandardScaler()),
                    ('pca', PCA(n_components=2, whiten=True)),
                    ('clf', LinearDiscriminantAnalysis())])

rf_Clf = Pipeline([('scaling', StandardScaler()),
                   ('pca', PCA(n_components=2, whiten=True)),
                   ('clf', RandomForestClassifier(max_depth=2, random_state=0))])

models = []
models.append(('LR', l_Clf))
models.append(('LDA', lda_Clf))
models.append(('KNN', knn_Clf))
models.append(('RF', rf_Clf))
models.append(('NB', nb_Clf))
models.append(('SVM', svm_Clf))

results = []
names = []
scoring = 'accuracy'
df1 = df
df1 = df.drop(['poi'], axis=1)
X = df1.values
Y = df['poi']
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: Mean Accuracy %f" % (name, cv_results.mean())
    print(msg)

### Task 5: Tune your classifier to achfieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
clf = RandomForestClassifier(max_depth=100, min_samples_split=2)
test_classifier(clf, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)
