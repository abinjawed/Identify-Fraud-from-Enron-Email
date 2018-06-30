#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing, svm
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
from sklearn.preprocessing import FunctionTransformer, minmax_scale


import matplotlib.pyplot as plt

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'to_messages',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi', 'shared_receipt_with_poi']  # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df.drop(['email_address'], axis=1)
print(df.groupby('poi').count())
df = df.replace("NaN", np.nan, regex=True)
df['poi'] = df['poi'].astype('uint8')
print(df.isnull().sum(axis = 0).sum())

d_poi = df['poi']
df = df.drop(['poi'], axis=1)
df = df.fillna(0)
#pt = PowerTransformer(method='box-cox', standardize=False)
rng = np.random.RandomState(304)

fig = plt.figure()

# plt.subplot(2, 2, 1)
# plt.scatter(df["salary"], df["bonus"])
# plt.xlabel("Salary")
# plt.ylabel("Bonus")
#
# plt.subplot(2, 2, 2)
# plt.scatter(df["total_payments"], df["expenses"])
# plt.xlabel("Total Payments")
# plt.ylabel("Expenses")
#
# plt.subplot(2, 2, 3)
# plt.scatter(df["total_stock_value"], df["deferred_income"])
# plt.xlabel("Total Stock Value")
# plt.ylabel("Deferred Income")
#
# plt.subplot(2, 2, 4)
# plt.scatter(df["from_poi_to_this_person"], df["from_this_person_to_poi"])
# plt.xlabel("From POI to this person")
# plt.ylabel("From this person to POI")
#
# plt.show()


df = df[10000000 > df["bonus"]]
df = df[df["total_payments"] < 10000000]

# fig = plt.figure()
# #
# plt.subplot(2, 2, 1)
# plt.hist(df["bonus"])
# plt.title("Bonus")
#
# plt.subplot(2, 2, 2)
# plt.hist(df["salary"])
# plt.title("Salary")
# #
#
# plt.subplot(2, 2, 3)
# plt.hist(df["total_payments"])
# plt.title("Total Payments")
# #
# plt.subplot(2, 2, 4)
# plt.hist(df["expenses"])
# plt.title("Expenses")
#
# #
# plt.show()

# df["bonus"] = np.sqrt(df["bonus"])
# df["salary"] = np.sqrt(df["salary"])
# df["total_payments"] = np.sqrt(df["total_payments"])
# df["expenses"] = np.sqrt(df["expenses"])

# fig = plt.figure()
# #
# plt.subplot(2, 2, 1)
# plt.hist(df["bonus"])
# plt.title("Bonus")
#
# plt.subplot(2, 2, 2)
# plt.hist(df["salary"])
# plt.title("Salary")
# #
#
# plt.subplot(2, 2, 3)
# plt.hist(df["total_payments"])
# plt.title("Total Payments")
# #
# plt.subplot(2, 2, 4)
# plt.hist(df["expenses"])
# plt.title("Expenses")
#
# #
# plt.show()


from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(2)
#df = poly.fit_transform(df)
#df = pd.DataFrame(df)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
df['poi'] = d_poi
df = df.fillna(0)
#df["salary/bonus"] = df["salary"]/df["bonus"]
#df[['salary/bonus']] = df[['salary']].div(df.bonus, axis=0)

my_dataset = df.to_dict(orient='index')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

nb_Clf = Pipeline([('scaling', StandardScaler()),
                   ('pca', PCA()),
                   ('clf', GaussianNB())])

l_Clf = Pipeline([('scaling', StandardScaler()),
                  ('pca', PCA()),
                  ('clf', LogisticRegression())])

svm_Clf = Pipeline([('scaling', StandardScaler()),
                    ('pca', PCA()),
                    ('clf', SVC())])

knn_Clf = Pipeline([('scaling', StandardScaler()),
                    ('pca', PCA()),
                    ('clf', KNeighborsClassifier())])

lda_Clf = Pipeline([('scaling', StandardScaler()),
                    ('pca', PCA()),
                    ('clf', LinearDiscriminantAnalysis())])

rf_Clf = Pipeline([('scaling', StandardScaler()),
                   ('pca', PCA()),
                   ('clf', RandomForestClassifier())])

models = [('LR', l_Clf), ('LDA', lda_Clf), ('KNN', knn_Clf), ('RF', rf_Clf), ('NB', nb_Clf), ('SVM', svm_Clf)]

results = []
names = []
scoring = 'accuracy'
df1 = df

f_list = df.columns
f_list = f_list[1:len(f_list)-1]
f_list = map(str, f_list)
f_features = ['poi']
for fn in f_list:
    f_features.append(fn)
print (f_features)

test_classifier(l_Clf, my_dataset, f_features, folds=45)
test_classifier(lda_Clf, my_dataset, f_features, folds=45)
test_classifier(knn_Clf, my_dataset, f_features, folds=45)
test_classifier(rf_Clf, my_dataset, f_features, folds=45)

print("\n\n")
df['salary_bonus_ratio'] = df.salary.div(df.bonus)
df.loc[~np.isfinite(df['salary_bonus_ratio']), 'salary_bonus_ratio'] = 0
df['salary_expense_ratio'] = df.salary.div(df.expenses)
df.loc[~np.isfinite(df['salary_expense_ratio']), 'salary_expense_ratio'] = 0

features_list.append('salary_bonus_ratio')
features_list.append('salary_expense_ratio')

df = df[features_list]
df = df.apply(np.sqrt, axis = 1)
df = df.fillna(0)

my_dataset = df.to_dict(orient='index')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

nb_Clf = Pipeline([('scaling', StandardScaler()),
                   ('pca', PCA()),
                   ('clf', GaussianNB())])

l_Clf = Pipeline([('scaling', StandardScaler()),
                  ('pca', PCA()),
                  ('clf', LogisticRegression())])

svm_Clf = Pipeline([('scaling', StandardScaler()),
                    ('pca', PCA()),
                    ('clf', SVC())])

knn_Clf = Pipeline([('scaling', StandardScaler()),
                    ('pca', PCA()),
                    ('clf', KNeighborsClassifier())])

lda_Clf = Pipeline([('scaling', StandardScaler()),
                    ('pca', PCA()),
                    ('clf', LinearDiscriminantAnalysis())])

rf_Clf = Pipeline([('scaling', StandardScaler()),
                   ('pca', PCA()),
                   ('clf', RandomForestClassifier())])

f_list = df.columns
f_list = f_list[1:len(f_list)-1]
f_list = map(str, f_list)

models = [('LR', l_Clf), ('LDA', lda_Clf), ('KNN', knn_Clf), ('RF', rf_Clf), ('NB', nb_Clf), ('SVM', svm_Clf)]

results = []
names = []
scoring = 'accuracy'
df1 = df
#df1 = df.drop(['poi'], axis=1)
X = df1.values
Y = df['poi']

print("\n\n")
test_classifier(l_Clf, my_dataset, features_list, folds=45)
test_classifier(lda_Clf, my_dataset, features_list, folds=45)
test_classifier(knn_Clf, my_dataset, features_list, folds=45)
test_classifier(rf_Clf, my_dataset, features_list, folds=45)

print("\n\n")


nb_Clf = Pipeline([('scaling', StandardScaler()),
                   ('pca', PCA(n_components=3, whiten=True)),
                   ('clf', GaussianNB())])

penalty = ['l1', 'l2']
C = [0.001, 0.01, 0.1, 1, 2, 5, 10, 100, 1000]
param_grid = {'C': C, 'penalty': penalty}
ls_clf = GridSearchCV(LogisticRegression(), param_grid)
X = df1.values
Y = df['poi']
ls_clf.fit(X, Y)
print("Best score: %0.3f" % ls_clf.best_score_)
print("Best parameters set:")
best_parameters = ls_clf.best_estimator_.get_params()
print (best_parameters)

l_Clf = Pipeline([
    #LogisticRegression(C=10, penalty='l1', intercept_scaling= 1, solver='liblinear', multi_class='ovr')
    # ('scaling', StandardScaler()),
                  ('pca', PCA()),
                  ('clf', LogisticRegression(C=10, penalty='l1', max_iter=100, intercept_scaling=1, fit_intercept=False, solver='liblinear', multi_class='ovr'))])

svm_Clf = Pipeline([('scaling', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', SVC())])

params = {"n_neighbors": np.arange(1, 31, 2),
	"metric": ["euclidean", "cityblock"]}

k_Clf = GridSearchCV(KNeighborsClassifier(), params, cv=5, verbose=0)
knn_Clf = Pipeline([('scaling', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', KNeighborsClassifier())])

lda_Clf = Pipeline([('scaling', StandardScaler()),
                    ('pca', PCA(n_components=4, whiten=True)),
                    ('clf', LinearDiscriminantAnalysis(n_components=50))])

rf_Clf = Pipeline([('scaling', StandardScaler()),
                   ('pca', PCA(n_components=2, whiten=True)),
                   ('clf', RandomForestClassifier(max_features=0.2))])

models = [('LR', l_Clf), ('LDA', lda_Clf), ('KNN', knn_Clf), ('RF', rf_Clf), ('NB', nb_Clf), ('SVM', svm_Clf)]

results = []
names = []
scoring = 'accuracy'
df1 = df
#df1 = df.drop(['poi'], axis=1)

print ("\n\n")
test_classifier(l_Clf, my_dataset, features_list, folds=45)
test_classifier(lda_Clf, my_dataset, features_list, folds=45)
test_classifier(knn_Clf, my_dataset, features_list, folds=45)
test_classifier(rf_Clf, my_dataset, features_list, folds=45)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(l_Clf, my_dataset, features_list)
