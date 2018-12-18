# 
# roc_graph3.py
#
# Authors:
# Vibhu Jawa
# Kenneth Roe
#
from __future__ import print_function, division
import os
import time
import shutil
import math
import pickle
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score

import directories

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

dir_11=directories.pickled_data+'11_clinincally_viable_features/'
svc_grid_11 = pickle.load(open(dir_11+'logistic_regression.pickle', "rb" ))
x_pr_test_11 = pickle.load(open(dir_11+'X_pr_test.pickle', "rb" ))
y_pr_test_11 = pickle.load(open(dir_11+'Y_np_test.pickle', "rb" ))
y_probas_11_l = svc_grid_11.best_estimator_.predict_proba(x_pr_test_11)

dir_11d=directories.pickled_data_demographics+'11_clinincally_viable_features/'
svc_grid_11d = pickle.load(open(dir_11d+'logistic_regression.pickle', "rb" ))
x_pr_test_11d = pickle.load(open(dir_11d+'X_pr_test.pickle', "rb" ))
y_pr_test_11d = pickle.load(open(dir_11d+'Y_np_test.pickle', "rb" ))
y_probas_11d_l = svc_grid_11d.best_estimator_.predict_proba(x_pr_test_11d)

dir_11i=directories.pickled_data_interventions+'11_clinincally_viable_features/'
svc_grid_11i = pickle.load(open(dir_11i+'logistic_regression.pickle', "rb" ))
x_pr_test_11i = pickle.load(open(dir_11i+'X_pr_test.pickle', "rb" ))
y_pr_test_11i = pickle.load(open(dir_11i+'Y_np_test.pickle', "rb" ))
y_probas_11i_l = svc_grid_11i.best_estimator_.predict_proba(x_pr_test_11i)

dir_11t=directories.pickled_data_triples+'11_clinincally_viable_features/'
svc_grid_11t = pickle.load(open(dir_11t+'logistic_regression.pickle', "rb" ))
x_pr_test_11t = pickle.load(open(dir_11t+'X_pr_test.pickle', "rb" ))
y_pr_test_11t = pickle.load(open(dir_11t+'Y_np_test.pickle', "rb" ))
y_probas_11t_l = svc_grid_11t.best_estimator_.predict_proba(x_pr_test_11t)

dir_42=directories.pickled_data+'top_42_features/'
svc_grid_42 = pickle.load(open(dir_42+'logistic_regression.pickle', "rb" ))
x_pr_test_42 = pickle.load(open(dir_42+'X_pr_test.pickle', "rb" ))
y_pr_test_42 = pickle.load(open(dir_42+'Y_np_test.pickle', "rb" ))
y_probas_42_l = svc_grid_42.best_estimator_.predict_proba(x_pr_test_42)

dir_42d=directories.pickled_data_demographics+'top_42_features/'
svc_grid_42d = pickle.load(open(dir_42d+'logistic_regression.pickle', "rb" ))
x_pr_test_42d = pickle.load(open(dir_42d+'X_pr_test.pickle', "rb" ))
y_pr_test_42d = pickle.load(open(dir_42d+'Y_np_test.pickle', "rb" ))
y_probas_42d_l = svc_grid_42d.best_estimator_.predict_proba(x_pr_test_42d)

dir_42i=directories.pickled_data_interventions+'top_42_features/'
svc_grid_42i = pickle.load(open(dir_42i+'logistic_regression.pickle', "rb" ))
x_pr_test_42i = pickle.load(open(dir_42i+'X_pr_test.pickle', "rb" ))
y_pr_test_42i = pickle.load(open(dir_42i+'Y_np_test.pickle', "rb" ))
y_probas_42i_l = svc_grid_42i.best_estimator_.predict_proba(x_pr_test_42i)

fig=plt.figure( figsize=(10.5,5), dpi= 100)
ax = plt.subplot(121)

from matplotlib.font_manager import FontProperties

# # #Logistic Regresion ##


# # method I: plt
fontP = FontProperties()
fontP.set_size('small')
plt.title('Receiver Operating Characteristic (Logistic regression)')

preds = y_probas_11_l[:,1]
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_pr_test_11, preds)
roc_auc = sklearn.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label = str('Logistic Regression [num_f = 11 viable] = %0.3f' % roc_auc))

preds = y_probas_11d_l[:,1]
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_pr_test_11d, preds)
roc_auc1 = sklearn.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label = str('Logistic Regression [num_f = 11d viable] = %0.3f' % roc_auc1))

preds = y_probas_11i_l[:,1]
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_pr_test_11i, preds)
roc_auc2 = sklearn.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label = str('Logistic Regression [num_f = 11i viable] = %0.3f' % roc_auc2))

preds = y_probas_11t_l[:,1]
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_pr_test_11t, preds)
roc_auc3 = sklearn.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label = str('Logistic Regression [num_f = 11t viable] = %0.3f' % roc_auc3))

preds = y_probas_42_l[:,1]
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_pr_test_42, preds)
roc_auc4 = sklearn.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label = str('Logistic Regression [num_f = 42] = %0.3f' % roc_auc4))

preds = y_probas_42d_l[:,1]
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_pr_test_42d, preds)
roc_auc5 = sklearn.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label = str('Logistic Regression [num_f = 42d] = %0.3f' % roc_auc5))

preds = y_probas_42i_l[:,1]
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_pr_test_42i, preds)
roc_auc6 = sklearn.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label = str('Logistic Regression [num_f = 42i] = %0.3f' % roc_auc6))

s1 = str('Labs, 11 filtered features (AUC %0.3f)' % roc_auc)
s2 = str('Labs+demo, 11 filtered features (AUC %0.3f)' % roc_auc1)
s3 = str('Labs+demo+events, 11 filtered features (AUC %0.3f)' % roc_auc2)
s4 = str('Labs+demo+events+triples, 11 filtered features (AUC %0.3f)' % roc_auc3)
s5 = str('Labs, 42 features (AUC %0.3f)' % roc_auc4)
s6 = str('Labs+demo, 42 features (AUC %0.3f)' % roc_auc5)
s7 = str('Labs+demo+events, 42 features (AUC %0.3f)' % roc_auc6)

#plt.legend(("num_f = 42", "num_f = 8"), loc='upper left', prop=fontP,bbox_to_anchor=(1.05, 1.05))
plt.legend((s1,s2,s3,s4,s5,s6,s7), loc=2, borderaxespad=0., bbox_to_anchor=(1.05, 1))



plt.grid()
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

plt.savefig("roc3.png")
