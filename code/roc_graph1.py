# 
# roc_graph1.py
#
# Authors:
# Vibhu Jawa
# Kenneth Roe
#
# creates the roc curve for logistic regression classifiers
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

dir_42=directories.pickled_data+'top_42_features/'
svc_grid_42 = pickle.load(open(dir_42+'logistic_regression.pickle', "rb" ))
x_pr_test_42 = pickle.load(open(dir_42+'X_pr_test.pickle', "rb" ))
y_pr_test_42 = pickle.load(open(dir_42+'Y_np_test.pickle', "rb" ))
y_probas_42_l = svc_grid_42.best_estimator_.predict_proba(x_pr_test_42)

dir_32=directories.pickled_data+'top_32_features/'
svc_grid_32 = pickle.load(open(dir_32+'logistic_regression.pickle', "rb" ))
x_pr_test_32 = pickle.load(open(dir_32+'X_pr_test.pickle', "rb" ))
y_pr_test_32 = pickle.load(open(dir_32+'Y_np_test.pickle', "rb" ))
y_probas_32_l = svc_grid_32.best_estimator_.predict_proba(x_pr_test_32)

dir_16=directories.pickled_data+'top_16_features/'
svc_grid_16 = pickle.load(open(dir_16+'logistic_regression.pickle', "rb" ))
x_pr_test_16 = pickle.load(open(dir_16+'X_pr_test.pickle', "rb" ))
y_pr_test_16 = pickle.load(open(dir_16+'Y_np_test.pickle', "rb" ))
y_probas_16_l = svc_grid_16.best_estimator_.predict_proba(x_pr_test_16)

dir_8=directories.pickled_data+'top_8_features/'
svc_grid_8 = pickle.load(open(dir_8+'logistic_regression.pickle', "rb" ))
x_pr_test_8 = pickle.load(open(dir_8+'X_pr_test.pickle', "rb" ))
y_pr_test_8 = pickle.load(open(dir_8+'Y_np_test.pickle', "rb" ))
y_probas_8_l = svc_grid_8.best_estimator_.predict_proba(x_pr_test_8)

dir_4=directories.pickled_data+'top_4_features/'
svc_grid_4 = pickle.load(open(dir_4+'logistic_regression.pickle', "rb" ))
x_pr_test_4 = pickle.load(open(dir_4+'X_pr_test.pickle', "rb" ))
y_pr_test_4 = pickle.load(open(dir_4+'Y_np_test.pickle', "rb" ))
y_probas_4_l = svc_grid_4.best_estimator_.predict_proba(x_pr_test_4)

dir_2=directories.pickled_data+'top_2_features/'
svc_grid_2 = pickle.load(open(dir_2+'logistic_regression.pickle', "rb" ))
x_pr_test_2 = pickle.load(open(dir_2+'X_pr_test.pickle', "rb" ))
y_pr_test_2 = pickle.load(open(dir_2+'Y_np_test.pickle', "rb" ))
y_probas_2_l = svc_grid_2.best_estimator_.predict_proba(x_pr_test_2)

fig=plt.figure( figsize=(10.5,5), dpi= 100)

from matplotlib.font_manager import FontProperties

# # #Logistic Regresion ##


# # method I: plt
fontP = FontProperties()
fontP.set_size('small')

ax = plt.subplot(121)

plt.title('Receiver Operating Characteristic')

preds = y_probas_11_l[:,1]
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_pr_test_11, preds)
roc_auc = sklearn.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label = str('Logistic Regression [num_f = 11 viable] = %0.3f' % roc_auc))

preds = y_probas_42_l[:,1]
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_pr_test_42, preds)
roc_auc1 = sklearn.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label = str('Logistic Regression [num_f = 42] = %0.3f' % roc_auc1))

preds = y_probas_32_l[:,1]
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_pr_test_32, preds)
roc_auc2 = sklearn.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label = str('Logistic Regression [num_f = 32] = %0.3f' % roc_auc2))

preds = y_probas_16_l[:,1]
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_pr_test_16, preds)
roc_auc3 = sklearn.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label = str('Logistic Regression [num_f = 16] = %0.3f' % roc_auc3))

preds = y_probas_8_l[:,1]
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_pr_test_8, preds)
roc_auc4 = sklearn.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label = str('Logistic Regression [num_f = 8] = %0.3f' % roc_auc4))

preds = y_probas_4_l[:,1]
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_pr_test_4, preds)
roc_auc5 = sklearn.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label = str('Logistic Regression [num_f = 4] = %0.3f' % roc_auc5))

preds = y_probas_2_l[:,1]
fpr, tpr, threshold = sklearn.metrics.roc_curve(y_pr_test_2, preds)
roc_auc6 = sklearn.metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label = str('Logistic Regression [num_f = 2] = %0.3f' % roc_auc6))

s1 = str('Logistic regression (Labs, 11 filtered features, AUC %0.3f)' % roc_auc)
s2 = str('Logistic regression (Labs, 42 features, AUC %0.3f)' % roc_auc1)
s3 = str('Logistic regression (Labs, 32 features, AUC %0.3f)' % roc_auc2)
s4 = str('Logistic regression (Labs, 16 features, AUC %0.3f)' % roc_auc3)
s5 = str('Logistic regression (Labs, 8 features, AUC %0.3f)' % roc_auc4)
s6 = str('Logistic regression (Labs, 4 features, AUC %0.3f)' % roc_auc5)
s7 = str('Logistic regression (Labs, 2 features, AUC %0.3f)' % roc_auc6)

#plt.legend(("num_f = 42", "num_f = 8"), loc='upper left', prop=fontP,bbox_to_anchor=(1.05, 1.05))
plt.legend((s1,s2,s3,s4,s5,s6,s7), loc=2, borderaxespad=0., bbox_to_anchor=(1.05, 1))



plt.grid()
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

plt.savefig("roc1.png")
