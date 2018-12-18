# 
# logistic_regression.py
#
# Authors:
# Vibhu Jawa
# Kenneth Roe
#
# This file applies Logistic Regression to given train/test pickle
# It applies basic grid search with 10 fold cross validation

import argparse
import pickle
import directories
import time

from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score

parser = argparse.ArgumentParser(description='Parser to pass number of top features')
parser.add_argument('--top-fc', default=42, type=int, help='top_feature_count')
parser.add_argument('--data-set', default=0, type=int, help='Data set to pickle (0-3)')
args = parser.parse_args()

if args.top_fc==0:
    tail = "11_clinincally_viable_features/"
else:
    tail = "top_{}_features/".format(args.top_fc)

if args.data_set==1:
    feature_dir = directories.pickled_data_demographics+tail
elif args.data_set==2:
    feature_dir = directories.pickled_data_interventions+tail
elif args.data_set==3:
    feature_dir = directories.pickled_data_triples+tail
else:
    feature_dir = directories.pickled_data+tail
print(feature_dir)

from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

X_pr_train=[]
X_pr_test=[]
Y_np_train=[]
Y_np_test=[]

with open(feature_dir+"X_pr_train.pickle", 'rb') as handle:
  X_pr_train = pickle.load(handle)

with open(feature_dir+"X_pr_test.pickle", 'rb') as handle:
  X_pr_test = pickle.load(handle)

with open(feature_dir+"Y_np_train.pickle", 'rb') as handle:
  Y_np_train = pickle.load(handle)

with open(feature_dir+"Y_np_test.pickle", 'rb') as handle:
  Y_np_test = pickle.load(handle)


penalties = ['l1','l2']

c_list = [i*0.01 for i in range(1,21)]
clf = LogisticRegression()
svc_grid = GridSearchCV(estimator=clf,scoring = 'roc_auc',param_grid=dict(penalty=penalties,C =c_list,max_iter=[100,200,400]),cv =10,n_jobs =10)

print(svc_grid)
print("---"*10)

## %%time
import pickle
import os

print("Started Training")
print("---"*10)

svc_grid.fit(X_pr_train, Y_np_train)
save_dir = os.path.join(feature_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
pickle.dump(svc_grid,open(save_dir+'/logistic_regression.pickle',"wb"))
print("Saved File")

#---- Train Scores ----#

ll = svc_grid.best_estimator_.coef_.tolist()[0]

c = 0
for l in ll:
    if l<-0.00001 or l>0.0001:
        c = c+1

print("Coefficients")
print(ll)
print("Coefficients %d"%len(ll))
print("Non-zero coefficients %d"%c)

y_pred = svc_grid.predict(X_pr_train)
print("Train Scores")
print("---"*10)
print('Classification Report \n',classification_report(Y_np_train, y_pred))
print("-"*10)
print('CONFUSION MATRIX \n',confusion_matrix(Y_np_train, y_pred))
print("-"*10)
pred_scores = svc_grid.best_estimator_.predict_proba(X_pr_train)
print('Roc value\n',roc_auc_score(Y_np_train,pred_scores[:,1]))

#-- Test Scores---#
print("Test Scores")
print("---"*10)

y_pred = svc_grid.predict(X_pr_test)
print('Classification Report \n',classification_report(Y_np_test, y_pred))
print("-"*10)
print('CONFUSION MATRIX \n',confusion_matrix(Y_np_test, y_pred))
print("-"*10)
pred_scores = svc_grid.predict_proba(X_pr_test)
print('Roc value\n',roc_auc_score(Y_np_test,pred_scores[:,1]))


