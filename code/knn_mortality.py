# 
# knn_mortality.py
#
# Authors:
# Vibhu Jawa
# Kenneth Roe
#
# This file applies knn model for mortality using knn and unweighted features
from __future__ import print_function, division
import os
import time
import shutil
import pickle
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
import argparse

import directories

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # inte

import directories

parser = argparse.ArgumentParser(description='Parser to pass number of top features')
parser.add_argument('--data-set', default=0, type=int, help='Data set to pickle (0-3)')
args = parser.parse_args()

if args.data_set==1:
    data_source = directories.processed_data_demographics
elif args.data_set==2:
    data_source = directories.processed_data_interventions
elif args.data_set==3:
    data_source = directories.processed_data_triples
else:
    data_source = directories.processed_data
print(data_source)


patient_id_mortailty_df = pd.read_csv(directories.processed_csv+"died.csv");

patient_id_mortailty_df.head()

death_percentage = np.sum((patient_id_mortailty_df['died']=='Y').values)/len(patient_id_mortailty_df)
print("Death Percentage {}".format(death_percentage))

patient_id_mortailty_df = pd.read_csv(directories.processed_csv+"died.csv");
orginal_patient_list = (patient_id_mortailty_df['id'].values);
orginal_patient_list = [str(patient_id) for patient_id in orginal_patient_list]
cleaned_patients = os.listdir(data_source);
print("Cleaned Patients: ", len(cleaned_patients))
print("Orignal Patients: ", len(orginal_patient_list))

# Cleaned patients

final_patients = set(cleaned_patients).intersection(set(orginal_patient_list))
len(final_patients)
final_patients = list(final_patients)
final_patients = [int(patient_id) for patient_id in final_patients]
print(len(final_patients))

# Divide into train and test

final_df = patient_id_mortailty_df[patient_id_mortailty_df['id'].isin(final_patients)]
train_df=final_df.sample(frac=0.8,random_state=200)
test_df =final_df.drop(train_df.index)

train_df['died'][train_df["died"]=='Y']=1
train_df['died'][train_df["died"]=='N']=0


test_df['died'][test_df["died"]=='Y']=1
test_df['died'][test_df["died"]=='N']=0

train_df.to_csv(directories.processed_csv+'mortality_gradient_boosting_train.csv')
test_df.to_csv(directories.processed_csv+'mortality_gradient_boosting_test.csv')

# checking on ratio of  label= 1 for test and train
print("Train Death/Total ratio {}".format(np.sum(train_df["died"]==1)/len(train_df)))
print("Test Death/Total Ratio {}".format(np.sum(test_df["died"]==1)/len(test_df)))

# Read Dataset into a single variable
def find_file(data_location):
    episodes = os.listdir(data_location)
    if len(episodes)==0:
        print(data_location)
        return "no_data"
    episodes.sort()
    return os.path.join(data_location,episodes[-1])

find_file(directories.processed_data+"52695")

X_train = []
Y_train = []
c = 0 
total_rows = len(train_df)
for index,row in train_df.iterrows():
    data_location = os.path.join(data_source,str(row['id']))
    ts_data = find_file(data_location)
    if ts_data == 'no_data':
        continue;
    ts = pd.read_csv(ts_data)
    ts = ts.iloc[:,1:]
    X_train.append(ts)
    Y_train.append(row["died"])
    if c%200==0:
        print("Completed->{} / {}".format(c+1,total_rows))
    c=c+1

X_test = []
Y_test = []
c = 0 
total_rows = len(test_df)
for index,row in test_df.iterrows():
    data_location = os.path.join(data_source,str(row['id']))
    ts_data = find_file(data_location)
    if ts_data == 'no_data':
        continue;
    ts = pd.read_csv(ts_data)
    ts = ts.iloc[:,1:]
    X_test.append(ts)
    Y_test.append(row["died"])
    if c%200==0:
        print("Completed->{} / {}".format(c+1,total_rows))
    c=c+1

print("Pre processing steps")
X_np_train = np.asarray([np.asarray(x).flatten() for x in X_train])
Y_np_train = np.asarray(Y_train)

X_np_test = np.asarray([np.asarray(x).flatten() for x in X_test])
Y_np_test = np.asarray(Y_test)

# Pre processing steps
scaler = StandardScaler()
scaler.fit(X_np_train)
X_pr_train = scaler.transform(X_np_train)
X_pr_test = scaler.transform(X_np_test)

# Model training
# %time
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import KFold
# from sklearn.model_selection import GridSearchCV



clf = KNeighborsClassifier()
svc_grid = GridSearchCV(estimator=clf,scoring = 'roc_auc', param_grid=dict(n_neighbors=[1,3,5,7,9,11,13]),cv =10,n_jobs =-1)

svc_grid

print("Main learning")
#time
import pickle
import os

svc_grid.fit(X_pr_train, Y_np_train)

print("Saving results")
#curr_dir = 'mortality_models/saved_sklearn_models'
#save_dir = os.path.join(os.path.expanduser('~'),curr_dir)
save_dir = directories.profile_directory
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print(save_dir)
pickle.dump(svc_grid,open(save_dir+'knn.pickle',"wb"))

#curr_dir = 'mortality_models/saved_sklearn_models'
#save_dir = os.path.join(os.path.expanduser('~'),curr_dir)

#svc_grid = pickle.load(open( save_dir+'/knn.pickle', "rb" ))

#pd.DataFrame(svc_grid.cv_results_)

# Train scores
y_pred = svc_grid.predict(X_pr_train)
print('Classification Report \n',classification_report(Y_np_train, y_pred))
print("-"*10)
print('CONFUSION MATRIX \n',confusion_matrix(Y_np_train, y_pred))
print("-"*10)
pred_scores = svc_grid.best_estimator_.predict_proba(X_pr_train)
print('Roc value\n',roc_auc_score(Y_np_train,pred_scores[:,1]))

# Test scores
y_pred = svc_grid.predict(X_pr_test)
print('Classification Report \n',classification_report(Y_np_test, y_pred))
print("-"*10)
print('CONFUSION MATRIX \n',confusion_matrix(Y_np_test, y_pred))
print("-"*10)
pred_scores = svc_grid.predict_proba(X_pr_test)
print('Roc value\n',roc_auc_score(Y_np_test,pred_scores[:,1]))

