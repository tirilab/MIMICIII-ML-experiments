# 
# knn_mortality_weighted_ken_features.py
#
# Authors:
# Vibhu Jawa
# Kenneth Roe
#
# This file applies knn model for mortality using knn and weighted features  using mI score

from __future__ import print_function, division
import os
import time
import shutil
import pickle
import math
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
import argparse
import directories

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

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


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # inte

import directories

old_lab_measures = '''
Troponin-T,0.0170921
Basophils,0.0164497
Lymphocytes,0.0159563
Neutrophils,0.0154964
Lactate dehydrogenase,0.0149952
Alkaline phosphate,0.0134134
Platelets,0.0114053
Troponin-I,0.010767
Alanine aminotransferase,0.0102186
Hemoglobin,0.0101575
Bilirubin,0.0100716
Albumin,0.00954687
Monocytes,0.00942504
Blood urea nitrogen,0.00922118
Asparate aminotransferase,0.00914682
Eosinophils,0.00792003
Hematocrit,0.00737431
Mean corpuscular volume,0.00643329
Bicarbonate,0.00636398
Phosphate,0.0063263
Mean corpuscular hemoglobin concentration,0.00615807
Sodium,0.00470533
Anion gap,0.00466117
Positive end-expiratory pressure,0.00417201
White blood cell count,0.00416711
CO2 (ETCO2- PCO2- etc.),0.00384952
Oxygen saturation,0.00373707
Magnesium,0.00368481
Prothrombin time,0.00319972
Partial thromboplastin time,0.00278742
nan,0.00264979
Mean corpuscular hemoglobin,0.00240326
Partial pressure of carbon dioxide,0.00236824
Chloride,0.0023635
Lactate,0.00225831
Calcium,0.00201718
Potassium,0.00161767
pH,0.00157639
Cholesterol,0.00155214
Glucose,0.00150908
Red blood cell count,0.000738692
Creatinine,0.000734525
Blood culture,0.000471454
'''

lab_measures = '''Hematocrit,3.207493
Mean corpuscular hemoglobin,2.295840
Partial thromboplastin time,0.767677
Bilirubin,1.182294
Lactate,0.866825
Alanine aminotransferase,1.031776
Lymphocytes,1.929227
Alkaline phosphate,0.932960
Monocytes,1.850016
Anion gap,3.075508
pH,1.244274
Albumin,0.053121
Chloride,2.654501
Glucose,3.053178
Red blood cell count,1.325231
Bicarbonate,3.351181
White blood cell count,2.323280
Sodium,2.331572
Platelets,1.226600
Partial pressure of carbon dioxide,1.594997
Lactate dehydrogenase,2.030204
Oxygen saturation,1.485634
Troponin-T,0.064730
Eosinophils,0.137325
Mean corpuscular hemoglobin concentration,3.891900
Asparate aminotransferase,2.488124
Calcium,2.592858
Phosphate,2.015443
Magnesium,1.400330
Cholesterol,0.693147
Basophils,1.614006
Neutrophils,1.301875
Troponin-I,0.094101
CO2 (ETCO2- PCO2- etc.),1.975516
Prothrombin time,2.686391
Hemoglobin,2.488968
Blood culture,0.000000
Blood urea nitrogen,1.504837
Mean corpuscular volume,2.564504
Positive end-expiratory pressure,1.184735
Creatinine,0.866140
Potassium,1.779490'''

lab_values = dict([(lab_measure.split(',')[0], lab_measure.split(',')[1]) for lab_measure in lab_measures.split('\n')])
#del lab_values['nan']
lab_list = list(lab_values.keys())
lab_list.sort()

# Lab measures weighted
weights = np.asarray([np.asarray([[float(lab_mesausre)]* 6 for lab_mesausre in lab_values.values()]).flatten()] * 48).flatten()

# Lab_list to be dropped
all_functions = [' min', ' max', ' mean',' std', ' skew',' not_null_len']
my_cols = ["Calcium ionized","Calcium Ionized","Capillary refill rate","Diastolic blood pressure","Heart Rate","Lactic acid","Mean blood pressure",
           "Partial pressure of oxygen","Peak inspiratory pressure",
           "Pupillary response left","Pupillary response right",\
           "Pupillary size left","Pupillary size right","Respiratory rate","Systolic blood pressure",
          "Urine Appearance","Urine Color","Urine output"]
cols_to_drop = []
for col in my_cols:
    for function in all_functions:
        cols_to_drop.append(col+function)
cols_to_drop+=['Hours','Unnamed: 0']

patient_id_mortailty_df = pd.read_csv(directories.processed_csv+"died.csv");

patient_id_mortailty_df.head()

death_percentage = np.sum((patient_id_mortailty_df['died']=='Y').values)/len(patient_id_mortailty_df)
print("Death Percentage {}".format(death_percentage))

patient_id_mortailty_df = pd.read_csv("processed_csv/died.csv");
orginal_patient_list = (patient_id_mortailty_df['id'].values);
orginal_patient_list = [str(patient_id) for patient_id in orginal_patient_list]
cleaned_patients = os.listdir(data_source);
print("Cleaned Patients: ", len(cleaned_patients))
print("Orignal Patients: ", len(orginal_patient_list))

# Cleaned_patients
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

# Read Datset into a single variable
def find_file(data_location):
    episodes = os.listdir(data_location)
    if len(episodes)==0:
        print(data_location)
        return "no_data"
    episodes.sort()
    return os.path.join(data_location,episodes[-1])

train_df.columns

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
    ts = ts.drop(cols_to_drop,axis = 1)
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
    ts = ts.drop(cols_to_drop,axis = 1)
    X_test.append(ts)
    Y_test.append(row["died"])
    if c%200==0:
        print("Completed->{} / {}".format(c+1,total_rows))
    c=c+1

print("Pre processing")

len(X_test[0].columns)

X_np_train = np.asarray([np.asarray(x).flatten() for x in X_train])
Y_np_train = np.asarray(Y_train)

X_np_test = np.asarray([np.asarray(x).flatten() for x in X_test])
Y_np_test = np.asarray(Y_test)

# Pre Processing Steps
scaler = StandardScaler()
scaler.fit(X_np_train)
# weighing step
X_pr_train = scaler.transform(X_np_train) * weights
X_pr_test = scaler.transform(X_np_test) * weights

print("Main learning")
# Model Training
#time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV



clf = KNeighborsClassifier()
svc_grid = GridSearchCV(estimator=clf,scoring = 'roc_auc', param_grid=dict(n_neighbors=[7,9,11,13,15]),cv =10,n_jobs =15)

svc_grid

#%%time
import pickle
import os

svc_grid.fit(X_pr_train, Y_np_train)
#curr_dir = 'mortality_models/ken_features/saved_sklearn_models'
#save_dir = os.path.join(os.path.expanduser('~'),curr_dir)
save_dir = directories.profile_directory
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print(save_dir)
pickle.dump(svc_grid,open(save_dir+'/knn_weighted.pickle',"wb"))

#curr_dir = 'mortality_models/ken_features/saved_sklearn_models'
#save_dir = os.path.join(os.path.expanduser('~'),curr_dir)

#svc_grid = pickle.load(open( save_dir+'/knn_weighted.pickle', "rb" ))

#pd.DataFrame(svc_grid.cv_results_)

# Train scores
# print Model Metrics
y_pred = svc_grid.predict(X_pr_train)
print('Classification Report \n',classification_report(Y_np_train, y_pred))
print("-"*10)
print('CONFUSION MATRIX \n',confusion_matrix(Y_np_train, y_pred))
print("-"*10)
pred_scores = svc_grid.best_estimator_.predict_proba(X_pr_train)
print('Roc value\n',roc_auc_score(Y_np_train,pred_scores[:,1]))

len(X_pr_test[0])
# Test scores
y_pred = svc_grid.predict(X_pr_test)
print('Classification Report \n',classification_report(Y_np_test, y_pred))
print("-"*10)
print('CONFUSION MATRIX \n',confusion_matrix(Y_np_test, y_pred))
print("-"*10)
pred_scores = svc_grid.predict_proba(X_pr_test)
print('Roc value\n',roc_auc_score(Y_np_test,pred_scores[:,1]))


