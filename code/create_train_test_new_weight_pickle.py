#
# create_train_test_new_weight_pickle.py
#
# Authors:
# Vibhu Jawa
# Kenneth Roe
#
from __future__ import print_function, division
# This file creates train, test files for the downstream ml algorithms
# It creates 4 type of train/test files for the  following --dataset_input
# 0: labs,
# 1: labs + demographics,
# 2: labs + demographics+interventions,
# 3: labs + demographics+interventions+triples'""")from __future__ import print_function, division
#
# It creates Pickle files with the 11 clinically relevant features
import os
import time
import shutil
import math
import pickle
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score

import directories
import argparse

parser = argparse.ArgumentParser(description='Parser to pass number of top features')
parser.add_argument('--data-set', default=0, type=int, help='Data set to pickle (0-3)')
args = parser.parse_args()

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # inte


if args.data_set==1: 
    data_source = directories.processed_data_demographics 
elif args.data_set==2:
    data_source = directories.processed_data_interventions
elif args.data_set==3:
    data_source = directories.processed_data_triples
else:
    data_source = directories.processed_data

patient_id_mortailty_df = pd.read_csv("processed_csv/died.csv");
death_percentage = np.sum((patient_id_mortailty_df['died']=='Y').values)/len(patient_id_mortailty_df)
print("Death Percentage {}".format(death_percentage))

#--Patient Intersection--#
patient_id_mortailty_df = pd.read_csv("processed_csv/died.csv");
orginal_patient_list = (patient_id_mortailty_df['id'].values);
orginal_patient_list = [str(patient_id) for patient_id in orginal_patient_list]
cleaned_patients = os.listdir(data_source);
print("Cleaned Patients: ", len(cleaned_patients))
print("Orignal Patients: ", len(orginal_patient_list))

#---Cleanup with Ken Features---#

all_functions = [' min', ' max', ' mean',' std', ' skew',' not_null_len']
my_cols = ["Calcium ionized","Calcium Ionized","Capillary refill rate","Diastolic blood pressure","Heart Rate","Lactic acid","Mean blood pressure",
           "Partial pressure of oxygen","Peak inspiratory pressure",'Lactate dehydrogenase',
           "Pupillary response left","Pupillary response right",\
           "Pupillary size left","Pupillary size right","Respiratory rate","Systolic blood pressure",
          "Urine Appearance","Urine Color","Urine output"]
cols_to_drop = []
for col in my_cols:
    for function in all_functions:
        cols_to_drop.append(col+function)
        
cols_to_drop+=['Hours','Unnamed: 0']

#--Lab Values--#

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
CO2 (ETCO2* PCO2* etc.),1.975516
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
lab_values = sorted(lab_values.items(), key=lambda x: x[1])
#--Final Patients---#

final_patients = set(cleaned_patients).intersection(set(orginal_patient_list))
len(final_patients)
final_patients = list(final_patients)
final_patients = [int(patient_id) for patient_id in final_patients]

# Divide into train and test #


final_df = patient_id_mortailty_df[patient_id_mortailty_df['id'].isin(final_patients)]
train_df=final_df.sample(frac=0.8,random_state=200)
test_df =final_df.drop(train_df.index)

train_df['died'][train_df["died"]=='Y']=1
train_df['died'][train_df["died"]=='N']=0


test_df['died'][test_df["died"]=='Y']=1
test_df['died'][test_df["died"]=='N']=0

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

new_lab_measures =[ 'Bicarbonate',
 'Blood urea nitrogen',
 'CO2 (ETCO2* PCO2* etc.)',
 'Creatinine',
 'Lactate',
 'Oxygen saturation',
 'Partial pressure of carbon dioxide',
 'Positive end-expiratory pressure',
 'Potassium',
 'White blood cell count',
 'pH']
sorted(new_lab_measures)

#--- Divide into top x features --#
for top_k_feature in [42]:
  print("Selection out of {}  Features".format(top_k_feature))
  print("---"*10)
  #--- Read into X_train and Y_train ---#

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
      temp_df_cols = ts.columns.values
      len_cols = [x for x in temp_df_cols if 'not_null_len' in x]
      ts = ts.drop(cols_to_drop+len_cols,axis = 1)
      cols_to_keep = new_lab_measures
      ts_cols_to_keep = [ ]
      for col in cols_to_keep:
        for tss_col in ts.columns.values:
          ts_col = tss_col.replace(",","*")
          if col in ts_col or (len(ts_col)>3 and ts_col[0:4]=="DRUG") or (len(ts_col)>3 and ts_col[0:4]=="PROC") or ts_col=="age" or ts_col=="sex" or ts_col=="ethnicity":
            if not(ts_col in ts_cols_to_keep):
              ts_cols_to_keep.append(ts_col.replace("*",","))
      ts=ts[ts_cols_to_keep]        
      X_train.append(ts)
      Y_train.append(row["died"])
      if c%200==0:
          print("Completed->{} / {}".format(c+1,total_rows))
      c=c+1

  #--- Read into X_test and Y_test --#
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
      temp_df_cols = ts.columns.values
      len_cols = [x for x in temp_df_cols if 'not_null_len' in x]
      ts = ts.drop(cols_to_drop+len_cols,axis = 1)
      
      ts_cols_to_keep = [ ]
      cols_to_keep = new_lab_measures
      for col in cols_to_keep:
        for tss_col in ts.columns.values:
          ts_col = tss_col.replace(",","*")
          if col in ts_col or (len(ts_col)>3 and ts_col[0:4]=="DRUG") or (len(ts_col)>3 and ts_col[0:4]=="PROC") or ts_col=="age" or ts_col=="sex" or ts_col=="ethnicity":
            if not(ts_col in ts_cols_to_keep):
              ts_cols_to_keep.append(ts_col.replace("*",","))
      ts=ts[ts_cols_to_keep]      

      X_test.append(ts)
      Y_test.append(row["died"])
      if c%200==0:
          print("Completed->{} / {}".format(c+1,total_rows))
      c=c+1

  X_np_train = np.asarray([np.asarray(x).flatten() for x in X_train])
  Y_np_train = np.asarray(Y_train)
  X_np_test = np.asarray([np.asarray(x).flatten() for x in X_test])
  Y_np_test = np.asarray(Y_test)

  print("X Test[0]",len(X_np_test[0]))
  print("X Train[0]",len(X_np_train[0]))

  # Pre-Processing Steps
  scaler = StandardScaler()
  scaler.fit(X_np_train)

  X_pr_train = scaler.transform(X_np_train)
  X_pr_test = scaler.transform(X_np_test)


  if args.data_set==1:
      pickle_save_dir = directories.pickled_data_demographics + "11_clinincally_viable_features/"
  elif args.data_set==2:
      pickle_save_dir = directories.pickled_data_interventions + "11_clinincally_viable_features/" 
  elif args.data_set==3:
      pickle_save_dir = directories.pickled_data_triples + "11_clinincally_viable_features/"
  else:
      pickle_save_dir = directories.pickled_data + "11_clinincally_viable_features/"

  if not os.path.exists(pickle_save_dir):
    os.makedirs(pickle_save_dir)

          
  pickle.dump(X_pr_train,open(pickle_save_dir+"X_pr_train.pickle", "wb"))
  pickle.dump(Y_np_train,open(pickle_save_dir+"Y_np_train.pickle", "wb"))

  pickle.dump(X_pr_test,open(pickle_save_dir+"X_pr_test.pickle", "wb"))
  pickle.dump(Y_np_test,open(pickle_save_dir+"Y_np_test.pickle", "wb"))

