#
# create_train_test_pickle.py
#
# Authors:
# Vibhu Jawa
# Kenneth Roe
#
# This file creates train, test files for the downstream ml algorithms
# It creates 4 type of train/test files for the  following --dataset_input
# 0: labs,
# 1: labs + demographics,
# 2: labs + demographics+interventions,
# 3: labs + demographics+interventions+triples'""")
#
# It creates features for the top 42,32,16,8,4,2 and 1 features

from __future__ import print_function, division
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import directories
import argparse

parser = argparse.ArgumentParser(description='Parser to pass number of top features')
parser.add_argument('--data-set', default=0, type=int,
                    help = """ Data set to pickle (0-3):
                    0: labs, 
                    1: labs + demographics,
                    2: labs + demographics+interventions,
                    3: labs + demographics+interventions+triples'""")
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

patient_id_mortailty_df = pd.read_csv(directories.processed_csv + "died.csv");
death_percentage = np.sum((patient_id_mortailty_df['died']=='Y').values)/len(patient_id_mortailty_df)
print("Death Percentage {}".format(death_percentage))

#--Patient Intersection to run on asthama patients preesent in died.csv--#
patient_id_mortailty_df = pd.read_csv(directories.processed_csv + "died.csv");
orginal_patient_list = (patient_id_mortailty_df['id'].values);
orginal_patient_list = [str(patient_id) for patient_id in orginal_patient_list]
cleaned_patients = os.listdir(data_source);
print("Cleaned Patients: ", len(cleaned_patients))
print("Orignal Patients: ", len(orginal_patient_list))

#---Functions to apply on lab values ---#
all_functions = [' min', ' max', ' mean',' std', ' skew',' not_null_len']

# columns not to drop
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

#--Lab Values with MI score to keep--#

lab_measures = '''Troponin-T,0.0170921
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
CO2 (ETCO2* PCO2* etc.),0.00384952
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
Blood culture,0.000471454'''

lab_values = dict([(lab_measure.split(',')[0], lab_measure.split(',')[1]) for lab_measure in lab_measures.split('\n')])
del lab_values['nan']
lab_list = list(lab_values.keys())
lab_list.sort()
# sorted lab values by MI score
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


#--- Divide into top x features --#
for top_k_feature in [1,2,4,8,16,32,42]:
  print("Top {}  Features".format(top_k_feature))
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
      cols_to_keep = [lab_values[i][0] for i in range(0,top_k_feature)]
      ts_cols_to_keep = [ ]
      for col in cols_to_keep:
        for tss_col in ts.columns.values:
          ts_col = tss_col.replace(",", "*")
          if (len(col)<len(ts_col) and len(col)>len(ts_col)-6 and (col==ts_col[0:len(col)])) or (len(ts_col)>3 and ts_col[0:4]=="DRUG") or (len(ts_col)>3 and ts_col[0:4]=="PROC") or ts_col=="age" or ts_col=="sex" or ts_col=="ethnicity":
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
      cols_to_keep = [lab_values[i][0] for i in range(0,top_k_feature)]
      for col in cols_to_keep:
        for tss_col in ts.columns.values:
          ts_col = tss_col.replace(",", "*")
          if (len(col)<len(ts_col) and len(col)>len(ts_col)-6 and (col==ts_col[0:len(col)])) or (len(ts_col)>3 and ts_col[0:4]=="DRUG") or (len(ts_col)>3 and ts_col[0:4]=="PROC") or ts_col=="age" or ts_col=="sex" or ts_col=="ethnicity":
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

  # write to correct directory according to the set of features to extract
  if args.data_set==1:
      pickle_save_dir = directories.pickled_data_demographics + "top_{}_features/".format(top_k_feature)
  elif args.data_set==2:
      pickle_save_dir = directories.pickled_data_interventions + "top_{}_features/".format(top_k_feature)
  elif args.data_set==3:
      pickle_save_dir = directories.pickled_data_triples + "top_{}_features/".format(top_k_feature)
  else:
      pickle_save_dir = directories.pickled_data + "top_{}_features/".format(top_k_feature)
  if not os.path.exists(pickle_save_dir):
    os.makedirs(pickle_save_dir)

          
  pickle.dump(X_pr_train,open(pickle_save_dir+"X_pr_train.pickle", "wb"))
  pickle.dump(Y_np_train,open(pickle_save_dir+"Y_np_train.pickle", "wb"))

  pickle.dump(X_pr_test,open(pickle_save_dir+"X_pr_test.pickle", "wb"))
  pickle.dump(Y_np_test,open(pickle_save_dir+"Y_np_test.pickle", "wb"))

