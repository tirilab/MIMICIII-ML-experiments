# 
# extract_hourly.py
#
# Authors:
# Vibhu Jawa
# Kenneth Roe
#
# This file extracts  the features for each hourly time series buckets
# by appluing the following functions to the reading during that hour
# [(' min',min), (' max',max), (' mean',np.mean),
# (' std',np.std), (' skew',skew)


import pandas as pd
import os
import numpy as np
import os
from scipy.stats import skew
import directories

data_source_dir  = directories.episode_data
data_target_dir = "c/"
data_ts_dir = directories.processed_data

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


all_functions = [(' min',min), (' max',max), (' mean',np.mean),
                 (' std',np.std), (' skew',skew), (' not_null_len',len)]

functions_map = {
    "all": all_functions,
    "len": [len],
    "all_but_len": all_functions[:-1]
}

periods_map = {
    "all": (0, 0, 1, 0),
    "first4days": (0, 0, 0, 4*24),
    "first8days": (0, 0, 0, 8*24),
    "last12hours": (1, -12, 1, 0),
    "first25percent": (2, 25),
    "first50percent": (2, 50)
}

patient_ids  = os.listdir(data_source_dir)
len(patient_ids)

# Imputes time series data for asthama patients , (backward, forward)

count_patient = 0 
for i  in range(0,len(patient_ids)):
    patient_id=patient_ids[i]
    patient_target_dir = data_target_dir + patient_id
    if not os.path.exists(patient_target_dir):
        os.makedirs(patient_target_dir)
        
    episode_timeseries = [x for x in  os.listdir(data_source_dir+patient_id) \
                          if ('timeseries' in x) and ('extracted_values') not in x]
    for episode in  episode_timeseries:
        episode_pd = pd.read_csv(data_source_dir+patient_id+'/'+episode)
        event_hour_list = []
        for hours in range(0,48):
            episode_hour_dict={}
            curr_event_pd = episode_pd[(episode_pd['Hours']>=hours)  & (episode_pd['Hours']<hours+1)]
            for col in curr_event_pd.columns:
                if col!='Hours':
                    curr_array = curr_event_pd[curr_event_pd[col].notnull()][col].values
                    curr_array = [float(x) for x in curr_array if isfloat(x)]
                    for function in all_functions:
                        column = col+str(function[0])
                        if len(curr_array)!=0:
                            episode_hour_dict[column]=np.apply_along_axis(function[1], 0, curr_array)
                        else:
                             episode_hour_dict[column]=0
                else:
                    episode_hour_dict[col]=hours
            event_hour_list.append(episode_hour_dict)

        event_hour_df = pd.DataFrame(event_hour_list)
        for col in event_hour_df.columns:
            if 'not_null_len' not in col:
                event_hour_df[col] = event_hour_df[col].fillna(method = 'backfill')
                event_hour_df[col] = event_hour_df[col].fillna(method = 'ffill')
        event_hour_df=event_hour_df.fillna(0)
        event_hour_df.head()
        event_hour_df.to_csv(patient_target_dir+'/extracted_values_'+episode)
    count_patient =  count_patient + 1
    if count_patient%10==0:
        print("Processed Patents {} out of {}".format(count_patient,len(patient_ids)))

not_extracted_ids = []
for i in range(0,len(patient_ids)):
    curr_files=os.listdir(data_target_dir+patient_ids[i])
    if all("extracted_values" not in file for file in  curr_files) :
        not_extracted_ids.append((patient_ids[i],i))

not_first_stay = []
for i in range(0,len(patient_ids)):
    curr_files=os.listdir(data_source_dir+patient_ids[i])
    if all("extracted_valuesepisode1_timeseries" not in file for file in  curr_files) :
        not_first_stay.append((patient_ids[i],i))

#os.listdir(data_source_dir+"1620")

#episode_pd = pd.read_csv(data_source_dir+"1620"+"/episode1_timeseries.csv")
#len(episode_pd.columns)

patient_ids_not_written_f = open(directories.processed_csv+'patient_ids_not_written.txt', 'w')
for patient_id,_ in not_extracted_ids:
    patient_ids_not_written_f.write("%s\n" % patient_id)
patient_ids_not_written_f.close()
os.getcwd()  

file = open("processed_csv/patient_ids_not_written.txt", "r") 
print(file.read())

# copy first occurence to another directory
import shutil
curr_dir = data_target_dir
new_dir = data_ts_dir
for patient in  patient_ids:
    ls = os.listdir(curr_dir+patient)
    extracted_ts = [x for x in ls if 'extracted_values_episode' in x ]
    if len(extracted_ts) == 0:
        print("no episode found for patiend_if = {}".format(patient))
    else:
        if not os.path.exists(new_dir+patient):
            os.makedirs(new_dir+patient)
        shutil.copy(curr_dir+patient+"/"+extracted_ts[0],new_dir+patient+"/ts1.csv")


