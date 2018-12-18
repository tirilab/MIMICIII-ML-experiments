# 
# extract_hourly_intervention.py
#
# Authors:
# Vibhu Jawa
# Kenneth Roe
#
# This file extracts the hourly intervation for patients
import pandas as pd
import os
import numpy as np
import os
from scipy.stats import skew
import directories
import csv
import argparse

parser = argparse.ArgumentParser(description='Parser to pass number of top features')
parser.add_argument('--data-set', default=0, type=int, help='Data set package (1-3)')
args = parser.parse_args()

if args.data_set==1:
    augment_interventions = False
    augment_triples = False
    augment_demographics = True
    data_source_dir  = directories.episode_data
    data_target_dir = "c/"
    data_ts_dir = directories.processed_data_demographics
elif args.data_set==2:
    augment_interventions = True
    augment_triples = False
    augment_demographics = True
    data_source_dir  = directories.episode_data
    data_target_dir = "d/"
    data_ts_dir = directories.processed_data_interventions
elif args.data_set==3:
    augment_interventions = False
    augment_triples = True
    augment_demographics = True
    data_source_dir  = directories.episode_data
    data_target_dir = "e/"
    data_ts_dir = directories.processed_data_triples
else:
    exit()


def read_itemid_to_variable_map(fn, variable_column='LEVEL2'):
    var_map = pd.DataFrame.from_csv(fn, index_col=None).fillna('').astype(str)
    #var_map[variable_column] = var_map[variable_column].apply(lambda s: s.lower())
    var_map.COUNT = var_map.COUNT.astype(int)
    var_map = var_map.ix[(var_map[variable_column] != '') & (var_map.COUNT>0)]
    var_map = var_map.ix[(var_map.STATUS == 'ready')]
    var_map.ITEMID = var_map.ITEMID.astype(int)
    var_map = var_map[[variable_column, 'ITEMID', 'MIMIC LABEL']].set_index('ITEMID')
    return var_map.rename_axis({variable_column: 'VARIABLE', 'MIMIC LABEL': 'MIMIC_LABEL'}, axis=1)

def map_itemids_to_variables(events, var_map):
    return events.merge(var_map, left_on='ITEMID', right_index=True)

def get_events_for_stay(events, icustayid, intime=None, outtime=None):
    idx = (events.ICUSTAY_ID == icustayid)
    if intime is not None and outtime is not None:
        idx = idx | ((events.CHARTTIME >= intime) & (events.CHARTTIME <= outtime))
    events = events.ix[idx]
    del events['ICUSTAY_ID']
    return events

def add_hours_elpased_to_events(events, dt, remove_charttime=True):
    events['HOURS'] = (pd.to_datetime(events.CHARTTIME) - dt).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60
    if remove_charttime:
        del events['CHARTTIME']
    return events

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

# Read procedure and medication connections
medsConnections = pd.DataFrame.from_csv(directories.processed_csv+"MedsConnections.csv",index_col=None).fillna('').astype(str)
proceduresConnection = pd.DataFrame.from_csv(directories.processed_csv+"ProceduresConnection.csv",index_col=None).fillna('').astype(str)
medicationTriples = {}
for x in range(0,medsConnections.shape[0]):
    medicationTriples.setdefault(medsConnections["Prescription"][x],[]).append(medsConnections["Lab"][x])
procedureTriples = {}
for x in range(0,proceduresConnection.shape[0]):
    procedureTriples.setdefault(proceduresConnection["Procedure"][x],[]).append(proceduresConnection["Lab"][x])
# Imputes time series data for asthama patients , (backward, forward)
#print(medicationTriples)
print(procedureTriples)

count_patient = 0 

procedures = pd.DataFrame.from_csv(directories.mimic_iii_data+"PROCEDUREEVENTS_MV.csv",index_col=None).fillna('').astype(str)
procedures.STARTTIME = pd.to_datetime(procedures.STARTTIME)
procedures.ENDTIME = pd.to_datetime(procedures.ENDTIME)

d_item = pd.DataFrame.from_csv(directories.mimic_iii_data+"D_ITEMS.csv",index_col=None).fillna('').astype(str)
d_item = d_item[["ITEMID","LABEL"]]
#procedures = procedures.merge(d_item, left_on='ITEMID', right_index=True)
procedure_map = {}
for x in range(0,d_item.shape[0]):
    procedure_map[d_item["ITEMID"][x]] = d_item["LABEL"][x]

prescriptions = pd.DataFrame.from_csv(directories.mimic_iii_data+"PRESCRIPTIONS.csv",index_col=None).fillna('').astype(str)
prescriptions.STARTDATE = pd.to_datetime(prescriptions.STARTDATE)
prescriptions.ENDDATE = pd.to_datetime(prescriptions.ENDDATE)

for i  in range(0,len(patient_ids)):
    patient_id=patient_ids[i]
    print("Patient "+str(patient_id))
    idx = procedures.SUBJECT_ID==str(patient_id)
    p = procedures[idx]
    idx = prescriptions.SUBJECT_ID==str(patient_id)
    pr = prescriptions[idx]
    #print("Procedures")
    #print(p)
    #print("Prescriptions")
    #print(pr)

    patient_target_dir = data_target_dir + patient_id
    if not os.path.exists(patient_target_dir):
        os.makedirs(patient_target_dir)
        
    episode_timeseries = [x for x in  os.listdir(data_source_dir+patient_id) \
                          if ('timeseries' in x) and ('extracted_values') not in x]
    try:
        stays = pd.read_csv(data_source_dir+patient_id+"/stays.csv")
        events = pd.read_csv(data_source_dir+patient_id+"/events.csv")
        events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
        stays.INTIME = pd.to_datetime(stays.INTIME)
        stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
        map = read_itemid_to_variable_map("mimic3benchmarks/resources/itemid_to_variable_map.csv")
        events = map_itemids_to_variables(events, map)

        variables = map.VARIABLE.unique
        #print(stays)
    except FileNotFoundError:
        episode_timeseries = []

    for episode in  episode_timeseries:
        episode_pd = pd.read_csv(data_source_dir+patient_id+'/'+episode)
        s = episode
        while (s[0]<'0'or s[0]>'9'):
            s = s[1:]
        n = 0
        while n < len(s) and s[n]>='0' and s[n]<='9': 
            n = n+1
        #print(n)
        #print(s)
        n = int(s[0:n])-1
        #print(n)
        intime = stays['INTIME'][n]
        outtime = stays['OUTTIME'][n]
        ethnicity = stays['ETHNICITY'][n]
        ethnicity_n = 0
        if len(ethnicity)>=5 and ethnicity[:5]=="WHITE":
            ethnicity_n = 1
        elif len(ethnicity)>=5 and ethnicity[0:5]=="BLACK":
            ethnicity_n = 2
        elif len(ethnicity)>=5 and ethnicity[0:5]=="ASIAN":
            ethnicity_n = 3
        elif len(ethnicity)>=8 and ethnicity[0:8]=="HISPANIC":
            ethnicity_n = 4
        elif len(ethnicity)>=5 and ethnicity[0:5]=="MULTI":
            ethnicity_n = 4

        age = stays['AGE'][n]
        if age < 2:
            age_bucket = 0
        elif age < 17:
            age_bucket = 1
        elif age < 35:
            age_bucket = 2 
        elif age < 50:
            age_bucket = 3
        elif age < 70:
            age_bucket = 4
        else:
            age_bucket = 5
        if stays['GENDER'][n]=="F":
            sex = 0
        else:
            sex = 1

        icu_stay_id = stays['ICUSTAY_ID'][n]
        #print(intime)
        #print(outtime)
        #print(ethnicity)
        #print(age)
        #print(icu_stay_id)
        idx = (p.ICUSTAY_ID == icu_stay_id)
        if intime is not None and outtime is not None:
            idx = idx | ((p.STARTTIME >= intime) & (p.STARTTIME <= outtime))
        p_stay = p[idx]
        p_stay['HOURS'] = (pd.to_datetime(p.STARTTIME) - intime).apply(lambda s: s / np.timedelta64(1, 's')) / 60./60
        idx = (pr.ICUSTAY_ID == icu_stay_id)
        if intime is not None and outtime is not None:
            idx = idx | ((pr.STARTDATE >= intime) & (pr.STARTDATE <= outtime))
        pr_stay = pr[idx]
        #print("Procedures and drugs")
        #print(p_stay)
        #print(pr_stay)

        event_hour_list = []
        for hours in range(0,48):
            #print("HOUR %d"%hours)

            episode_hour_dict={}
            curr_event_pd = episode_pd[(episode_pd['Hours']>=hours)  & (episode_pd['Hours']<hours+1)]
            for col in curr_event_pd.columns:
                #print(col)
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
            #print(episode_hour_dict)
            event_hour_list.append(episode_hour_dict)
        event_hour_df = pd.DataFrame(event_hour_list)
        for col in event_hour_df.columns:
            if 'not_null_len' not in col:
                event_hour_df[col] = event_hour_df[col].fillna(method = 'backfill')
                event_hour_df[col] = event_hour_df[col].fillna(method = 'ffill')
        event_hour_df=event_hour_df.fillna(0)
        event_hour_df.head()
        if augment_interventions:
            drug_used = {}
            #print("DRUGS")
            for d in medicationTriples.keys():
                drug_used[d] = 0
            for x in pr_stay["DRUG"]:
                try:
                    drug_used[x] = drug_used[x]+1
                    #print("Using drug "+str(x))
                except KeyError:
                    pass
                #print(x)
            for d in drug_used.keys():
                l = []
                if drug_used[d] > 0:
                    print("Using "+d)
                for x in range(0,48):
                    l.append(drug_used[d])
                event_hour_df["DRUG_USED_"+d] = l
            procedure_used = {}
            #print("PROCEEDURES")
            for d in procedureTriples.keys():
                procedure_used[d] = 0
            for x in p_stay["ITEMID"]:
                try:
                    xx = procedure_map[x]
                    print("Testing procedure "+str(xx))
                    try:
                        procedure_used[xx] = procedure_used[xx]+1
                        print("Using procedure "+str(xx))
                    except KeyError:
                        pass
                except KeyError:
                    print("Procedure key error "+str(x))
                #print(x)
            for d in procedure_used.keys():
                l = []
                if procedure_used[d] > 0:
                    print("Using procedure2 "+d)
                for x in range(0,48):
                    l.append(procedure_used[d])
                event_hour_df["PROCEDURE_USED_"+d] = l
        if augment_triples:
            drug_used = {}
            #print("DRUGS")
            for d in medicationTriples.keys():
                 drug_used[d] = 0
            for x in pr_stay["DRUG"]:
                try:
                    print("Using drug "+str(x))
                    drug_used[x] = drug_used[x]+1
                except KeyError:
                    pass
                #print(x)
            for d in drug_used.keys():
                #print("LABS FOR "+d)
                for lab in medicationTriples[d]:
                    #print(lab)
                    l = []
                    if drug_used[d] > 0:
                        for x in range(0,48):
                            try:
                                l.append(event_hour_df[lab+" mean"][x])
                            except KeyError:
                                l.append(1)
                                #print("No lab")
                    else:
                        for x in range(0,48):
                            l.append(0)
                    event_hour_df["DRUG_LAB_"+d+"_"+lab] = l
            procedure_used = {}
            #print("PROCEDURES")
            for d in procedureTriples.keys():
                 procedure_used[d] = 0
            for x in p_stay["ITEMID"]:
                try:
                    xx = procedure_map[x]
                    print("Testing procedure "+str(xx))
                    try:
                        print("Using procedure "+str(xx))
                        procedure_used[xx] = procedure_used[xx]+1
                    except KeyError:
                        pass
                except KeyError:
                    print("Procedure key error "+str(x))
                #print(x)
            for d in procedure_used.keys():
                #print("LABS FOR "+d)
                for lab in procedureTriples[d]:
                    #print(lab)
                    l = []
                    if procedure_used[d] > 0:
                        for x in range(0,48):
                            try:
                                l.append(event_hour_df[lab+" mean"][x])
                            except KeyError:
                                l.append(1)
                                #print("No lab")
                    else:
                        for x in range(0,48):
                            l.append(0)
                    event_hour_df["PROCEDURE_LAB_"+d+"_"+lab] = l
        if augment_demographics:
            ages = []
            ethnicities = []
            sexes = []
            for x in range(0,48):
                ages.append(age_bucket)
                sexes.append(sex)
                ethnicities.append(ethnicity_n)
            event_hour_df["sex"] = pd.Series(sexes)
            event_hour_df["age"] = pd.Series(ages)
            event_hour_df["ethnicity"] = pd.Series(ethnicities)
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


