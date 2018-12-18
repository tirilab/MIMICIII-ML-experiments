# 
# extract_asthma.py
#
# Author:
# Kenneth Roe
#
# This file extracts the asthama patients using those present in died.csv
import os
import csv
import directories

data_source_dir  = directories.all_episode_data
data_target_dir = directories.episode_data

#os.mkdir(data_target_dir)

with open(directories.processed_csv+'died.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row)
        if len(row)>0:
            source = data_source_dir+"/"+row['id']
            target = data_target_dir+"/"+row['id']
            os.spawnlp(os.P_WAIT, 'cp', 'cp', '-r', source, target)

