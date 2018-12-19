# Running Machine Learning experiments

Machine learning experiments to understand the impact of ranking and filtering laboratory result features according to discriminative score, on the interpretability and performance of logistic regression and gradient boosting machine learning models. Experiments are conducted with a severe asthma mortality prediction case study.

## Requirments
* Python 3.6  
* Pandas
* Sklearn library

## Getting Paitent Mortality Labels and Discriminative Scores
Three preprocessed data files MedsConnections.csv, ProcedureConnection.csv and died.csv can be downloaded from [Here](data/preprocessed).

## Steps to run the experiments:

1. Copy the MIMIC III .csv files described [Here](data/input/README) into the directory data/input

2. Carry out steps 2-5 of the [Mimic 3 benchmark](mimic3benchmarks) scripts to generate episode
   data using the version of the scripts in this repository. (Note: MIMIC III
   benchmark scripts come from here with edits:
   https://github.com/YerevaNN/mimic3-benchmarks)
```
       cd mimic3benchmarks
       export PYTHONPATH=$PYTHONPATH:[PATH TO DIRECTORY]
       python scripts/extract_subjects.py ../data/input ../data/all_patients
       python scripts/validate_events.py ../data/all_patients
       python scripts/extract_episodes_from_subjects.py ../data/all_patients
```

3. Run the following [script](code/extract_asthma.py) to exctract asthma patients
```
       cd code
       python extract_asthma.py
```

   
4. Run the following scripts in the [codes](code) directory to summarize hourly data for 48 hours in the ICU

```
       # extracts just lab values
       python extract_hourly.py
       
       # extracts labs+data demographics
       python extract_hourly_intervention.py --data-set=1
       
       # excracts labs+demographics+interventions
       python extract_hourly_intervention.py --data-set=2
       
       # extracts labs+demographics+interventions+triples
       python extract_hourly_intervention.py --data-set=3
```

5. Create (Train/Test) split in  pickle file format  by  running the following [script](code/create_train_test_new_weight_pickle.py)
with these parameters where the following correspond: 
  * 0: labs
  * 1: labs + demographics
  * 2: labs + demographics+interventions
  * 3: labs+demographics+interventions+triples
```
       python create_train_test_new_weight_pickle.py
       python create_train_test_pickle.py
       python create_train_test_new_weight_pickle.py --data-set=1
       python create_train_test_pickle.py --data-set=1
       python create_train_test_new_weight_pickle.py --data-set=2
       python create_train_test_pickle.py --data-set=2
       python create_train_test_new_weight_pickle.py --data-set=3
       python create_train_test_pickle.py --data-set=3
```
6. Run the followin python scripts to run the machine learning modelling experiments (each take 12-20 hours to run):

### [Logistic Regression](code/logistic_regression.py): 
```   
# logistic regression, 11 clinically relevant labs:
python logistic_regression.py --top-fc=0 > ../data/output/logistic_regression0.out

#logistic regression, 42 lab features:      
python logistic_regression.py > ../data/output/logistic_regression1.out

# logistic regression, top 32 lab features:
python logistic_regression.py --top-fc=32 > ../data/output/logistic_regression2.out

#logistic regression, top 16 lab features:
python logistic_regression.py --top-fc=16 > ../data/output/logistic_regression3.out

# logistic regression, top 8 lab features:
python logistic_regression.py --top-fc=8 > ../data/output/logistic_regression4.out
   
# logistic regression, top 4 lab features:
python logistic_regression.py --top-fc=4 > ../data/output/logistic_regression5.out
   
# logistic regression, top 2 lab features:
python logistic_regression.py --top-fc=2 > ../data/output/logistic_regression6.out
   
# logistic regression, top 1 lab feature:
python logistic_regression.py --top-fc=1 > ../data/output/logistic_regression7.out
   
# logistic regression, 11 clinically relevant labs, demographics:
python logistic_regression.py --top-fc=0 --data-set=1 > ../data/output/logistic_regression0_d.out

# logistic regression, 42 labs, demographics:
python logistic_regression.py --data-set=1 > ../data/output/logistic_regression1_d.out

# logistic regression, top 32 labs, demographics:
python logistic_regression.py --top-fc=32 --data-set=1 > ../data/output/logistic_regression2_d.out

# logistic regression, top 16 labs, demographics:
python logistic_regression.py --top-fc=16 --data-set=1 > ../data/output/logistic_regression3_d.out

# logistic regression, 11 clinically relevant labs, demographics+interventions:
python logistic_regression.py --top-fc=0 --data-set=2 > ../data/output/logistic_regression0_i.out
   
# logistic regression, 42 labs, demographics+interventions:
python logistic_regression.py --data-set=2 > ../data/output/logistic_regression1_i.out

# logistic regression, top 32 labs, demographics+interventions:
python logistic_regression.py --top-fc=32 --data-set=2 > ../data/output/logistic_regression2_i.out
   
# logistic regression, top 16 labs, demographics+interventions:
python logistic_regression.py --top-fc=16 --data-set=2 > ../data/output/logistic_regression3_i.out

# logistic regression, 11 clinically relevant labs, demographics+interventions+triples:
python logistic_regression.py --top-fc=0 --data-set=3 > ../data/output/logistic_regression0_t.out

# logistic regression, 42 labs, demographics+interventions+triples:
python logistic_regression.py --data-set=3 > ../data/output/logistic_regression1_t.out
   
# logistic regression, top 32 labs, demographics+interventions+triples:
python logistic_regression.py --top-fc=32 --data-set=3 > ../data/output/logistic_regression2_t.out

# logistic regression, top 16 labs, demographics+interventions+triples:
python logistic_regression.py --top-fc=16 --data-set=3 > ../data/output/logistic_regression3_t.out
```

### [Gradient Boosting](code/gradient_boosting.py): 

```
# gradient boosting, 11 clinically relevant labs:
python gradient_boosting.py --top-fc=0 > ../data/output/gradient_boosting0.out

# gradient boosting, 42 labs:
python gradient_boosting.py > ../data/output/gradient_boosting1.out
   
# gradient boosting, top 32 labs:
python gradient_boosting.py --top-fc=32 > ../data/output/gradient_boosting2.out
   
# gradient boosting, top 16 labs:
python gradient_boosting.py --top-fc=16 > ../data/output/gradient_boosting3.out

# gradient boosting, top 8 labs:
python gradient_boosting.py --top-fc=8 > ../data/output/gradient_boosting4.out
   
# gradient boosting, top 4 labs:
python gradient_boosting.py --top-fc=4 > ../data/output/gradient_boosting5.out

# gradient boosting, top 2 labs:
python gradient_boosting.py --top-fc=2 > ../data/output/gradient_boosting6.out

# gradient boosting, top 1 lab:
python gradient_boosting.py --top-fc=1 > ../data/output/gradient_boosting7.out
   
# gradient boosting, 11 clinically relevant labs+demographics:
python gradient_boosting.py --top-fc=0 --data-set=1 > ../data/output/gradient_boosting0_d.out
   
# gradient boosting, 42 labs+demographics:
python gradient_boosting.py --data-set=1 > ../data/output/gradient_boosting1_d.out
   
# gradient boosting, 32 labs+demographics:
python gradient_boosting.py --top-fc=32 --data-set=1 > ../data/output/gradient_boosting2_d.out
   
# gradient boosting, 16 labs+demographics:
python gradient_boosting.py --top-fc=16 --data-set=1 > ../data/output/gradient_boosting3_d.out
   
# gradient boosting, 11 clinically relevant labs+demographics+interventions:
python gradient_boosting.py --top-fc=0 --data-set=2 > ../data/output/gradient_boosting0_i.out
   
# gradient boosting, 42 labs+demographics+interventions:
python gradient_boosting.py --data-set=2 > ../data/output/gradient_boosting1_i.out

# gradient boosting, top 32 labs+demographics+interventions:
python gradient_boosting.py --top-fc=32 --data-set=2 > ../data/output/gradient_boosting2_i.out
   
# gradient boosting, top 16 labs+demographics+interventions:
python gradient_boosting.py --top-fc=16 --data-set=2 > ../data/output/gradient_boosting3_i.out

# gradient boosting, 11 clinically relevant labs+demographics+interventions+triples:
python gradient_boosting.py --top-fc=0 --data-set=3 > ../data/output/gradient_boosting0_t.out
 
# gradient boosting, 42 labs+demographics+interventions+triples:
python gradient_boosting.py --data-set=3 > ../data/output/gradient_boosting1_t.out

# gradient boosting, 32 labs+demographics+interventions+triples:
python gradient_boosting.py --top-fc=32 --data-set=3 > ../data/output/gradient_boosting2_t.out

# gradient boosting, 16 labs+demographics+interventions+triples:
python gradient_boosting.py --top-fc=16 --data-set=3 > ../data/output/gradient_boosting3_t.out
```

## License
By committing your code to the MIMICIII-ML-experiments code repository you agree to release the code under the [MIT License](https://github.com/translational-informatics/MIMICIII-ML-experiments/blob/master/LICENSE) attached to the repository.
