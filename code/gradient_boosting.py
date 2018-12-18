# This file applies GradientBoosting to given train/test pickle
# It applies basic grid search with 10 fold cross validation
# It also tabulates the weights for trained model

import argparse
import pickle
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
import directories

parser = argparse.ArgumentParser(description='Parser to pass number of top features')
parser.add_argument('--top-fc', default=42, type=int, help='top_feature_count')
parser.add_argument('--data-set', default=0, type=int, help='Data set pickle to use (0-3)')

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

from sklearn.ensemble import GradientBoostingClassifier
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

clf = GradientBoostingClassifier()
svc_grid = GridSearchCV(estimator=clf,scoring = 'roc_auc', param_grid=dict(max_leaf_nodes=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]),cv =10,n_jobs =10)
# svc_grid = GridSearchCV(estimator=clf,scoring = 'roc_auc', param_grid=dict(max_leaf_nodes=[3]),cv =10,n_jobs =15)

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
pickle.dump(svc_grid,open(save_dir+'/gradient_boosting_tree.pickle',"wb"))

print("Saved File")

#---- Train Scores ----#

#print(svc_grid.best_estimator_)
#print(svc_grid.best_estimator_.get_params())

#ll = svc_grid.best_estimator_.coef_.tolist()[0]

#c = 0
#for l in ll:
#    if l<-0.00001 or l>0.0001:
#        c = c+1

#print("Coefficients")
#print(ll)
#print("Coefficients %d"%len(ll))
#print("Non-zero coefficients %d"%c)

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

f_imp = svc_grid.best_estimator_.feature_importances_
print("Feature importance")
print(f_imp)
print("[")
for l in f_imp:
    print(str(l)+",")
print("]")


def tabulate_weights(features,weights):
    pos = 0
    non_zero_count = 0
    feature_total = {}
    feature_count = {}
    for x in features:
        l = len(x)
        n = x
        if x[l-4:l]==" min":
            n = x[0:l-4]
        elif x[l-4:l]==" max":
            n = x[0:l-4]
        elif x[l-5:l]==" mean":
            n = x[0:l-5]
        elif x[l-5:l]==" skew":
            n = x[0:l-5]
        elif x[l-4:l]==" std":
            n = x[0:l-4]
        c = 48
        while c > 0:
            c = c-1
            feature_total[n] = feature_total.setdefault(n,0)+weights[pos]
            feature_count[n] = feature_count.setdefault(n,0)+1
            if weights[pos]>0:
                non_zero_count = non_zero_count+1
            pos = pos+1
    k = list(feature_total.keys())
    k.sort(key= lambda x: feature_total[x])
    l = []
    for q in k:
        l.append((q,feature_total[q],feature_count[q]))
    return (l,feature_total,feature_count,non_zero_count)

if args.data_set==0:
    if args.top_fc==0:
        features = [
                     "Bicarbonate max",
                     "Bicarbonate mean",
                     "Bicarbonate min",
                     "Bicarbonate skew",
                     "Bicarbonate std",
                     "Blood urea nitrogen max",
                     "Blood urea nitrogen mean",
                     "Blood urea nitrogen min",
                     "Blood urea nitrogen skew",
                     "Blood urea nitrogen std",
                     "CO2 (ETCO2, PCO2, etc.) max",
                     "CO2 (ETCO2, PCO2, etc.) mean",
                     "CO2 (ETCO2, PCO2, etc.) min",
                     "CO2 (ETCO2, PCO2, etc.) skew",
                     "CO2 (ETCO2, PCO2, etc.) std",
                     "Creatinine max",
                     "Creatinine mean",
                     "Creatinine min",
                     "Creatinine skew",
                     "Creatinine std",
                     "Lactate max",
                     "Lactate mean",
                     "Lactate min",
                     "Lactate skew",
                     "Lactate std",
                     "Oxygen saturation max",
                     "Oxygen saturation mean",
                     "Oxygen saturation min",
                     "Oxygen saturation skew",
                     "Oxygen saturation std",
                     "Partial pressure of carbon dioxide max",
                     "Partial pressure of carbon dioxide mean",
                     "Partial pressure of carbon dioxide min",
                     "Partial pressure of carbon dioxide skew",
                     "Partial pressure of carbon dioxide std",
                     "Positive end-expiratory pressure max",
                     "Positive end-expiratory pressure mean",
                     "Positive end-expiratory pressure min",
                     "Positive end-expiratory pressure skew",
                     "Positive end-expiratory pressure std",
                     "Potassium max",
                     "Potassium mean",
                     "Potassium min",
                     "Potassium skew",
                     "Potassium std",
                     "White blood cell count max",
                     "White blood cell count mean",
                     "White blood cell count min",
                     "White blood cell count skew",
                     "White blood cell count std",
                     "pH max",
                     "pH mean",
                     "pH min",
                     "pH skew",
                     "pH std",
                   ]
    elif args.top_fc==42:
            features = [
                     "Blood culture max",
                     "Blood culture mean",
                     "Blood culture min",
                     "Blood culture skew",
                     "Blood culture std",
                     "Creatinine max",
                     "Creatinine mean",
                     "Creatinine min",
                     "Creatinine skew",
                     "Creatinine std",
                     "Red blood cell count max",
                     "Red blood cell count mean",
                     "Red blood cell count min",
                     "Red blood cell count skew",
                     "Red blood cell count std",
                     "Glucose max",
                     "Glucose mean",
                     "Glucose min",
                     "Glucose skew",
                     "Glucose std",
                     "Cholesterol max",
                     "Cholesterol mean",
                     "Cholesterol min",
                     "Cholesterol skew",
                     "Cholesterol std",
                     "pH max",
                     "pH mean",
                     "pH min",
                     "pH skew",
                     "pH std",
                     "Potassium max",
                     "Potassium mean",
                     "Potassium min",
                     "Potassium skew",
                     "Potassium std",
                     "Calcium max",
                     "Calcium mean",
                     "Calcium min",
                     "Calcium skew",
                     "Calcium std",
                     "Lactate dehydrogenase max",
                     "Lactate dehydrogenase mean",
                     "Lactate dehydrogenase min",
                     "Lactate dehydrogenase skew",
                     "Lactate dehydrogenase std",
                     "Lactate max",
                     "Lactate mean",
                     "Lactate min",
                     "Lactate skew",
                     "Lactate std",
                     "Chloride max",
                     "Chloride mean",
                     "Chloride min",
                     "Chloride skew",
                     "Chloride std",
                     "Partial pressure of carbon dioxide max",
                     "Partial pressure of carbon dioxide mean",
                     "Partial pressure of carbon dioxide min",
                     "Partial pressure of carbon dioxide skew",
                     "Partial pressure of carbon dioxide std",
                     "Mean corpuscular hemoglobin concentration max",
                     "Mean corpuscular hemoglobin concentration mean",
                     "Mean corpuscular hemoglobin concentration min",
                     "Mean corpuscular hemoglobin concentration skew",
                     "Mean corpuscular hemoglobin concentration std",
                     "Mean corpuscular hemoglobin max",
                     "Mean corpuscular hemoglobin mean",
                     "Mean corpuscular hemoglobin min",
                     "Mean corpuscular hemoglobin skew",
                     "Mean corpuscular hemoglobin std",
                     "Partial thromboplastin time max",
                     "Partial thromboplastin time mean",
                     "Partial thromboplastin time min",
                     "Partial thromboplastin time skew",
                     "Partial thromboplastin time std",
                     "Prothrombin time max",
                     "Prothrombin time mean",
                     "Prothrombin time min",
                     "Prothrombin time skew",
                     "Prothrombin time std",
                     "Magnesium max",
                     "Magnesium mean",
                     "Magnesium min",
                     "Magnesium skew",
                     "Magnesium std",
                     "Oxygen saturation max",
                     "Oxygen saturation mean",
                     "Oxygen saturation min",
                     "Oxygen saturation skew",
                     "Oxygen saturation std",
                     "CO2 (ETCO2, PCO2, etc.) max",
                     "CO2 (ETCO2, PCO2, etc.) mean",
                     "CO2 (ETCO2, PCO2, etc.) min",
                     "CO2 (ETCO2, PCO2, etc.) skew",
                     "CO2 (ETCO2, PCO2, etc.) std",
                     "White blood cell count max",
                     "White blood cell count mean",
                     "White blood cell count min",
                     "White blood cell count skew",
                     "White blood cell count std",
                     "Positive end-expiratory pressure max",
                     "Positive end-expiratory pressure mean",
                     "Positive end-expiratory pressure min",
                     "Positive end-expiratory pressure skew",
                     "Positive end-expiratory pressure std",
                     "Anion gap max",
                     "Anion gap mean",
                     "Anion gap min",
                     "Anion gap skew",
                     "Anion gap std",
                     "Sodium max",
                     "Sodium mean",
                     "Sodium min",
                     "Sodium skew",
                     "Sodium std",
                     "Phosphate max",
                     "Phosphate mean",
                     "Phosphate min",
                     "Phosphate skew",
                     "Phosphate std",
                     "Bicarbonate max",
                     "Bicarbonate mean",
                     "Bicarbonate min",
                     "Bicarbonate skew",
                     "Bicarbonate std",
                     "Mean corpuscular volume max",
                     "Mean corpuscular volume mean",
                     "Mean corpuscular volume min",
                     "Mean corpuscular volume skew",
                     "Mean corpuscular volume std",
                     "Hematocrit max",
                     "Hematocrit mean",
                     "Hematocrit min",
                     "Hematocrit skew",
                     "Hematocrit std",
                     "Eosinophils max",
                     "Eosinophils mean",
                     "Eosinophils min",
                     "Eosinophils skew",
                     "Eosinophils std",
                     "Asparate aminotransferase max",
                     "Asparate aminotransferase mean",
                     "Asparate aminotransferase min",
                     "Asparate aminotransferase skew",
                     "Asparate aminotransferase std",
                     "Blood urea nitrogen max",
                     "Blood urea nitrogen mean",
                     "Blood urea nitrogen min",
                     "Blood urea nitrogen skew",
                     "Blood urea nitrogen std",
                     "Monocytes max",
                     "Monocytes mean",
                     "Monocytes min",
                     "Monocytes skew",
                     "Monocytes std",
                     "Albumin max",
                     "Albumin mean",
                     "Albumin min",
                     "Albumin skew",
                     "Albumin std",
                     "Bilirubin max",
                     "Bilirubin mean",
                     "Bilirubin min",
                     "Bilirubin skew",
                     "Bilirubin std",
                     "Hemoglobin max",
                     "Hemoglobin mean",
                     "Hemoglobin min",
                     "Hemoglobin skew",
                     "Hemoglobin std",
                     "Alanine aminotransferase max",
                     "Alanine aminotransferase mean",
                     "Alanine aminotransferase min",
                     "Alanine aminotransferase skew",
                     "Alanine aminotransferase std",
                     "Troponin-I max",
                     "Troponin-I mean",
                     "Troponin-I min",
                     "Troponin-I skew",
                     "Troponin-I std",
                     "Platelets max",
                     "Platelets mean",
                     "Platelets min",
                     "Platelets skew",
                     "Platelets std",
                     "Alkaline phosphate max",
                     "Alkaline phosphate mean",
                     "Alkaline phosphate min",
                     "Alkaline phosphate skew",
                     "Alkaline phosphate std",
                     "Neutrophils max",
                     "Neutrophils mean",
                     "Neutrophils min",
                     "Neutrophils skew",
                     "Neutrophils std",
                     "Lymphocytes max",
                     "Lymphocytes mean",
                     "Lymphocytes min",
                     "Lymphocytes skew",
                     "Lymphocytes std",
                     "Basophils max",
                     "Basophils mean",
                     "Basophils min",
                     "Basophils skew",
                     "Basophils std",
                     "Troponin-T max",
                     "Troponin-T mean",
                     "Troponin-T min",
                     "Troponin-T skew",
                     "Troponin-T std",
                   ]
    else:
        features = []
elif args.data_set==1:
    if args.top_fc==0:
       features = [
                    "Bicarbonate max",
                    "Bicarbonate mean",
                    "Bicarbonate min",
                    "Bicarbonate skew",
                    "Bicarbonate std",
                    "sex",
                    "age",
                    "ethnicity",
                    "Blood urea nitrogen max",
                    "Blood urea nitrogen mean",
                    "Blood urea nitrogen min",
                    "Blood urea nitrogen skew",
                    "Blood urea nitrogen std",
                    "CO2 (ETCO2, PCO2, etc.) max",
                    "CO2 (ETCO2, PCO2, etc.) mean",
                    "CO2 (ETCO2, PCO2, etc.) min",
                    "CO2 (ETCO2, PCO2, etc.) skew",
                    "CO2 (ETCO2, PCO2, etc.) std",
                    "Creatinine max",
                    "Creatinine mean",
                    "Creatinine min",
                    "Creatinine skew",
                    "Creatinine std",
                    "Lactate max",
                    "Lactate mean",
                    "Lactate min",
                    "Lactate skew",
                    "Lactate std",
                    "Oxygen saturation max",
                    "Oxygen saturation mean",
                    "Oxygen saturation min",
                    "Oxygen saturation skew",
                    "Oxygen saturation std",
                    "Partial pressure of carbon dioxide max",
                    "Partial pressure of carbon dioxide mean",
                    "Partial pressure of carbon dioxide min",
                    "Partial pressure of carbon dioxide skew",
                    "Partial pressure of carbon dioxide std",
                    "Positive end-expiratory pressure max",
                    "Positive end-expiratory pressure mean",
                    "Positive end-expiratory pressure min",
                    "Positive end-expiratory pressure skew",
                    "Positive end-expiratory pressure std",
                    "Potassium max",
                    "Potassium mean",
                    "Potassium min",
                    "Potassium skew",
                    "Potassium std",
                    "White blood cell count max",
                    "White blood cell count mean",
                    "White blood cell count min",
                    "White blood cell count skew",
                    "White blood cell count std",
                    "pH max",
                    "pH mean",
                    "pH min",
                    "pH skew",
                    "pH std"
                  ]
    elif args.top_fc==42:
        features = [
                     "Blood culture max",
                     "Blood culture mean",
                     "Blood culture min",
                     "Blood culture skew",
                     "Blood culture std",
                     "sex",
                     "age",
                     "ethnicity",
                     "Creatinine max",
                     "Creatinine mean",
                     "Creatinine min",
                     "Creatinine skew",
                     "Creatinine std",
                     "Red blood cell count max",
                     "Red blood cell count mean",
                     "Red blood cell count min",
                     "Red blood cell count skew",
                     "Red blood cell count std",
                     "Glucose max",
                     "Glucose mean",
                     "Glucose min",
                     "Glucose skew",
                     "Glucose std",
                     "Cholesterol max",
                     "Cholesterol mean",
                     "Cholesterol min",
                     "Cholesterol skew",
                     "Cholesterol std",
                     "pH max",
                     "pH mean",
                     "pH min",
                     "pH skew",
                     "pH std",
                     "Potassium max",
                     "Potassium mean",
                     "Potassium min",
                     "Potassium skew",
                     "Potassium std",
                     "Calcium max",
                     "Calcium mean",
                     "Calcium min",
                     "Calcium skew",
                     "Calcium std",
                     "Lactate dehydrogenase max",
                     "Lactate dehydrogenase mean",
                     "Lactate dehydrogenase min",
                     "Lactate dehydrogenase skew",
                     "Lactate dehydrogenase std",
                     "Lactate max",
                     "Lactate mean",
                     "Lactate min",
                     "Lactate skew",
                     "Lactate std",
                     "Chloride max",
                     "Chloride mean",
                     "Chloride min",
                     "Chloride skew",
                     "Chloride std",
                     "Partial pressure of carbon dioxide max",
                     "Partial pressure of carbon dioxide mean",
                     "Partial pressure of carbon dioxide min",
                     "Partial pressure of carbon dioxide skew",
                     "Partial pressure of carbon dioxide std",
                     "Mean corpuscular hemoglobin concentration max",
                     "Mean corpuscular hemoglobin concentration mean",
                     "Mean corpuscular hemoglobin concentration min",
                     "Mean corpuscular hemoglobin concentration skew",
                     "Mean corpuscular hemoglobin concentration std",
                     "Mean corpuscular hemoglobin max",
                     "Mean corpuscular hemoglobin mean",
                     "Mean corpuscular hemoglobin min",
                     "Mean corpuscular hemoglobin skew",
                     "Mean corpuscular hemoglobin std",
                     "Partial thromboplastin time max",
                     "Partial thromboplastin time mean",
                     "Partial thromboplastin time min",
                     "Partial thromboplastin time skew",
                     "Partial thromboplastin time std",
                     "Prothrombin time max",
                     "Prothrombin time mean",
                     "Prothrombin time min",
                     "Prothrombin time skew",
                     "Prothrombin time std",
                     "Magnesium max",
                     "Magnesium mean",
                     "Magnesium min",
                     "Magnesium skew",
                     "Magnesium std",
                     "Oxygen saturation max",
                     "Oxygen saturation mean",
                     "Oxygen saturation min",
                     "Oxygen saturation skew",
                     "Oxygen saturation std",
                     "CO2 (ETCO2, PCO2, etc.) max",
                     "CO2 (ETCO2, PCO2, etc.) mean",
                     "CO2 (ETCO2, PCO2, etc.) min",
                     "CO2 (ETCO2, PCO2, etc.) skew",
                     "CO2 (ETCO2, PCO2, etc.) std",
                     "White blood cell count max",
                     "White blood cell count mean",
                     "White blood cell count min",
                     "White blood cell count skew",
                     "White blood cell count std",
                     "Positive end-expiratory pressure max",
                     "Positive end-expiratory pressure mean",
                     "Positive end-expiratory pressure min",
                     "Positive end-expiratory pressure skew",
                     "Positive end-expiratory pressure std",
                     "Anion gap max",
                     "Anion gap mean",
                     "Anion gap min",
                     "Anion gap skew",
                     "Anion gap std",
                     "Sodium max",
                     "Sodium mean",
                     "Sodium min",
                     "Sodium skew",
                     "Sodium std",
                     "Phosphate max",
                     "Phosphate mean",
                     "Phosphate min",
                     "Phosphate skew",
                     "Phosphate std",
                     "Bicarbonate max",
                     "Bicarbonate mean",
                     "Bicarbonate min",
                     "Bicarbonate skew",
                     "Bicarbonate std",
                     "Mean corpuscular volume max",
                     "Mean corpuscular volume mean",
                     "Mean corpuscular volume min",
                     "Mean corpuscular volume skew",
                     "Mean corpuscular volume std",
                     "Hematocrit max",
                     "Hematocrit mean",
                     "Hematocrit min",
                     "Hematocrit skew",
                     "Hematocrit std",
                     "Eosinophils max",
                     "Eosinophils mean",
                     "Eosinophils min",
                     "Eosinophils skew",
                     "Eosinophils std",
                     "Asparate aminotransferase max",
                     "Asparate aminotransferase mean",
                     "Asparate aminotransferase min",
                     "Asparate aminotransferase skew",
                     "Asparate aminotransferase std",
                     "Blood urea nitrogen max",
                     "Blood urea nitrogen mean",
                     "Blood urea nitrogen min",
                     "Blood urea nitrogen skew",
                     "Blood urea nitrogen std",
                     "Monocytes max",
                     "Monocytes mean",
                     "Monocytes min",
                     "Monocytes skew",
                     "Monocytes std",
                     "Albumin max",
                     "Albumin mean",
                     "Albumin min",
                     "Albumin skew",
                     "Albumin std",
                     "Bilirubin max",
                     "Bilirubin mean",
                     "Bilirubin min",
                     "Bilirubin skew",
                     "Bilirubin std",
                     "Hemoglobin max",
                     "Hemoglobin mean",
                     "Hemoglobin min",
                     "Hemoglobin skew",
                     "Hemoglobin std",
                     "Alanine aminotransferase max",
                     "Alanine aminotransferase mean",
                     "Alanine aminotransferase min",
                     "Alanine aminotransferase skew",
                     "Alanine aminotransferase std",
                     "Troponin-I max",
                     "Troponin-I mean",
                     "Troponin-I min",
                     "Troponin-I skew",
                     "Troponin-I std",
                     "Platelets max",
                     "Platelets mean",
                     "Platelets min",
                     "Platelets skew",
                     "Platelets std",
                     "Alkaline phosphate max",
                     "Alkaline phosphate mean",
                     "Alkaline phosphate min",
                     "Alkaline phosphate skew",
                     "Alkaline phosphate std",
                     "Neutrophils max",
                     "Neutrophils mean",
                     "Neutrophils min",
                     "Neutrophils skew",
                     "Neutrophils std",
                     "Lymphocytes max",
                     "Lymphocytes mean",
                     "Lymphocytes min",
                     "Lymphocytes skew",
                     "Lymphocytes std",
                     "Basophils max",
                     "Basophils mean",
                     "Basophils min",
                     "Basophils skew",
                     "Basophils std",
                     "Troponin-T max",
                     "Troponin-T mean",
                     "Troponin-T min",
                     "Troponin-T skew",
                     "Troponin-T std",
                   ]
    else:
        features = []
else:
    features = []

print(tabulate_weights(features,f_imp))
count = 0
nz = 0
for x in f_imp:
    count = count+1
    if x > 0:
        nz = nz+1

print("Weight count = "+str(count))
print("Non-zero count = "+str(nz))


