import uproot
import math
import time
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from matplotlib import rcParams
from sklearn_genetic import GASearchCV
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import shuffle
import skopt
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.metrics import plot_roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.layers import BatchNormalization


import warnings
seed = 42 # Set seed for reproducibility purposes
metric = 'accuracy' # See other options https://scikit-learn.org/stable/modules/model_evaluation.html
kFoldSplits = 3

signal_events = []
bkg_events1 , bkg_events2 = [] , []

train_file = uproot.open("training.root")
train_file2 = uproot.open('training_bkg.root')
train_file3 = uproot.open('testing_bkg.root')


train_signal_tree = train_file["sig_tree"]
train_bkg_tree = train_file2["bkg_tree"]

test_bkg_tree = train_file3["bkg_tree"]


test_file_1 = uproot.open("test-1.root")
test_tree_1 = test_file_1["sig_tree"]

test_file_05 = uproot.open("test-0.5.root")
test_tree_05 = test_file_05["sig_tree"]

test_file05 = uproot.open("test+0.5.root")
test_tree05 = test_file05["sig_tree"]

test_file1 = uproot.open("test+1.root")
test_tree1 = test_file1["sig_tree"]

test_file_15 = uproot.open("test-15.root")
test_tree_15 = test_file1["sig_tree"]

test_file_15_2 = uproot.open("test-15_2.root")
test_tree_15_2 = test_file1["sig_tree"]

testsm1_file = uproot.open("SM_vis.root")
testsm1_tree = testsm1_file["sig_tree"]
testsm1_bkg_tree = testsm1_file["bkg_tree"]

test_sm1_file = uproot.open("SM_vis_neg.root")
test_sm1_tree = test_sm1_file["sig_tree"]
test_sm1_bkg_tree = test_sm1_file["bkg_tree"]



def file_read(evt_no,tree_name,class_value):
    events = []
    for i in range(evt_no):
        evt = []
        evt.extend([int(tree_name["n_jets"].array()[i]),tree_name["max_eta"].array()[i],int(tree_name["charge_sum"].array()[i]),int(tree_name["eta_1"].array()[i]),tree_name["del_eta_lib"].array()[i]
            ,tree_name["del_eta_libs"].array()[i],tree_name["del_eta_lil"].array()[i],tree_name["del_phi_ssl"].array()[i],tree_name["min_del_R"].array()[i],tree_name["pt_subl"].array()[i],class_value])

        events.append(evt)
        events2 = np.array(events)
        dataframe = pd.DataFrame(events2,
                   columns=['n_jets', 'max_eta', 'charge_sum' , 'eta_1' , 'del_eta_lib' , 'del_eta_libs' , 'del_eta_lil' , 'del_phi_ssl', 'min_del_R' , 'pt_subl','target'])

    return dataframe

def df_format(dataframe):
    
    dataframe.n_jets = dataframe.n_jets.astype(int)
    dataframe.charge_sum = dataframe.charge_sum.astype(int)
    dataframe.eta_1 = dataframe.eta_1.astype(int)

    columns_titles = ["n_jets","charge_sum","eta_1","max_eta","del_eta_lib","del_eta_libs","del_eta_lil","del_phi_ssl","min_del_R","pt_subl","target"]
    dataframe=dataframe.reindex(columns=columns_titles)


    def std_norm(dataframe, column):
        c = dataframe[column]
        dataframe[column] = (c - c.mean())/c.std()

    std_norm(dataframe, 'max_eta')
    std_norm(dataframe, 'del_eta_lib')
    std_norm(dataframe, 'del_eta_libs')
    std_norm(dataframe, 'del_eta_lil')
    std_norm(dataframe, 'del_phi_ssl')
    std_norm(dataframe, 'min_del_R')
    std_norm(dataframe, 'pt_subl')

    return dataframe

train_signal = df_format(file_read(10000,train_signal_tree,1))
train_bkg = df_format(file_read(10000,train_bkg_tree,0))



test_signal_1 = df_format(file_read(1000,test_tree_1,1))


test_signal_05 = df_format(file_read(1000,test_tree_05,1))
test_signal05 = df_format(file_read(1000,test_tree05,1))
test_signal1 = df_format(file_read(1000,test_tree1,1))
test_signal_15 = df_format(file_read(370,test_tree_15,1))
test_signal_15_2 = df_format(file_read(370,test_tree_15_2,1))
testsm1 = df_format(file_read(500,testsm1_tree,1))
test_sm1 = df_format(file_read(500,test_sm1_tree,1))


test_bkgsm1 = df_format(file_read(500,testsm1_bkg_tree,0))
test_bkg_sm1 = df_format(file_read(500,test_sm1_bkg_tree,0))

test_bkg = df_format(file_read(10000,test_bkg_tree,0))


test_bkg_1 = test_bkg.iloc[0:1000,0:11]
test_bkg_05 = test_bkg.iloc[2000:3000,0:11]
test_bkg05 = test_bkg.iloc[4000:5000,0:11]
test_bkg1 = test_bkg.iloc[7000:8000,0:11]
test_bkg_15 = test_bkg.iloc[9000:9370,0:11]
test_bkg_15_2 = test_bkg.iloc[9500:9870,0:11]

train_dataframe = shuffle(train_signal.append(train_bkg),random_state=seed)

test_dataframe_1 = shuffle(test_signal_1.append(test_bkg_1),random_state=seed)
test_dataframe_05 = shuffle(test_signal_05.append(test_bkg_05),random_state=seed)
test_dataframe05 = shuffle(test_signal05.append(test_bkg05),random_state=seed)
test_dataframe1 = shuffle(test_signal1.append(test_bkg1),random_state=seed)
test_dataframe_15_sig = shuffle(test_signal_15.append(test_signal_15_2),random_state=seed)
test_dataframe_15_bkg = shuffle(test_bkg_15.append(test_bkg_15_2),random_state=seed)
test_dataframe_15 =[test_dataframe_15_sig,test_dataframe_15_bkg]
test_dataframe_15=shuffle(pd.concat(test_dataframe_15),random_state=seed)


test_dataframesm1 = shuffle(testsm1.append(test_bkgsm1),random_state=seed)
test_dataframe_sm1 = shuffle(test_sm1.append(test_bkg_sm1),random_state=seed)




X_train = train_dataframe.iloc[:,0:10].values
Y_train = train_dataframe[train_dataframe.columns[10]]


X_test_1 = test_dataframe_1.iloc[:,0:10].values
Y_test_1 = test_dataframe_1.iloc[:,10:11].values


X_test_05 = test_dataframe_05.iloc[:,0:10].values
Y_test_05 = test_dataframe_05.iloc[:,10:11].values

X_test05 = test_dataframe05.iloc[:,0:10].values
Y_test05 = test_dataframe05.iloc[:,10:11].values

X_test1 = test_dataframe1.iloc[:,0:10].values
Y_test1 = test_dataframe1.iloc[:,10:11].values

X_test_15 = test_dataframe_15.iloc[:,0:10].values
Y_test_15 = test_dataframe_15.iloc[:,10:11].values


X_testsm1 = test_dataframesm1.iloc[:,0:10].values
Y_testsm1 = test_dataframesm1.iloc[:,10:11].values

X_test_sm1 = test_dataframe_sm1.iloc[:,0:10].values
Y_test_sm1 = test_dataframe_sm1.iloc[:,10:11].values

# split data into train and test sets


print(Y_test_1.shape)

#parameters = {
#    "n_estimators":[600,700,800,900,1000],
#    "max_depth":[5,7,9,10],
#    "learning_rate":[0.03,0.05,0.07,0.1,0.3]
#}

#cv = GridSearchCV(gb_clf,parameters,cv=5, scoring = 'accuracy', n_jobs = 5, verbose=4)
#cv.fit(X_train,Y_train)

#def display(results):
#    print(f'Best parameters are: {results.best_params_}')
#    print("\n")
#    mean_score = results.cv_results_['mean_test_score']
#    std_score = results.cv_results_['std_test_score']
#    params = results.cv_results_['params']
#    for mean,std,params in zip(mean_score,std_score,params):
#        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')








bayes_cv_tuner = BayesSearchCV(
    estimator = xgb.XGBClassifier(
        n_jobs = -1,
        objective = 'binary:logistic',
        eval_metric = 'logloss',
        tree_method='approx'
    ),

search_spaces = {
        'learning_rate': (0.01, 0.1, 'log-uniform'),
        'max_depth': (2, 5,'uniform'),
        'max_delta_step': (0, 10,'uniform'),
        'subsample': (0.5, 1.0, 'uniform'),
        'colsample_bytree': (0.5, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'gamma' : (0.1,5.0, 'uniform'),
        'lambda': (2, 50, 'log-uniform'),
        'alpha': (0, 4.0, 'uniform'),
        'min_child_weight': (0, 200),
        'n_estimators': (10, 120),
        #'scale_pos_weight': (1e-6, 500, 'log-uniform')
    },    
scoring = 'roc_auc',
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    ),
    n_jobs = 4,
    n_iter = 10,
    verbose = 2,
    refit = True,
    random_state = seed
)


bayes_cv_tuner.fit(X_train, Y_train) #callback=status_print)
best_params = bayes_cv_tuner.best_params_

def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
       print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')


display(bayes_cv_tuner)

gb_clf = xgb.XGBClassifier(**best_params,random_state=seed).fit(X_train,Y_train)


eval_set = [(X_train, Y_train), (X_test_05, Y_test_05), (X_test05, Y_test05),(X_test1, Y_test1),(X_test_sm1, Y_test_sm1)]
gb_clf.fit(X_train, 
          Y_train, 
          eval_metric="error", 
          eval_set=eval_set, 
          verbose=False)
results = gb_clf.evals_result()


epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)


rcParams['figure.figsize'] = 45, 30
rcParams['legend.fontsize'] = 60
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['error'], color='darkred',linestyle='solid', lw=2.0, label='Training error BP1')
ax.plot(x_axis, results['validation_1']['error'], color='darkorange',linestyle='dotted', lw=2.5, label='Test error BP2')
ax.plot(x_axis, results['validation_2']['error'], color='darkgreen',linestyle='dashed', lw=2.0, label='Test error BP3')
ax.plot(x_axis, results['validation_3']['error'], color='black',linestyle=(5, (10, 3)), lw=2.0, label='Test error BP4')
ax.plot(x_axis, results['validation_4']['error'], color='blueviolet',linestyle=(0, (3, 1, 1, 1, 1, 1)), lw=2.0, label='Test error SM $ \kappa_{t} = -1.0 $')
ax.legend()
plt.xlabel('Epochs', fontsize=70)
plt.ylabel('Error', fontsize=70)
plt.setp(ax.get_xticklabels(), fontsize=70)
plt.setp(ax.get_yticklabels(), fontsize=70)
plt.savefig('/Users/kuntalpal/Dropbox/Higgs-EFT/Note/error.pdf')
plt.show()


from sklearn.metrics import roc_curve, auc

decisions_1 = gb_clf.predict_proba(X_test_1)[:,1]
decisions_05 = gb_clf.predict_proba(X_test_05)[:,1]
decisions05 = gb_clf.predict_proba(X_test05)[:,1]
decisions1 = gb_clf.predict_proba(X_test1)[:,1]
decisions_sm1 = gb_clf.predict_proba(X_test_sm1)[:,1]
decisionssm1 = gb_clf.predict_proba(X_testsm1)[:,1]

# Compute ROC curve and area under the curve
fpr_1, tpr_1, thresholds_1 = roc_curve(Y_test_1, decisions_1)
fpr_05, tpr_05, thresholds_05 = roc_curve(Y_test_05, decisions_05)
fpr05, tpr05, thresholds05 = roc_curve(Y_test05, decisions05)
fpr1, tpr1, thresholds1 = roc_curve(Y_test1, decisions1)
fpr_sm1, tpr_sm1, thresholds_sm1 = roc_curve(Y_test_sm1, decisions_sm1)
fprsm1, tprsm1, thresholdssm1 = roc_curve(Y_testsm1, decisionssm1)

roc_auc_1 = auc(fpr_1, tpr_1)
roc_auc_05 = auc(fpr_05, tpr_05)
roc_auc05 = auc(fpr05, tpr05)
roc_auc1 = auc(fpr1, tpr1)
roc_auc_sm1 = auc(fpr_sm1, tpr_sm1)
roc_aucsm1 = auc(fprsm1, tprsm1)

rcParams['figure.figsize'] = 45, 30
rcParams['legend.fontsize'] = 60
plt.plot(fpr_1, tpr_1, lw=5.0, linestyle='solid', color='darkred',label='Training BP1 ROC (area = %0.2f)'%(roc_auc_1))
plt.plot(fpr_05, tpr_05, lw=5.5, linestyle='dashdot', color='darkorange', label='Test BP2 ROC (area = %0.2f)'%(roc_auc_05))
plt.plot(fpr05, tpr05, lw=5.0, linestyle='dashed', color='darkgreen', label='Test BP3 ROC (area = %0.2f)'%(roc_auc05))
plt.plot(fpr1, tpr1, lw=5.0, linestyle='dotted', color='black', label='Test BP4 ROC (area = %0.2f)'%(roc_auc1))
plt.plot(fpr_sm1, tpr_sm1, lw=5.0, linestyle= (0, (3, 1, 1, 1, 1, 1)), color='blueviolet', label='Test SM $ \kappa_{t} = -1.0 $ ROC (area = %0.2f)'%(roc_auc_sm1))
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=70)
plt.ylabel('True Positive Rate', fontsize=70)
plt.legend(loc="lower right")
plt.grid()
plt.setp(ax.get_xticklabels(), fontsize=70)
plt.setp(ax.get_yticklabels(), fontsize=70)
plt.savefig('/Users/kuntalpal/Dropbox/Higgs-EFT/Note/roc.pdf')
plt.show()




y_predicted_1 = gb_clf.predict(X_test_1)
print(classification_report(Y_test_1, y_predicted_1,
                            target_names=["background", "signal"]))
matrix_1 = confusion_matrix(Y_test_1, y_predicted_1)
print(matrix_1.diagonal()/matrix_1.sum(axis=1))

y_predicted_05 = gb_clf.predict(X_test_05)
print(classification_report(Y_test_05, y_predicted_05,
                            target_names=["background", "signal"]))
matrix_05 = confusion_matrix(Y_test_05, y_predicted_05)
print(matrix_05.diagonal()/matrix_05.sum(axis=1))

y_predicted05 = gb_clf.predict(X_test05)
print(classification_report(Y_test05, y_predicted05,
                            target_names=["background", "signal"]))
matrix05 = confusion_matrix(Y_test05, y_predicted05)
print(matrix05.diagonal()/matrix05.sum(axis=1))

y_predicted1 = gb_clf.predict(X_test1)
print(classification_report(Y_test1, y_predicted1,
                            target_names=["background", "signal"]))

matrix1 = confusion_matrix(Y_test1, y_predicted1)
print(matrix1.diagonal()/matrix1.sum(axis=1))

y_predicted_15 = gb_clf.predict(X_test_15)
print(classification_report(Y_test_15, y_predicted_15,
                            target_names=["background", "signal"]))

y_predictedsm1 = gb_clf.predict(X_testsm1)
print(classification_report(Y_testsm1, y_predictedsm1,
                            target_names=["background", "signal"]))

matrixsm1 = confusion_matrix(Y_testsm1, y_predictedsm1)
print(matrixsm1.diagonal()/matrixsm1.sum(axis=1))

y_predicted_sm1 = gb_clf.predict(X_test_sm1)
print(classification_report(Y_test_sm1, y_predicted_sm1,
                            target_names=["background", "signal"]))

matrix_sm1 = confusion_matrix(Y_test_sm1, y_predicted_sm1)
print(matrix_sm1.diagonal()/matrix_sm1.sum(axis=1))



print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, Y_train)))
print("Accuracy score (test) -1: {0:.3f}".format(gb_clf.score(X_test_1, Y_test_1)))
print("Accuracy score (test) -0.5: {0:.3f}".format(gb_clf.score(X_test_05, Y_test_05)))
print("Accuracy score (test) 0.5: {0:.3f}".format(gb_clf.score(X_test05, Y_test05)))
print("Accuracy score (test) 1.0: {0:.3f}".format(gb_clf.score(X_test1, Y_test1)))
print("Accuracy score (test) -1.5: {0:.3f}".format(gb_clf.score(X_test_15, Y_test_15)))
print("Accuracy score (test) sm 1: {0:.3f}".format(gb_clf.score(X_testsm1, Y_testsm1)))
print("Accuracy score (test) sm -1: {0:.3f}".format(gb_clf.score(X_test_sm1, Y_test_sm1)))








