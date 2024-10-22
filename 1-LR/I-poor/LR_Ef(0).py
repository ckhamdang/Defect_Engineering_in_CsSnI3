#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 20:19:51 2024

@author: chada
"""

import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

data = pd.read_excel('Training dataset_Ef(0).xlsx')

csvdata = data.to_numpy()

dopant = csvdata[1:, 0]
feature = csvdata[0, 1:-1]
E_form = csvdata[1:, -1]
X = csvdata[1:, 1:-1]

### Reading outside data ###

data_out = pd.read_excel('Testing dataset.xlsx')

csvdata_out = data_out.to_numpy()

dopant_out = csvdata_out[1:, 0]
X_out = csvdata_out[1:, 1:]
n_out = dopant_out.size

data_out = copy.deepcopy(csvdata_out)

### Data preparation ###

XX = copy.deepcopy(X)
prop = copy.deepcopy(E_form)
n = feature.size
m = prop.size

t = 0.2

X_train, X_test, E_form_train, E_form_test = train_test_split(XX,prop, test_size=t)
n_tr = E_form_train.size
n_te = E_form_test.size

### Converting data types ###

X_train_fl = [[0.0 for a in range(n)] for b in range(n_tr)]
for i in range(0,n_tr):
    for j in range(0,n):
        X_train_fl[i][j] = float(X_train[i][j])
        
X_test_fl = [[0.0 for a in range(n)] for b in range(n_te)]
for i in range(0,n_te):
    for j in range(0,n):
        X_test_fl[i][j] = float(X_test[i][j])
        
### Defining hyperparameter ###

param_grid = {'fit_intercept': [False],
              'copy_X': [True],
              'n_jobs': [2,4,8,10,20],
              'positive': [False]
}

### Train the model ###

Prop_train = copy.deepcopy(E_form_train)
Prop_test = copy.deepcopy(E_form_test)

Prop_train_fl = np.zeros(n_tr)
for i in range(0,n_tr):
    Prop_train_fl[i] = copy.deepcopy(float(Prop_train[i]))

Prop_test_fl = np.zeros(n_te)
for i in range(0,n_te):
    Prop_test_fl[i] = copy.deepcopy(float(Prop_test[i]))
    
lreg_opt = RandomizedSearchCV(LinearRegression(), param_distributions=param_grid, cv=5)

lreg_opt.fit(X_train_fl, Prop_train_fl)
print('Best estimator by GridSearchCV:', lreg_opt.best_estimator_)

Pred_train_fl = lreg_opt.predict(X_train_fl)
Pred_test_fl = lreg_opt.predict(X_test_fl)

feature_importances = np.abs(lreg_opt.best_estimator_.coef_)
feature_names = ['EA',  '1st IE', '2nd IE', '3rd IE', 'D', 't (AR) 100%', 'u (IR) 100%', 'OS', 'HF (cal)', 'S', 'HV']

feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

print('Feature importances =', feature_importances_df)

plt.figure(figsize=(8, 8))
plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'], color='#371B58')
plt.xlabel('Relative feature importance', fontsize=22)
plt.title('Feature Importances', fontsize=22)
plt.gca().invert_yaxis()
plt.xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5], fontsize=20)
plt.yticks(fontsize=20)
#plt.savefig('feature importance.png', dpi=450)
plt.show()

### Outside prediction ###

m_out = dopant_out.size
X_out_fl = [[0.0 for a in range(n)] for b in range(n_out)]
for i in range(0,n_out):
    for j in range(0,n):
        X_out_fl[i][j] = float(X_out[i][j])


Pred_out_fl  =  [[0.0 for a in range(1)] for b in range(n_out)]
Pred_out_str  =  [[0.0 for a in range(1)] for b in range(n_out)]
err_up_out   =  [[0.0 for a in range(1)] for b in range(n_out)]
err_down_out =  [[0.0 for a in range(1)] for b in range(n_out)]


Pred_out = lreg_opt.predict(X_out_fl)
for i in range(0,n_out):
    Pred_out_fl[i] = float(Pred_out[i])
    
for i in range(0,n_out):
    Pred_out_str[i] = str(Pred_out[i])
    
dopant_out_array = np.array(dopant_out)
Pred_out_array = np.array(Pred_out)


Pred_dopant_out = [[dopant_out_array[i], Pred_out_array[i]] for i in range(len(dopant_out_array))]
Pred_dopant_out = np.column_stack((dopant_out_array, Pred_out_array))


### Matrics calculation ###

Prop_train_form = copy.deepcopy(Prop_train_fl)
Pred_train_form = copy.deepcopy(Pred_train_fl)
Prop_test_form  = copy.deepcopy(Prop_test_fl)
Pred_test_form  = copy.deepcopy(Pred_test_fl)

mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_form, Pred_test_form)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_form, Pred_train_form)
rmse_test_form  = np.sqrt(mse_test_prop)
rmse_train_form = np.sqrt(mse_train_prop)
print('rmse_test_form  = ', np.sqrt(mse_test_prop))
print('rmse_train_form = ', np.sqrt(mse_train_prop))

# Open a text file to save the predictions
with open('Predictions_LR_Ef0.txt', 'w') as f:
    f.write("Dopant, Predicted Formation Energy\n")
    
    # Iterate through the predictions and dopants and write to file
    for i in range(n_out):
        f.write(f"{dopant_out[i]}, {Pred_out_fl[i]:.4f}\n")

print("Predictions have been saved to 'Predictions_LR_Ef0.txt'.")


### Plotting results ###

fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(8,8))

fig.text(0.5, -0.01, 'DFT Calculation', ha='center', fontsize=26)
fig.text(-0.01, 0.5, 'ML Prediction', va='center', rotation='vertical', fontsize=26)

plt.subplots_adjust(left=0.12, bottom=0.10, right=0.97, top=0.95, wspace=0.25, hspace=0.4)

Prop_train_temp = copy.deepcopy(Prop_train_form)
Pred_train_temp = copy.deepcopy(Pred_train_form)
Prop_test_temp  = copy.deepcopy(Prop_test_form)
Pred_test_temp  = copy.deepcopy(Pred_test_form)

a = [-175,0,125]
b = [-175,0,125]
ax.plot(b, a, c='k', ls='-')

ax.xaxis.set_tick_params(labelsize=26)
ax.yaxis.set_tick_params(labelsize=26)

ax.scatter(Prop_test_temp[:], Pred_test_temp[:], c='#615EFC', marker='o', s=80, label='Test')
ax.scatter(Prop_train_temp[:], Pred_train_temp[:], c='#FA7070', marker='o', s=80, label='Train')

te = '%.2f' % rmse_test_form
tr = '%.2f' % rmse_train_form

ax.set_ylim([-0.5, 2])
ax.set_xlim([-0.5, 2])

ax.text(0.75, -0.2, 'Test_rmse = ', c='black', fontsize=18)
ax.text(1.4, -0.2, te, c='black', fontsize=18)
ax.text(1.65, -0.2, 'eV', c='black', fontsize=18)
ax.text(0.75, -0.3, 'Train_rmse = ', c='black', fontsize=18)
ax.text(1.4, -0.3, tr, c='black', fontsize=18)
ax.text(1.65, -0.3, 'eV', c='black', fontsize=18)

ax.set_xticks([0, 0.5, 1, 1.5, 2])
ax.set_yticks([-0.5, 0, 0.5, 1, 1.5, 2])

ax.set_title('Formation energy (q=0)', c='k', fontsize=26, pad=10)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles,labels, fontsize=18)
plt.tick_params(axis='y', width=2,length=4, labelsize=24) 
plt.tick_params(axis='x', width=2, length=4,  labelsize=24) 

ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

plt.savefig('rfreg-EF(1).png', dpi=450)
plt.show()
