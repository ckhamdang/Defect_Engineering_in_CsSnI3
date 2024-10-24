#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 20:19:51 2024

@author: chada
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Lasso

### Reading data ###

data = pd.read_excel('Training dataset_Ef(1).xlsx')

csvdata = data.to_numpy()

dopant = csvdata[1:, 0]
feature = csvdata[0, 1:-1]
E_form = csvdata[1:, -1]
X = csvdata[1:, 1:-1]

X_train, X_test, E_form_train, E_form_test = train_test_split(X, E_form, test_size=0.2)
        
### Defining hyperparameter ###

param_grid = {"alpha": [1e-2,1e-1, 1e0, 1e1,1e2],
              "max_iter": [50,100,500,1000],
              "selection": ['random', 'cyclic']
}

### Train the model ###
    
lassoreg_opt = RandomizedSearchCV(Lasso(), param_distributions=param_grid, cv=5)

lassoreg_opt.fit(X_train, E_form_train)
print('Best estimator by GridSearchCV:', lassoreg_opt.best_estimator_)

Pred_train = lassoreg_opt.predict(X_train)
Pred_test = lassoreg_opt.predict(X_test)

feature_importances = np.abs(lassoreg_opt.best_estimator_.coef_)
feature_names = ['EA',  '1st IE', '2nd IE', '3rd IE', 'D', 't (AR) 100%', 'u (IR) 100%', 'OS', 'HF (cal)', 'S', 'HV']

feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

print('Feature importances =', feature_importances_df)

plt.figure(figsize=(8, 8))
plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'], color='#371B58')
plt.xlabel('Relative feature importance', fontsize=22)
plt.title('Feature Importances', fontsize=22)
plt.gca().invert_yaxis()
plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],fontsize=20)
plt.yticks(fontsize=20)
plt.show()

### Matrics calculation ###

mse_test_prop  = sklearn.metrics.mean_squared_error(E_form_test, Pred_test)
mse_train_prop = sklearn.metrics.mean_squared_error(E_form_train, Pred_train)
rmse_test_form  = np.sqrt(mse_test_prop)
rmse_train_form = np.sqrt(mse_train_prop)
print('rmse_test_form  = ', np.sqrt(mse_test_prop))
print('rmse_train_form = ', np.sqrt(mse_train_prop))

### Plotting results ###

fig, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
fig.text(0.5, -0.01, 'DFT Calculation', ha='center', fontsize=26)
fig.text(-0.01, 0.5, 'ML Prediction', va='center', rotation='vertical', fontsize=26)

plt.subplots_adjust(left=0.12, bottom=0.10, right=0.97, top=0.95, wspace=0.25, hspace=0.4)

a = [-175,0,125]
b = [-175,0,125]
ax.plot(b, a, c='k', ls='-')

ax.xaxis.set_tick_params(labelsize=26)
ax.yaxis.set_tick_params(labelsize=26)

ax.scatter(E_form_test[:], Pred_test[:], c='#615EFC', marker='o', s=80, label='Test')
ax.scatter(E_form_train[:], Pred_train[:], c='#FA7070', marker='o', s=80, label='Train')

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

ax.set_title('Formation energy (q=+1)', c='k', fontsize=26, pad=10)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles,labels, fontsize=18)
plt.tick_params(axis='y', width=2,length=4, labelsize=24) 
plt.tick_params(axis='x', width=2, length=4,  labelsize=24) 

ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
plt.show()
