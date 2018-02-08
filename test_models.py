# -*- coding: utf-8 -*-
"""
This is a script for fitting models to behavioral data, with the aim of identifying
a process by which the animals make choices. 

@author: bart
"""
##Run these prior to running any code. 
%load_ext autoreload
%autoreload 2

import Plotter
import Analyzer
import Helper
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.backends.backend_pdf import PdfPages
from functools import reduce
import pickle as pkl
import time

dbfloc = 'D:\\Bart\\Dropbox\\science\\leelab\\projects\\Arithmetic\\data\\_curatedData\\'
conn = sqlite3.connect(database=dbfloc+'arithmeticstudy.db')
cur = conn.cursor()

#%%
###########Fit approximate number model from Dehaene 2007

#Make SQl query
query = '''
        SELECT session,animal,chose_sum,augend,addend,singleton,
        augend+addend-singleton as diff
        FROM behavioralstudy
        WHERE experiment = 'FlatLO' and animal='Ruffio'
        ORDER BY animal,session
'''
#Execute query, then convert to pandas table
data = Helper.getData(cur,query)

realmin = Analyzer.realmin
pracmin = .00001 #a practical minimum, so that we don't accidentally square realmin.
models = Analyzer.get_models()

bounds = ((pracmin,1),(pracmin,1),(pracmin,1),)
parrange = [np.linspace(pracmin,1,10),np.linspace(pracmin,1,10),np.linspace(pracmin,1,10)]
dm_mout,_ = Analyzer.fit_grid_search(models['dm_full'],data,data['chose_sum'],parrange,cost='default',bounds=bounds)

linear_mout = Analyzer.fit_model(models['linear'],data,data['chose_sum'],x0=[1,1,1,1])

#test linear_model
lr = Analyzer.logistic_regression(data,model='chose_sum ~ diff+singleton')
print(lr.bic)

#predicted data (lines) vs. actual data (dots)
data['pred'] = models['dm_prior_optimal'](dm_mout.x,data)
Plotter.panelplots(data,plotvar='pred',scattervar='chose_sum',groupby=['augend','addend','diff'],
                   ylim=[0,1],xlim=[-2,2],xlabel='diff',ylabel='p(choose sum)',
                    xticks=[-2,-1,0,1,2],yticks=[0,.25,.5,.75,1])
plt.tight_layout()

#%%
#compare ratio to difference model

flatlo = Helper.getFlatLOTrialset()
query = '''
        SELECT session,animal,chose_sum,augend,addend,singleton,
        augend+addend-singleton as diff,(augend+addend)/singleton as ratio,
        augend+addend as sum
        FROM behavioralstudy
        WHERE experiment = 'FlatLO' and animal='Ruffio'
        ORDER BY animal,session
'''
data = Helper.getData(cur,query)



data['sing_prior'] = [flatlo['pcs'][np.where(flatlo['using']==x)[0][0]] for x in data['singleton']]
data['logratio'] = np.log(data['ratio'])

lr_diff = Analyzer.logistic_regression(data,model='chose_sum ~ diff + sing_prior')
lr_ratio = Analyzer.logistic_regression(data,model='chose_sum ~ ratio + sing_prior')
lr_addition_prior = Analyzer.logistic_regression(data,model='chose_sum ~ augend+addend+singleton + sing_prior')
lr_addition = Analyzer.logistic_regression(data,model='chose_sum ~ augend + addend + singleton')

models = Analyzer.get_models()

wds = Analyzer.fit_model(models['weighted_diff_singprior'],data,data['chose_sum'],[.1,.1])

#%%

query = '''
        SELECT session,animal,chose_sum,augend,addend,singleton,
        augend+addend-singleton as diff,(augend+addend)/singleton as ratio,
        augend+addend as sum
        FROM behavioralstudy
        WHERE experiment = 'FlatLO'
        ORDER BY animal,session
'''
data = Helper.getData(cur,query)

resolution = 5
groupby = ['animal']#['animal','session']
method = 'Nelder-Mead'
realmin = Analyzer.realmin
pracmin = .00001 #a practical minimum, so that we don't accidentally square realmin.
models = Analyzer.get_models()

#Pica et al. 2004, Cantlon & Brannon 2007, Dehaene 2007 model
print('Dehaene')
bounds = ((pracmin,1),(pracmin,1),(pracmin,1),)
parrange = [np.linspace(pracmin,1,resolution),np.linspace(pracmin,1,resolution),
            np.linspace(pracmin,1,resolution)]
dm_mout = Analyzer.fit_grid_search(models['dm_full'],data,data['chose_sum'],
                        parrange,groupby=groupby,bounds=bounds)

#Livingstone et al. 2014 model
print('Livingstone')
bounds = ((0,None),(pracmin,None),(None,None))
parrange = [np.linspace(0,10,resolution),np.linspace(-10,10,resolution),
            np.linspace(-10,10,resolution)]
livingstone_mout = Analyzer.fit_grid_search(models['livingstone_norm'],data,
                        data['chose_sum'],parrange,bounds=bounds,groupby=groupby)

#Linear model
print('Linear')
parrange = [np.linspace(-10,10,resolution),np.linspace(-10,10,resolution),
            np.linspace(-10,10,resolution),np.linspace(-10,10,resolution)]
linear_mout = Analyzer.fit_grid_search(models['linear'],data,data['chose_sum'],
                        parrange,groupby=groupby,method=method)

#non-strawman logarithmic model
print('Logarithmic (log(b*sum))')
parrange = [np.linspace(0,10,resolution),np.linspace(0,10,resolution),
            np.linspace(0,10,resolution),np.linspace(0,10,resolution)]
bounds = ((None,None),(pracmin,None),(pracmin,None),(pracmin,None),)
logarithmic_mout = Analyzer.fit_grid_search(models['logarithmic'],data,data['chose_sum'],
                        parrange,groupby=groupby,bounds=bounds)

print('Logarithmic 2 (b*log(sum))')
parrange = [np.linspace(0,10,resolution),np.linspace(0,10,resolution),
            np.linspace(0,10,resolution)]
logarithmic2_mout = Analyzer.fit_grid_search(models['logarithmic2'],data,data['chose_sum'],
                        parrange,groupby=groupby,method=method)

print('Logarithmic 3 (AAS)')
parrange = [np.linspace(-10,10,resolution),np.linspace(-10,10,resolution),
            np.linspace(-10,10,resolution),np.linspace(-10,10,resolution)]
logaas_mout = Analyzer.fit_grid_search(models['log_aas'],data,data['chose_sum'],
                        parrange,groupby=groupby,bounds=bounds)



#mouts = [dm_mout,livingstone_mout,linear_mout,logarithmic_mout]                  
#results = linear_mout[groupby]
#results['dm_bic'] = dm_mout['bic']
#results['livingstone_bic'] = livingstone_mout['bic']
#results['logarithmic_bic'] = logarithmic_mout['bic']
#results['linear_bic'] = linear_mout['bic']



#%%
#test cross-validation code

query = '''
        SELECT session,animal,chose_sum,augend,addend,singleton,
        augend+addend-singleton as diff,(augend+addend)/singleton as ratio,
        augend+addend as sum
        FROM behavioralstudy
        WHERE experiment = 'FlatLO'
        ORDER BY animal,session
'''
data = Helper.getData(cur,query)

resolution = 1
realmin = Analyzer.realmin
pracmin = .00001 #a practical minimum, so that we don't accidentally square realmin.
models = Analyzer.get_models()

#Linear model
print('Livingstone')
import time
start = time.time()
parrange = [np.linspace(-10,10,resolution),np.linspace(-10,10,resolution),
            np.linspace(-10,10,resolution),np.linspace(-10,10,resolution)]
linear_mout = Analyzer.fit_grid_loo_validate(models['linear'],data,data['chose_sum'],
                        parrange,validateby=['session'],groupby=['animal'],method='Nelder-Mead')
stop = time.time()
runtime = (stop-start)/60

#%%
#Fit models to entire data, to each session, and cross-validate them.

query = '''
        SELECT session,animal,chose_sum,augend,addend,singleton,
        augend+addend-singleton as diff,(augend+addend)/singleton as ratio,
        augend+addend as sum
        FROM behavioralstudy
        WHERE experiment = 'FlatLO'
        ORDER BY animal,session
'''
data = Helper.getData(cur,query)

#Some parameters for fitting
resolution = 5              #grid search resolution
method = 'Nelder-Mead'      #Fitting algorithm

#Some invariants for fitting
realmin = Analyzer.realmin#smallest floating point number supported by numpy
pracmin = .00001 #a practical minimum, so that we don't accidentally square realmin.
models = Analyzer.get_models()#the models

model_names = ['dm_full','livingstone_norm','linear','logarithmic2']
bounds = (((pracmin,1),(pracmin,1),(pracmin,1),),
          ((0,None),(pracmin,None),(None,None)),
          ((None,None),(None,None),(None,None),(None,None)),
          ((None,None),(None,None),(None,None)))
parrange = ((np.linspace(pracmin,1,resolution),np.linspace(pracmin,1,resolution),np.linspace(pracmin,1,resolution)),
            (np.linspace(0,10,resolution),np.linspace(-10,10,resolution),np.linspace(-10,10,resolution)),
            (np.linspace(-10,10,resolution),np.linspace(-10,10,resolution),np.linspace(-10,10,resolution),np.linspace(-10,10,resolution)),
            (np.linspace(0,10,resolution),np.linspace(-10,10,resolution),np.linspace(-10,10,resolution)))

nmodel = len(model_names)
assert len(bounds)==nmodel
assert len(parrange)==nmodel
model_out = []
for i in range(0,nmodel):
    print(model_names[i])
    
    #Fit combined data
    fit_all_out = Analyzer.fit_grid_search(models[model_names[i]],data,data['chose_sum'],
                        parrange[i],groupby=['animal'],bounds=bounds[i])
    
    #Set the initial guesses as the best-fitting values from all-session fit
    new_init = [(x,y) for x,y in zip(fit_all_out['par'][0],fit_all_out['par'][1])]
    
    #Fit each session separately
    fit_sess_out = Analyzer.fit_grid_search(models[model_names[i]],data,data['chose_sum'],
                        new_init,groupby=['animal','session'],bounds=bounds[i])
    
    #Cross-validation
    cv_out = Analyzer.fit_grid_loo_validate(models[model_names[i]],data,data['chose_sum'],
                        new_init,validateby=['session'],groupby=['animal'],method='Nelder-Mead')

    #Store it
    model_out.append((fit_all_out,fit_sess_out,cv_out))

fname = 'arithmetic_modeling_'+str(int(np.floor(time.time())))
floc = 'D:\\Bart\\dropbox\\'
pkl.dump(model_out,open(floc+fname,'wb'))


#%%
#Fit models to entire Addition/Subtraction data, to each session, and cross-validate them.

query = '''
        SELECT session,animal,experiment,chose_sum,augend,addend,singleton,
        augend+addend-singleton as diff,(augend+addend)/singleton as ratio,
        augend+addend as sum
        FROM behavioralstudy
        WHERE experiment in ('Addition','Subtraction')
        ORDER BY animal,session
'''
data = Helper.getData(cur,query)


#Some parameters for fitting
resolution = 1              #grid search resolution
method = None      #Fitting algorithm
groupby = ['animal','experiment']
validateby = ['session']
sess_groupby = ['animal','experiment','session']

#Some invariants for fitting
realmin = Analyzer.realmin#smallest floating point number supported by numpy
pracmin = .00001 #a practical minimum, so that we don't accidentally square realmin.
models = Analyzer.get_models()#the models

model_names = ['dm_full','livingstone_norm','linear','logarithmic2']
bounds = (((pracmin,1),(pracmin,1),(pracmin,1),),
          ((0,None),(None,None),(None,None)),
          ((None,None),(None,None),(None,None),(None,None)),
          ((None,None),(None,None),(None,None)))
parrange = ((np.linspace(pracmin,1,resolution),np.linspace(pracmin,1,resolution),np.linspace(pracmin,1,resolution)),
            (np.linspace(pracmin,10,resolution),np.linspace(1,10,resolution),np.linspace(-10,10,resolution)),
            (np.linspace(-10,10,resolution),np.linspace(-10,10,resolution),np.linspace(-10,10,resolution),np.linspace(-10,10,resolution)),
            (np.linspace(-10,10,resolution),np.linspace(-10,10,resolution),np.linspace(-10,10,resolution)))

nmodel = len(model_names)
assert len(bounds)==nmodel
assert len(parrange)==nmodel
model_out = []
for i in range(0,nmodel):
    print(model_names[i])
    
    #Fit combined data
    fit_all_out = Analyzer.fit_grid_search(models[model_names[i]],data,data['chose_sum'],
                        parrange[i],groupby=groupby,bounds=bounds[i],method=method)
    
    #Set the initial guesses as the best-fitting values from all-session fit
    new_init = [(x,y) for x,y in zip(fit_all_out['par'][0],fit_all_out['par'][1])]
    
    #Fit each session separately
    fit_sess_out = Analyzer.fit_grid_search(models[model_names[i]],data,data['chose_sum'],
                        new_init,groupby=sess_groupby,bounds=bounds[i])
    
    #Cross-validation
    cv_out = Analyzer.fit_grid_loo_validate(models[model_names[i]],data,data['chose_sum'],
                        new_init,validateby=validateby,groupby=groupby,method=method)

    #Store it
    model_out.append((fit_all_out,fit_sess_out,cv_out))
    
    
fname = 'AddSubtract_modeling_'+str(int(np.floor(time.time())))
floc = 'D:\\Bart\\dropbox\\'
pkl.dump(model_out,open(floc+fname,'wb'))


w = fit_all_out['par'][1]
models['livingstone_norm'](w,data.loc[data['experiment']=='Subtraction'])