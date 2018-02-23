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
resolution = 4              #grid search resolution
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
          ((0,None),(.001,100),(None,None)),
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
    model_out.append((model_names[i],fit_all_out,fit_sess_out,cv_out))
    
    
fname = 'AddSubtract_modeling_'+str(int(np.floor(time.time())))
floc = 'D:\\Bart\\dropbox\\'
pkl.dump(model_out,open(floc+fname,'wb'))

cond = (data['animal']=='Ruffio') & (data['experiment']=='Addition')
models['cost'](data['chose_sum'].loc[cond],models['livingstone_norm'](model_out[1][0]['par'][0],data.loc[cond]))


floc = 'D:\\Bart\\dropbox\\'
model_out_in = pkl.load(open(floc+'AddSubtract_modeling_1518389483','rb'))

w = model_out_in[1][0]['par'][3]
w = [0,.001,-2]
models['livingstone_norm'](w,data)

#CV accuracy
[d[2].groupby(['animal','experiment'])['cv_acc'].mean() for d in model_out_in]
 
 
[d[2]['cv_acc'].loc[(d[2]['animal']=='Xavier') & (d[2]['experiment']=='Addition')] for d in model_out_in]


#%%

#Fit Dehaene model w/ term for non-linear relationship between variance and
#quantity.
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

#################################################################################
#The conclusion of this work is that, with the exception of Ruffio's addition data,
#I can't get the dehaene model to beat the linear model with one or 2nd order terms.
#################################################################################

#Restrictions for dm_full_nonlinear
#bounds = ((None,None),(None,None),(None,None),(None,None),
#          (pracmin,None),(pracmin,None),(pracmin,None),(-4,None))
#parrange = (np.linspace(1,10,resolution),np.linspace(1,10,resolution),
#            np.linspace(1,10,resolution),np.linspace(1,10,resolution),
#            np.linspace(pracmin,1,resolution),np.linspace(pracmin,1,resolution),
#            np.linspace(pracmin,1,resolution),
#            np.linspace(1,10,resolution))
#fit_all_out = Analyzer.fit_grid_search(models['dm_full_nonlinear'],data,data['chose_sum'],
#                        parrange,groupby=groupby,bounds=bounds,method=method)

#Same weight for all numbers in numerator. 
#dm_nonlinear = lambda w,data: models['dm_full_nonlinear']([w[0],1,1,1,w[1],w[2],w[3],w[4]],data)
#bounds = ((None,None),(pracmin,None),(pracmin,None),(pracmin,None),(-10,None))
#parrange = (np.linspace(1,10,resolution),np.linspace(pracmin,1,resolution),
#            np.linspace(pracmin,1,resolution),np.linspace(pracmin,1,resolution),
#            np.linspace(1,10,resolution))
#fit_all_out = Analyzer.fit_grid_search(dm_nonlinear,data,data['chose_sum'],
#                        parrange,groupby=groupby,bounds=bounds,method=method)

#Same weight for quantity variance
#dm_nonlinear_oneweight = lambda w,data: models['dm_full_nonlinear']([0,w[0],w[1],w[2],w[3],w[3],w[3],w[4]],data)
#bounds = ((None,None),(None,None),(None,None),(pracmin,None),(pracmin,None))
#parrange = (np.linspace(1,3,resolution),
#            np.linspace(1,3,resolution),np.linspace(1,3,resolution),
#            np.linspace(.1,3,resolution),np.linspace(.1,1,resolution))
#fit_all_out = Analyzer.fit_grid_search(dm_nonlinear_oneweight,data,data['chose_sum'],
#                        parrange,groupby=groupby,bounds=bounds,method=method,tol=.0000001)

#Same weight for quantity variance and numerator. Variable intercept in numerator
#dm_nonlinear_oneweight = lambda w,data: models['dm_full_nonlinear']([w[0],1,1,1,w[1],w[1],w[1],w[2]],data)
#bounds = ((None,None),(pracmin,None),(pracmin,None))
#parrange = (np.linspace(1,3,resolution),np.linspace(.1,3,resolution),np.linspace(.1,1,resolution))
#fit_all_out = Analyzer.fit_grid_search(dm_nonlinear_oneweight,data,data['chose_sum'],
#                        parrange,groupby=groupby,bounds=bounds,method=method,tol=.0000001)

#intercept and weight for difference, same weight for all quantity variance.
#dm_nonlinear_oneweight = lambda w,data: models['dm_full_nonlinear']([w[0],w[1],w[1],w[1],w[2],w[2],w[2],w[3]],data)
#bounds = ((None,None),(None,None),(pracmin,None),(pracmin,None))
#parrange = (np.linspace(1,3,resolution),np.linspace(1,3,resolution),np.linspace(.1,3,resolution),np.linspace(.1,1,resolution))
#fit_all_out = Analyzer.fit_grid_search(dm_nonlinear_oneweight,data,data['chose_sum'],
#                        parrange,groupby=groupby,bounds=bounds,method=method)

#intercept and weight for difference, diff weight for all quantity variance.
#dm_nonlinear_oneweight = lambda w,data: models['dm_full_nonlinear']([w[0],w[1],w[1],w[1],w[2],w[3],w[4],w[5]],data)
#bounds = ((None,None),(None,None),(pracmin,None),(pracmin,None),(pracmin,None),(pracmin,None))
#parrange = (np.linspace(-3,3,resolution),np.linspace(.1,3,resolution),
#            np.linspace(.1,1,resolution),np.linspace(.1,1,resolution),
#            np.linspace(.1,1,resolution),np.linspace(.1,1,resolution))
#fit_all_out = Analyzer.fit_grid_search(dm_nonlinear_oneweight,data,data['chose_sum'],
#                        parrange,groupby=groupby,bounds=bounds,method=method)

#linear numerator, diff weight for all quantity variance.
dm_nonlinear_oneweight = lambda w,data: models['dm_full_nonlinear']([0,1,1,1,w[0],w[1],w[2],w[3]],data)
bounds = ((pracmin,None),(pracmin,None),(pracmin,None),(pracmin,None))
parrange = (np.linspace(.1,1,resolution),np.linspace(.1,1,resolution),
            np.linspace(.1,1,resolution),np.linspace(.1,1,resolution))
fit_all_out = Analyzer.fit_grid_search(dm_nonlinear_oneweight,data,data['chose_sum'],
                        parrange,groupby=groupby,bounds=bounds,method=method)


#Linear numerator, same weight for all quantity variance.
dm_nonlinear_oneweight = lambda w,data: models['dm_full_nonlinear']([0,1,1,1,w[0],w[0],w[0],w[1]],data)
bounds = ((pracmin,None),(pracmin,None))
parrange = (np.linspace(.1,1,resolution),np.linspace(.1,2,resolution))
fit_all_out = Analyzer.fit_grid_search(dm_nonlinear_oneweight,data,data['chose_sum'],
                        parrange,groupby=groupby,bounds=bounds,method=method)


#Compare to linear models
lr_model = 'chose_sum ~ augend + addend + singleton'
lr_out = Analyzer.logistic_regression(data,lr_model,groupby=groupby)

#Compare to linear models
lr_2ndorder_model = 'chose_sum ~ augend + addend + singleton + np.power(augend,2) + np.power(addend,2) + np.power(singleton,2)'
lr_2ndorder_out = Analyzer.logistic_regression(data,lr_2ndorder_model,groupby=groupby)

cond = (data['animal']=='Xavier') & (data['experiment'] == 'Addition')
w = fit_all_out['par'][2]
d = data.loc[cond]

pred = dm_nonlinear_oneweight(w,d)
y = data['chose_sum'].loc[cond]

models['cost'](y,pred)



#%%
#Fit Dehaene model w/ term for non-linear relationship between variance and
#quantity. Examine relationship to p(correct)
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
resolution = 3              #grid search resolution
method = None      #Fitting algorithm
groupby = ['animal','experiment']
validateby = ['session']
sess_groupby = ['animal','experiment','session']

#Some invariants for fitting
realmin = Analyzer.realmin#smallest floating point number supported by numpy
pracmin = .001 #a practical minimum, so that we don't accidentally square realmin.
expmin = .01
models = Analyzer.get_models()#the models

#linear numerator, diff weight for all quantity variance.
dm_nonlinear_oneweight = lambda w,data: models['dm_full_nonlinear']([0,1,1,1,w[0],w[1],w[2],w[3]],data)
dm_var_nonlinear = lambda w,data: np.power((w[0]**(1.0/w[3]))*data['augend']**2 
            + (w[1]**(1.0/w[3]))*data['addend']**2 + (w[2]**(1.0/w[3]))*data['singleton']**2,w[3])

#fit the model.
bounds = ((pracmin,None),(pracmin,None),(pracmin,None),(expmin,None))
parrange = (np.linspace(.1,1,resolution),np.linspace(.1,1,resolution),
            np.linspace(.1,1,resolution),np.linspace(.1,5,resolution))
fit_all_out = Analyzer.fit_grid_search(dm_nonlinear_oneweight,data,data['chose_sum'],
                        parrange,groupby=groupby,bounds=bounds,method=method)

#fit linear model for comparison
lr_out = Analyzer.logistic_regression(data,'chose_sum ~ augend+addend+singleton',groupby=groupby)


#Plot p(correct) as a function of fitted internal variance
query = '''
        SELECT animal,experiment,augend,addend,singleton,AVG(chose_sum==((augend+addend)>singleton)) as pcor
        FROM behavioralstudy
        WHERE experiment in ('Addition','Subtraction')
        GROUP BY animal,experiment,augend,addend,singleton
        ORDER BY animal,session
'''
data2 = Helper.getData(cur,query)

height_per_panel = 6
width_per_panel = 6
h,ax = plt.subplots(2,2,figsize=[2*width_per_panel,2*height_per_panel])

#iterate through experiments and plot correlation between p(correct) and internal variance
animals = ['Ruffio','Xavier']
experiments = ['Addition','Subtraction']
corr = []
for i in range(0,len(animals)):
    for j in range(0,len(experiments)):
        cond = (data2['experiment']==experiments[j]) & (data2['animal']==animals[i])
        d = data2.loc[cond]
        d['dm_var'] = dm_var_nonlinear(fit_all_out['par'][i*2 + j],d)
        Plotter.scatter(ax[i,j],xdata=d['dm_var'],ydata=d['pcor'],
                        xlabel='Internal variance',ylabel='accuracy',
                title=animals[i]+', '+experiments[j],ylim=[0,1],xlim=[],xticks=[],
                yticks=[0,0.25,0.5,0.75,1],corrline='on')
        corr.append((animals[i],experiments[j],scipy.stats.spearmanr(d['dm_var'],d['pcor'])))





cond = (data['animal']=='Xavier') & (data['experiment'] == 'Addition')
w = [.4,.4,.4,.01]#fit_all_out['par'][2]
d = data.loc[cond]
dm_nonlinear_oneweight(w,d)

pred = dm_nonlinear_oneweight(w,d)
y = data['chose_sum'].loc[cond]

#%%
#Backwards elimination on data w/ 
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
resolution = 3              #grid search resolution
method = None      #Fitting algorithm
validateby = ['session']
groupby = ['animal','experiment','session']


be_out = logistic_backwardselimination_sessionwise(df,model,groupby=groupby,
          groupby_thresh=.05,pthresh=.05)


#%%
#Fit power function model to data
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
resolution = 2              #grid search resolution
method = None      #Fitting algorithm
groupby = ['animal','experiment']
validateby = ['session']
sess_groupby = ['animal','experiment','session']

#Some invariants for fitting
realmin = Analyzer.realmin#smallest floating point number supported by numpy
pracmin = .001 #a practical minimum, so that we don't accidentally square realmin.
models = Analyzer.get_models()#the models

#N is a free parameter for everything
power_full = lambda w,data: models['power'](w,data)
bounds = ((None,None),(None,None),(None,None),(None,None),(pracmin,None),(pracmin,None),(pracmin,None))
parrange = (np.linspace(-2,2,resolution),np.linspace(-2,2,resolution),
            np.linspace(-2,2,resolution),np.linspace(-2,2,resolution),
            np.linspace(pracmin,2,resolution),np.linspace(pracmin,2,resolution),
            np.linspace(pracmin,2,resolution))
fit_all_out = Analyzer.fit_grid_search(power_full,data,data['chose_sum'],
                        parrange,groupby=groupby,bounds=bounds,method=method) 


#N is different for visible items (addend,singleton) and items in WM (augend)
power_vis_nonvis = lambda w,data: models['power']([w[0],w[1],w[2],w[3],w[4],w[5],w[5]],data)
bounds = ((None,None),(None,None),(None,None),(None,None),(pracmin,None),(pracmin,None))
parrange = (np.linspace(-2,2,resolution),np.linspace(-2,2,resolution),
            np.linspace(-2,2,resolution),np.linspace(-2,2,resolution),
            np.linspace(pracmin,2,resolution),np.linspace(pracmin,2,resolution))
fit_all_out = Analyzer.fit_grid_search(power_vis_nonvis,data,data['chose_sum'],
                        parrange,groupby=groupby,bounds=bounds,method=method)


#N is the same for all terms
power_single = lambda w,data: models['power']([w[0],w[1],w[2],w[3],w[4],w[4],w[4]],data)
bounds = ((None,None),(None,None),(None,None),(None,None),(pracmin,None))
parrange = (np.linspace(-2,2,resolution),np.linspace(-2,2,resolution),
            np.linspace(-2,2,resolution),np.linspace(-2,2,resolution),
            np.linspace(pracmin,2,resolution))
fit_all_out = Analyzer.fit_grid_search(power_vis_nonvis,data,data['chose_sum'],
                        parrange,groupby=groupby,bounds=bounds,method=method)


#N is one
linear = lambda w,data: models['linear'](w,data)
bounds = ((None,None),(None,None),(None,None),(None,None))
parrange = (np.linspace(-2,2,resolution),np.linspace(-2,2,resolution),
            np.linspace(-2,2,resolution),np.linspace(-2,2,resolution))
fit_all_out = Analyzer.fit_grid_search(linear,data,data['chose_sum'],
                        parrange,groupby=groupby,bounds=bounds,method=method)