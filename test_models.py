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