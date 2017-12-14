# -*- coding: utf-8 -*-
"""
Testing ground for running analysis on data from the Arithmetic study.

@author: bart
creation date: 12-11-17
"""


##Run these prior to running any code here. 
#cd D:\\Bart\\Dropbox\\code\\python\\leelab\\thesis
#%load_ext autoreload
#%autoreload 2

import sqlite3
import pandas as pd
import Plotter
import analyzer
import matplotlib.pyplot as plt
import numpy as np
import scipy

def getData(cur,query):
    cur.execute(query)
    dataout = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    data = pd.DataFrame.from_records(dataout,columns=colnames)
    return data

dbfloc = 'D:\\Bart\\Dropbox\\science\\leelab\\projects\\Arithmetic\\data\\_curatedData\\'
conn = sqlite3.connect(database=dbfloc+'arithmeticstudy.db')
cur = conn.cursor()

#%%
###########Generate plot of accuracy in enforced addition and non-enforced addition trials.
#EA trials are those where sum > singleton, but augend < singleton and addend < singleton.
#First generate a SQL query to the database
#this query gets accuracy in EA trials and accuracy in non-EA trials
query1 = '''
        SELECT session,animal,experiment,
        
        AVG(CASE WHEN (augend+addend) > singleton AND augend<singleton AND addend<singleton
        THEN CAST(CASE WHEN (augend+addend) > singleton THEN chose_sum ELSE (NOT chose_sum) END AS int)
        ELSE NULL END) AS EA_pcorrect,
        
        AVG(CASE WHEN NOT ((augend+addend) > singleton AND augend<singleton AND addend<singleton)
        THEN CAST(CASE WHEN (augend+addend) > singleton THEN chose_sum ELSE (NOT chose_sum) END AS int)
        ELSE NULL END) AS nonEA_pcorrect
        
        FROM behavioralstudy
        GROUP BY session,animal,experiment
        HAVING experiment = 'Addition'
        ORDER BY animal,session;
        '''
#execute query, and convert output into a pandas table
cur.execute(query1)
dataout1 = cur.fetchall()
colnames1 = [desc[0] for desc in cur.description]
data1 = pd.DataFrame.from_records(dataout1,columns=colnames1)

#Generate a scatterplot of the results. 
h,axes = plt.subplots(1,1)
Plotter.scatter(axes,xdata=data1['nonEA_pcorrect'],xlabel='Accuracy in non-EA trials',xlim=(.5,1),
                ydata=data1['EA_pcorrect'],ylabel='Accuracy in EA trials',ylim=(.5,1),title='Animal Accuracy')


#%%
###########Fit a regression model to animals' choice data, and then plot coefficients.
#SQL query
query2 = '''
        SELECT session,animal,CAST(chose_sum AS int) as chose_sum,augend,addend,singleton
        FROM behavioralstudy
        WHERE experiment = 'Addition'
        ORDER BY animal,session
'''
#Execute query, then convert to pandas table
cur.execute(query2)
dataout2 = cur.fetchall()
colnames2 = [desc[0] for desc in cur.description]
data2 = pd.DataFrame.from_records(dataout2,columns=colnames2)

#Fit regression model to separate sessions
output = analyzer.logistic_regression(data2,model='chose_sum ~ augend + addend + singleton',
                                      groupby=['animal','session'])

#setup plotting structures

h,axes = plt.subplots(2,2)
#plot augend vs. addend
Plotter.scatter(axes[0,0],xdata=output['b_augend'].loc[output['animal']=='Ruffio'],xlabel = 'Augend coefficient',
                ydata=output['b_addend'].loc[output['animal']=='Ruffio'],ylabel = 'Addend coefficient',
                title='Aug vs. Add Weight',color=[1,0,0])
Plotter.scatter(axes[0,0],xdata=output['b_augend'].loc[output['animal']=='Xavier'],xlabel = 'Augend coefficient',
                ydata=output['b_addend'].loc[output['animal']=='Xavier'],ylabel = 'Addend coefficient',
                title='Aug vs. Add Weight',color=[0,0,1])
#plot augend vs. singleton
Plotter.scatter(axes[0,1],xdata=output['b_augend'].loc[output['animal']=='Ruffio'],xlabel = 'Augend coefficient',
                ydata=output['b_singleton'].loc[output['animal']=='Ruffio'],ylabel = 'Singleton coefficient',
                title='Aug vs. Sing Weight',color=[1,0,0])
Plotter.scatter(axes[0,1],xdata=output['b_augend'].loc[output['animal']=='Xavier'],xlabel = 'Augend coefficient',
                ydata=output['b_singleton'].loc[output['animal']=='Xavier'],ylabel = 'Singleton coefficient',
                title='Aug vs. Sing Weight',color=[0,0,1])
#plot addend vs. singleton
Plotter.scatter(axes[1,1],xdata=output['b_addend'].loc[output['animal']=='Ruffio'],xlabel = 'Addend coefficient',
                ydata=output['b_singleton'].loc[output['animal']=='Ruffio'],ylabel = 'Singleton coefficient',
                title='Add vs. Sing Weight',color=[1,0,0])
Plotter.scatter(axes[1,1],xdata=output['b_addend'].loc[output['animal']=='Xavier'],xlabel = 'Addend coefficient',
                ydata=output['b_singleton'].loc[output['animal']=='Xavier'],ylabel = 'Singleton coefficient',
                title='Add vs. Sing Weight',color=[0,0,1])
plt.tight_layout()



#%%
###########Perform backwards elimination on the full logistic regression model.
#SQL query
query3 = '''
        SELECT session,animal,CAST(chose_sum AS int) as chose_sum,augend,addend,singleton
        FROM behavioralstudy
        WHERE experiment = 'FlatLO' and animal = 'Ruffio'
        ORDER BY animal,session
'''
#Execute query, then convert to pandas table
cur.execute(query3)
dataout3 = cur.fetchall()
colnames3 = [desc[0] for desc in cur.description]
data3 = pd.DataFrame.from_records(dataout3,columns=colnames3)

#Perform backwards elimination on full model. All interactions/terms with augend,
#addend, and singleton.
model = 'chose_sum ~ augend * addend * singleton'
be_results = analyzer.logistic_backwardselimination_sessionwise(data3,model=model,
                groupby=['animal','session'],groupby_thresh=.05,pthresh=.05)

#%%
###########Fit approximate number model from Dehaene 2007

#Make SQl query
query3 = '''
        SELECT session,animal,chose_sum,augend,addend,singleton
        FROM behavioralstudy
        WHERE experiment = 'FlatLO' and animal='Ruffio'
        ORDER BY animal,session
'''
#Execute query, then convert to pandas table
data = getData(cur,query3)

#decision variable is p(sum correct)
#linearDV = data['augend']+data['addend']-data['singleton']
#scale = w*np.sqrt(data['augend']**2. + data['addend']**2 + data['singleton']**2)
#predResponse = 1-scipy.stats.norm(linearDV,scale).cdf(0)

#setup cost function for fitting
realmin = np.finfo(np.double).tiny #smallest possible floating point number
cost = lambda y,h: -np.sum(y*np.log(np.max([h,np.ones(len(h))*realmin],axis=0))+
                    (1-y)*np.log(np.max([1-h,np.ones(len(h))*realmin],axis=0))); 

#setup gaussian approximate number model from Dehaene 2007, single scaling factor
dm_onescale = lambda w,data: 1-scipy.stats.norm(data['augend']+data['addend']-data['singleton'],
         w*np.sqrt(data['augend']**2. + data['addend']**2 + data['singleton']**2)).cdf(0)

#Dehaene 2007, separate scaling factor for augend.
dm_augscale = lambda w,data: 1-scipy.stats.norm(data['augend']+data['addend']-data['singleton'],
         np.sqrt((w[0]**2)*data['augend']**2 + (w[1]**2)*data['addend']**2 + (w[1]**2)*data['singleton']**2)).cdf(0)

#setup  objective for fitting
objfun_dm_onescale = lambda w: cost(data['chose_sum'],dm_onescale(w,data))
objfun_dm_augscale = lambda w: cost(data['chose_sum'],dm_augscale(w,data))

#perform fitting
opt_dm_onescale = scipy.optimize.minimize(fun=objfun_dm_onescale,x0=(1.0),bounds=((.00001,None),))
opt_dm_augscale = scipy.optimize.minimize(fun=objfun_dm_augscale,x0=(1.0,1.0),bounds=((.00001,None),(.00001,None),))

w_best_dm_onescale = opt_dm_onescale.x
bic_dm_onescale = opt_dm_onescale.fun*2 + np.log(data.shape[0])*1
w_best_dm_augscale = opt_dm_augscale.x
bic_dm_augscale = opt_dm_augscale.fun*2 + np.log(data.shape[0])*2
                                                 
#add predictions to data structure
data['pred_dm_onescale'] = dm_onescale(w_best_dm_onescale,data)
data['pred_dm_augscale'] = dm_augscale(w_best_dm_augscale,data)
data['ratio'] = (data['augend']+data['addend'])/data['singleton']

uratio = data['ratio'].drop_duplicates()

ratiopred_dm_onescale = [np.mean(data['pred_dm_onescale'].loc[data['ratio'] == ur]) for ur in uratio]
ratiopred_dm_augscale = [np.mean(data['pred_dm_augscale'].loc[data['ratio'] == ur]) for ur in uratio]
choices = [np.mean(data['chose_sum'].loc[data['ratio']==ur]) for ur in uratio]

h,axes = plt.subplots(1,2)
Plotter.scatter(axes[0],choices,xlabel = 'Actual p(choose sum)',
                ydata= ratiopred_dm_onescale,ylabel = 'Predicted p(choose sum)',
                title='DM2007, single scaling factor',color=[0,0,0],xlim=[0,1],ylim=[0,1])
Plotter.scatter(axes[1],choices,xlabel = 'Actual p(choose sum)',
                ydata= ratiopred_dm_augscale,ylabel = 'Predicted p(choose sum)',
                title='DM2007, separate aug scaling factor',color=[0,0,0],xlim=[0,1],ylim=[0,1])
plt.tight_layout()