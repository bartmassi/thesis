# -*- coding: utf-8 -*-
"""
Testing ground for running analysis on data from the Arithmetic study.

@author: bart
creation date: 12-11-17
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
output = Analyzer.logistic_regression(data2,model='chose_sum ~ augend + addend + singleton',
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
be_results = Analyzer.logistic_backwardselimination_sessionwise(data3,model=model,
                groupby=['animal','session'],groupby_thresh=.05,pthresh=.05)

#%%
###########Fit approximate number model from Dehaene 2007

#Make SQl query
query3 = '''
        SELECT session,animal,chose_sum,augend,addend,singleton,
        augend+addend-singleton as diff
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

Plotter.panelplots(data,plotvar='pred_dm_onescale',scattervar='chose_sum',groupby=['augend','addend','diff'],
                   ylim=[0,1],xlim=[-2,2],xlabel='diff',ylabel='p(choose sum)',
                    xticks=[-2,-1,0,1,2],yticks=[0,.25,.5,.75,1])
plt.tight_layout()

h,axes = plt.subplots(1,2)
Plotter.scatter(axes[0],choices,xlabel = 'Actual p(choose sum)',
                ydata= ratiopred_dm_onescale,ylabel = 'Predicted p(choose sum)',
                title='DM2007, single scaling factor',color=[0,0,0],xlim=[0,1],ylim=[0,1])
Plotter.scatter(axes[1],choices,xlabel = 'Actual p(choose sum)',
                ydata= ratiopred_dm_augscale,ylabel = 'Predicted p(choose sum)',
                title='DM2007, separate aug scaling factor',color=[0,0,0],xlim=[0,1],ylim=[0,1])
plt.tight_layout()


#%%
#Fit a model that weights base rate probabilities and approximate sum representation
#Make SQl query
query3 = '''
        SELECT session,animal,chose_sum,augend,addend,singleton,
        augend+addend-singleton as diff
        FROM behavioralstudy
        WHERE experiment = 'FlatLO' and animal='Xavier'
        ORDER BY animal,session
'''
#Execute query, then convert to pandas table
data = Helper.getData(cur,query3)
#get p(correct) for each value of singleton in the flatLO experiment
flatlo = Helper.getFlatLOTrialset()
pcs = flatlo['pcs']
using = flatlo['using']
avgratios = flatlo['avgratio']


#Specify model function
#setup gaussian approximate number model from Dehaene 2007, single scaling factor
dm_onescale = lambda w,data: 1-scipy.stats.norm(data['augend']+data['addend']-data['singleton'],
         w*np.sqrt(data['augend']**2. + data['addend']**2 + data['singleton']**2)).cdf(0)
#setup dynamic base-rate weighting function
dm_prior_weighting_model = lambda w,data: (1-w[data['singleton']])*dm_onescale(w[0],data) \
                                           + w[data['singleton']]*pcs[data['singleton']-1]

prior_weighting = lambda w: (w[0] + w[1]*(data['augend']+data['addend'])/data['singleton'] \
                             + w[2]*np.abs(pcs[data['singleton']-1]-.5))
dm_prior_usefulness_model = lambda w,data: pcs[data['singleton']-1]*prior_weighting(w[1:]) \
                            + dm_onescale(w[0],data)*(1-prior_weighting(w[1:]))
 

#Fit model w/ 12 separate weights for each singleton
realmin = np.finfo(np.double).tiny
bounds = ((.00001,None),(realmin,1-realmin),(realmin,1-realmin),(realmin,1-realmin),(realmin,1-realmin),
                    (realmin,1-realmin),(realmin,1-realmin),(realmin,1-realmin),(realmin,1-realmin),(realmin,1-realmin),
                    (realmin,1-realmin),(realmin,1-realmin),(realmin,1-realmin),)
x0 = (1,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5)#initial parameter values
opt1 = Analyzer.fit_model(model=dm_prior_weighting_model,X=data,y=data['chose_sum'],x0=x0,bounds=bounds)

#fit model w/ 3 term function for weighting
bounds = ((.00001,None),(None,None),(None,None),(None,None),)
x0 = (.5,0,.5,.5)
opt2 = Analyzer.fit_model(model=dm_prior_usefulness_model,X=data,y=data['chose_sum'],x0=x0,bounds=bounds)

h,ax = plt.subplots(1,1)
avgweights = np.array([np.mean(prior_weighting(opt2.x[1:])[data['singleton']==si]) for si in using])
Plotter.lineplot(ax,xdata=using,ydata=avgweights,xlabel='Singleton',ylabel='P(correct|singleton) weight',
                 title='prior-informativeness weighting',color=[1,0,0],label='informativeness weighting')
Plotter.lineplot(ax,xdata=using,ydata=opt1.x[1:],xlabel='Singleton',ylabel='P(correct|singleton) weight',
                 title='individual singleton weighting',color=[0,0,0],label='individual weighting')

#test grid search
prange = [np.linspace(0,1,3),np.linspace(-1,1,4),np.linspace(-1,1,4),np.arange(-5,5,4)]
prange = [1.0,-0.3,0.3,-1]
Analyzer.fit_grid_search(model=dm_prior_usefulness_model,X=data,y=data['chose_sum'],
                         parrange=prange,bounds=bounds)


#test linear_model
lr = Analyzer.logistic_regression(data,model='chose_sum ~ augend * addend * singleton')
print(lr.bic)

logistic = lambda x:1./(1.+np.exp(-x))
linear_model = lambda w,data: w[0]+w[1]*data['augend']+w[2]*data['addend']+w[3]*data['singleton']
logistic_model = lambda w,data: logistic(linear_model(w,data))
x0 = [1,1,1,1]
opt_test = Analyzer.fit_model(model=logistic_model,X=data,y=data['chose_sum'],x0=x0)

#Plot weights aside prior informativeness and avg difficulty
h,ax = plt.subplots(1,1)
Plotter.lineplot(ax,xdata=using,ydata=opt1.x[1:],xlabel='Singleton',ylabel='P(correct|singleton) weight',
                 title='dynamic singleton weighting',color=[0,0,0],label='Weights')
Plotter.lineplot(ax,xdata=using,ydata=pcs,color=[1,0,0],label='p(sum correct)')
Plotter.lineplot(ax,xdata=using,ydata=avgratios,color=[0,0,1],label='avg ratio')
plt.legend(loc=0,fontsize='x-small')
                                                 
#add predictions to data structure
data['pred'] = lr.fittedvalues[0]#logistic_model(opt_test.x,data)
data['ratio'] = (data['augend']+data['addend'])/data['singleton']

#Make predictions at each ratio
uratio = data['ratio'].drop_duplicates()
ratiopred = [np.mean(data['pred'].loc[data['ratio'] == ur]) for ur in uratio]
choices = [np.mean(data['chose_sum'].loc[data['ratio']==ur]) for ur in uratio]

Plotter.panelplots(data,plotvar='pred',scattervar='chose_sum',groupby=['augend','addend','diff'],
                   ylim=[0,1],xlim=[-2,2],xlabel='diff',ylabel='p(choose sum)',
                    xticks=[-2,-1,0,1,2],yticks=[0,.25,.5,.75,1])
plt.tight_layout()

h,axes = plt.subplots(1,1)
Plotter.scatter(axes,choices,xlabel = 'Actual p(choose sum)',
                ydata= ratiopred,ylabel = 'Predicted p(choose sum)',
                title='DM2007+baserate',color=[0,0,0],xlim=[0,1],ylim=[0,1])
plt.tight_layout()

#%%
#determine how often the animal chooses sum as a function of the product of aug,
#add, and sing
query = '''
        SELECT animal,augend*addend*singleton as prod,AVG(chose_sum) as pchosesum
        FROM behavioralstudy
        WHERE experiment='FlatLO'
        GROUP BY animal,prod
        ORDER BY animal,prod
'''
#Execute query, then convert to pandas table
data = Helper.getData(cur,query)

h,ax = plt.subplots(1,2)
Plotter.scatter(ax[0],xdata=data['prod'].loc[data['animal']=='Xavier'],
                ydata=data['pchosesum'].loc[data['animal']=='Xavier'],
            xlabel='aug*add*sing',ylabel='p(chose sum)',ylim=[0,1],xlim=[1,288],color=[1,0,0])
Plotter.scatter(ax[1],xdata=data['prod'].loc[data['animal']=='Ruffio'],
                ydata=data['pchosesum'].loc[data['animal']=='Ruffio'],
            xlabel='aug*add*sing',ylabel='p(chose sum)',ylim=[0,1],xlim=[1,288],color=[0,0,1])



#%%
#determine how often the animals choose the sum as a function of sum and singleton
#show this as a matrix.

query = '''
        SELECT animal,augend+addend as sum,singleton,AVG(chose_sum) as chose_sum
        FROM behavioralstudy
        WHERE experiment='Addition'
        GROUP BY animal,sum,singleton
        ORDER BY animal,sum,singleton
'''

data = Helper.getData(cur,query)

usum = np.unique(data['sum'])
nusum = len(usum)
using = np.unique(data['singleton'])
nusing = len(using)

#
xperf = np.array([[data['chose_sum'].loc[(data['singleton']==si) & (data['sum']==su) & (data['animal']=='Xavier')] if su != si else np.nan for su in usum] for si in using])
rperf = np.array([[data['chose_sum'].loc[(data['singleton']==si) & (data['sum']==su) & (data['animal']=='Ruffio')] if su != si else np.nan for su in usum] for si in using])


with PdfPages('D:\\Bart\\Dropbox\\pdf_test.pdf') as pdf:
    h,ax = plt.subplots(1,1)
    Plotter.gridplot(ax,xperf,cmap=plt.cm.seismic,title='Monkey X',
        xticks=np.arange(0,len(usum),1),xticklabels=usum,xlabel='Sum',
        yticks=np.arange(0,len(using),1),yticklabels=using,ylabel='Singleton',
        cticks=[0,.5,1],clabel='p(choose sum)')
    pdf.savefig()
    
    h,ax = plt.subplots(1,1)
    Plotter.gridplot(ax,rperf,cmap=plt.cm.seismic,title='Monkey R',
        xticks=np.arange(0,len(usum),1),xticklabels=usum,xlabel='Sum',
        yticks=np.arange(0,len(using),1),yticklabels=using,ylabel='Singleton',
        cticks=[0,.5,1],clabel='p(choose sum)')
    pdf.savefig()
    
#get psychometric curve data
query2 = '''
        SELECT animal,augend+addend-singleton as diff,chose_sum,'animal'
        FROM behavioralstudy
        WHERE experiment='Addition'
        ORDER BY animal,diff
'''

data2 = Helper.getData(cur,query2)


udiffs = np.unique(data2['diff'])
x_psum = [np.mean(data2['chose_sum'].loc[(data2['diff']==di) & (data2['animal']=='Xavier')]) for di in udiffs]
r_psum = [np.mean(data2['chose_sum'].loc[(data2['diff']==di) & (data2['animal']=='Ruffio')]) for di in udiffs]
x_sem = [scipy.stats.sem(data2['chose_sum'].loc[(data2['diff']==di) & (data2['animal']=='Xavier')]) for di in udiffs]
r_sem = [scipy.stats.sem(data2['chose_sum'].loc[(data2['diff']==di) & (data2['animal']=='Ruffio')]) for di in udiffs]


h,ax = plt.subplots(1,1)
Plotter.lineplot(ax,xdata=udiffs,
    ydata=x_psum,sem=x_sem,title='Monkey X',xlabel='Sum - Singleton',xticks = [-8,-4,0,4,8],
    ylabel='P(choose sum)',yticks=[0,.25,.5,.75,1])
Plotter.scatter(ax,xdata=udiffs,
    ydata=x_psum,identity='off')



#%%
###########Look at the trials where the animals' behavior deviates most from optimal. 

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

data['irrationality'] = (data['diff']>0)-data['chose_sum']

with PdfPages('D:\\Bart\\Dropbox\\irrationality.pdf') as pdf:
    Plotter.panelplots(data,plotvar='irrationality',groupby=['augend','addend','diff'],
                       ylim=[-1,1],xlim=[-2,2],xlabel='diff',ylabel='P(sum correct) - P(choose sum)',
                        xticks=[-2,-1,0,1,2],yticks=[-1,-.5,0,.5,1],horiz=0)
    pdf.savefig()
    Plotter.panelplots(data,plotvar='irrationality',groupby=['addend','augend','diff'],
                       ylim=[-1,1],xlim=[-2,2],xlabel='diff',ylabel='P(sum correct) - P(choose sum)',
                        xticks=[-2,-1,0,1,2],yticks=[-1,-.5,0,.5,1],horiz=0)
    pdf.savefig()

