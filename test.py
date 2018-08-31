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
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import pdb
from sklearn.decomposition import PCA

#this ensures that text is editable in illustrator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Arial'

try:
    dbfloc = 'D:\\Bart\\Dropbox\\science\\leelab\\projects\\Arithmetic\\data\\_curatedData\\'
    conn = sqlite3.connect(database=dbfloc+'arithmeticstudy.db')
    cur = conn.cursor()
except:
    print('Work computer detected.')
    dbfloc = 'C:\\Users\\Bart\\Dropbox\\science\\leelab\\projects\\Arithmetic\\data\\_curatedData\\'
    conn = sqlite3.connect(database=dbfloc+'arithmeticstudy.db')
    cur = conn.cursor()
    
    
def gettimestr():
    return str(int(np.floor(time.time())))

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


query = '''
        SELECT DISTINCT augend+addend as sum,singleton,trialset
        FROM behavioralstudy
        WHERE experiment = 'Subtraction' and animal='Xavier' and trialset==1
        ORDER BY animal,session
'''

data = Helper.getData(cur,query);



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


#%%
#Make addend-augend grid plot
#Plots the animals' performance as a function of singleton for each aug,add pair, and prior in lines.


flatlo = Helper.getFlatLOTrialset()
query = '''
        SELECT augend,addend,singleton,chose_sum,augend+addend-singleton as diff,
        animal,session,trial,(augend+addend-singleton)>0 as sum_correct,
        1.0*(augend+addend)/singleton as ratio
        FROM behavioralstudy
        WHERE experiment='FlatLO' AND animal='Ruffio'
        ORDER BY animal,session
'''

data = Helper.getData(cur,query)

uaug = np.unique(data['augend'])
uadd = np.unique(data['addend'])
using = np.unique(data['singleton'])
usingadddiff = np.unique(data['singleton']-data['addend']);
usingaddrat = np.unique(data['singleton']/data['addend']);
naug = len(uaug)
nadd = len(uadd)
nsing = len(using)
nsingadddiff = len(usingadddiff)
nsingaddrat = len(usingaddrat)

#iterate through augend and addend, find unique singleton and performance at each triplet.
usings = np.empty([naug,nadd,4])
perf = np.empty([naug,nadd,4,2])
for i in range(0,naug):
    for j in range(0,nadd):
        if(i>0 or j>0):
            usings[i,j,:] = np.unique(data['singleton'].loc[(data['augend']==uaug[i]) & (data['addend']==uadd[j])])
            for k in range(0,len(usings[i,j,:])):
                cond = (data['singleton']==usings[i,j,k]) & (data['augend']==uaug[i]) & (data['addend']==uadd[j])
                perf[i,j,k,0] = np.mean(data['chose_sum'].loc[cond])
                perf[i,j,k,1] = scipy.stats.sem(data['chose_sum'].loc[cond])
        else:
            usings[i,j,1:3] = np.unique(data['singleton'].loc[(data['augend']==uaug[i]) & (data['addend']==uadd[j])])
            usings[i,j,[0,3]] = np.nan
            perf[i,j,1,:] = np.nan
            perf[i,j,2,:] = np.nan
            for k in range(1,3):
                cond = (data['singleton']==usings[i,j,k]) & (data['augend']==uaug[i]) & (data['addend']==uadd[j])
                perf[i,j,k,0] = np.mean(data['chose_sum'].loc[cond])
                perf[i,j,k,1] = scipy.stats.sem(data['chose_sum'].loc[cond])
                
#make panel plots
nrow = nadd
ncol = naug
h,ax = plt.subplots(nrow,ncol,figsize=[ncol*6,nrow*6])
yt = [0,.25,.5,.75,1]
for i in range(0,naug):
    for j in range(0,nadd):
        if(i>0 or j>0):
            #plot priors
            #addsing prior
            Plotter.lineplot(ax[j,i],xdata=usings[i,j,:],
                ydata=flatlo['add_sing_prior'][j,usings[i,j,:].astype(int)-1],
                title='Aug='+str(uaug[i])+',Add='+str(uadd[j]),
                xlabel='Singleton',xticks=usings[i,j,:].tolist(),
                ylabel='P(choose sum)',yticks=yt,ylim=[0,1],label='add,sing prior',color=[1,0,0])
            Plotter.lineplot(ax[j,i],xdata=usings[i,j,:],
                ydata=flatlo['pcs'][usings[i,j,:].astype(int)-1],
                title='Aug='+str(uaug[i])+',Add='+str(uadd[j]),
                xlabel='Singleton',xticks=usings[i,j,:].tolist(),
                ylabel='P(choose sum)',yticks=yt,ylim=[0,1],label='sing prior',color=[0,0,1])
            Plotter.lineplot(ax[j,i],xdata=usings[i,j,:],
                ydata=[flatlo['singadddiff_prior'][usingadddiff.tolist().index(x)]
                for x in usings[i,j,:].astype(int)-uadd[j]],
                title='Aug='+str(uaug[i])+',Add='+str(uadd[j]),
                xlabel='Singleton',xticks=usings[i,j,:].tolist(),
                ylabel='P(choose sum)',yticks=yt,ylim=[0,1],label='sing minus add prior',color=[0,1,0])
            Plotter.lineplot(ax[j,i],xdata=usings[i,j,:],
                ydata=[flatlo['singaddrat_prior'][usingaddrat.tolist().index(x)]
                for x in usings[i,j,:]/uadd[j]],
                title='Aug='+str(uaug[i])+',Add='+str(uadd[j]),
                xlabel='Singleton',xticks=usings[i,j,:].tolist(),
                ylabel='P(choose sum)',yticks=yt,ylim=[0,1],label='sing/add prior',color=[0,1,1])
            #plot data
            Plotter.lineplot(ax[j,i],xdata=usings[i,j,:],
                ydata=perf[i,j,:,0],ls='none',sem=perf[i,j,:,1],yticks=[0,.25,.5,.75,1],ylim=[0,1])
            Plotter.scatter(ax[j,i],xdata=usings[i,j,:],
                ydata=perf[i,j,:,0],ylim=[0,1],label='Animal',identity='off')
        else:
            #plot priors
            #sing prior
            Plotter.lineplot(ax[j,i],xdata=usings[i,j,1:3],
                ydata=flatlo['add_sing_prior'][j,usings[i,j,1:3].astype(int)-1],
                title='Aug='+str(uaug[i])+',Add='+str(uadd[j]),
                xlabel='Singleton',xticks=usings[i,j,1:3].tolist(),
                ylabel='P(choose sum)',yticks=yt,ylim=[0,1],label='add,sing prior',xlim=[0,4],color=[1,0,0])
            Plotter.lineplot(ax[j,i],xdata=usings[i,j,1:3],
                ydata=flatlo['pcs'][usings[i,j,1:3].astype(int)-1],
                title='Aug='+str(uaug[i])+',Add='+str(uadd[j]),
                xlabel='Singleton',xticks=usings[i,j,1:3].tolist(),
                ylabel='P(choose sum)',yticks=yt,ylim=[0,1],label='sing prior',xlim=[0,4],color=[0,0,1])
            Plotter.lineplot(ax[j,i],xdata=usings[i,j,1:3],
                ydata=[flatlo['singadddiff_prior'][usingadddiff.tolist().index(x)]
                for x in usings[i,j,1:3].astype(int)-uadd[j]],
                title='Aug='+str(uaug[i])+',Add='+str(uadd[j]),
                xlabel='Singleton',xticks=usings[i,j,1:3].tolist(),
                ylabel='P(choose sum)',yticks=yt,ylim=[0,1],label='sing minus add prior',xlim=[0,4],color=[0,1,0])
            Plotter.lineplot(ax[j,i],xdata=usings[i,j,1:3],
                ydata=[flatlo['singaddrat_prior'][usingaddrat.tolist().index(x)]
                for x in usings[i,j,1:3]/uadd[j]],
                title='Aug='+str(uaug[i])+',Add='+str(uadd[j]),
                xlabel='Singleton',xticks=usings[i,j,1:3].tolist(),
                ylabel='P(choose sum)',yticks=yt,ylim=[0,1],label='sing/add prior',xlim=[0,4],color=[0,1,1])
            ax[j,i].legend(loc='center left',fontsize='small',frameon=False)
            #plot data
            Plotter.lineplot(ax[j,i],xdata=usings[i,j,:],
                ydata=perf[i,j,:,0],sem=perf[i,j,:,1],ls='none',yticks=[0,.25,.5,.75,1],ylim=[0,1])
            Plotter.scatter(ax[j,i],xdata=usings[i,j,:],
                ydata=perf[i,j,:,0],ylim=[0,1],label='Animal',identity='off')
            
data['addaug'] = data['addend']*10 + data['augend']
data['const'] = data['augend']-data['augend']
data['sing_prior']
data['sing_prior'] = sing_prior
Plotter.panelplots(data=data,plotvar='chose_sum',groupby=['addaug','const','singleton'],
               axes=None,scattervar=[],xlim=[],ylim=[0,1],xlabel='Singleton',ylabel='P(choose sum)',
               xticks=[],yticks=[],horiz=None,maxcol=6,legend='off')
            
#for every combo of aug add sing, compute ratio and look at deviation from add-sing prior
#also look at singleton prior
ratio_perf_addsing = []
ratio_perf_sing = []
for i in range(0,naug):
    for j in range(0,nadd):
        for k in range(0,nsing):
            cond = (data['augend']==uaug[i]) & (data['addend']==uadd[j]) & (data['singleton']==using[k])
            ratio = (uaug[i]+uadd[j])/using[k]
            perfdiff_addsing = np.mean(data['chose_sum'].loc[cond])-flatlo['add_sing_prior'][j,k]
            perfdiff_sing = np.mean(data['chose_sum'].loc[cond])-flatlo['pcs'][k]
            ratio_perf_addsing.append([ratio,perfdiff_addsing])
            ratio_perf_sing.append([ratio,perfdiff_sing])


#reorganize data
ratperf_addsing = np.array(ratio_perf_addsing)
ratperf_addsing = ratperf_addsing[~np.isnan(ratperf_addsing).any(axis=1)]
Xaddsing = np.vstack([np.log(ratperf_addsing[:,0]),np.ones(len(ratperf_addsing[:,0]))]).T
m_addsing,c_addsing = np.linalg.lstsq(Xaddsing,ratperf_addsing[:,1])[0]
                                
#get data for just singleton
ratperf_sing = np.array(ratio_perf_sing)
ratperf_sing = ratperf_sing[~np.isnan(ratperf_sing).any(axis=1)]
X = np.vstack([np.log(ratperf_sing[:,0]),np.ones(len(ratperf_sing[:,0]))]).T
m_sing,c_sing = np.linalg.lstsq(X,ratperf_sing[:,1])[0]

h,ax = plt.subplots(1,2,figsize=[2*6,6])
Plotter.scatter(ax[0],xdata=np.log(ratperf_addsing[:,0]),ydata=ratperf_addsing[:,1],title='Actual vs. Addend,Singleton Prior choice',
                xlabel='log(sum/sing)',ylabel='p(choose sum) - p(sum>sing|addend,sing)',
                ylim=[-.75,.75],xlim=[-1.25,1.25],identity='cross')
Plotter.lineplot(ax[0],xdata=[-1.25,1.25],ydata=m_addsing*np.array([-1.25,1.25]) + c_addsing)

Plotter.scatter(ax[1],xdata=np.log(ratperf_sing[:,0]),ydata=ratperf_sing[:,1],title='Actual vs. Singleton Prior choice',
                xlabel='log(sum/sing)',ylabel='p(choose sum) - p(sum>sing|sing)',
                ylim=[-.75,.75],xlim=[-1.25,1.25],identity='cross')
Plotter.lineplot(ax[1],xdata=[-1.25,1.25],ydata=m_sing*np.array([-1.25,1.25]) + c_sing)

x = np.log(ratperf_sing[:,0])
y = ratperf_sing[:,1]
scipy.stats.pearsonr(x,y)


#Look at deviation from prior as a function of log ratio, for unique ratios.
uratio = np.unique(data['ratio'])

addsing_prior = [flatlo['add_sing_prior'][uadd.tolist().index(x),using.tolist().index(y)]
         for x,y in zip(data['addend'],data['singleton'])]
sing_prior = [flatlo['pcs'][using.tolist().index(x)] for x in data['singleton']]
data['addsing_prior'] = addsing_prior
data['sing_prior'] = sing_prior

uratio_perf_addsing = []
uratio_perf_sing = []
for ur in uratio:
    cond = data['ratio']==ur
    uratio_perf_addsing.append(np.mean(data['chose_sum'].loc[cond] - data['addsing_prior'].loc[cond]))
    uratio_perf_sing.append(np.mean(data['chose_sum'].loc[cond] - data['sing_prior'].loc[cond]))

h,ax = plt.subplots(1,2,figsize=[2*6,6])
Plotter.scatter(ax[0],xdata=np.log(uratio),ydata=uratio_perf_addsing,title='Actual vs. Addend,Singleton Prior choice',
                xlabel='log(sum/sing)',ylabel='p(choose sum) - p(sum>sing|addend,sing)',
                ylim=[-.75,.75],xlim=[-1.25,1.25],identity='cross')
Plotter.scatter(ax[1],xdata=np.log(uratio),ydata=uratio_perf_sing,title='Actual vs. Singleton Prior choice',
                xlabel='log(sum/sing)',ylabel='p(choose sum) - p(sum>sing|sing)',
                ylim=[-.75,.75],xlim=[-1.25,1.25],identity='cross')


scipy.stats.pearsonr(np.log(uratio),uratio_perf_sing)

#%%
#Make addend-augend grid plot using panelplots, and visualize prior.


flatlo = Helper.getFlatLOTrialset()
query = '''
        SELECT augend,addend,singleton,chose_sum,augend+addend-singleton as diff,
        animal,session,trial,(augend+addend-singleton)>0 as sum_correct,
        1.0*(augend+addend)/singleton as ratio
        FROM behavioralstudy
        WHERE experiment='FlatLO' AND animal='Xavier'
        ORDER BY animal,session
'''

data = Helper.getData(cur,query) 

#setup panelplots to make aug/add grid
data['addaug'] = data['addend']*10 + data['augend']
data['const'] = data['augend']-data['augend']
addsing_prior = [flatlo['add_sing_prior'][flatlo['uadd'].tolist().index(x),flatlo['using'].tolist().index(y)]
         for x,y in zip(data['addend'],data['singleton'])]
sing_prior = [flatlo['pcs'][flatlo['using'].tolist().index(x)] for x in data['singleton']]
data['sing_prior'] = sing_prior
data['addsing_prior'] = addsing_prior
Plotter.panelplots(data=data,plotvar='addsing_prior',scattervar='chose_sum',
    groupby=['addaug','const','singleton'],axes=None,
    xlim=[],ylim=[0,1],xlabel='Singleton',ylabel='P(choose sum)',
    xticks=[],yticks=[],horiz=None,maxcol=6,legend='off')


#%%
#Look at best fit of logistic regression model to aug-add grid plot.
flatlo = Helper.getFlatLOTrialset()
query = '''
        SELECT augend,addend,singleton,chose_sum,augend+addend-singleton as diff,
        animal,session,trial,(augend+addend-singleton)>0 as sum_correct,
        1.0*(augend+addend)/singleton as ratio,
        aug_num_green,add_num_green,sing_num_green,
        aug_num_quad,add_num_quad,sing_num_quad
        FROM behavioralstudy
        WHERE experiment='FlatLO' AND animal='Ruffio'
        ORDER BY animal,session
'''

data = Helper.getData(cur,query) 

#setup panelplots to make aug/add grid
data['addaug'] = data['addend']*10 + data['augend']
data['const'] = data['augend']-data['augend']

model = 'chose_sum ~ aug_num_green + add_num_green + sing_num_green + aug_num_quad '
lr = Analyzer.logistic_regression(data,model=model)
data['pred'] = lr.fittedvalues[0]

with PdfPages('D:\\Bart\\Dropbox\\Ruffio_regression_perf.pdf') as pdf:
    Plotter.panelplots(data=data,plotvar='pred',scattervar='chose_sum',
        groupby=['addaug','const','singleton'],axes=None,
        xlim=[],ylim=[0,1],xlabel='Singleton',ylabel='P(choose sum)',
        xticks=[],yticks=[],horiz=None,maxcol=6,legend='off')
    pdf.savefig()

#%%
#Look at best fit of logistic regression model to aug-add grid plot.
flatlo = Helper.getFlatLOTrialset()
query = '''
        SELECT augend,addend,singleton,chose_sum,augend+addend-singleton as diff,
        animal,session,trial,(augend+addend-singleton)>0 as sum_correct,
        1.0*(augend+addend)/singleton as ratio,
        aug_num_green,add_num_green,sing_num_green,
        aug_num_quad,add_num_quad,sing_num_quad
        FROM behavioralstudy
        WHERE experiment='FlatLO' AND animal='Xavier'
        ORDER BY animal,session
'''

data = Helper.getData(cur,query) 

#setup panelplots to make aug/add grid
data['addaug'] = data['addend']*10 + data['augend']
data['const'] = data['augend']-data['augend']

model = 'chose_sum ~ aug_num_green * add_num_green * sing_num_green * aug_num_quad '
lr_be = Analyzer.logistic_backwardselimination_sessionwise(df=data,model=model,
                groupby=['animal','session'],groupby_thresh=.05,pthresh=.05)
data['pred'] = lr_be['final_modelout'].fittedvalues[0]


with PdfPages('D:\\Bart\\Dropbox\\Xavier_regression_perf_be.pdf') as pdf:
    Plotter.panelplots(data=data,plotvar='pred',scattervar='chose_sum',
        groupby=['addaug','const','singleton'],axes=None,
        xlim=[],ylim=[0,1],xlabel='Singleton',ylabel='P(choose sum)',
        xticks=[],yticks=[],horiz=None,maxcol=6,legend='off')
    pdf.savefig()


#%%
#analyze effects of previous trial
#drops trial 1 of each session

#query involves a LEFT SELF JOIN
query = '''
    SELECT bs1.animal,bs1.session,bs1.trial,bs1.aug_num_green,bs1.add_num_green,
    bs1.sing_num_green,bs1.aug_num_quad,bs1.chose_sum,
    bs1.augend,bs1.addend,bs1.singleton,
    bs2.chose_sum as chose_sum_m1,
    ((bs2.augend+bs2.addend)>bs2.singleton) as sum_correct_m1,
    bs2.chose_sum==((bs2.augend+bs2.addend)>bs2.singleton) as chose_correct_m1
    FROM behavioralstudy as bs1
    LEFT JOIN behavioralstudy as bs2
    ON (bs1.session = bs2.session) AND (bs1.animal = bs2.animal) AND (bs1.trial = (bs2.trial+1))
    AND (bs1.experiment = bs2.experiment)
    WHERE bs1.animal='Xavier' AND bs1.experiment='FlatLO' AND (chose_sum_m1 IS NOT NULL)
    ORDER BY bs1.animal,bs1.session
'''

data = Helper.getData(cur,query)

data['addaug'] = data['addend']*10 + data['augend']
data['sumcorrectm1_chosesumm1'] = data['sum_correct_m1']*10 + data['chose_sum_m1']

with PdfPages('D:\\Bart\\Dropbox\\Xavier_prevtrial_perf.pdf') as pdf:
    Plotter.panelplots(data=data,plotvar='chose_sum',
        groupby=['addaug','sumcorrectm1_chosesumm1','singleton'],axes=None,
        xlim=[],ylim=[0,1],xlabel='Singleton',ylabel='P(choose sum)',
        xticks=[],yticks=[],horiz=None,maxcol=6,legend='on')
    pdf.savefig()


#%%
#Backwards elimination on flat log-odds trialset for model w/ terms for previous trial.

query = '''
    SELECT bs1.animal,bs1.session,bs1.trial,bs1.aug_num_green,bs1.add_num_green,
    bs1.sing_num_green,bs1.aug_num_quad,bs1.chose_sum,
    bs1.augend,bs1.addend,bs1.singleton,
    bs2.chose_sum as chose_sum_m1,
    ((bs2.augend+bs2.addend)>bs2.singleton) as sum_correct_m1,
    bs2.chose_sum==((bs2.augend+bs2.addend)>bs2.singleton) as chose_correct_m1
    FROM behavioralstudy as bs1
    LEFT JOIN behavioralstudy as bs2
    ON (bs1.session = bs2.session) AND (bs1.animal = bs2.animal) AND (bs1.trial = (bs2.trial+1))
    AND (bs1.experiment = bs2.experiment)
    WHERE bs1.animal='Ruffio' AND bs1.experiment='FlatLO' AND (chose_sum_m1 IS NOT NULL)
    ORDER BY bs1.animal,bs1.session
'''

data = Helper.getData(cur,query)

#recode binary variables as 1/-1 instead of 0/1
data['chose_sum_m1'].loc[data['chose_sum_m1']==0] = -1
data['sum_correct_m1'].loc[data['sum_correct_m1']==0] = -1
model = 'chose_sum ~ aug_num_green + aug_num_quad + add_num_green + sing_num_green + chose_sum_m1*sum_correct_m1'
lr_be = Analyzer.logistic_backwardselimination_sessionwise(df=data,model=model,
                groupby=['animal','session'],groupby_thresh=.05,pthresh=.05)


#%%
#quad learning (uni/quad ratio) as a function of session
query = '''
        SELECT animal,session,chose_sum,
        aug_num_green,add_num_green,sing_num_green,
        aug_num_quad,add_num_quad,sing_num_quad
        FROM behavioralstudy
        WHERE experiment='QuadDots'
        ORDER BY animal,session
'''

data = Helper.getData(cur,query) 


model = 'chose_sum ~ aug_num_green + add_num_green + sing_num_green + aug_num_quad + add_num_quad + sing_num_quad'
lr = Analyzer.logistic_regression(df=data,model=model,groupby=['animal','session'])

lr['aug_ratio'] = lr['b_aug_num_quad']/lr['b_aug_num_green']
lr['add_ratio'] = lr['b_add_num_quad']/lr['b_add_num_green']
lr['sing_ratio'] = lr['b_sing_num_quad']/lr['b_sing_num_green']

h,ax = plt.subplots(1,2,figsize=[2*6,4])
animal = 'Xavier'
Plotter.lineplot(ax[0],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['aug_ratio'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls='solid',xlabel='Session',ylabel='quad-dots coef / uni-dots coef',
             xticks=[],yticks=[],color=[0,1,0],title=[],label='aug-ratio')
Plotter.lineplot(ax[0],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['add_ratio'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls='solid',xlabel='Session',ylabel='quad-dots coef / uni-dots coef',
             xticks=[],yticks=[],color=[1,0,0],title=[],label='aug-ratio')
Plotter.lineplot(ax[0],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['sing_ratio'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls='solid',xlabel='Session',ylabel='quad-dots coef / uni-dots coef',
             xticks=[],yticks=[],color=[0,0,1],title=[],label='sing-ratio')
ax[0].legend(loc='lower right',fontsize='small',scatterpoints=1,frameon=False)

animal = 'Ruffio'
Plotter.lineplot(ax[1],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['aug_ratio'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls='solid',xlabel='Session',ylabel='quad-dots coef / uni-dots coef',
             xticks=[],yticks=[],color=[0,1,0],title=[],label='aug-ratio')
Plotter.lineplot(ax[1],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['add_ratio'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls='solid',xlabel='Session',ylabel='quad-dots coef / uni-dots coef',
             xticks=[],yticks=[],color=[1,0,0],title=[],label='aug-ratio')
Plotter.lineplot(ax[1],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['sing_ratio'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls='solid',xlabel='Session',ylabel='quad-dots coef / uni-dots coef',
             xticks=[],yticks=[],color=[0,0,1],title=[],label='sing-ratio')

#Plot raw coefficients
h,ax = plt.subplots(1,2,figsize=[6*2,4])
animal = 'Xavier'
Plotter.lineplot(ax[0],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['b_aug_num_green'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls=':',xlabel='Session',ylabel='reg. coef.',
             xticks=[],yticks=[],color=[0,1,0],title=animal,label='augend',identity='zero')
Plotter.lineplot(ax[0],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['b_add_num_green'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls=':',xlabel='Session',ylabel=[],
             xticks=[],yticks=[],color=[1,0,0],title=[],label='addend')
Plotter.lineplot(ax[0],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['b_sing_num_green'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls=':',xlabel='Session',ylabel=[],
             xticks=[],yticks=[],color=[0,0,1],title=[],label='singleton')
ax[0].legend(loc='lower right',fontsize='small',scatterpoints=1,frameon=False)
Plotter.lineplot(ax[0],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['b_aug_num_quad'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls='solid',xlabel='Session',ylabel=[],
             xticks=[],yticks=[],color=[0,1,0],title=[],label=[])
Plotter.lineplot(ax[0],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['b_add_num_quad'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls='solid',xlabel='Session',ylabel=[],
             xticks=[],yticks=[],color=[1,0,0],title=[],label=[])
Plotter.lineplot(ax[0],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['b_sing_num_quad'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls='solid',xlabel='Session',ylabel=[],
             xticks=[],yticks=[],color=[0,0,1],title=[],label=[])

animal = 'Ruffio'
Plotter.lineplot(ax[1],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['b_aug_num_green'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls=':',xlabel='Session',ylabel='reg. coef.',
             xticks=[],yticks=[],color=[0,1,0],title=animal,label='augend',identity='zero')
Plotter.lineplot(ax[1],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['b_add_num_green'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls=':',xlabel='Session',ylabel=[],
             xticks=[],yticks=[],color=[1,0,0],title=[],label='addend')
Plotter.lineplot(ax[1],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['b_sing_num_green'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls=':',xlabel='Session',ylabel=[],
             xticks=[],yticks=[],color=[0,0,1],title=[],label='singleton')
Plotter.lineplot(ax[1],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['b_aug_num_quad'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls='solid',xlabel='Session',ylabel=[],
             xticks=[],yticks=[],color=[0,1,0],title=[],label=[])
Plotter.lineplot(ax[1],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['b_add_num_quad'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls='solid',xlabel='Session',ylabel=[],
             xticks=[],yticks=[],color=[1,0,0],title=[],label=[])
Plotter.lineplot(ax[1],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['b_sing_num_quad'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls='solid',xlabel='Session',ylabel=[],
             xticks=[],yticks=[],color=[0,0,1],title=[],label=[])


#%%
#Subtraction learning as a function of session
query = '''
        SELECT animal,session,chose_sum,
        augend,-addend as addend,singleton
        FROM behavioralstudy
        WHERE experiment='Subtraction'
        ORDER BY animal,session
'''

data = Helper.getData(cur,query) 


#recode binary variables as 1/-1 instead of 0/1
model = 'chose_sum ~ augend + addend + singleton'
lr = Analyzer.logistic_regression(df=data,model=model,groupby=['animal','session'])

#Plot raw coefficients
h,ax = plt.subplots(1,2,figsize=[6*2,4])
animal = 'Xavier'
Plotter.lineplot(ax[0],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['b_augend'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls='-',xlabel='Session',ylabel='reg. coef.',
             xticks=[],yticks=[],color=[0,1,0],title=animal,label='minuend',identity='zero')
Plotter.lineplot(ax[0],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['b_addend'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls='-',xlabel='Session',ylabel=[],
             xticks=[],yticks=[],color=[1,0,0],title=[],label='subtrahend')
Plotter.lineplot(ax[0],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['b_singleton'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls='-',xlabel='Session',ylabel=[],
             xticks=[],yticks=[],color=[0,0,1],title=[],label='singleton')
ax[0].legend(loc='lower right',fontsize='small',scatterpoints=1,frameon=False)

animal = 'Ruffio'
Plotter.lineplot(ax[1],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['b_augend'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls='-',xlabel='Session',ylabel='reg. coef.',
             xticks=[],yticks=[],color=[0,1,0],title=animal,label='minuend',identity='zero')
Plotter.lineplot(ax[1],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['b_addend'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls='-',xlabel='Session',ylabel=[],
             xticks=[],yticks=[],color=[1,0,0],title=[],label='subtrahend')
Plotter.lineplot(ax[1],xdata=lr['session'].loc[lr['animal']==animal],
                 ydata=lr['b_singleton'].loc[lr['animal']==animal],
                sem=None,xlim=[],ylim=[],ls='-',xlabel='Session',ylabel=[],
             xticks=[],yticks=[],color=[0,0,1],title=[],label='singleton')
ax[0].legend(loc='lower right',fontsize='small',scatterpoints=1,frameon=False)



query = '''
    SELECT animal,experiment,MAX(session)
    FROM behavioralstudy
    GROUP BY animal,experiment
    ORDER BY animal,experiment
'''

data = Helper.getData(cur,query)


#%%

flatlo = Helper.getFlatLOTrialset()
query = '''
        SELECT augend,addend,singleton,chose_sum,augend+addend-singleton as diff,
        animal,session,trial,(augend+addend-singleton)>0 as sum_correct,
        1.0*(augend+addend)/singleton as ratio,
        aug_num_green,add_num_green,sing_num_green,
        aug_num_quad,add_num_quad,sing_num_quad
        FROM behavioralstudy
        WHERE experiment='FlatLO' AND animal='Xavier'
        ORDER BY animal,session
'''

data = Helper.getData(cur,query) 

#setup panelplots to make aug/add grid
data['addaug'] = data['addend']*10 + data['augend']
data['const'] = data['augend']-data['augend']

model = 'chose_sum ~ aug_num_green * add_num_green * sing_num_green * aug_num_quad '
lr_be = Analyzer.logistic_backwardselimination_sessionwise(df=data,model=model,
                groupby=['animal','session'],groupby_thresh=.05,pthresh=.05)
data['pred'] = lr_be['final_modelout'].fittedvalues[0]



#%%
#Try PCA & cluster on choices in flat log odds trialset
query = '''
        SELECT animal,session,trial,augend,addend,singleton,chose_sum,aug_num_green,aug_num_quad,
        chose_sum=chose_right as sum_right
        FROM behavioralstudy
        WHERE experiment = 'FlatLO' and animal='Xavier'
        ORDER BY animal,session
'''
data = Helper.getData(cur,query)

#the number of unique conditions in the flat log odds experiment
#this corresponds to unique combinations of aug_num_green, aug_num_quad, addend, singleton, and sum_right
n_unique_trials = 212 
animal = 'Xavier'
usess = np.unique(data['session'].loc[data['animal']==animal])
nsess = len(usess)
chose_sum_vec = []
trial_types = []
for i in range(0,nsess):
    t = data.loc[(data['animal']==animal) & (data['session']==usess[i])]
    ntrial = t.shape[0]
    nfullblocks = np.floor(ntrial/n_unique_trials).astype(int)
    
    for j in range(0,nfullblocks):
        trials = t[['aug_num_green','aug_num_quad','addend','singleton','sum_right','chose_sum']].iloc[(j*n_unique_trials):((j+1)*n_unique_trials)].sort(['aug_num_green','aug_num_quad','addend','singleton','sum_right'])
        
        if(len(trials)==len(trials.drop_duplicates())):
            chose_sum_vec.append(trials['chose_sum'])
            trial_types.append(trials)
        else:
            print('sess:'+str(i)+', block:'+str(j))

chose_sum = np.array(chose_sum_vec)#obs,features
pca = PCA(n_components=3)
pca.fit(chose_sum)
exp_var = pca.explained_variance_ratio_
transformed_data = pca.transform(chose_sum)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(transformed_data[:,0],transformed_data[:,1],transformed_data[:,2])

#%%

query = '''
        SELECT animal,experiment,augend,addend,singleton,AVG(chose_sum==((augend+addend)>singleton)) as pcor
        FROM behavioralstudy
        WHERE experiment in ('Addition','Subtraction')
        GROUP BY animal,experiment,augend,addend,singleton
        ORDER BY animal,session
'''
data = Helper.getData(cur,query)

dm_var = lambda w,data: np.sqrt((w[0]**2)*data['augend']**2 + (w[1]**2)*data['addend']**2 + (w[2]**2)*data['singleton']**2)
data['dm_var'] = dm_var([1,1,1],data)#np.power(dm_var([1,1,1],data),2)
data['ratio'] = np.min([data['augend']+data['addend'],data['singleton']],axis=0)/np.max([data['augend']+data['addend'],data['singleton']],axis=0)
data['diff'] = data['augend']+data['addend']-data['singleton']

#
data.groupby(['dm_var','diff'])['chose_sum'].mean()


height_per_panel = 6
width_per_panel = 6
h,ax = plt.subplots(1,2,figsize=[2*width_per_panel,1*height_per_panel])
Plotter.scatter(ax[0],xdata = data['diff'].loc[data['experiment']=='Addition'],
                ydata = data['dm_var'].loc[data['experiment']=='Addition'],
                xlabel='Sum-Singleton',ylabel='sqrt(var)')
Plotter.scatter(ax[1],xdata = data['diff'].loc[data['experiment']=='Subtraction'],
                ydata = data['dm_var'].loc[data['experiment']=='Subtraction'],
                xlabel='Sum-Singleton',ylabel='sqrt(var)')

scipy.stats.spearmanr(data['ratio'].loc[data['experiment']=='Addition'],
                      data['dm_var'].loc[data['experiment']=='Addition'])
scipy.stats.spearmanr(data['ratio'].loc[data['experiment']=='Subtraction'],
                      data['dm_var'].loc[data['experiment']=='Subtraction'])

[scipy.stats.pearsonr(data['dm_var'].loc[(data['experiment']==x) & (data['animal']==y)],data['pcor'].loc[(data['experiment']==x) & (data['animal']==y)])
     for x,y in zip(['Addition','Subtraction','Addition','Subtraction'],['Xavier','Xavier','Ruffio','Ruffio'])]

[(x,y) for x,y in zip(['Addition','Subtraction','Addition','Subtraction'],['Xavier','Xavier','Ruffio','Ruffio'])]


height_per_panel = 6
width_per_panel = 6
h,ax = plt.subplots(2,2,figsize=[2*width_per_panel,2*height_per_panel])
Plotter.scatter(ax[0,0],xdata=data['dm_var'].loc[(data['experiment']=='Addition') & (data['animal']=='Xavier')],
                ydata=data['pcor'].loc[(data['experiment']=='Addition') & (data['animal']=='Xavier')],xlabel='Internal variance',ylabel='accuracy',
                title='Monkey X, Addition',ylim=[0,1],xlim=[],xticks=[],yticks=[0,0.25,0.5,0.75,1],corrline='on')
Plotter.scatter(ax[0,1],xdata=data['dm_var'].loc[(data['experiment']=='Subtraction') & (data['animal']=='Xavier')],
                ydata=data['pcor'].loc[(data['experiment']=='Subtraction') & (data['animal']=='Xavier')],xlabel='Internal variance',ylabel='accuracy',
                title='Monkey X, Subtraction',ylim=[0,1],xlim=[],xticks=[],yticks=[0,0.25,0.5,0.75,1],corrline='on')
Plotter.scatter(ax[1,0],xdata=data['dm_var'].loc[(data['experiment']=='Addition') & (data['animal']=='Ruffio')],
                ydata=data['pcor'].loc[(data['experiment']=='Addition') & (data['animal']=='Ruffio')],xlabel='Internal variance',ylabel='accuracy',
                title='Monkey R, Addition',ylim=[0,1],xlim=[],xticks=[],yticks=[0,0.25,0.5,0.75,1],corrline='on')
Plotter.scatter(ax[1,1],xdata=data['dm_var'].loc[(data['experiment']=='Subtraction') & (data['animal']=='Ruffio')],
                ydata=data['pcor'].loc[(data['experiment']=='Subtraction') & (data['animal']=='Ruffio')],xlabel='Internal variance',ylabel='accuracy',
                title='Monkey R, Subtraction',ylim=[0,1],xlim=[],xticks=[],yticks=[0,0.25,0.5,0.75,1],corrline='on')


#%%


query = '''
        SELECT animal,experiment,augend,addend,singleton,chose_sum==((augend+addend)>singleton) as cor,
        chose_sum
        FROM behavioralstudy
        WHERE experiment in ('Addition','Subtraction')
        ORDER BY animal,session
'''
data = Helper.getData(cur,query)

dm_var = lambda w,data: np.sqrt((w[0]**2)*data['augend']**2 + (w[1]**2)*data['addend']**2 + (w[2]**2)*data['singleton']**2)
data['dm_var'] = np.power(dm_var([1,1,1],data),2)
data['ratio'] = np.min([data['augend']+data['addend'],data['singleton']],axis=0)/np.max([data['augend']+data['addend'],data['singleton']],axis=0)
data['diff'] = data['augend']+data['addend']-data['singleton']

#For unique combinations of diff and dm_var, look at p(choose_sum)
choice_data = data.groupby(['dm_var','diff','animal','experiment'],as_index=False)['chose_sum'].mean()
choice_data['const'] = 1


height_per_panel = 6
width_per_panel = 6
udiff = np.unique(choice_data['diff'])
h,ax = plt.subplots(2,2,figsize=[2*width_per_panel,2*height_per_panel])
animals = ['Xavier','Ruffio']
experiments = ['Addition','Subtraction']
cmap = matplotlib.cm.get_cmap('YlOrRd')
for i in range(0,len(animals)):
    for j in range(0,len(experiments)):
        expcond = (choice_data['animal']==animals[i]) & (choice_data['experiment']==experiments[j])
        uvar = np.unique(choice_data['dm_var'].loc[expcond])
        for d in udiff:
            cond = (choice_data['diff']==d) & expcond
            for v in uvar:
                norm = matplotlib.colors.Normalize(vmin=choice_data['dm_var'].loc[expcond].min(),
                                                   vmax=choice_data['dm_var'].loc[expcond].max())
                color = cmap(norm(v))
                if(len(choice_data['chose_sum'].loc[cond & (choice_data['dm_var']==v)])>0):
                    Plotter.scatter(ax[i,j],xdata=d,ydata=choice_data['chose_sum'].loc[cond & (choice_data['dm_var']==v)],color=color,
                                    title=animals[i]+', '+experiments[j],xlabel='Diff',ylabel='P(choose sum)')

Plotter.panelplots(data=choice_data,plotvar='chose_sum',groupby=['diff','const','dm_var'],axes=None,scattervar=[],xlim=[0,300],ylim=[0,1],xlabel=[],ylabel=[],
               xticks=None,horiz=None,maxcol=4,legend='on')


#%%
#Plot p(choose sum) as a function of internal variance
query = '''
        SELECT animal,experiment,augend,addend,singleton,chose_sum==((augend+addend)>singleton) as cor,
        chose_sum
        FROM behavioralstudy
        WHERE experiment in ('Addition','Subtraction')
        ORDER BY animal,session
'''
data = Helper.getData(cur,query)

dm_var = lambda w,data: np.sqrt((w[0]**2)*data['augend']**2 + (w[1]**2)*data['addend']**2 + (w[2]**2)*data['singleton']**2)
data['dm_var'] = np.power(dm_var([1,1,1],data),2)
data['ratio'] = np.min([data['augend']+data['addend'],data['singleton']],axis=0)/np.max([data['augend']+data['addend'],data['singleton']],axis=0)
data['diff'] = data['augend']+data['addend']-data['singleton']

Plotter.panelplots(data=choice_data.loc[choice_data['experiment']=='Subtraction'],plotvar='chose_sum',groupby=['diff','animal','dm_var'],
                   axes=None,scattervar=[],xlim=[0,300],ylim=[0,1],xlabel='Int.Var.',ylabel='p(choose sum)',
               xticks=None,horiz=None,maxcol=4,legend='on')
Plotter.panelplots(data=choice_data.loc[choice_data['experiment']=='Addition'],plotvar='chose_sum',groupby=['diff','animal','dm_var'],
                   axes=None,scattervar=[],xlim=[0,300],ylim=[0,1],xlabel='Int.Var.',ylabel='p(choose sum)',
               xticks=None,horiz=None,maxcol=4,legend='on')


#%%
#Plot performance in all trial types for addition & subtraction experiments

query = '''
        SELECT animal,experiment,augend,addend,singleton,chose_sum==((augend+addend)>singleton) as cor,
        chose_sum,augend+addend-singleton as diff
        FROM behavioralstudy
        WHERE experiment in ('Addition','Subtraction')
        ORDER BY animal,session
'''
data = Helper.getData(cur,query)

groupby = ['animal','experiment']
lr_model = 'chose_sum ~ augend + addend + singleton + np.power(augend,2) + np.power(addend,2) + np.power(singleton,2)'
lr_out = Analyzer.logistic_regression(data,lr_model,groupby=groupby)

with PdfPages('D:\\Bart\\Dropbox\\AddSubtract_Study.pdf') as pdf:

    cond = (data['experiment']=='Subtraction') & (data['animal']=='Xavier')
    Plotter.panelplots(data=data.loc[cond],plotvar='chose_sum',groupby=['augend','addend','diff'],
                       axes=None,scattervar=[],xlim=[-8,8],ylim=[0,1],xlabel='Difference - Singleton',ylabel='p(choose sum)',
                   xticks=[],horiz=None,title='Subtraction, Xavier',maxcol=3,legend='on')
    pdf.savefig()

    cond = (data['experiment']=='Subtraction') & (data['animal']=='Ruffio')
    Plotter.panelplots(data=data.loc[cond],plotvar='chose_sum',groupby=['augend','addend','diff'],
                       axes=None,scattervar=[],xlim=[-8,8],ylim=[0,1],xlabel='Difference - Singleton',ylabel='p(choose sum)',
                   xticks=[],horiz=None,title='Subtraction, Ruffio',maxcol=3,legend='on')
    pdf.savefig()
    
    cond = (data['experiment']=='Addition') & (data['animal']=='Xavier')
    Plotter.panelplots(data=data.loc[cond],plotvar='chose_sum',groupby=['augend','addend','diff'],
                       axes=None,scattervar=[],xlim=[-8,8],ylim=[0,1],xlabel='Sum - Singleton',ylabel='p(choose sum)',
                   xticks=[],horiz=None,title='Addition, Xavier',maxcol=3,legend='on')
    pdf.savefig()
        
    cond = (data['experiment']=='Addition') & (data['animal']=='Ruffio')
    Plotter.panelplots(data=data.loc[cond],plotvar='chose_sum',groupby=['augend','addend','diff'],
                       axes=None,scattervar=[],xlim=[-8,8],ylim=[0,1],xlabel='Sum - Singleton',ylabel='p(choose sum)',
                   xticks=[],horiz=None,title='Addition, Ruffio',maxcol=3,legend='on')
    pdf.savefig()

#%%
#Compute subjective value for each pair of augend & addend.
query = '''
        SELECT animal,experiment,augend,addend,singleton,chose_sum==((augend+addend)>singleton) as cor,
        chose_sum,augend+addend-singleton as diff
        FROM behavioralstudy
        WHERE experiment in ('Addition','Subtraction','FlatLO')
        ORDER BY animal,session
'''
data = Helper.getData(cur,query)

def SVPlot(experiment,animal):
    datain = data.loc[(data['experiment']==experiment) & (data['animal']==animal)]
    #Do a logistic regression on p(choose_sum) vs. singleton for each augend/addend pair
    model = 'chose_sum ~ singleton'
    aug_add_pairs = datain[['augend','addend']].drop_duplicates().sort_values(by=['augend','addend']).get_values()
    lr_out = pd.concat([Analyzer.logistic_regression(datain.loc[(datain['augend']==x[0])&(datain['addend']==x[1])],
              model) for x in aug_add_pairs])
    
    #Solve for the point of subjective equality to get each aug/add's PSE. 
    out = pd.DataFrame(aug_add_pairs,columns=['augend','addend'])
    out['PSE'] = (-lr_out['b_Intercept']/lr_out['b_singleton']).get_values()
    out['sum_bias'] = out['PSE'] - (out['augend']+out['addend'])
    
    augs = out['augend'].drop_duplicates().get_values()
    naug = augs.shape[0]
    adds = out['addend'].drop_duplicates().get_values()
    nadd = adds.shape[0]
    
    #Plot it all
    height_per_panel = 6
    width_per_panel = 6
    h,ax = plt.subplots(1,2,figsize=[2*width_per_panel,height_per_panel])
    cmap = [matplotlib.cm.get_cmap('rainbow')(x/naug) for x in range(0,naug)]
    for i in range(0,naug):
        cond = (out['augend']==augs[i]) & ((out['augend']+out['addend']) != 10)
        Plotter.lineplot(ax[0],xdata=out['addend'].loc[cond],
             ydata=out['sum_bias'].loc[cond],sem=None,color=cmap[i],
             label='Aug='+str(augs[i]),xlabel='Addend',ylabel='SV(Sum) - Sum',ls='solid')
    ax[0].legend(fontsize='x-small',loc=0)
    for i in range(0,naug):
        cond = (out['augend']==augs[i]) & ((out['augend']+out['addend']) != 10)
        Plotter.scatter(ax[0],xdata=out['addend'].loc[cond],
                 ydata=out['sum_bias'].loc[cond],color=cmap[i])
    for i in range(0,nadd):
        cond = (out['addend']==adds[i]) & ((out['augend']+out['addend']) != 10)
        Plotter.lineplot(ax[1],xdata=out['augend'].loc[cond],
             ydata=out['sum_bias'].loc[cond],sem=None,color=cmap[i],
             label='Add='+str(adds[i]),xlabel='Augend',ylabel='SV(Sum) - Sum',ls='solid')
    ax[1].legend(fontsize='x-small',loc=0)
    for i in range(0,nadd):
        cond = (out['addend']==adds[i]) & ((out['augend']+out['addend']) != 10)
        Plotter.scatter(ax[1],xdata=out['augend'].loc[cond],
                 ydata=out['sum_bias'].loc[cond],color=cmap[i])
    plt.suptitle(experiment+', '+animal)
    
    
#Filter to a particular experiment
with PdfPages('D:\\Bart\\Dropbox\\SubjectiveValue.pdf') as pdf:
    experiment = 'Addition'
    animal = 'Ruffio'
    SVPlot(experiment,animal)
    pdf.savefig()
    
    experiment = 'Addition'
    animal = 'Xavier'
    SVPlot(experiment,animal)
    pdf.savefig()
    
    experiment = 'Subtraction'
    animal = 'Ruffio'
    SVPlot(experiment,animal)
    pdf.savefig()
    
    experiment = 'Subtraction'
    animal = 'Xavier'
    SVPlot(experiment,animal)
    pdf.savefig()

#%%
#Look at the trials that the animal tends to get the most wrong in the addition and subtraction expts.
query = '''
        SELECT animal,experiment,augend,addend,singleton,
        AVG(chose_sum) as cs,
        AVG((((augend+addend)>singleton)+0) - (chose_sum+0)) as ccs,
        AVG(chose_sum==((augend+addend)>singleton)) as cor
        FROM behavioralstudy
        WHERE experiment in ('Addition','Subtraction')
        GROUP BY animal,experiment,augend,addend,singleton
        ORDER BY animal,experiment,ccs
'''
data = Helper.getData(cur,query)

filtercond = abs(data['ccs'])>.3

#Plot it
plot_data = data.loc[(data['animal']=='Xavier') & (data['experiment']=='Addition') & filtercond]
                     
fig,ax = plt.subplots(1,1,figsize=[1*15,1*10])
#plot data
plt.plot(range(0,len(plot_data['ccs'])),
         plot_data['ccs'],'-b',label='Sum Correct - P(chose sum)')
#Plot .5 line
plt.plot(range(0,len(plot_data['ccs'])),[0 for i in range(0,plot_data.shape[0])],'--k')
ax.set_xticks(range(0,len(plot_data['ccs'])))
ax.set_xticklabels(['('+str(ag)+','+str(ad)+
','+str(si)+')' for ag,ad,si in zip(plot_data['augend'],plot_data['addend'],plot_data['singleton'])],rotation=90)
ax.set_ylabel('Sum Correct - P(chose sum)')
ax.set_xlabel('AAS')
ax.set_title('Singleton Bias on particular trials (Xavier)')
#ax.legend(loc=9)
#ax.set_xlim([0,33])

#%%
#Look at the "tie effect" - performance as a function of the difference between augend and addend.
#Tie effect predicts better performance when aug and add are similar.
query = '''
        SELECT animal,experiment,session,augend,addend,singleton,chose_sum,
        chose_sum==((augend+addend)>singleton) as cor,
        augend+addend-singleton as diff,augend-addend as aadiff,augend+addend as sum,
        (augend+addend)>singleton as sum_correct
        FROM behavioralstudy
        WHERE experiment in ('Addition','Subtraction')
        ORDER BY animal,session
'''
data = Helper.getData(cur,query)

full = 'chose_sum ~ augend+addend+singleton'
animal = 'Xavier'
experiment = 'Addition'

#Compute regression model
out = Analyzer.logistic_regression(model=full,df=data,groupby=['animal','experiment'])
logistic = lambda x:1./(1.+np.exp(-x));

cond = data['experiment']=='Addition'
Plotter.panelplots(data=data.loc[cond],plotvar='cor',groupby=['animal','experiment','aadiff'],
                        axes=None,scattervar=[],xlim=[-8,8],ylim=[0,1],
                        xlabel='Augend-Addend',ylabel='p(correct)',horiz=.5,
                        maxcol=4,legend='on',cmap=plt.get_cmap('jet'))


#Plot Actual data
with PdfPages('D:\\Bart\\Dropbox\\TieEffect.pdf') as pdf:
    cond = (data['experiment']=='Addition') & (data['animal']=='Ruffio')
    Plotter.panelplots(data=data.loc[cond],plotvar='cor',groupby=['singleton','sum','aadiff'],
                        axes=None,scattervar=[],xlim=[-8,8],ylim=[0,1],
                        xlabel='Augend-Addend',ylabel='p(correct)',horiz=.5,
                        maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Ruffio, Addition')
    pdf.savefig()
    
    cond = (data['experiment']=='Addition') & (data['animal']=='Ruffio')
    Plotter.panelplots(data=data.loc[cond],plotvar='chose_sum',groupby=['singleton','sum','aadiff'],
                        axes=None,scattervar=[],xlim=[-8,8],ylim=[0,1],
                        xlabel='Augend-Addend',ylabel='p(chose_sum)',horiz=.5,
                        maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Ruffio, Addition')
    pdf.savefig()
    
    cond = (data['experiment']=='Addition') & (data['animal']=='Xavier')
    Plotter.panelplots(data=data.loc[cond],plotvar='cor',groupby=['singleton','sum','aadiff'],
                        axes=None,scattervar=[],xlim=[-8,8],ylim=[0,1],
                        xlabel='Augend-Addend',ylabel='p(correct)',horiz=.5,
                        maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Xavier, Addition')
    pdf.savefig()
    
    cond = (data['experiment']=='Addition') & (data['animal']=='Xavier')
    Plotter.panelplots(data=data.loc[cond],plotvar='chose_sum',groupby=['singleton','sum','aadiff'],
                        axes=None,scattervar=[],xlim=[-8,8],ylim=[0,1],
                        xlabel='Augend-Addend',ylabel='p(chose_sum)',horiz=.5,
                        maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Xavier, Addition')
    pdf.savefig()

#Make predictions, plot against data.
with PdfPages('D:\\Bart\\Dropbox\\TieEffect_predictions.pdf') as pdf:
    animal = 'Ruffio'
    experiment = 'Addition'
    cond_out = (out['animal']==animal) & (out['experiment']==experiment)
    cond_data = (data['animal']==animal) & (data['experiment']==experiment)
    thisdata = data.loc[cond_data]
    X = np.concatenate((np.ones((sum(cond_data),1)),
        thisdata[['augend','addend','singleton']].values),axis=1)
    thisdata['predictions'] = logistic(np.matmul(X,out.loc[cond_out]['params'][0].values))
    thisdata['predicted_cor'] = thisdata['sum_correct']*thisdata['predictions'] + (1-thisdata['sum_correct'])*(1-thisdata['predictions'])
    Plotter.panelplots(data=thisdata,plotvar='predicted_cor',groupby=['singleton','sum','aadiff'],
                            axes=None,scattervar='cor',xlim=[-8,8],ylim=[0,1],
                            xlabel='Augend-Addend',ylabel='p(correct)',horiz=.5,
                            maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Ruffio, Addition')
    pdf.savefig()
    
    animal = 'Xavier'
    experiment = 'Addition'
    cond_out = (out['animal']==animal) & (out['experiment']==experiment)
    cond_data = (data['animal']==animal) & (data['experiment']==experiment)
    thisdata = data.loc[cond_data]
    X = np.concatenate((np.ones((sum(cond_data),1)),
        thisdata[['augend','addend','singleton']].values),axis=1)
    thisdata['predictions'] = logistic(np.matmul(X,out.loc[cond_out]['params'][2].values))
    thisdata['predicted_cor'] = thisdata['sum_correct']*thisdata['predictions'] + (1-thisdata['sum_correct'])*(1-thisdata['predictions'])
    Plotter.panelplots(data=thisdata,plotvar='predicted_cor',groupby=['singleton','sum','aadiff'],
                            axes=None,scattervar='cor',xlim=[-8,8],ylim=[0,1],
                            xlabel='Augend-Addend',ylabel='p(correct)',horiz=.5,
                            maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Xavier, Addition')
    pdf.savefig()

#%%
#Look at the trials that the arithmetic model fails to predict.
query = '''
        SELECT animal,experiment,session,augend,addend,singleton,chose_sum,
        chose_sum==((augend+addend)>singleton) as cor,
        augend+addend-singleton as diff,augend-addend as aadiff,augend+addend as sum,
        (augend+addend)>singleton as sum_correct
        FROM behavioralstudy
        WHERE experiment in ('Addition','Subtraction')
        ORDER BY animal,session
'''
data = Helper.getData(cur,query)

full = 'chose_sum ~ augend+addend+singleton'
animal = 'Xavier'
experiment = 'Addition'

#Compute regression model
out = Analyzer.logistic_regression(model=full,df=data,groupby=['animal','experiment'])
logistic = lambda x:1./(1.+np.exp(-x));


#get predictions of regression model
animal = 'Ruffio'
experiment = 'Addition'
cond_out = (out['animal']==animal) & (out['experiment']==experiment)
cond_data = (data['animal']==animal) & (data['experiment']==experiment)
thisdata = data.loc[cond_data]
X = np.concatenate((np.ones((sum(cond_data),1)),
    thisdata[['augend','addend','singleton']].values),axis=1)
thisdata['predictions'] = logistic(np.matmul(X,out.loc[cond_out]['params'][0].values))
gbdata = thisdata.groupby(['augend','addend','singleton'],as_index=False)[['predictions','chose_sum']].mean()

#Compare predictions to actual data
gbdata['delta'] = gbdata['predictions'] - gbdata['chose_sum'];
gbdata.sort_values(by='delta',inplace=True)

#Filter gbdata to include just the most extreme offenders
filtercond = abs(gbdata['delta'])>.15

plot_data = gbdata.loc[filtercond]
#
fig,ax = plt.subplots(1,1,figsize=[1*15,1*10])
#plot data
plt.plot(range(0,len(plot_data['delta'])),
         plot_data['delta'],'-b',label='Sum Correct - P(chose sum)')
#Plot .5 line
plt.plot(range(0,len(plot_data['delta'])),[0 for i in range(0,plot_data.shape[0])],'--k')
ax.set_xticks(range(0,len(plot_data['delta'])))
ax.set_xticklabels(['('+str(ag)+','+str(ad)+
','+str(si)+')' for ag,ad,si in zip(plot_data['augend'],plot_data['addend'],plot_data['singleton'])],rotation=90)
ax.set_ylabel('Predicted P(CS) - Actual P(CS)')
ax.set_xlabel('AAS')
ax.set_title('Predictions of linear model vs. actual data (Ruffio)')
#ax.legend(loc=9)


#%%
#Look at p(correct) and average juice/trial over the course of subtraction experiment.
query = '''
        SELECT animal,session,experiment,AVG(chose_sum==((augend+addend)>singleton)) as pcor,
        AVG((chose_sum+0)*(augend+addend) + (1-(chose_sum+0))*singleton) as mrew
        FROM behavioralstudy
        WHERE experiment in ('Subtraction')
        GROUP BY animal,session
        ORDER BY animal,session
'''
data = Helper.getData(cur,query)

cond = data['experiment']=='Subtraction'
Plotter.panelplots(data=data.loc[cond],plotvar='mrew',groupby=['animal','experiment','session'],
                        axes=None,scattervar=[],
                        xlabel='Augend-Addend',ylabel='p(correct)',horiz=.5,
                        maxcol=4,legend='on',cmap=plt.get_cmap('jet'))

#%%
#plot deviation from best linear model for each aas
query = '''
        SELECT animal,experiment,session,augend,addend,singleton,chose_sum,
        chose_sum==((augend+addend)>singleton) as cor,
        augend+addend-singleton as diff,augend-addend as aadiff,augend+addend as sum,
        (augend+addend)>singleton as sum_correct
        FROM behavioralstudy
        WHERE experiment in ('Addition','Subtraction')
        ORDER BY animal,session
'''
data = Helper.getData(cur,query)

def delta_linear(animal,experiment):
#    animal = 'Ruffio'
#    experiment = 'Addition'
    cond = (data['animal']==animal) & (data['experiment']==experiment)
    thisdata = data.loc[cond]
    logistic = lambda x:1./(1.+np.exp(-x))
    
    #Compute regression model
    model = 'chose_sum ~ augend+addend+singleton'
    out = Analyzer.logistic_regression(model=model,df=thisdata,groupby=['animal','experiment'])
    X = np.concatenate((np.ones((sum(cond),1)),
            thisdata[['augend','addend','singleton']].values),axis=1)
    thisdata['predictions'] = logistic(np.matmul(X,out['params'][0].values))
    thisdata['delta_linear'] = thisdata['chose_sum']-thisdata['predictions']
    return thisdata

    
with PdfPages('D:\\Bart\\Dropbox\\Linear_mispredictions.pdf') as pdf:
    thisdata = delta_linear('Xavier','Addition')
    Plotter.panelplots(data=thisdata,plotvar='delta_linear',groupby=['augend','addend','diff'],
                            axes=None,scattervar=[],xlim=[-8,8],ylim=[-.5,.5],
                            xlabel='Sum-Sing',ylabel='Actual - Predicted P(CS)',horiz=0,
                            maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Xavier, Addition')
    pdf.savefig()
    
    thisdata = delta_linear('Xavier','Addition')
    Plotter.panelplots(data=thisdata,plotvar='delta_linear',groupby=['addend','augend','diff'],
                            axes=None,scattervar=[],xlim=[-8,8],ylim=[-.5,.5],
                            xlabel='Sum-Sing',ylabel='Actual - Predicted P(CS)',horiz=0,
                            maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Xavier, Addition')
    pdf.savefig()
    
    thisdata = delta_linear('Ruffio','Addition')
    Plotter.panelplots(data=thisdata,plotvar='delta_linear',groupby=['augend','addend','diff'],
                            axes=None,scattervar=[],xlim=[-8,8],ylim=[-.5,.5],
                            xlabel='Sum-Sing',ylabel='Actual - Predicted P(CS)',horiz=0,
                            maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Ruffio, Addition')
    pdf.savefig()
    
    thisdata = delta_linear('Ruffio','Addition')
    Plotter.panelplots(data=thisdata,plotvar='delta_linear',groupby=['addend','augend','diff'],
                            axes=None,scattervar=[],xlim=[-8,8],ylim=[-.5,.5],
                            xlabel='Sum-Sing',ylabel='Actual - Predicted P(CS)',horiz=0,
                            maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Ruffio, Addition')
    pdf.savefig()

    
#Plot wrt/ singleton
with PdfPages('D:\\Bart\\Dropbox\\Linear_mispredictions_aas.pdf') as pdf:
    thisdata = delta_linear('Xavier','Addition')
    Plotter.panelplots(data=thisdata,plotvar='delta_linear',groupby=['augend','addend','singleton'],
                            axes=None,scattervar=[],xlim=[2,10],ylim=[-.5,.5],
                            xlabel='Sum-Sing',ylabel='Actual - Predicted P(CS)',horiz=0,
                            maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Xavier, Addition')
    pdf.savefig()
    
    thisdata = delta_linear('Xavier','Addition')
    Plotter.panelplots(data=thisdata,plotvar='delta_linear',groupby=['addend','augend','singleton'],
                            axes=None,scattervar=[],xlim=[2,10],ylim=[-.5,.5],
                            xlabel='Sum-Sing',ylabel='Actual - Predicted P(CS)',horiz=0,
                            maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Xavier, Addition')
    pdf.savefig()
    
    thisdata = delta_linear('Ruffio','Addition')
    Plotter.panelplots(data=thisdata,plotvar='delta_linear',groupby=['augend','addend','singleton'],
                            axes=None,scattervar=[],xlim=[2,10],ylim=[-.5,.5],
                            xlabel='Sum-Sing',ylabel='Actual - Predicted P(CS)',horiz=0,
                            maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Ruffio, Addition')
    pdf.savefig()
    
    thisdata = delta_linear('Ruffio','Addition')
    Plotter.panelplots(data=thisdata,plotvar='delta_linear',groupby=['addend','augend','singleton'],
                            axes=None,scattervar=[],xlim=[2,10],ylim=[-.5,.5],
                            xlabel='Sum-Sing',ylabel='Actual - Predicted P(CS)',horiz=0,
                            maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Ruffio, Addition')
    pdf.savefig()
    
    
#%%
#Look at sum bias as a function of augend and addend.
query = '''
        SELECT animal,experiment,session,augend,addend,singleton,chose_sum,
        chose_sum==((augend+addend)>singleton) as cor,
        augend+addend-singleton as diff,augend-addend as aadiff,augend+addend as sum,
        (augend+addend)>singleton as sum_correct
        FROM behavioralstudy
        WHERE experiment in ('Addition')
        ORDER BY animal,session
'''
data = Helper.getData(cur,query)

#Find best fitting linear predictions
model = 'chose_sum ~ augend + addend + singleton'
data['linear_predictions'] = Analyzer.fit_predict_linear(data,model)
#Group the data by augend, addend, and diff.
data_choice = data.groupby(['animal','experiment','augend','addend','diff'],
                       as_index=False)[['sum_correct','chose_sum','linear_predictions']].mean()

data_choice['sum_bias'] = data_choice['chose_sum'] - data_choice['sum_correct']
data_choice['predicted_sum_bias'] = data_choice['linear_predictions'] - data_choice['sum_correct']

#Look at sum bias as a function of augend & difference.
plot_data = data_choice.loc[data_choice['experiment']=='Addition']
Plotter.panelplots(data=plot_data,plotvar='sum_bias',groupby=['animal','augend','diff'],
                            axes=None,scattervar=[],xlim=[-8,8],ylim=[-.6,.6],
                            xlabel='Sum-Sing',ylabel='P(choose sum) - P(sum > sing)',horiz=0,
                            maxcol=4,legend='on',cmap=plt.get_cmap('jet'))

Plotter.panelplots(data=plot_data,plotvar='predicted_sum_bias',groupby=['animal','augend','diff'],
                            axes=None,scattervar=[],xlim=[-8,8],ylim=[-.6,.6],
                            xlabel='Sum-Sing',ylabel='P(choose sum) - P(sum > sing)',horiz=0,
                            maxcol=4,legend='on',cmap=plt.get_cmap('jet'))

#Look at sum bias as a function of augend & addend
plot_data = data_choice.loc[data_choice['experiment']=='Addition']
Plotter.panelplots(data=plot_data,plotvar='sum_bias',groupby=['animal','augend','addend'],
                            axes=None,scattervar=[],xlim=[1,9],ylim=[-.35,.35],
                            xlabel='Addend',ylabel='P(choose sum) - P(sum > sing)',horiz=0,
                            maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
plt.suptitle('Data')

Plotter.panelplots(data=plot_data,plotvar='predicted_sum_bias',groupby=['animal','augend','addend'],
                            axes=None,scattervar=[],xlim=[1,9],ylim=[-.35,.35],
                            xlabel='Addend',ylabel='P(choose sum) - P(sum > sing)',horiz=0,
                            maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
plt.suptitle('Linear Model')


#Look at sum bias as a function of augend & addend
plot_data = data_choice.loc[data_choice['experiment']=='Addition']
Plotter.panelplots(data=plot_data,plotvar='sum_bias',groupby=['animal','addend','augend'],
                            axes=None,scattervar=[],xlim=[1,9],ylim=[-.35,.35],
                            xlabel='Augend',ylabel='P(choose sum) - P(sum > sing)',horiz=0,
                            maxcol=4,legend='on',cmap=plt.get_cmap('jet'))

Plotter.panelplots(data=plot_data,plotvar='predicted_sum_bias',groupby=['animal','addend','augend'],
                            axes=None,scattervar=[],xlim=[1,9],ylim=[-.35,.35],
                            xlabel='Augend',ylabel='P(choose sum) - P(sum > sing)',horiz=0,
                            maxcol=4,legend='on',cmap=plt.get_cmap('jet'))



#%%
#Look at sum bias as a function of augend and addend in all experiments. 
query = '''
        SELECT animal,experiment,session,augend,addend,singleton,chose_sum,
        chose_sum==((augend+addend)>singleton) as cor,
        augend+addend-singleton as diff,augend-addend as aadiff,augend+addend as sum,
        (augend+addend)>singleton as sum_correct
        FROM behavioralstudy
        WHERE experiment in ('Addition','FlatLO','Subtraction')
        ORDER BY animal,session
'''
data = Helper.getData(cur,query)

#Find best fitting linear predictions
model = 'chose_sum ~ augend + addend + singleton'
#Group the data by augend, addend, and diff.
#data_choice = data.groupby(['animal','experiment','augend','addend','diff'],
#                       as_index=False)[['sum_correct','chose_sum','linear_predictions']].mean()
data['sum_bias'] = data['chose_sum'] - data['sum_correct']

with PdfPages('D:\\Bart\\Dropbox\\SumBias.pdf') as pdf:
    #Look at sum bias as a function of augend & addend
    plot_data = data.loc[data['experiment']=='Addition']
    plot_data.ix[plot_data['animal']=='Ruffio','linear_predictions'] = Analyzer.fit_predict_linear(plot_data.loc[plot_data['animal']=='Ruffio'],model)
    plot_data.ix[plot_data['animal']=='Xavier','linear_predictions'] = Analyzer.fit_predict_linear(plot_data.loc[plot_data['animal']=='Xavier'],model)
    plot_data['predicted_sum_bias'] = plot_data['linear_predictions'] - plot_data['sum_correct']
    Plotter.panelplots(data=plot_data,plotvar='sum_bias',groupby=['animal','augend','addend'],
                                axes=None,scattervar=[],xlim=[.5,9.5],ylim=[-.35,.35],
                                xlabel='Addend',ylabel='P(choose sum) - P(sum > sing)',horiz=0,
                                maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Data (Addition)')
    pdf.savefig()
    Plotter.panelplots(data=plot_data,plotvar='predicted_sum_bias',groupby=['animal','augend','addend'],
                                sem=None,axes=None,scattervar=[],xlim=[.5,9.5],ylim=[-.35,.35],
                                xlabel='Addend',ylabel='P(choose sum) - P(sum > sing)',horiz=0,
                                maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Linear Model (Addition)')
    pdf.savefig()
    
    
    #Look at sum bias as a function of augend & addend
    plot_data = data.loc[data['experiment']=='FlatLO']
    plot_data.ix[plot_data['animal']=='Ruffio','linear_predictions'] = Analyzer.fit_predict_linear(plot_data.loc[plot_data['animal']=='Ruffio'],model)
    plot_data.ix[plot_data['animal']=='Xavier','linear_predictions'] = Analyzer.fit_predict_linear(plot_data.loc[plot_data['animal']=='Xavier'],model)
    plot_data['predicted_sum_bias'] = plot_data['linear_predictions'] - plot_data['sum_correct']
    Plotter.panelplots(data=plot_data,plotvar='sum_bias',groupby=['animal','augend','addend'],
                                axes=None,scattervar=[],xlim=[.5,4.5],ylim=[-.35,.35],
                                xlabel='Addend',ylabel='P(choose sum) - P(sum > sing)',horiz=0,
                                maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Data (FlatLO)')
    pdf.savefig()
    Plotter.panelplots(data=plot_data,plotvar='predicted_sum_bias',groupby=['animal','augend','addend'],
                                sem=None,axes=None,scattervar=[],xlim=[.5,4.5],ylim=[-.35,.35],
                                xlabel='Addend',ylabel='P(choose sum) - P(sum > sing)',horiz=0,
                                maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Linear Model (FlatLO)')
    pdf.savefig()
    
    
    #Look at sum bias as a function of augend & addend
    plot_data = data.loc[data['experiment']=='Subtraction']
    plot_data.ix[plot_data['animal']=='Ruffio','linear_predictions'] = Analyzer.fit_predict_linear(plot_data.loc[plot_data['animal']=='Ruffio'],model)
    plot_data.ix[plot_data['animal']=='Xavier','linear_predictions'] = Analyzer.fit_predict_linear(plot_data.loc[plot_data['animal']=='Xavier'],model)
    plot_data['predicted_sum_bias'] = plot_data['linear_predictions'] - plot_data['sum_correct']
    Plotter.panelplots(data=plot_data,plotvar='sum_bias',groupby=['animal','augend','addend'],
                                axes=None,scattervar=[],xlim=[-6.5,-2.5],ylim=[-.35,.35],
                                xlabel='Subtrahend',ylabel='P(choose sum) - P(sum > sing)',horiz=0,
                                maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Data (Subtraction)')
    pdf.savefig()
    Plotter.panelplots(data=plot_data,plotvar='predicted_sum_bias',groupby=['animal','augend','addend'],
                                sem=None,axes=None,scattervar=[],xlim=[-6.5,-2.5],ylim=[-.35,.35],
                                xlabel='Subtrahend',ylabel='P(choose sum) - P(sum > sing)',horiz=0,
                                maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    plt.suptitle('Linear Model (Subtraction)')
    pdf.savefig()

    
#%%
#Examine the sum-weighting curves in each experiment for each animal

#Load in modeling data
floc = ''
fname_flatlo = 'linearauggain_FlatLO_1521970212.pkl'
fname_addition = 'linearauggain_Addition_1521968826.pkl'
fname_quad = 'linearauggain_Quad_1522016060.pkl'
data_wm_flatlo = pd.read_pickle(floc+fname_flatlo)
data_wm_addition = pd.read_pickle(floc+fname_addition)
data_wm_quad = pd.read_pickle(floc+fname_quad)

uaug_full = np.array([i for i in range(1,10)])
data_wm_flatlo[0][1]['par'][0]
data_wm_flatlo[0][1]['par'][1]
data_wm_addition[0][1]['par'][0]
data_wm_addition[0][1]['par'][1]
data_wm_quad[0][1]['par'][0]
data_wm_quad[0][1]['par'][1]
aug_weighting_fun = lambda w,x: w[2] + w[3]*(x-np.mean(uaug))

with PdfPages('D:\\Bart\\Dropbox\\AddendWeighting.pdf') as pdf:

    h,ax = plt.subplots(1,2,figsize=[2*6,6])
    uaug = np.array([i for i in range(1,10)])
    Plotter.lineplot(ax[0],xdata=uaug_full,ydata=aug_weighting_fun(data_wm_addition[0][1]['par'][0],uaug_full),
                     xlabel='Augend',ylabel='Addend Weight',xlim=[.5,9.5],color=[1,0,0],
                     ylim=[-.1,2],title='Monkey R',label='Experiment 1')
    Plotter.lineplot(ax[0],xdata=uaug_full,ydata=aug_weighting_fun(data_wm_quad[0][1]['par'][0],uaug_full),
                     xlabel='Augend',ylabel='Addend Weight',xlim=[.5,9.5],color=[0,0,1],
                     ylim=[-.1,2],title='Monkey R',label='Experiment 2')
    uaug = np.array([1,2,4])
    Plotter.lineplot(ax[0],xdata=uaug_full,ydata=aug_weighting_fun(data_wm_flatlo[0][1]['par'][0],uaug_full),
                     xlabel='Augend',ylabel='Addend Weight',xlim=[.5,9.5],color=[0,1,0],
                     ylim=[-.1,2],title='Monkey R',label='Experiment 3')
    ax[0].legend(loc=0,fontsize='x-small')
    ax[0].plot([uaug_full[0]-.5,uaug_full[-1]+.5],[0,0],'--k')
    uaug = np.array([i for i in range(1,10)])
    Plotter.lineplot(ax[1],xdata=uaug_full,ydata=aug_weighting_fun(data_wm_addition[0][1]['par'][1],uaug_full),
                     xlabel='Augend',ylabel='Addend Weight',xlim=[.5,9.5],color=[1,0,0],
                     ylim=[-.1,2],title='Monkey X',label='Experiment 1')
    Plotter.lineplot(ax[1],xdata=uaug_full,ydata=aug_weighting_fun(data_wm_quad[0][1]['par'][1],uaug_full),
                     xlabel='Augend',ylabel='Addend Weight',xlim=[.5,9.5],color=[0,0,1],
                     ylim=[-.1,2],title='Monkey X',label='Experiment 2')
    uaug = np.array([1,2,4])
    Plotter.lineplot(ax[1],xdata=uaug_full,ydata=aug_weighting_fun(data_wm_flatlo[0][1]['par'][1],uaug_full),
                     xlabel='Augend',ylabel='Addend Weight',xlim=[.5,9.5],color=[0,1,0],
                     ylim=[-.1,2],title='Monkey X',label='Experiment 3')
    ax[1].legend(loc=0,fontsize='x-small')
    ax[1].plot([uaug_full[0]-.5,uaug_full[-1]+.5],[0,0],'--k')
    uaug = np.array([i for i in range(1,10)])
    pdf.savefig()


#%%
#Look at animals fractional reward gain as a function of session
query = '''
        SELECT session,animal,experiment,chose_sum,augend,addend,singleton,
        augend+addend-singleton as diff,(augend+addend)/singleton as ratio,
        augend+addend as sum,trialset,((augend+addend)>singleton) as sum_correct
        FROM behavioralstudy
        WHERE experiment in ('Subtraction')
        ORDER BY animal,session,experiment
'''
data = Helper.getData(cur,query)
data['frac_available_reward'] = ( data['chose_sum']*data['sum'] + (1-data['chose_sum'])*data['singleton'] ) / np.max(data[['sum','singleton']],axis=1)

gb_data = data.groupby(['animal','session','experiment'],as_index=False)['frac_available_reward'].mean()


h,ax = plt.subplots(1,2,figsize=[2*6,4])
Plotter.lineplot(ax[0],xdata=gb_data['session'].loc[gb_data['animal']=='Xavier'],
    ydata=gb_data['frac_available_reward'].loc[gb_data['animal']=='Xavier'],
    ylim=[.85,1])
Plotter.scatter(ax[0],xdata=gb_data['session'].loc[gb_data['animal']=='Xavier'],
    ydata=gb_data['frac_available_reward'].loc[gb_data['animal']=='Xavier'],
    ylim=[.85,1],xlabel='Session',ylabel='Fractional reward received',title='Monkey X',
    yticks=[.85,.9,.95,1])

Plotter.lineplot(ax[1],xdata=gb_data['session'].loc[gb_data['animal']=='Ruffio'],
    ydata=gb_data['frac_available_reward'].loc[gb_data['animal']=='Ruffio'],
    ylim=[.85,1])
Plotter.scatter(ax[1],xdata=gb_data['session'].loc[gb_data['animal']=='Ruffio'],
    ydata=gb_data['frac_available_reward'].loc[gb_data['animal']=='Ruffio'],
    ylim=[.85,1],xlabel='Session',ylabel='Fractional reward received',title='Monkey R',
    yticks=[.85,.9,.95,1])

#%%
#Examine performance separately in matched trials where there are quad dots and not.
query = '''
        SELECT animal,experiment,session,augend,addend,singleton,chose_sum,
        chose_sum==((augend+addend)>singleton) as cor,
        augend+addend-singleton as diff,augend-addend as aadiff,augend+addend as sum,
        (augend+addend)>singleton as sum_correct,
        (aug_num_quad+add_num_quad+sing_num_quad)>0 as quadtrial,
        aug_num_quad,add_num_quad,sing_num_quad,
        (aug_num_quad>0) as quadaug,(add_num_quad>0) as quadadd,
        augend*10+addend as augadd,
        ((augend>4) OR (addend>4) OR (singleton>4))qeligible
        FROM behavioralstudy
        WHERE experiment in ('QuadDots')
        ORDER BY animal,session
'''
data = Helper.getData(cur,query)
data['quad_group'] = 0
data['quad_group'].loc[data['aug_num_quad']>0] = 'Aug'
data['quad_group'].loc[data['add_num_quad']>0] = 'Add'
data['quad_group'].loc[data['sing_num_quad']>0] = 'Sing'
data['quad_group'].loc[data['quadtrial']==0] = 'None'

with PdfPages('D:\\Bart\\Dropbox\\QuadPerformance.pdf') as pdf:
    plot_data = data.loc[data['animal']=='Ruffio']
    Plotter.panelplots(data=plot_data,plotvar='chose_sum',groupby=['quad_group','sum','singleton'],
                                axes=None,scattervar=[],xlim=[2,10],ylim=[0,1],sem=None,
                                xlabel='sum',ylabel='pcs',horiz=0,
                                maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    pdf.savefig()
    plot_data = data.loc[data['animal']=='Xavier']
    Plotter.panelplots(data=plot_data,plotvar='chose_sum',groupby=['quad_group','sum','singleton'],
                                axes=None,scattervar=[],xlim=[2,10],ylim=[0,1],sem=None,
                                xlabel='sum',ylabel='pcs',horiz=0,
                                maxcol=4,legend='on',cmap=plt.get_cmap('jet'))
    pdf.savefig()
##Examine difference in performance for each sum/singleton
##This doesn't look good.
#gdata = data.loc[(data['quadtrial']==0) & (data['qeligible']==1)].groupby(['animal','sum','singleton'],as_index=False)['cor'].mean()
#qdata = data.loc[(data['quadtrial']==1) & (data['qeligible']==1)].groupby(['animal','sum','singleton'],as_index=False)['cor'].mean()
#
#
##Make sure these data structures are properly aligned
#for i in range(0,gdata.shape[0]):
#    assert(gdata.shape == qdata.shape)
#    assert(gdata['animal'].iloc[i] == qdata['animal'].iloc[i])
#    assert(gdata['sum'].iloc[i] == qdata['sum'].iloc[i])
#    assert(gdata['singleton'].iloc[i] == qdata['singleton'].iloc[i])
#print('passed')
#
#combdata = gdata
#combdata['corq'] = qdata['cor']
#combdata['cordiff'] = combdata['cor']-combdata['corq']
#
#plot_data = combdata
#Plotter.panelplots(data=plot_data,plotvar='cordiff',groupby=['animal','singleton','sum'],
#                            axes=None,scattervar=[],xlim=[2,10],ylim=[-.3,.3],sem=None,
#                            xlabel='sum',ylabel='pcs',horiz=0,
#                            maxcol=9,legend='on',cmap=plt.get_cmap('jet'))


#%%
#Make schematic for log vs. linear representation
res = 1000000
x = np.linspace(0,100,res)
quantities = [1,2,3]
w = .25
xlim = [0,7]
lxlim = [pow(10,-.7),pow(10,.84)]

linear = [scipy.stats.norm(q,w*q).pdf(x) for q in quantities]
log = [scipy.stats.norm(np.log(q),w).pdf(np.log(x)) for q in quantities]
fig,ax = plt.subplots(2,2,figsize=[2*6,2*3])
#Linear model, linear domain
for l in linear:
    Plotter.lineplot(ax[0,0],xdata=x,ydata=l/max(l),xlim=xlim)
#Linear model, log domain
for l in linear:
    Plotter.lineplot(ax[0,1],xdata=x,ydata=l/max(l),xlim=lxlim)
ax[0,1].set_xscale('log')
#Log model, linear domain
for l in log:
    Plotter.lineplot(ax[1,0],xdata=x,ydata=l/max(l),xlim=xlim)
#Log model, log domain
for l in log:
    Plotter.lineplot(ax[1,1],xdata=x,ydata=l/max(l),xlim=lxlim)
ax[1,1].set_xscale('log')


#%%
description1997 = 'Testing whether addition in quad trials is noisier than addition in non-quad trials'
query1997 = '''
        SELECT animal,experiment,session,augend,addend,singleton,chose_sum,
        chose_sum==((augend+addend)>singleton) as cor,
        augend+addend-singleton as diff,augend-addend as aadiff,augend+addend as sum,
        (aug_num_quad+add_num_quad+sing_num_quad)>0 as quadtrial,
        ((augend>4) OR (addend>4) OR (singleton>4))as qeligible,
        augend+addend+singleton as total
        FROM behavioralstudy
        WHERE experiment in ('QuadDots') AND qeligible==1
        ORDER BY animal,session
'''
data1997 = Helper.getData(cur,query1997)
#q = data.loc[data['quadtrial']==1]
#u = data.loc[data['quadtrial']==0]

model1997 = 'chose_sum ~ total:augend + total:addend + total:singleton - 1'
out1997 = Analyzer.logistic_regression(data1997,model1997,groupby=['animal','quadtrial','session'])

sessx1997_nqt = out1997[['session','b_total:augend']].loc[(out1997['animal']=='Xavier') & (out1997['quadtrial']==0)]
sessx1997_qt = out1997[['session','b_total:augend']].loc[(out1997['animal']=='Xavier') & (out1997['quadtrial']==1)]
sessr1997_nqt = out1997[['session','b_total:augend']].loc[(out1997['animal']=='Ruffio') & (out1997['quadtrial']==0)]
sessr1997_qt = out1997[['session','b_total:augend']].loc[(out1997['animal']=='Ruffio') & (out1997['quadtrial']==1)]

n1997_x = len(sessx1997_nqt)
t1997_x = scipy.stats.ttest_rel(sessx1997_nqt['b_total:augend'],sessx1997_qt['b_total:augend'])
n1997_r = len(sessr1997_nqt)
t1997_r = scipy.stats.ttest_rel(sessr1997_nqt['b_total:augend'],sessr1997_qt['b_total:augend'])

tests = []
tests.append({'description':description1997+', Augend (Xavier)','p':t1997_x.pvalue,
              'stat':t1997_x.statistic,'mean':(np.mean(sessx1997_nqt),np.mean(sessx1997_qt)),
              'n':n1997_x,'df':n1997_x-1})
tests.append({'description':description1997+', Augend (Xavier)','p':t1997_r.pvalue,
              'stat':t1997_r.statistic,'mean':(np.mean(sessr1997_nqt),np.mean(sessr1997_qt)),
              'n':n1997_r,'df':n1997_r-1})
    
    
    
#%%
#Look at relationship between singleton magntiude and p(singleton correct)
    
query = '''
        SELECT singleton,AVG((augend+addend)>singleton) as sumcorrect
        FROM behavioralstudy
        WHERE experiment=='QuadDots'
        GROUP BY singleton
        ORDER BY singleton
'''
data = Helper.getData(cur,query)
#q = data.loc[data['quadtrial']==1]
#u = data.loc[data['quadtrial']==0]

fig,ax = plt.subplots(1,1,figsize=[1*6,1*4])
Plotter.lineplot(ax,xdata=data['singleton'],ydata=data['sumcorrect'],ylabel='p(sum correct)',xlabel='Singleton')
scipy.stats.pearsonr(data['singleton'],data['sumcorrect'])


#%%

#Is the relationship between total size and discriminability scalar?
query_sssum = '''
        SELECT animal,augend+addend as sum,singleton,AVG(chose_sum) as chose_sum,
        augend+addend+singleton as ss_sum,augend+addend-singleton as diff
        FROM behavioralstudy
        WHERE experiment='Addition'
        GROUP BY animal,sum,singleton
        ORDER BY animal,sum,singleton
'''
data_sssum = Helper.getData(cur,query_sssum)

usssum = np.unique(data_sssum['ss_sum'])

model = 'chose_sum ~ diff'
ss_sum_slopes = []
for ss in usssum:
     try:
         lr_out = Analyzer.logistic_regression(data_sssum.loc[data_sssum['ss_sum']==ss],
                        model,groupby=['animal'])
         ss_sum_slopes.append((ss,lr_out['b_diff'].get_values()))
     except:
         pass
     
h,ax = plt.subplots(1,1,figsize=[6,4])
Plotter.lineplot(ax,xdata=[s[0] for s in ss_sum_slopes],ydata=[s[1][0] for s in ss_sum_slopes])
Plotter.lineplot(ax,xdata=[s[0] for s in ss_sum_slopes],ydata=[s[1][1] for s in ss_sum_slopes])

scipy.stats.pearsonr(x=[s[0] for s in ss_sum_slopes],
                     y=[s[1][0] for s in ss_sum_slopes])
scipy.stats.pearsonr(x=[s[0] for s in ss_sum_slopes],
                     y=[s[1][1] for s in ss_sum_slopes])