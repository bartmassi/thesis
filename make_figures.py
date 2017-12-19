# -*- coding: utf-8 -*-
"""
Make figures for thesis. 

@author: bart
"""


##Run these prior to running any code. 
cd D:\\Bart\\Dropbox\\code\\python\\leelab\\thesis
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
#determine how often the animals choose the sum as a function of sum and singleton
#show this as a matrix.

#get data
query = '''
        SELECT animal,augend+addend as sum,singleton,AVG(chose_sum) as chose_sum
        FROM behavioralstudy
        WHERE experiment='Addition'
        GROUP BY animal,sum,singleton
        ORDER BY animal,sum,singleton
'''
data = Helper.getData(cur,query)

#pull out unique values
usum = np.unique(data['sum'])
nusum = len(usum)
using = np.unique(data['singleton'])
nusing = len(using)

#get performance in each condition, make into matrix
xperf = np.array([[data['chose_sum'].loc[(data['singleton']==si) & (data['sum']==su) & (data['animal']=='Xavier')] if su != si else np.nan for su in usum] for si in using])
rperf = np.array([[data['chose_sum'].loc[(data['singleton']==si) & (data['sum']==su) & (data['animal']=='Ruffio')] if su != si else np.nan for su in usum] for si in using])


    
#get psychometric curve data
query2 = '''
        SELECT animal,augend+addend-singleton as diff,chose_sum,trialset,session,
        (((augend+addend)>singleton) AND (augend<singleton) AND (addend<singleton)) as ea,
        (chose_sum = ((augend+addend)>singleton)) as chose_correct
        FROM behavioralstudy
        WHERE experiment='Addition'
        ORDER BY animal,session,diff
'''
data2 = Helper.getData(cur,query2)

#compute mean/sem for each sum-sing and animal
udiffs = np.unique(data2['diff'])
x_psum = [np.mean(data2['chose_sum'].loc[(data2['diff']==di) & (data2['animal']=='Xavier')]) for di in udiffs]
r_psum = [np.mean(data2['chose_sum'].loc[(data2['diff']==di) & (data2['animal']=='Ruffio')]) for di in udiffs]
x_sem = [scipy.stats.sem(data2['chose_sum'].loc[(data2['diff']==di) & (data2['animal']=='Xavier')]) for di in udiffs]
r_sem = [scipy.stats.sem(data2['chose_sum'].loc[(data2['diff']==di) & (data2['animal']=='Ruffio')]) for di in udiffs]

#get performance in EA, non-EA, set1, and set2 trials.
usess = np.unique(data2['session'])
x_ea = [np.mean(data2['chose_correct'].loc[(data2['animal']=='Xavier') & (data2['session']==s) &  (data2['ea']==True)]) for s in usess]
x_nea = [np.mean(data2['chose_correct'].loc[(data2['animal']=='Xavier') & (data2['session']==s) &  (data2['ea']==False)]) for s in usess]
x_set1 = [np.mean(data2['chose_correct'].loc[(data2['animal']=='Xavier') & (data2['session']==s) &  (data2['trialset']==1)]) for s in usess]
x_set2 = [np.mean(data2['chose_correct'].loc[(data2['animal']=='Xavier') & (data2['session']==s) &  (data2['trialset']==2)]) for s in usess]
r_ea = [np.mean(data2['chose_correct'].loc[(data2['animal']=='Ruffio') & (data2['session']==s) &  (data2['ea']==True)]) for s in usess]
r_nea = [np.mean(data2['chose_correct'].loc[(data2['animal']=='Ruffio') & (data2['session']==s) &  (data2['ea']==False)]) for s in usess]
r_set1 = [np.mean(data2['chose_correct'].loc[(data2['animal']=='Ruffio') & (data2['session']==s) &  (data2['trialset']==1)]) for s in usess]
r_set2 = [np.mean(data2['chose_correct'].loc[(data2['animal']=='Ruffio') & (data2['session']==s) &  (data2['trialset']==2)]) for s in usess]


#%%
#put it all in a PDF
with PdfPages('D:\\Bart\\Dropbox\\pdf_test.pdf') as pdf:
    h,ax = plt.subplots(1,2)

    Plotter.gridplot(ax[0],xperf,cmap=plt.cm.seismic,title='Monkey X',
        xticks=np.arange(0,len(usum),1),xticklabels=usum,xlabel='Sum',
        yticks=np.arange(0,len(using),1),yticklabels=using,ylabel='Singleton',
        cticks=[0,.5,1],clabel='p(choose sum)')    

    Plotter.gridplot(ax[1],rperf,cmap=plt.cm.seismic,title='Monkey R',
        xticks=np.arange(0,len(usum),1),xticklabels=usum,xlabel='Sum',
        yticks=np.arange(0,len(using),1),yticklabels=using,ylabel='Singleton',
        cticks=[0,.5,1],clabel='p(choose sum)')
    
    plt.tight_layout()
    pdf.savefig()
    
    
    h,ax = plt.subplots(1,2)
    Plotter.lineplot(ax[0],xdata=udiffs,
        ydata=x_psum,sem=x_sem,title='Monkey X',xlabel='Sum - Singleton',xticks = [-8,-4,0,4,8],
        ylabel='P(choose sum)',yticks=[0,.25,.5,.75,1])
    Plotter.scatter(ax[0],xdata=udiffs,
        ydata=x_psum,identity='off')
    
    Plotter.lineplot(ax[1],xdata=udiffs,
        ydata=r_psum,sem=r_sem,title='Monkey R',xlabel='Sum - Singleton',xticks = [-8,-4,0,4,8],
        ylabel='P(choose sum)',yticks=[0,.25,.5,.75,1])
    Plotter.scatter(ax[1],xdata=udiffs,
        ydata=r_psum,identity='off')
    
    plt.tight_layout()
    pdf.savefig()
    
    h,ax = plt.subplots(1,2)
    Plotter.scatter(ax[0],xdata=x_nea,ydata=x_ea,xlabel='Performance in non-EA',ylabel='Performance in EA',
                    title='Monkey X')