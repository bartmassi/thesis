# -*- coding: utf-8 -*-
"""
Testing ground for running analysis on data from the Arithmetic study.

@author: bart
creation date: 12-11-17
"""
#cd D:\\Bart\\Dropbox\\code\\python\\leelab\\thesis
#%load_ext autoreload
#%autoreload 2

import sqlite3
import pandas as pd
import Plotter
import analyzer
import matplotlib.pyplot as plt
import numpy as np

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
        WHERE experiment = 'Addition'
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
