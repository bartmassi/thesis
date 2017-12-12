# -*- coding: utf-8 -*-
"""
Testing ground for running analysis on data from the Arithmetic study.

@author: bart
creation date: 12-11-17
"""
#cd D:\\Bart\\Dropbox\\code\\python\\leelab\\thesis
#%load_ext autoreload
#%autoreload 2

import psycopg2 as pg2
import pandas as pd
import Plotter
import analyzer

conn = pg2.connect(database='arithmeticstudy', user='postgres',
                   password=input('pw: '))
cur = conn.cursor()

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
colnames1 = [desc.name for desc in cur.description]
data = pd.DataFrame.from_records(dataout1,columns=colnames1)

#Generate a scatterplot of the results. 
xdict = {'data':data['nonea_pcorrect'],
         'label':'Accuracy in non-EA trials',
         'limits':(.5,1)}
ydict = {'data':data['ea_pcorrect'],
         'label':'Accuracy in EA trials',
         'limits':(.5,1)}

h,axes = plt.subplots(1,1)
Plotter.scatter(axes,xdict,ydict,title='Animal Accuracy')


###########Fit a regression model to animals' choice data, and then plot coefficients.
#SQL query
query2 = '''
        SELECT session,animal,CAST(chose_sum AS int),augend,addend,singleton
        FROM behavioralstudy
        WHERE experiment = 'Addition'
        ORDER BY animal,session
'''
#Execute query, then convert to pandas table
cur.execute(query2)
dataout2 = cur.fetchall()
colnames2 = [desc.name for desc in cur.description]
data = pd.DataFrame.from_records(dataout2,columns=colnames2)

#Fit regression model to separate sessions
output = analyzer.logistic_regression(data,model='chose_sum ~ augend + addend + singleton',
                                      groupby=['animal','session'])

#setup plotting structures

h,axes = plt.subplots(1,1)