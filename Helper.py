# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 16:59:41 2017

@author: Bart
"""
import numpy as np
import sqlite3
import pandas as pd

#This generates the contents of the trials in the FlatLO experiment. Also computes
#the base-rate probability that each singleton value is correct, which is useful
#for some modeling & analysis. 
def getFlatLOTrialset():
    #generate trialset
    aug = np.arange(1,7,1)
    add = np.array([1,2,4])
    diff = np.array([-2,-1,1,2])
    trials = []
    for i in range(0,len(aug)):
        if aug[i]<4:
            itercount = 1
        else:
            itercount = 2
        for it in range(0,itercount):
            for j in range(0,len(add)):
                if(aug[i]==1 and add[j]==1):
                    trials.append([aug[i],add[j],aug[i]+add[j]+1])
                    trials.append([aug[i],add[j],aug[i]+add[j]-1])
                else:
                    for k in range(0,len(diff)):
                        trials.append([aug[i],add[j],aug[i]+add[j]+diff[k]])
                        
    #compute p(correct) for each singleton
    tset = np.array(trials);#trialset in augend,addend,singleton format.
    using = np.unique(tset[:,2])
    pcorrect_sing = []
    for si in using:
        pcorrect_sing.append(np.mean((tset[tset[:,2]==si,0]+tset[tset[:,2]==si,1]) > tset[tset[:,2]==si,2]))
    
    #P(sum correct|singleton=y) for all values of y. 
    pcs= np.array(pcorrect_sing);

    return {'trialset':tset,'pcs':pcs,'using':using}


#executes SQL queries and returns results in a predictable and useful way. 
def getData(cur,query):
    cur.execute(query)
    dataout = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    data = pd.DataFrame.from_records(dataout,columns=colnames)
    return data