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
    #iterate through each augend, addend, and difference and append a trial.
    #when aug>=4,add two trials of each type (for quad dots trials)
    #when aug & add are 1, only add sing=1 and sing=3 (not 0,4)
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
    #and avg min(sum,sing)/max(sum,singleton) ratio for each singleton
    #and marginal frequency of each singleton value
    tset = np.array(trials);#trialset in augend,addend,singleton format.
    using = np.unique(tset[:,2])
    uprod = np.unique(np.prod(tset,axis=1))
    pcorrect_sing = []
    avgratio_sing = []
    singdist = []
    sum_correct = (tset[:,0]+tset[:,1])>tset[:,2]
                   
    (tset[:,0]+tset[:,1])*sum_correct + tset[:,2]*(np.negative(sum_correct))
    for si in using:
        thissing = tset[:,2]==si
        sc_ts = sum_correct[thissing]
        pcorrect_sing.append(np.mean((tset[thissing,0]+tset[thissing,1]) > tset[thissing,2]))
        avgratio_sing.append(np.mean(
                                     ((tset[thissing,0]+tset[thissing,1])*np.negative(sc_ts) + tset[thissing,2]*sc_ts)/#small of sum & sing
                                       ((tset[thissing,0]+tset[thissing,1])*sc_ts + tset[thissing,2]*(np.negative(sc_ts)))#large of sum & sing
                                      ))
        singdist.append(np.mean(thissing))
        
    pcorrect_prod = []
    avgratio_prod = []
    proddist = []
    for p in uprod:
        thisp = np.prod(tset,axis=1)==p
        sc_ts = sum_correct[thisp]
        pcorrect_prod.append(np.mean((tset[thisp,0]+tset[thisp,1]) > tset[thisp,2]))
        avgratio_prod.append(np.mean(
                                     ((tset[thisp,0]+tset[thisp,1])*np.negative(sc_ts) + tset[thisp,2]*sc_ts)/#small of sum & sing
                                       ((tset[thisp,0]+tset[thisp,1])*sc_ts + tset[thisp,2]*(np.negative(sc_ts)))#large of sum & sing
                                      ))
        proddist.append(np.mean(thissing))
    
    #P(sum correct|singleton=y) for all values of y. 
    pcs= np.array(pcorrect_sing);
    avgratio = np.array(avgratio_sing);
    singdist = np.array(singdist);
    

    return {'trialset':tset,'pcs':pcs,'using':using,'avgratio':avgratio,'singdist':singdist,
            'uprod':uprod,'pcp':np.array(pcorrect_prod),'avgratio_prod':np.array(avgratio_prod),'proddist':np.array(proddist)}


#executes SQL queries and returns results in a predictable and useful way. 
def getData(cur,query):
    cur.execute(query)
    dataout = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    data = pd.DataFrame.from_records(dataout,columns=colnames)
    return data