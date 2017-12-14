# -*- coding: utf-8 -*-
"""

Simplified functions for running common analyses, such as general linear models.


@author: bart
creation date: 12-11-17
"""

import numpy as np #numerical operations and matrix operations
import pandas as pd #data structures
import statsmodels.genmod.generalized_linear_model as sreg
import statsmodels as sm
import patsy #specify models in R-like way.  
import scipy

#This runs a logistic regression on data.
#df: the data frame containing the variables that are parts of the model
#model: the model. Uses patsy (i.e., R) formatting.
#groupby
def logistic_regression(df,model,groupby='None',compute_cpd=True):
    
    #should we use group by?
    usegroupby = groupby != 'None' and groupby != 'none'
    
    #If we're using group by, find unique values to group by.
    if(usegroupby):
        gb_u = df[groupby].drop_duplicates()#unique values of groupby variable
        ncond = gb_u.shape[0]
    else:#otherwise, we're not using group by.
        ncond = 1;
        
    mout = []
    for i in range(0,ncond):
        
        #gets the subset of data given by groupby variable. 
        #Otherwise, use entire dataframe
        if(usegroupby):
            thisdf = df.loc[np.sum(df[groupby] == gb_u.iloc[i,:],axis=1)==len(groupby)]
        else:
            thisdf = df
            
        #converts data into regressand (y) and regression matrix (X) based on model. 
        y, X = patsy.dmatrices(model, thisdf, return_type='dataframe')
        #create and fit model object
        mdl = sreg.GLM(endog=y,exog=X,family=sm.genmod.families.family.Binomial())
        thismout = mdl.fit()
        
        #placeholder for computing coefficient of partial determination
        if(compute_cpd):
            pass
        else:
            pass
        
        #store results
        mout.append(thismout)
        

    #convert output from GLMresults object into dictionary, which is later converted
    #into a pandas table.
    mout_dict = {'bic':[m.bic for m in mout],
    	'deviance':[m.deviance for m in mout],
    	'df_model':[m.df_model for m in mout],
    	'df_resid':[m.df_resid for m in mout],
    	'fittedvalues':[m.fittedvalues for m in mout],
    	'llf':[m.llf for m in mout],
    	'mu':[m.mu for m in mout],
    	'null_deviance':[m.null_deviance for m in mout],
    	'resid_deviance':[m.resid_deviance for m in mout],
    	'scale':[m.scale for m in mout]}

    #flatten parameter/pvalue output into 1 parameter per column.
    for i in range(0,X.shape[1]):
        mout_dict['b_'+X.columns[i]] = [m.params[i] for m in mout]
        mout_dict['p_'+X.columns[i]] = [m.pvalues[i] for m in mout]

    #add groupby information to output data structure
    if(usegroupby):
        for i in range(0,ncond):
            for gbcond in groupby:
                if(i==0):
                    mout_dict[gbcond] = []
                mout_dict[gbcond].append(gb_u[gbcond].iloc[i])

    #convert dictionary into dataframe
    return pd.DataFrame(mout_dict)
    
    
#Starts with the full model specified by 'model', and performs backwards induction.
#The criteria is that each term must be significant (at pthresh) in statistically
#more than groupby_thresh sessions.
#Groupby must be specified. 
def logistic_backwardselimination_sessionwise(df,model,groupby,groupby_thresh=.05,pthresh=.05):
    
    #get out column names and store removed columns
    y, X = patsy.dmatrices(model,df)
    all_columns = X.design_info.column_names
    remaining_columns = all_columns
    removed_columns = ['']

    
    #perform backwards elimination.
    #terminate when all terms are significant in more than groupby_thresh sessions.
    flag = 1;
    while(flag):
        
        #update model by removing bad terms
        thismodel = model + ' - '.join(removed_columns)
        #start by performing logistic regression
        outdf = logistic_regression(df,thismodel,groupby,compute_cpd=False)
        total_sessions = outdf.shape[0];

        #get the number of sessions that each term is significant in
        n_sig_sess = [np.sum(outdf['p_'+rc]<pthresh) for rc in remaining_columns]
        term_p = []

        #binomial test on p(significant) for each term.
        for i in range(0,len(remaining_columns)):
            term_p.append(scipy.stats.binom_test(n_sig_sess[i],total_sessions,
                            p=groupby_thresh,alternative='greater'))
            
        #find index of worst performing term
        worstind = np.argmax(term_p)
        if(term_p[worstind] > groupby_thresh):
            removed_columns.append(remaining_columns.pop(worstind))
        else:
            flag = 0;        
    
    #return final model
    final_model = model = model + ' - '.join(removed_columns);
    return {'final_model':final_model,'final_modelout':outdf,
    'remaining_terms':remaining_columns,'removed_columns':removed_columns[1:]}