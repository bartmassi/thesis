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