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
import itertools
import Helper
import pdb

realmin = np.finfo(np.double).tiny


#This function returns a dictionary with anonymous functions for models of
#arithmetic data. All models return a probability of choosing sum.
#Models are hand-chosen based on literature or intuition.
#
#The models expect specific naming schemes for columns in 'data', that correspond
#to the column names in the SQLite DB. 
#The models also take, in order, their parameters and the data.
def get_models():
    #setup cost function for fitting
    #realmin = np.finfo(np.double).tiny #smallest possible floating point number
    cost = lambda y,h: -np.sum(y*np.log(np.max([h,np.ones(len(h))*realmin],axis=0))+
                        (1-y)*np.log(np.max([1-h,np.ones(len(h))*realmin],axis=0))); 

    #get information about singleton prior in flat log odds trialset
    flatlo = Helper.getFlatLOTrialset()
    pcs = flatlo['pcs']
    pcs_var = pcs*(1-pcs)
                                               
    #setup gaussian approximate number model from Dehaene 2007, single scaling factor and intercept
    #2 parameter
    dm_onescale = lambda w,data: 1-scipy.stats.norm(w[0] + data['augend']+data['addend']-data['singleton'],
             w[1]*np.sqrt(data['augend']**2. + data['addend']**2 + data['singleton']**2)).cdf(0)
    
    #Full Dehaene 2007, separate scaling factor for each group
    #3 parameters
    dm_var = lambda w,data: np.sqrt((w[0]**2)*data['augend']**2 + (w[1]**2)*data['addend']**2 + (w[2]**2)*data['singleton']**2)
    dm_full = lambda w,data: 1-scipy.stats.norm(data['augend']+data['addend']-data['singleton'],
             dm_var(w,data)).cdf(0)
    
    #Dehaene model w/ one scaling factor, and separate weight for each element of prior. 
    #Meant for FlatLO experiment only.
    #13 parameters
    dm_prior_weighting = lambda w,data: (1-w[data['singleton']])*dm_onescale(w[[0,1]],data) \
                                           + w[data['singleton']+1]*pcs[data['singleton']-1]

    logistic = lambda x:1./(1.+np.exp(-x))
    
    #Dehaene model plus prior w/ optimal weighting of each
    #3 parameters
    weightfun = lambda w,data: dm_var(w,data)/(dm_var(w,data)+pcs_var[data['singleton']-1])
    dm_prior_optimal = lambda w,data: (1-weightfun(w,data))*dm_full(w,data) + weightfun(w,data)*pcs[data['singleton']-1]

    #Linear addition model.
    #4 parameters
    linear = lambda w,data: logistic(w[0] + w[1]*data['augend'] + w[2]*data['addend'] + w[3]*data['singleton'])
    
    #Average of the difference and the prior's predictions, weighted by the ratio of sum and singleton.
    #2 parameters
    ratiofun = lambda data: data[['sum','singleton']].min(axis=1)/data[['sum','singleton']].max(axis=1)
    weighted_diff_singprior = lambda w,data: logistic(w[0] + 
                (1-ratiofun(data)*w[1])*(data['sum']-data['singleton']) + ratiofun(data)*w[1]*data['sing_prior'])
    
    #2 parameters
    norm_coef = lambda w,q1,q2: 1.0/(1.0+w[0]*np.power(np.power(np.abs(q1),w[1]) + np.power(np.abs(q2),w[1]),(1.0/w[1])) )
    livingstone_norm = lambda w,data: logistic(w[2]*(norm_coef(w[0:2],data['addend'],data['singleton'])*data['augend'] 
        + norm_coef(w[0:2],data['augend'],data['singleton'])*data['addend'] 
        - norm_coef(w[0:2],data['augend'],data['addend'])*data['singleton']))
    
    #a model in which the ratio of the sum and singleton are compared.
    #coefficients are on the aug,add,sing prior to log transform.
    #4 parameters
    logarithmic = lambda w,data: logistic(w[0] + np.log(w[1]*data['augend']+w[2]*data['addend']) 
        - np.log(w[3]*data['singleton']))
    
    #a model in which the ratio of the sum and singleton are compared.
    #coefficients are on the log of the sum and singleton.
    #4 parameters
    logarithmic2 = lambda w,data: logistic(w[0] + w[1]*np.log(data['augend']+data['addend']) 
        - w[2]*np.log(data['singleton']))
    
    #a model where the log of the aug, add, sing are combined
    #4 parameters
    log_aas = lambda w,data: logistic(w[0] + w[1]*np.log(data['augend']) + w[2]*np.log(data['addend'])
        + w[3]*np.log(data['singleton']))
    
    models = {'cost':cost,'dm_onescale':dm_onescale,
    'dm_full':dm_full,
    'dm_prior_weighting':dm_prior_weighting,
    'linear':linear,
    'dm_prior_optimal':dm_prior_optimal,
    'weightfun':weightfun,
    'weighted_diff_singprior':weighted_diff_singprior,
    'livingstone_norm':livingstone_norm,
    'logarithmic':logarithmic,
    'logarithmic2':logarithmic2,
    'log_aas':log_aas}
    
    return models

    
#This runs a logistic regression on data.
#df: the data frame containing the variables that are parts of the model
#model: the model. Uses patsy (i.e., R) model specification.
#groupby: List of field names. fits model separately to each unique combination 
#of groupby variable values.
def logistic_regression(df,model,groupby=None,compute_cpd=True,standardize=False):
    
    #should we use group by?
    usegroupby = groupby != None
    
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
        if(standardize):
            for c in X.columns:
                if(c!='Intercept'):
                    X[c] = scipy.stats.mstats.zscore(X[c])
        #create and fit model object
        mdl = sreg.GLM(endog=y,exog=X,family=sm.genmod.families.family.Binomial())
        thismout = mdl.fit()
        thismout.bic = thismout.deviance+np.log(X.shape[0])*len(thismout.params)
        thismout.rank = np.linalg.matrix_rank(X)
        thismout.npar = X.shape[1]
        thismout.fullrank = thismout.rank==thismout.npar
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
    'npar':[m.npar for m in mout],
    	'null_deviance':[m.null_deviance for m in mout],
    'rank':[m.rank for m in mout],
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

#fits a model to given data.
#model: function handle to the model. Must take parameters,data as arguments (in that order)
#X: input data to the model.
#y: variable to be predicted by model
#cost: the cost function used to evaluate the model. If not specified, sum of
#log likelihood is used.
#Other arguments are passed to scipy.optimize.minimize.
def fit_model(model,X,y,x0,cost='default',**kwargs):
    
    
    realmin = np.finfo(np.double).tiny #smallest possible floating point number

    #define cost function    
    if(cost=='default'):
        cost = lambda y,h: -np.sum(y*np.log(np.max([h,np.ones(len(h))*realmin],axis=0))+
                    (1-y)*np.log(np.max([1-h,np.ones(len(h))*realmin],axis=0))); 
                                                   
    objfun = lambda par: cost(y,model(par,X))
    opt = scipy.optimize.minimize(fun=objfun,x0=x0,
            **kwargs)
    
    opt['bic'] = opt.fun*2 + np.log(len(y))*len(x0)
    opt['init'] = x0
    
    return opt
    
#Fits a model to given data using a range of initializations.
#model: function handle to the model. Must take parameters,data as arguments (in that order)
#X: input data to the model.
#y: variable to be predicted by model
#parrange: a tuple (or list) of initial parameter values to be used for each 
#parameter. Must be exactly the length of the number of free parameters that 
#model takes. 
#cost: the cost function used to evaluate the model. If not specified, sum of
#log likelihood is used.
#Other arguments are passed to scipy.optimize.minimize.
#TODO: this is not scalable, because itertools.product returns the entire 
#cartesian product of the lists in parrange. Likewise, all outputs are stored.
#def fit_grid_search(model,X,y,parrange,cost='default',verbose=True,**kwargs):
#    
#    #get all combinations of initial values
#    parcombs = itertools.product(*parrange)
#    nparcomb = np.prod([len(x) for x in parrange])
#    #perform grid search
#    opt = []
#    i = 0
#    for p in parcombs:
#        opt.append(fit_model(model=model,X=X,y=y,x0=p,cost=cost,**kwargs))    
#        i += 1        
#        if(verbose):
#            if(i % 10 == 0):
#                print(str(100*i/nparcomb)+'% done')
#                
#    #figure out best fitting. 
#    best_ind = np.argmin([o.fun for o in opt])  
#    
#    return opt[best_ind],opt

#Fits a model to given data using a range of initializations.
#model: function handle to the model. Must take parameters,data as arguments (in that order)
#X: input data to the model.
#y: variable to be predicted by model
#parrange: a tuple (or list) of initial parameter values to be used for each 
#groupby: a list of strings corresponing to variables in X. The analysis will be
#duplicated for each unique combination of these variables.
#parameter. Must be exactly the length of the number of free parameters that 
#model takes. 
#cost: the cost function used to evaluate the model. If not specified, sum of
#log likelihood is used.
#Other arguments are passed to scipy.optimize.minimize.
#
#TODO: this is not scalable, because itertools.product returns the entire 
#cartesian product of the lists in parrange. Likewise, all outputs are stored.
def fit_grid_search(model,X,y,parrange,groupby=None,cost='default',verbose=True,**kwargs):
    #should we use group by?
    usegroupby = groupby != None 
    
    #If we're using group by, find unique values to group by.
    if(usegroupby):
        gb_u = X[groupby].drop_duplicates()#unique values of groupby variable
        ncond = gb_u.shape[0]
    else:#otherwise, we're not using group by.
        ncond = 1;
    
    gb_opt = []
    for i in range(0,ncond):
        #get all combinations of initial values
        parcombs = itertools.product(*parrange)
        
        #pick out this groupby
        if(usegroupby):
            thisx = X.loc[np.sum(X[groupby] == gb_u.iloc[i,:],axis=1)==len(groupby)]
            thisy = y.loc[np.sum(X[groupby] == gb_u.iloc[i,:],axis=1)==len(groupby)]
        else:
            thisx = X
            thisy = y
            
        nparcomb = np.prod([len(x) for x in parrange])
        #perform grid search
        opt = []
        j = 0
        if(verbose):
            percentdone = round(1000*i/ncond)/10
            print(str(percentdone)+'% done' )
        for p in parcombs:
            opt.append(fit_model(model=model,X=thisx,y=thisy,x0=p,cost=cost,**kwargs))    
            j += 1     
            
#            #this is barely functional. Needs work.
#            if(verbose):
#                if(((j+1)*(i+1)) % np.floor(nparcomb*ncond/10) == 0):
#                    print(str(100*((j+1)*(i+1))/(nparcomb*ncond))+'% done')

                    
        #figure out best fitting. 
        best_ind = np.argmin([o.fun for o in opt])  
        
        if(usegroupby):
            gb_opt.append((opt[best_ind],gb_u.iloc[i,:]))
        else:
            gb_opt.append((opt[best_ind],))
            
    #convert output from GLMresults object into dictionary, which is later converted
    #into a pandas table.
    #pdb.set_trace()
    mout_dict = {'bic':[m[0].bic for m in gb_opt],
    	'fun':[m[0].fun for m in gb_opt],#'hess_inv':[m[0].hess_inv for m in gb_opt],
    	'init':[m[0].init for m in gb_opt],#'jac':[m[0].jac for m in gb_opt],
    	'message':[m[0].message for m in gb_opt],
    	'nfev':[m[0].nfev for m in gb_opt],
    'nit':[m[0].nit for m in gb_opt],
    'status':[m[0].status for m in gb_opt],
    	'success':[m[0].success for m in gb_opt],
    	'par':[m[0].x for m in gb_opt]}

    #add groupby information to output data structure
    if(usegroupby):
        for i in range(0,ncond):
            for gbcond in groupby:
                if(i==0):
                    mout_dict[gbcond] = []
                mout_dict[gbcond].append(gb_u[gbcond].iloc[i])

    #convert dictionary into dataframe
    return pd.DataFrame(mout_dict)
    
    return gb_opt
    
    
def predict(model,par,X,y=None):
    
    predictions = model(par,X)
    
    if(y is None):
        acc = np.nan
    else:
        acc = np.mean((predictions>.5)==y)
    
    return predictions,acc

#Perform a leave-one-out validation on a model, separately for each groupby variable.
#X: input data to the model.
#y: variable to be predicted by model
#parrange: a tuple (or list) of initial parameter values to be used for each parameter in model
#validateby: variable whose values will be used for leave-one-out approach.
#groupby: a list of strings corresponing to variables in X. The analysis will be
#duplicated for each unique combination of these variables.
#parameter. Must be exactly the length of the number of free parameters that 
#model takes. 
#cost: the cost function used to evaluate the model. If not specified, sum of
#log likelihood is used.
#Other arguments are passed to scipy.optimize.minimize.
#
#TODO: add support for groupby='None'
def fit_grid_loo_validate(model,X,y,parrange,validateby,groupby=None,cost='default',verbose=True,**kwargs):
    
        #should we use group by?
    usegroupby = groupby != None
    
    #If we're using group by, find unique values to group by.
    if(usegroupby):
        gb_u = X[groupby].drop_duplicates()#unique values of groupby variable
        ncond = gb_u.shape[0]
    else:#otherwise, we're not using group by.
        ncond = 1;
        
    results = []
    for i in range(0,ncond):
        gblabel = np.sum(X[groupby] == gb_u.iloc[i,:],axis=1)==len(groupby)
        validate_u = X[validateby].loc[gblabel].drop_duplicates()
        nvalidate = validate_u.shape[0]
        for j in range(0,nvalidate):
            if(verbose):
                percentdone = round(1000*((i*nvalidate + j))/(ncond*nvalidate),1)/10
                print(str(percentdone)+'% done' )
            
            vlabel = np.sum(X[validateby] == validate_u.iloc[j,:],axis=1)==len(validateby)
            test_x = X.loc[gblabel & vlabel]
            test_y = y.loc[gblabel & vlabel]
            train_x = X.loc[gblabel & np.logical_not(vlabel)]
            train_y = y.loc[gblabel & np.logical_not(vlabel)]
            mout = fit_grid_search(model,train_x,train_y,parrange,groupby=None,cost=cost,verbose=False,**kwargs)
            _,acc = predict(model,mout['par'][0],test_x,test_y)
            
            #store results
            results.append([x for x in gb_u.iloc[i,:]] + [y for y in validate_u.iloc[j,:]] + [acc])

    colnames = groupby + validateby + ['cv_acc']
    loo_out = pd.DataFrame(results,columns=colnames)
    
    return loo_out
    
    