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



#get logistic regression data for Addition study
query3 = '''
        SELECT animal,session,augend,addend,singleton,chose_sum,trial % 2 as odd_trial
        FROM behavioralstudy
        WHERE experiment='Addition'
        ORDER BY animal,session
'''
data3 = Helper.getData(cur,query3)

#perform simple logistic regression
model = 'chose_sum ~ augend + addend + singleton'
mout1 = Analyzer.logistic_regression(data3,model,groupby=['animal','session','odd_trial'])



#get logistic regression data for quad dots study
query4 = '''
        SELECT animal,session,aug_num_green,add_num_green,sing_num_green,
        aug_num_quad,add_num_quad,sing_num_quad,chose_sum,trial % 2 as odd_trial
        FROM behavioralstudy
        WHERE experiment='QuadDots'
        ORDER BY animal,session
'''
data4 = Helper.getData(cur,query4)

#perform simple logistic regression
model = 'chose_sum ~ aug_num_green + add_num_green + sing_num_green \
            + aug_num_quad + add_num_quad + sing_num_quad'
mout2 = Analyzer.logistic_regression(data4,model,groupby=['animal','session','odd_trial'])


#compare visual trials to non-visual trials
query5 = '''
        SELECT animal,session,
        AVG(CASE WHEN ((aug_num_green+aug_num_quad+add_num_green+add_num_quad) > (sing_num_green+sing_num_quad))==((augend+addend)>singleton)
        THEN chose_sum == ((augend+addend)>singleton) ELSE NULL END) as pc_vt,
        AVG(CASE WHEN ((aug_num_green+aug_num_quad+add_num_green+add_num_quad) > (sing_num_green+sing_num_quad))!=((augend+addend)>singleton)
        THEN chose_sum == ((augend+addend)>singleton) ELSE NULL END) as pc_nvt
        FROM behavioralstudy
        WHERE experiment='QuadDots'
        GROUP BY animal,session
        ORDER BY animal,session
'''
data5 = Helper.getData(cur,query5)

#look at performance as a function of singleton
query6 = '''
        SELECT animal,augend+addend-singleton as diff,singleton,chose_sum
        FROM behavioralstudy 
        WHERE experiment='FlatLO'
        ORDER BY animal,singleton,diff
'''
data6 = Helper.getData(cur,query6)

udiff_flatlo = np.unique(data6['diff'])
using_flatlo = np.unique(data6['singleton'])
cm_inds = np.linspace(0,1,len(using_flatlo))
x_sing_perf = [[np.mean(data6['chose_sum'].loc[(data6['diff']==ud) & (data6['singleton']==si) &
                (data6['animal']=='Xavier')]) for ud in udiff_flatlo] for si in using_flatlo]
x_sing_perf_sem = [[scipy.stats.sem(data6['chose_sum'].loc[(data6['diff']==ud) & (data6['singleton']==si) &
                (data6['animal']=='Xavier')]) for ud in udiff_flatlo] for si in using_flatlo]
r_sing_perf = [[np.mean(data6['chose_sum'].loc[(data6['diff']==ud) & (data6['singleton']==si) &
                (data6['animal']=='Ruffio')]) for ud in udiff_flatlo] for si in using_flatlo]
r_sing_perf_sem = [[scipy.stats.sem(data6['chose_sum'].loc[(data6['diff']==ud) & (data6['singleton']==si) &
                (data6['animal']=='Ruffio')]) for ud in udiff_flatlo] for si in using_flatlo]

#%%
#put it all in a PDF
with PdfPages('D:\\Bart\\Dropbox\\pdf_test.pdf') as pdf:
    h,ax = plt.subplots(1,2)

    #Plot p(choose sum) as a function of sum and singleton.
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
    
    #Plot psychometric curves
    h,ax = plt.subplots(1,2)
    Plotter.lineplot(ax[0],xdata=udiffs,
        ydata=x_psum,sem=x_sem,title='Monkey X',xlabel='Sum - Singleton',xticks = [-8,-4,0,4,8],
        ylabel='P(choose sum)',yticks=[0,.25,.5,.75,1],ylim=[0,1],xlim=[-8,8])
    Plotter.scatter(ax[0],xdata=udiffs,
        ydata=x_psum,identity='off')
    
    Plotter.lineplot(ax[1],xdata=udiffs,
        ydata=r_psum,sem=r_sem,title='Monkey R',xlabel='Sum - Singleton',xticks = [-8,-4,0,4,8],
        ylabel='P(choose sum)',yticks=[0,.25,.5,.75,1],ylim=[0,1],xlim=[-8,8])
    Plotter.scatter(ax[1],xdata=udiffs,
        ydata=r_psum,identity='off')
    
    plt.tight_layout()
    pdf.savefig()
    
    
    #Plot accuracy in EA/non-EA trials
    h,ax = plt.subplots(1,2)
    Plotter.scatter(ax[0],xdata=x_nea,ydata=x_ea,xlabel='Performance in non-EA',ylabel='Performance in EA',
                    title='EA performance',ylim=[.6,1],xlim=[.6,1],xticks=[.6,.8,1],yticks=[.6,.8,1],color=[1,0,0])
    Plotter.scatter(ax[0],xdata=r_nea,ydata=r_ea,xlabel='Performance in non-EA',ylabel='Performance in EA',
                    title='EA performance',ylim=[.6,1],xlim=[.6,1],xticks=[.6,.8,1],yticks=[.6,.8,1],color=[0,0,1])
    
    #plot accuracy in set1 and set2
    Plotter.scatter(ax[1],xdata=x_set1,ydata=x_set2,xlabel='Performance in set 1',ylabel='Performance in set 2',
                    title='Set performance',ylim=[.8,1],xlim=[.8,1],xticks=[.8,.9,1],yticks=[.8,.9,1],color=[1,0,0])
    Plotter.scatter(ax[1],xdata=r_set1,ydata=r_set2,xlabel='Performance in set 1',ylabel='Performance in set 2',
                    title='Set performance',ylim=[.8,1],xlim=[.8,1],xticks=[.8,.9,1],yticks=[.8,.9,1],color=[0,0,1])
    plt.tight_layout()
    pdf.savefig()
    
    #Plot coefficients in Addition experiment
    h,ax = plt.subplots(1,2)
    Plotter.scatter(ax[0],xdata=mout1['b_augend'].loc[(mout1['odd_trial']==1) & (mout1['animal']=='Xavier')],
                    ydata=mout1['b_addend'].loc[(mout1['odd_trial']==0) & (mout1['animal']=='Xavier')],
                    xlabel='Augend coefficient',ylabel='Other coefficient',color=[1,0,0],label='Addend',
                    xlim=[-2.5,2.5],ylim=[-2.5,2.5],identity='full',title='Monkey X')
    Plotter.scatter(ax[0],xdata=mout1['b_augend'].loc[(mout1['odd_trial']==0) & (mout1['animal']=='Xavier')],
                    ydata=mout1['b_singleton'].loc[(mout1['odd_trial']==1) & (mout1['animal']=='Xavier')],
                    xlabel='Augend coefficient',ylabel='Other coefficient',color=[0,0,1],label='Singleton')
    #ax[0].legend(loc='center left',fontsize='x-small',scatterpoints=1,frameon=False)
    
    Plotter.scatter(ax[1],xdata=mout1['b_augend'].loc[(mout1['odd_trial']==1) & (mout1['animal']=='Ruffio')],
                    ydata=mout1['b_addend'].loc[(mout1['odd_trial']==0) & (mout1['animal']=='Ruffio')],
                    xlabel='Augend coefficient',ylabel='Other coefficient',color=[1,0,0],label='Addend',
                    xlim=[-2.5,2.5],ylim=[-2.5,2.5],identity='full',title='Monkey R')
    Plotter.scatter(ax[1],xdata=mout1['b_augend'].loc[(mout1['odd_trial']==0) & (mout1['animal']=='Ruffio')],
                    ydata=mout1['b_singleton'].loc[(mout['odd_trial']==1) & (mout1['animal']=='Ruffio')],
                    xlabel='Augend coefficient',ylabel='Other coefficient',color=[0,0,1],label='Singleton')
    ax[1].legend(loc='center right',fontsize='x-small',scatterpoints=1,frameon=False)
    plt.tight_layout();
    pdf.savefig()
    
    #plot uni dots vs. quad dots coefficients in quad dots experiment
    h,ax = plt.subplots(2,3)
    Plotter.scatter(ax[0,0],xdata=mout2['b_aug_num_green'].loc[(mout2['odd_trial']==1) & (mout2['animal']=='Xavier')],
                    ydata=mout2['b_aug_num_quad'].loc[(mout2['odd_trial']==0) & (mout2['animal']=='Xavier')],
                    ylabel='Quad-dots coef.',
                    xlim=[-7,7],ylim=[-7,7],identity='full',title='Monkey X,Augend')
    Plotter.scatter(ax[0,1],xdata=mout2['b_add_num_green'].loc[(mout2['odd_trial']==1) & (mout2['animal']=='Xavier')],
                    ydata=mout2['b_add_num_quad'].loc[(mout2['odd_trial']==0) & (mout2['animal']=='Xavier')],
                    xlim=[-7,7],ylim=[-7,7],identity='full',title='Monkey X,Addend')
    Plotter.scatter(ax[0,2],xdata=mout2['b_sing_num_green'].loc[(mout2['odd_trial']==1) & (mout2['animal']=='Xavier')],
                    ydata=mout2['b_sing_num_quad'].loc[(mout2['odd_trial']==0) & (mout2['animal']=='Xavier')],
                    xlim=[-7,7],ylim=[-7,7],identity='full',title='Monkey X, Singleton')
    
    Plotter.scatter(ax[1,0],xdata=mout2['b_aug_num_green'].loc[(mout2['odd_trial']==1) & (mout2['animal']=='Ruffio')],
                    ydata=mout2['b_aug_num_quad'].loc[(mout2['odd_trial']==0) & (mout2['animal']=='Ruffio')],
                    xlabel='Uni-dots coef.',ylabel='Quad-dots coef.',
                    xlim=[-6.5,6.5],ylim=[-6.5,7],identity='full',title='Monkey R,Augend')
    Plotter.scatter(ax[1,1],xdata=mout2['b_add_num_green'].loc[(mout2['odd_trial']==1) & (mout2['animal']=='Ruffio')],
                    ydata=mout2['b_add_num_quad'].loc[(mout2['odd_trial']==0) & (mout2['animal']=='Ruffio')],
                    xlabel='Uni-dots coef.',
                    xlim=[-7,7],ylim=[-6.5,7],identity='full',title='Monkey R,Addend')
    Plotter.scatter(ax[1,2],xdata=mout2['b_sing_num_green'].loc[(mout2['odd_trial']==1) & (mout2['animal']=='Ruffio')],
                    ydata=mout2['b_sing_num_quad'].loc[(mout2['odd_trial']==0) & (mout2['animal']=='Ruffio')],
                    xlabel='Uni-dots coef.',
                    xlim=[-7,6.5],ylim=[-7,7],identity='full',title='Monkey R, Singleton')
    plt.tight_layout()
    pdf.savefig()
    
    #plot performance in visual trials
    h,ax = plt.subplots(1,1)
    Plotter.scatter(ax,xdata=data5['pc_vt'].loc[data5['animal']=='Xavier'],
                    ydata=data5['pc_nvt'].loc[data5['animal']=='Xavier'],
                    xlabel='Performance in VT',ylabel='Performance in non-VT',label='Monkey X',
                    ylim=[.4,1],xlim=[.4,1],xticks=[.4,.6,.8,1],yticks=[.4,.6,.8,1],color=[1,0,0])
    Plotter.scatter(ax,xdata=data5['pc_vt'].loc[data5['animal']=='Ruffio'],
                    ydata=data5['pc_nvt'].loc[data5['animal']=='Ruffio'],
                    xlabel='Performance in VT',ylabel='Performance in non-VT',label='Monkey R',
                    ylim=[.4,1],xlim=[.4,1],xticks=[.4,.6,.8,1],yticks=[.4,.6,.8,1],color=[0,0,1])
    ax.legend(loc='center left',fontsize='x-small',scatterpoints=1,frameon=False)
    plt.tight_layout()
    pdf.savefig()
    
    #plot performance as a function of singleton
    h,ax = plt.subplots(1,3)
    for si in using_flatlo:
        Plotter.lineplot(ax[0],xdata=udiff,ydata=x_sing_perf[si-1],sem=x_sing_perf_sem[si-1],
            title='Monkey X',xlabel='Sum - Singleton',xticks = [-2,-1,0,1,2],ylim=[0,1],xlim=[-2,2],
            ylabel='P(choose sum)',yticks=[0,.25,.5,.75,1],color=plt.cm.jet(cm_inds[si-1]))
        Plotter.lineplot(ax[1],xdata=udiff,ydata=r_sing_perf[si-1],sem=r_sing_perf_sem[si-1],
            title='Monkey R',xlabel='Sum - Singleton',xticks = [-2,-1,0,1,2],ylim=[0,1],xlim=[-2,2],
            ylabel='P(choose sum)',yticks=[0,.25,.5,.75,1],color=plt.cm.jet(cm_inds[si-1]))
    #setup colorbar
    cmap_singplot = mpl.colors.ListedColormap(plt.cm.jet(cm_inds))
    norm = mpl.colors.BoundaryNorm(boundaries=np.insert(using_flatlo,0,.5)+.5,ncolors=cmap_singplot.N)
    mpl.colorbar.ColorbarBase(ax[2],cmap=cmap_singplot,norm=norm,ticks=using)
    plt.tight_layout()
    pdf.savefig()