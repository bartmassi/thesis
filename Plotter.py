# -*- coding: utf-8 -*-
"""

Plotter is a set of function wrappers for pyplot functions such that 
they produce plots with desired specifications.

This exists soley to reduce code complexity in scripts by encapsulating 
certain desired formatting decisions.


Created on Mon Dec 11 12:18:17 2017
@author: bart
creation date: 12-11-17
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import scipy
import pdb
#https://matplotlib.org/faq/howto_faq.html#save-multiple-plots-to-one-pdf-file

def standardize_ticks(ax,plotfont,fontsize):
    for tick in ax.get_yticklabels():
        tick.set_fontname(plotfont)
        tick.set_fontsize(fontsize)
    
    for tick in ax.get_xticklabels():
        tick.set_fontname(plotfont)
        tick.set_fontsize(fontsize)
        
    

def scatter(ax,xdata,ydata,xlim=[],ylim=[],xlabel=[],ylabel=[],xticks=[],yticks=[],
            color=[1,1,1],title=[],identity='on',label=[]):

    #font information
    plotfont = 'Arial'
    fontsize = 14


    #plot data    
    if(label):
        plothandle = ax.scatter(xdata,ydata,s=40,facecolor=color,edgecolor=[0,0,0],linewidth=2,label=label)        
    else:
        plothandle = ax.scatter(xdata,ydata,s=40,facecolor=color,edgecolor=[0,0,0],linewidth=2)    
    
        #set title
    if(title):
        ax.set_title(title,fontname=plotfont,fontsize=fontsize)
    
    
    
    #turn off top and right frame, and tick details
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(direction='out')
    
    #set axis limits
    if(xlim):
        ax.set_xlim(xlim)
        ax.set_autoscale_on(False)
    if(ylim):
        ax.set_ylim(ylim)
        ax.set_autoscale_on(False)
    
#    #make plot a square
    ylimits = ax.get_ylim()
    xlimits = ax.get_xlim()
#    yaxissize = -np.diff(ylimits)
#    xaxissize = -np.diff(xlimits)
#    #get axis handle and set plot aspect ratio
#    #ax.set_aspect('equal')
#    #ax.axis('equal')
#    ax.set_adjustable('box')
#    ax.set_aspect((xaxissize/yaxissize)[0])
        
        
    #set xticks
    if(xticks):
        ax.set_xticks(xticks)
    if(yticks):
        ax.set_yticks(yticks)
    
    standardize_ticks(ax,plotfont,fontsize)
    
    #add axis labels
    if(xlabel):
        ax.set_xlabel(xlabel,fontname=plotfont,fontsize=fontsize)
    if(ylabel):
        ax.set_ylabel(ylabel,fontname=plotfont,fontsize=fontsize)
        
        
    #add identity line
    if(identity.lower() == 'on'):
        #xlimits = ax.get_xlim()
        #ylimits = ax.get_ylim()
        limits = (min([xlimits[0],ylimits[0]]),max([xlimits[1],ylimits[1]]))
        ax.plot(limits,limits,'--',color='black')
    if(identity.lower() == 'full'):
        identityline = (min([xlimits[0],ylimits[0]]),max([xlimits[1],ylimits[1]]))
        invidentityline = (max([xlimits[1],ylimits[1]]),min([xlimits[0],ylimits[0]]))
        ax.plot(identityline,invidentityline,'--',color='black')
    if(identity.lower() == 'cross' or identity.lower() == 'full'):
        ax.plot(xlimits,[0,0],'--',color='black')
        ax.plot([0,0],ylimits,'--',color='black')
    else:
        pass

    #make plot a square
    ylimits = ax.get_ylim()
    xlimits = ax.get_xlim()
    yaxissize = -np.diff(ylimits)
    xaxissize = -np.diff(xlimits)
    #get axis handle and set plot aspect ratio
    #ax.set_aspect('equal')
    #ax.axis('equal')
    ax.set_adjustable('box')
    ax.set_aspect((xaxissize/yaxissize)[0])
    
    return plothandle

#makes a line plot with desired specifications
def lineplot(ax,xdata,ydata,sem=None,xlim=[],ylim=[],ls='solid',xlabel=[],ylabel=[],
             xticks=[],yticks=[],color=[0,0,0],title=[],label=[]):

    #font information
    plotfont = 'Arial'
    fontsize = 14


    #plot data
    
    if(label):
        plothandle = ax.errorbar(xdata,ydata,yerr=sem,xerr=None,ls=ls,color=color,linewidth=2,label=label)
    else:
        plothandle = ax.errorbar(xdata,ydata,yerr=sem,xerr=None,ls=ls,color=color,linewidth=2)      

    #set title
    if(title):
        ax.set_title(title,fontname=plotfont,fontsize=fontsize)
        

    #turn off top and right frame, and tick details
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(direction='out')
    
    #set axis limits
    if(xlim):
        ax.set_xlim(xlim)
        ax.set_autoscale_on(False)
    if(ylim):
        ax.set_ylim(ylim)
        ax.set_autoscale_on(False)
        
    #make plot a square
    ylimits = ax.get_ylim()
    xlimits = ax.get_xlim()
    yaxissize = -np.diff(ylimits)
    xaxissize = -np.diff(xlimits)
    #get axis handle and set plot aspect ratio
    #ax.set_aspect('equal')
    #ax.axis('equal')
    ax.set_adjustable('box')
    ax.set_aspect((xaxissize/yaxissize)[0])
        
    #set xticks
    if(xticks):
        ax.set_xticks(xticks)
    if(yticks):
        ax.set_yticks(yticks)
    
    #set tick font info
    standardize_ticks(ax,plotfont,fontsize)
        
    #add axis labels
    if(xlabel):
        ax.set_xlabel(xlabel,fontname=plotfont,fontsize=fontsize)
    if(ylabel):
        ax.set_ylabel(ylabel,fontname=plotfont,fontsize=fontsize)

    
    return plothandle
    

def gridplot(ax,datamat,title=[],xticks=[],yticks=[],xticklabels=[],yticklabels=[],xlabel=[],ylabel=[],
             cmap=plt.cm.jet,clim=[0,1],cticks=[],cticklabels=[],clabel=[]):
    
    #font information
    plotfont = 'Arial'
    fontsize = 14

    #make plot
    cax = ax.matshow(datamat,origin='lower',cmap=cmap)
    cax.set_clim(clim)
    
    ax.xaxis.set_ticks_position('bottom')
    
    #setup color bar
    cbar = plt.gcf().colorbar(cax,ax=ax,ticks=cticks,fraction=0.046, pad=0.04)
    cbar.ax.set_xticklabels([])
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontsize(fontsize)
        tick.set_fontname(plotfont)
    if(cticklabels):
        cbar.ax.set_yticklabels(cticklabels,fontsize=fontsize,fontname=plotfont)
    if(clabel):
        cbar.ax.set_ylabel(clabel,fontsize=fontsize,fontname=plotfont)
        cbar.ax.yaxis.set_label_position('right')
    
    #set title
    if(title):
        ax.set_title(title,fontname=plotfont,fontsize=fontsize)
    
    #set x and y axis ticks
    if(not isinstance(xticks,list)):
        xticks = xticks.tolist()
    if(not isinstance(yticks,list)):
        yticks = yticks.tolist()
    
    if(xticks):
        ax.set_xticks(xticks)
    if(yticks):
        ax.set_yticks(yticks)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(direction='out')
    
    #set tick labels
    if(not isinstance(xticklabels,list)):
        xticklabels = xticklabels.tolist()
    if(not isinstance(yticklabels,list)):
        yticklabels = yticklabels.tolist()
        
    if(xticklabels):
        ax.set_xticklabels(xticklabels)
    if(yticklabels):
        ax.set_yticklabels(yticklabels)
        
    standardize_ticks(ax,plotfont,fontsize)
    
    #add axis labels
    if(xlabel):
        ax.set_xlabel(xlabel,fontname=plotfont,fontsize=fontsize)
    if(ylabel):
        ax.set_ylabel(ylabel,fontname=plotfont,fontsize=fontsize)
#%%
#Below are functions that use the above plotting functions to make more complicated plots.

#This uses 3 groupby variables to plot a DV against one IV in subplots, a
#second IV in separate lines, and a third IV on the x-axis. 
def panelplots(data,plotvar,groupby,axes=None,scattervar=[],xlim=[],ylim=[],xlabel=[],ylabel=[],
               xticks=[],yticks=[],horiz=None,maxcol=3.0,legend='on'):
    
    assert len(groupby)==3

    if(not scattervar):
        scattervar = plotvar

    #get unique values of each groupby variable
    #groupby_unique = data[groupby].drop_duplicates()#unique values of groupby variable
    marginal_groupby_unique = []
    for gb in groupby:
        ustorage = data[gb].loc[~np.isnan(data[gb])].drop_duplicates().sort_values(inplace=False)
        marginal_groupby_unique.append(ustorage)
        #throw out nan values
        
    
    #get plot meta-data
    n_subplot = len(marginal_groupby_unique[0])
    n_lines = len(marginal_groupby_unique[1])
    n_x = len(marginal_groupby_unique[2])
    
    #setup colormap 
    if(n_lines>1):
        cmap = plt.get_cmap('cool');
        cmap_index = np.arange(0, 1+1/(n_lines-1), 1/(n_lines-1))
    else:
        cmap = plt.get_cmap('cool')
        cmap_index = np.array([1])
    #determine subplot dimensions
    nrow = int(np.ceil(n_subplot/maxcol));
    ncol = int(np.min([n_subplot,maxcol]))
    
    #make the plots
    if(axes is None):
        h,axes = plt.subplots(nrow,ncol,figsize=[6*ncol,6*nrow])
    for i in range(0,n_subplot):
        if(nrow==1):
            ax = axes[i]
        else:
            ax = axes[int(np.floor(i/ncol)),i % ncol]
        for j in range(0,n_lines):
            
            #get out the groupby data
            ydata1 = []
            ydata2 = []
            yerr = []
            for k in range(0,n_x):
                #collect scatter data
                #pdb.set_trace()
                scatterdata = data[scattervar].loc[np.sum(data[groupby] == 
                                    np.array([marginal_groupby_unique[0].iloc[i],
                                    marginal_groupby_unique[1].iloc[j],
                                    marginal_groupby_unique[2].iloc[k]]),axis=1)==len(groupby)]
                plotdata = data[plotvar].loc[np.sum(data[groupby] == 
                                    np.array([marginal_groupby_unique[0].iloc[i],
                                    marginal_groupby_unique[1].iloc[j],
                                    marginal_groupby_unique[2].iloc[k]]),axis=1)==len(groupby)]
                                
                #compute the mean and SEM - this might be removed in the future 
                #to enforce the data table to contain SEM and mean data explicitly.
                
                ydata1.append(np.mean(scatterdata))
                ydata2.append(np.mean(plotdata))
                yerr.append(scipy.stats.sem(scatterdata))
                
            #intelligently generate xlimits if they are not passed
            if(not xlim):
                
                ynotnan = np.where(~np.isnan(ydata1))
                bottomind = np.min(ynotnan)
                topind = np.max(ynotnan)
                xl = [marginal_groupby_unique[2].iloc[bottomind],marginal_groupby_unique[2].iloc[topind]]
                xt = marginal_groupby_unique[2].iloc[ynotnan].tolist()
                xl[0] = xl[0] - (xl[1]-xl[0])*.1      #10% of the space should be dead on the left    
                xl[1] = xl[1] + (xl[1]-xl[0])*.1      #10% of the space should be dead on the right
            else:
                xl = xlim
                xt = xticks
            
            #pdb.set_trace()
            #plot all data
            lineplot(ax,xdata=marginal_groupby_unique[2],ydata=ydata1,sem=yerr,
                     ls='none',color=cmap(cmap_index[j]),ylim=ylim,xlim=xl,ylabel=ylabel,xlabel=xlabel,
                                xticks=xt,yticks=yticks)
            
            scatter(ax,xdata=marginal_groupby_unique[2],ydata=ydata1
                     ,color=cmap(cmap_index[j]),ylim=ylim,xlim=xl,ylabel=ylabel,xlabel=xlabel,identity='off',
                                label=groupby[1]+'='+str(marginal_groupby_unique[1].iloc[j]),
                                title=groupby[0]+'='+str(marginal_groupby_unique[0].iloc[i]),
                                xticks=xt,yticks=yticks)
            
            nanmask = np.isnan(ydata2)
            #pdb.set_trace()
            
            lineplot(ax,xdata=marginal_groupby_unique[2].loc[~nanmask],ydata=np.array(ydata2)[~nanmask],
                          color=cmap(cmap_index[j]),ylim=ylim,xlim=xl,ylabel=ylabel,xlabel=xlabel,
                                xticks=xt,yticks=yticks)
            
            if(horiz is not None):
                ax.plot(xlim,[horiz for x in xlim],'--',color='black')
        #turn on legend
        if(legend.lower() == 'on'):
            ax.legend(fontsize='x-small',loc=0)
