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
from matplotlib.backends.backend_pdf import PdfPages
#https://matplotlib.org/faq/howto_faq.html#save-multiple-plots-to-one-pdf-file

def scatter(ax,xdata,ydata,xlim=[],ylim=[],xlabel=[],ylabel=[],xticks=[],yticks=[],color=[1,1,1],title='',identity='on'):

    #font information
    plotfont = 'Arial'
    fontsize = 14


    #plot data
    print(ax)
    
    ax.scatter(xdata,ydata,s=40,facecolor=color,edgecolor=[0,0,0],linewidth=2)        

    #set title
    ax.set_title(title,fontname=plotfont,fontsize=fontsize)
    
    #get axis handle and set plot aspect ratio
    ax.set_aspect('equal')
    
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
        
        
    #set xticks
    if(xticks):
        ax.set_xticks(xticks)
    if(yticks):
        ax.set_yticks(yticks)
    
    
    #add axis labels
    if(xlabel):
        ax.set_xlabel(xlabel,fontname=plotfont,fontsize=fontsize)
    if(ylabel):
        ax.set_ylabel(ylabel,fontname=plotfont,fontsize=fontsize)
        
    #add identity line
    if(identity.lower() == 'on'):
        xlimits = ax.get_xlim()
        ylimits = ax.get_ylim()
        limits = (min([xlimits[0],ylimits[0]]),max([xlimits[1],ylimits[1]]))
        ax.plot(limits,limits,'--',color='black')
    else:
        pass

    
    return ax