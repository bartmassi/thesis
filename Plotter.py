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

#def scatter(xdict,ydict,color=[1,1,1],title='',identity='on',new_plot = True):
#        
#    xdata = xdict['data']
#    ydata = ydict['data']
#
#    #font information
#    plotfont = 'Arial'
#    fontsize = 14
#
#    
#    #if new_plot is true, then a new figure handle will be created and much
#    #formatting will occur.
#    if(not new_plot):
#        h = plt.gcf();
#
#        #plot data
#        plt.scatter(xdata,ydata,s=40,facecolor=color,edgecolor=[0,0,0],linewidth=2)
#        
#    else:
#        h = plt.figure();
#        #plot data
#        plt.scatter(xdata,ydata,s=40,facecolor=color,edgecolor=[0,0,0],linewidth=2)        
#
#        #set title
#        plt.title(title,fontname=plotfont,fontsize=fontsize)
#        
#        #get axis handle and set plot aspect ratio
#        ax = plt.gca()
#        ax.set_aspect('equal')
#
#        #turn off top and right frame, and tick details
#        ax.spines['top'].set_color('none')
#        ax.spines['right'].set_color('none')
#        ax.xaxis.set_ticks_position('bottom')
#        ax.yaxis.set_ticks_position('left')
#        ax.tick_params(direction='out')
#        
#        #set axis limits
#        if('limits' in xdict):
#            plt.xlim(xdict['limits'])
#            ax.set_autoscale_on(False)
#        if('limits' in ydict):
#            plt.ylim(ydict['limits'])
#            ax.set_autoscale_on(False)
#        
#        
#        #add axis labels
#        if('label' in xdict):
#            plt.xlabel(xdict['label'],fontname=plotfont,fontsize=fontsize)
#        if('label' in ydict):
#            plt.ylabel(ydict['label'],fontname=plotfont,fontsize=fontsize)
#            
#        #add identity line
#        if(identity == 'on'):
#            xlimits = plt.xlim()
#            ylimits = plt.ylim()
#            limits = (min([xlimits[0],ylimits[0]]),max([xlimits[1],ylimits[1]]))
#            plt.plot(limits,limits,'--',color='black')
#        elif(identity=='off'):
#            pass
#        else:
#            error()
#            
#    return h
    
    

def scatter(ax,xdata,ydata,xlim=[],ylim=[],xlabel=[],ylabel=[],color=[1,1,1],title='',identity='on'):
        
    xdata
    ydata

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
    
    
    #add axis labels
    if(xlabel):
        ax.set_xlabel(xlabel,fontname=plotfont,fontsize=fontsize)
    if(ylabel):
        ax.set_ylabel(ylabel,fontname=plotfont,fontsize=fontsize)
        
    #add identity line
    if(identity == 'on'):
        xlimits = ax.get_xlim()
        ylimits = ax.get_ylim()
        limits = (min([xlimits[0],ylimits[0]]),max([xlimits[1],ylimits[1]]))
        ax.plot(limits,limits,'--',color='black')
    elif(identity=='off'):
        pass
    else:
        error()
    
    return ax