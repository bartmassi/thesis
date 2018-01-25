# -*- coding: utf-8 -*-
"""
Performs statistical analysis on dataset. All results to appear in thesis.
Stats are stored in a list with information about df, test stats, p-values, and test descriptions.

@author: bart
"""
#
#%load_ext autoreload
#%autoreload 2

import Plotter
import Analyzer
import Helper
import sqlite3
import pandas as pd
import numpy as np
import scipy

dbfloc = 'D:\\Bart\\Dropbox\\science\\leelab\\projects\\Arithmetic\\data\\_curatedData\\'
conn = sqlite3.connect(database=dbfloc+'arithmeticstudy.db')
cur = conn.cursor()

tests = [] #list to store output of tests

#==============================================EXP1

###############
description1 = 'one-sample t-test on sum - singleton coef in logistic regression'
query1 = '''
        SELECT augend,addend,singleton,augend+addend-singleton as diff,
        chose_sum,session,trial,trialset,animal
        FROM behavioralstudy
        WHERE experiment='Addition'
        ORDER BY animal,session,trial
'''
data1 = Helper.getData(cur,query1)
model1 = 'chose_sum ~ diff'
mout1 = Analyzer.logistic_regression(data1,model1,groupby=['animal','session'])
t1_x = scipy.stats.ttest_1samp(mout1['b_diff'].loc[mout1['animal']=='Xavier'],0)
n1_x = len(mout1['b_diff'].loc[mout1['animal']=='Xavier'])
t1_r = scipy.stats.ttest_1samp(mout1['b_diff'].loc[mout1['animal']=='Ruffio'],0)
n1_r = len(mout1['b_diff'].loc[mout1['animal']=='Ruffio'])
tests.append({'description':description1+'(Xavier)','p':t1_x.pvalue,'stat':t1_x.statistic,
              'mean':np.mean(mout1['b_diff'].loc[mout1['animal']=='Xavier']),'n':n1_x,'df':n1_x-1})
tests.append({'description':description1+'(Ruffio)','p':t1_r.pvalue,'stat':t1_r.statistic,
              'mean':np.mean(mout1['b_diff'].loc[mout1['animal']=='Ruffio']),'n':n1_r,'df':n1_r-1})


###############
description2 = 't-test on set 1 vs. set 2 accuracy for both animals'
query2 = '''
        SELECT animal,session,
        AVG(CASE WHEN trialset=1 THEN (chose_sum = ((augend+addend)>singleton)) ELSE NULL END) as set1perf,
        AVG(CASE WHEN trialset=2 THEN (chose_sum = ((augend+addend)>singleton)) ELSE NULL END) as set2perf
        FROM behavioralstudy
        WHERE experiment='Addition'
        GROUP BY animal,session
        HAVING set2perf IS NOT NULL
        ORDER BY animal,session
'''
data2 = Helper.getData(cur,query2)
t2_x = scipy.stats.ttest_rel(data2['set1perf'].loc[data2['animal']=='Xavier'],
                             data2['set2perf'].loc[data2['animal']=='Xavier'])
n2_x = len(data2['set2perf'].loc[data2['animal']=='Xavier'])
t2_r = scipy.stats.ttest_rel(data2['set1perf'].loc[data2['animal']=='Ruffio'],
                             data2['set2perf'].loc[data2['animal']=='Ruffio'])
n2_r = len(data2['set2perf'].loc[data2['animal']=='Ruffio'])
tests.append({'description':description2+'(Xavier)','p':t2_x.pvalue,'stat':t2_x.statistic,
              'mean':(np.mean(data2['set1perf'].loc[data2['animal']=='Xavier']),np.mean(data2['set2perf'].loc[data2['animal']=='Xavier'])),'n':n2_x,'df':n2_x-1})
tests.append({'description':description2+'(Ruffio)','p':t2_r.pvalue,'stat':t2_r.statistic,
              'mean':(np.mean(data2['set1perf'].loc[data2['animal']=='Ruffio']),np.mean(data2['set2perf'].loc[data2['animal']=='Ruffio'])),'n':n2_r,'df':n2_r-1})

###############
description3 = 'one sample t-test on set 2 accuracy vs. chance'
data3 = data2
t3_x = scipy.stats.ttest_1samp(data3['set2perf'].loc[data3['animal']=='Xavier'],.5)
n3_x = n2_x
t3_r = scipy.stats.ttest_1samp(data3['set2perf'].loc[data3['animal']=='Ruffio'],.5)
n3_r = n2_r
tests.append({'description':description3+'(Xavier)','p':t3_x.pvalue,'stat':t3_x.statistic,
              'mean':np.mean(data3['set2perf'].loc[data3['animal']=='Xavier']),'n':n3_x,'df':n3_x-1})
tests.append({'description':description3+'(Ruffio)','p':t3_r.pvalue,'stat':t3_r.statistic,
              'mean':np.mean(data3['set2perf'].loc[data3['animal']=='Ruffio']),'n':n3_r,'df':n3_r-1})


###############
description4 = 'binomial test on set 2 accuracy on first day'
query4 = '''
        SELECT animal,session,(chose_sum = ((augend+addend)>singleton)) as correct
        FROM behavioralstudy
        WHERE trialset=2 AND experiment='Addition' AND
        ((animal='Xavier' AND session=(SELECT session FROM behavioralstudy WHERE trialset=2 AND experiment='Addition' AND animal='Xavier' ORDER BY session LIMIT 1))
        OR (animal='Ruffio' AND session=(SELECT session FROM behavioralstudy WHERE trialset=2 AND experiment='Addition' AND animal='Ruffio' ORDER BY session LIMIT 1)))
        ORDER BY animal,session
        '''
data4 = Helper.getData(cur,query4)

n4_x = len(data4['correct'].loc[data4['animal']=='Xavier'])
n4_r = len(data4['correct'].loc[data4['animal']=='Ruffio'])
mux = np.sum(data4['correct'].loc[data4['animal']=='Xavier'])
mur = np.sum(data4['correct'].loc[data4['animal']=='Ruffio'])
t4_x = scipy.stats.binom_test(mux,n4_x,.5)
t4_r = scipy.stats.binom_test(mur,n4_r,.5)
tests.append({'description':description4+'(Xavier)','p':t4_x,'stat':None,'mean':mux/n4_x,
              'n':n4_x,'df':None})
tests.append({'description':description4+'(Ruffio)','p':t4_r,'stat':None,'mean':mur/n4_r,
              'n':n4_r,'df':None})

###############
description5 = 'EA vs. non-EA accuracy for both animals'
query5 = '''
        SELECT animal,session,
        AVG(CASE WHEN (((augend+addend)>singleton) AND (augend<singleton) AND (addend<singleton)) THEN (chose_sum = ((augend+addend)>singleton)) ELSE NULL END) as eaperf,
        AVG(CASE WHEN NOT (((augend+addend)>singleton) AND (augend<singleton) AND (addend<singleton)) THEN (chose_sum = ((augend+addend)>singleton)) ELSE NULL END) as noneaperf,
        SUM((((augend+addend)>singleton) AND (augend<singleton) AND (addend<singleton))) as totalea,
        SUM(NOT (((augend+addend)>singleton) AND (augend<singleton) AND (addend<singleton))) as totalnonea
        FROM behavioralstudy
        WHERE experiment='Addition'
        GROUP BY animal,session
        HAVING eaperf IS NOT NULL
        ORDER BY animal,session
'''
data5 = Helper.getData(cur,query5)
t5_x = scipy.stats.ttest_rel(data5['eaperf'].loc[data5['animal']=='Xavier'],
                             data5['noneaperf'].loc[data5['animal']=='Xavier'])
n5_x = len(data5['eaperf'].loc[data5['animal']=='Xavier'])
t5_r = scipy.stats.ttest_rel(data5['eaperf'].loc[data5['animal']=='Ruffio'],
                             data5['noneaperf'].loc[data5['animal']=='Ruffio'])
n5_r = len(data5['eaperf'].loc[data5['animal']=='Ruffio'])
tests.append({'description':description5+'(Xavier)','p':t5_x.pvalue,'stat':t5_x.statistic,
              'mean':(np.mean(data5['eaperf'].loc[data5['animal']=='Xavier']),np.mean(data5['noneaperf'].loc[data5['animal']=='Xavier'])),'n':n5_x,'df':n5_x-1})
tests.append({'description':description5+'(Ruffio)','p':t5_r.pvalue,'stat':t5_r.statistic,
              'mean':(np.mean(data5['eaperf'].loc[data5['animal']=='Ruffio']),np.mean(data5['noneaperf'].loc[data5['animal']=='Ruffio'])),'n':n5_r,'df':n5_r-1})

###############
description6 = 'binomial test on EA acc vs. chance in each session'
data6 = data5
#do binomial tests on EA acc vs. chance on every session
btest6x = np.array([scipy.stats.binom_test(x*y,y) for x,y in zip(data6['eaperf'].loc[data6['animal']=='Xavier'],
 data6['totalea'].loc[data6['animal']=='Xavier'])])
btest6r = np.array([scipy.stats.binom_test(x*y,y) for x,y in zip(data6['eaperf'].loc[data6['animal']=='Ruffio'],
     data6['totalea'].loc[data6['animal']=='Ruffio'])])
#do binomial test on p(EA>chance in session)
n6_x = len(btest6x)
n6_r = len(btest6r)
t6_x = scipy.stats.binom_test(np.sum(btest6x<.05),n6_x,.05)
t6_r = scipy.stats.binom_test(np.sum(btest6r<.05),n6_r,.05)
tests.append({'description':description6+'(Xavier)','p':t6_x,'stat':None,'mean':np.sum(btest6x<.05)/n6_x,
              'n':n6_x,'df':None})
tests.append({'description':description6+'(Ruffio)','p':t6_r,'stat':None,'mean':np.sum(btest6r<.05)/n6_r,
              'n':n6_r,'df':None})

###############
description7 = 'augend vs. addend coef, t-test'
data7 = data1

model7 = 'chose_sum ~ augend + addend + singleton'
mout7 = Analyzer.logistic_regression(data7,model7,groupby=['animal','session'])
aug7x = mout7['b_augend'].loc[(mout7['animal']=='Xavier')]
add7x = mout7['b_addend'].loc[(mout7['animal']=='Xavier')]
aug7r = mout7['b_augend'].loc[(mout7['animal']=='Ruffio')]
add7r = mout7['b_addend'].loc[(mout7['animal']=='Ruffio')]

n7_x = len(aug7x)
n7_r = len(aug7r)
t7_x = scipy.stats.ttest_rel(aug7x,add7x)
t7_r = scipy.stats.ttest_rel(aug7r,add7r)
tests.append({'description':description7+'(Xavier)','p':t7_x.pvalue,'stat':t7_x.statistic,'mean':(np.mean(aug7x),np.mean(add7x)),
              'n':n7_x,'df':n7_x-1})
tests.append({'description':description7+'(Ruffio)','p':t7_r.pvalue,'stat':t7_r.statistic,'mean':(np.mean(aug7r),np.mean(add7r)),
              'n':n7_r,'df':n7_r-1})

###############
description8 = 'augend vs. singleton coef, t-tests'
data8 = data7
mout8 = mout7
aug8x = aug7x
sing8x = mout8['b_singleton'].loc[(mout8['animal']=='Xavier')]
aug8r = aug7r
sing8r = mout8['b_singleton'].loc[(mout8['animal']=='Ruffio')]
n8_x = n7_x
n8_r = n7_r
t8_x = scipy.stats.ttest_rel(aug8x,np.abs(sing8x))
t8_r = scipy.stats.ttest_rel(aug8r,np.abs(sing8r))
tests.append({'description':description8+'(Xavier)','p':t8_x.pvalue,'stat':t8_x.statistic,'mean':(np.mean(aug8x),np.mean(sing8x)),
              'n':n8_x,'df':n8_x-1})
tests.append({'description':description8+'(Ruffio)','p':t8_r.pvalue,'stat':t8_r.statistic,'mean':(np.mean(aug8r),np.mean(sing8r)),
              'n':n8_r,'df':n8_r-1})


#==============================================EXP2
###############
description9 = 't-test on quad vs. uni coef for aug, add, sing'
query9 = '''
        SELECT animal,session,aug_num_green,add_num_green,sing_num_green,
        aug_num_quad,add_num_quad,sing_num_quad,chose_sum
        FROM behavioralstudy
        WHERE experiment='QuadDots'
        ORDER BY animal,session
'''
data9 = Helper.getData(cur,query9)

#perform simple logistic regression
model9 = 'chose_sum ~ aug_num_green + add_num_green + sing_num_green \
            + aug_num_quad + add_num_quad + sing_num_quad'
mout9 = Analyzer.logistic_regression(data9,model9,groupby=['animal','session'])

auguni9x = mout9['b_aug_num_green'].loc[(mout9['animal']=='Xavier')]
augquad9x = mout9['b_aug_num_quad'].loc[(mout9['animal']=='Xavier')]
adduni9x = mout9['b_add_num_green'].loc[(mout9['animal']=='Xavier')]
addquad9x = mout9['b_add_num_quad'].loc[(mout9['animal']=='Xavier')]
singuni9x = mout9['b_sing_num_green'].loc[(mout9['animal']=='Xavier')]
singquad9x = mout9['b_sing_num_quad'].loc[(mout9['animal']=='Xavier')]

auguni9r = mout9['b_aug_num_green'].loc[(mout9['animal']=='Ruffio')]
augquad9r = mout9['b_aug_num_quad'].loc[(mout9['animal']=='Ruffio')]
adduni9r = mout9['b_add_num_green'].loc[(mout9['animal']=='Ruffio')]
addquad9r = mout9['b_add_num_quad'].loc[(mout9['animal']=='Ruffio')]
singuni9r = mout9['b_sing_num_green'].loc[(mout9['animal']=='Ruffio')]
singquad9r = mout9['b_sing_num_quad'].loc[(mout9['animal']=='Ruffio')]

n9_aug_x = len(auguni9x)
t9_aug_x = scipy.stats.ttest_rel(auguni9x,augquad9x)
n9_aug_r = len(auguni9r)
t9_aug_r = scipy.stats.ttest_rel(auguni9r,augquad9r)

n9_add_x = len(adduni9x)
t9_add_x = scipy.stats.ttest_rel(adduni9x,addquad9x)
n9_add_r = len(adduni9r)
t9_add_r = scipy.stats.ttest_rel(adduni9r,addquad9r)

n9_sing_x = len(singuni9x)
t9_sing_x = scipy.stats.ttest_rel(singuni9x,singquad9x)
n9_sing_r = len(singuni9r)
t9_sing_r = scipy.stats.ttest_rel(singuni9r,singquad9r)

tests.append({'description':description9+', Augend (Xavier)','p':t9_aug_x.pvalue,'stat':t9_aug_x.statistic,'mean':(np.mean(auguni9x),np.mean(augquad9x)),
              'n':n9_aug_x,'df':n9_aug_x-1})
tests.append({'description':description9+', Augend (Ruffio)','p':t9_aug_r.pvalue,'stat':t9_aug_r.statistic,'mean':(np.mean(auguni9r),np.mean(augquad9r)),
              'n':n9_aug_r,'df':n9_aug_r-1})
tests.append({'description':description9+', Addend (Xavier)','p':t9_add_x.pvalue,'stat':t9_add_x.statistic,'mean':(np.mean(adduni9x),np.mean(addquad9x)),
              'n':n9_add_x,'df':n9_add_x-1})
tests.append({'description':description9+', Addend (Ruffio)','p':t9_add_r.pvalue,'stat':t9_add_r.statistic,'mean':(np.mean(adduni9r),np.mean(addquad9r)),
              'n':n9_add_r,'df':n9_add_r-1})
tests.append({'description':description9+', Singleton (Xavier)','p':t9_sing_x.pvalue,'stat':t9_sing_x.statistic,'mean':(np.mean(singuni9x),np.mean(singquad9x)),
              'n':n9_sing_x,'df':n9_sing_x-1})
tests.append({'description':description9+', Singleton (Ruffio)','p':t9_sing_r.pvalue,'stat':t9_sing_r.statistic,'mean':(np.mean(singuni9r),np.mean(singquad9r)),
              'n':n9_sing_r,'df':n9_sing_r-1})

###############
description10='one-sample t-test on quad/uni ratio vs. 4 in logistic regression (excluding first 2 sessions)'
data10 = data9

#exclude first 2 sessions
augratio10_x = augquad9x.iloc[2:-1]/auguni9x.iloc[2:-1]
augratio10_r = augquad9r.iloc[2:-1]/auguni9r.iloc[2:-1]
addratio10_x = addquad9x.iloc[2:-1]/adduni9x.iloc[2:-1]
addratio10_r = addquad9r.iloc[2:-1]/adduni9r.iloc[2:-1]
singratio10_x = singquad9x.iloc[2:-1]/singuni9x.iloc[2:-1]
singratio10_r = singquad9r.iloc[2:-1]/singuni9r.iloc[2:-1]

n10_x = len(augratio10_x)
n10_r = len(augratio10_r)

t10_aug_x = scipy.stats.ttest_1samp(augratio10_x,4)
t10_aug_r = scipy.stats.ttest_1samp(augratio10_r,4)
t10_add_x = scipy.stats.ttest_1samp(addratio10_x,4)
t10_add_r = scipy.stats.ttest_1samp(addratio10_r,4)
t10_sing_x = scipy.stats.ttest_1samp(singratio10_x,4)
t10_sing_r = scipy.stats.ttest_1samp(singratio10_r,4)

tests.append({'description':description10+', Augend (Xavier)','p':t10_aug_x.pvalue,'stat':t10_aug_x.statistic,'mean':np.mean(augratio10_x),'sem':scipy.stats.sem(augratio10_x),
              'n':n10_x,'df':n10_x-1})
tests.append({'description':description10+', Augend (Ruffio)','p':t10_aug_r.pvalue,'stat':t10_aug_r.statistic,'mean':np.mean(augratio10_r),'sem':scipy.stats.sem(augratio10_r),
              'n':n10_r,'df':n10_r-1})
tests.append({'description':description10+', Addend (Xavier)','p':t10_add_x.pvalue,'stat':t10_add_x.statistic,'mean':np.mean(addratio10_x),'sem':scipy.stats.sem(addratio10_x),
              'n':n10_x,'df':n10_x-1})
tests.append({'description':description10+', Addend (Ruffio)','p':t10_add_r.pvalue,'stat':t10_add_r.statistic,'mean':np.mean(addratio10_r),'sem':scipy.stats.sem(addratio10_r),
              'n':n10_r,'df':n10_r-1})
tests.append({'description':description10+', Singleton (Xavier)','p':t10_sing_x.pvalue,'stat':t10_sing_x.statistic,'mean':np.mean(singratio10_x),'sem':scipy.stats.sem(singratio10_x),
              'n':n10_x,'df':n10_x-1})
tests.append({'description':description10+', Singleton (Ruffio)','p':t10_sing_r.pvalue,'stat':t10_sing_r.statistic,'mean':np.mean(singratio10_r),'sem':scipy.stats.sem(singratio10_r),
              'n':n10_r,'df':n10_r-1})

###############
description11 = 't-test on VT vs. non-VT accuracy for both animals'
query11 = '''
        SELECT animal,session,
        AVG(CASE WHEN ((aug_num_green+aug_num_quad+add_num_green+add_num_quad) > (sing_num_green+sing_num_quad))==((augend+addend)>singleton)
        THEN chose_sum == ((augend+addend)>singleton) ELSE NULL END) AS pc_vt,
        AVG(CASE WHEN ((aug_num_green+aug_num_quad+add_num_green+add_num_quad) > (sing_num_green+sing_num_quad))!=((augend+addend)>singleton)
        THEN chose_sum == ((augend+addend)>singleton) ELSE NULL END) AS pc_nvt,
        SUM(((aug_num_green+aug_num_quad+add_num_green+add_num_quad) > (sing_num_green+sing_num_quad))==((augend+addend)>singleton)) AS totalvt,
        SUM(((aug_num_green+aug_num_quad+add_num_green+add_num_quad) > (sing_num_green+sing_num_quad))!=((augend+addend)>singleton)) AS totalnvt,
        COUNT(aug_num_green) AS total
        FROM behavioralstudy
        WHERE experiment='QuadDots'
        GROUP BY animal,session
        ORDER BY animal,session
'''
data11 = Helper.getData(cur,query11)

vtperf11x = data11['pc_vt'].loc[data11['animal']=='Xavier']
vtperf11r = data11['pc_vt'].loc[data11['animal']=='Ruffio']
nvtperf11x = data11['pc_nvt'].loc[data11['animal']=='Xavier']
nvtperf11r = data11['pc_nvt'].loc[data11['animal']=='Ruffio']

n11_x = len(vtperf11x)
n11_r = len(vtperf11r)
t11_x = scipy.stats.ttest_rel(vtperf11x,nvtperf11x)
t11_r = scipy.stats.ttest_rel(vtperf11r,nvtperf11r)
tests.append({'description':description11+', (Xavier)','p':t11_x.pvalue,'stat':t11_x.statistic,'mean':(np.mean(vtperf11x),np.mean(nvtperf11x)),
              'n':n11_x,'df':n11_x-1})
tests.append({'description':description11+', (Ruffio)','p':t11_r.pvalue,'stat':t11_r.statistic,'mean':(np.mean(vtperf11r),np.mean(nvtperf11r)),
              'n':n11_r,'df':n11_r-1})

###############
description12 = 'binomial test on NVT acc vs. chance in each session'
data12 = data11

btest12x = np.array([scipy.stats.binom_test(x*y,y,.5) for x,y in zip(data12['pc_nvt'].loc[data12['animal']=='Xavier'],
 data12['totalnvt'].loc[data12['animal']=='Xavier'])])
btest12r = np.array([scipy.stats.binom_test(x*y,y,.5) for x,y in zip(data12['pc_nvt'].loc[data12['animal']=='Ruffio'],
 data12['totalnvt'].loc[data12['animal']=='Ruffio'])])

n12_x = len(btest12x)
n12_r = len(btest12r)
t12_x = scipy.stats.binom_test(np.sum(btest12x<.05),n12_x,.05)
t12_r = scipy.stats.binom_test(np.sum(btest12r<.05),n12_r,.05)
tests.append({'description':description12+'(Xavier)','p':t12_x,'stat':None,'mean':np.sum(btest12x<.05)/n12_x,
              'n':n12_x,'df':None})
tests.append({'description':description12+'(Ruffio)','p':t12_r,'stat':None,'mean':np.sum(btest12r<.05)/n12_r,
              'n':n12_r,'df':None})


#==============================================EXP3
###############
description13 = 't-test on abs(SRC) magnitude for sum-singleton and singleton'
query13 = '''
        SELECT animal,session,augend,addend,singleton,augend+addend-singleton AS diff,
        chose_sum,aug_num_green,add_num_green,sing_num_green,
        aug_num_quad,add_num_quad,sing_num_quad
        FROM behavioralstudy
        WHERE experiment='FlatLO'
        ORDER BY animal,session
'''

data13 = Helper.getData(cur,query13)
model13 = 'chose_sum ~ diff+singleton'
mout13 = Analyzer.logistic_regression(data13,model13,groupby=['animal','session'],standardize=True)

bdiff13x = mout13['b_diff'].loc[mout13['animal']=='Xavier']
bdiff13r = mout13['b_diff'].loc[mout13['animal']=='Ruffio']
bsing13x = mout13['b_singleton'].loc[mout13['animal']=='Xavier']
bsing13r = mout13['b_singleton'].loc[mout13['animal']=='Ruffio']
n13_x = len(bdiff13x)
n13_r = len(bdiff13r)

t13_x = scipy.stats.ttest_rel(bdiff13x,np.abs(bsing13x))
t13_r = scipy.stats.ttest_rel(bdiff13r,np.abs(bsing13r))
tests.append({'description':description13+', (Xavier)','p':t13_x.pvalue,'stat':t13_x.statistic,'mean':(np.mean(bdiff13x),np.mean(np.abs(bsing13x))),
              'n':n13_x,'df':n13_x-1})
tests.append({'description':description13+', (Ruffio)','p':t13_r.pvalue,'stat':t13_r.statistic,'mean':(np.mean(bdiff13r),np.mean(np.abs(bsing13r))),
              'n':n13_r,'df':n13_r-1})


###############
description14 = 'chi square on significance of sum-singleton and singleton in LR'
data14 = data13
mout14 = mout13

contx = np.array([[np.sum((mout14['p_diff'].loc[mout14['animal']=='Xavier']<.05) & (mout14['p_singleton'].loc[mout14['animal']=='Xavier']<.05)),
  np.sum((mout14['p_diff'].loc[mout14['animal']=='Xavier']<.05) & (mout14['p_singleton'].loc[mout14['animal']=='Xavier']>.05))],
  [np.sum((mout14['p_diff'].loc[mout14['animal']=='Xavier']>.05) & (mout14['p_singleton'].loc[mout14['animal']=='Xavier']<.05)),
   np.sum((mout14['p_diff'].loc[mout14['animal']=='Xavier']>.05) & (mout14['p_singleton'].loc[mout14['animal']=='Xavier']>.05))]])

#g,p14_x,dof,expect = scipy.stats.chi2_contingency(contx)
p14_x = 'undef'
contr = np.array([[np.sum((mout14['p_diff'].loc[mout14['animal']=='Ruffio']<.05) & (mout14['p_singleton'].loc[mout14['animal']=='Ruffio']<.05)),
  np.sum((mout14['p_diff'].loc[mout14['animal']=='Ruffio']<.05) & (mout14['p_singleton'].loc[mout14['animal']=='Ruffio']>.05))],
  [np.sum((mout14['p_diff'].loc[mout14['animal']=='Ruffio']>.05) & (mout14['p_singleton'].loc[mout14['animal']=='Ruffio']<.05)),
   np.sum((mout14['p_diff'].loc[mout14['animal']=='Ruffio']>.05) & (mout14['p_singleton'].loc[mout14['animal']=='Ruffio']>.05))]])

g,p14_r,dof,expect = scipy.stats.chi2_contingency(contr)

tests.append({'description':description14+'(Xavier)','p':p14_x,'stat':None,'mean':contx,
              'n':len(mout14['p_diff'].loc[mout14['animal']=='Xavier']),'df':dof})
tests.append({'description':description14+'(Ruffio)','p':p14_r,'stat':None,'mean':contr,
              'n':len(mout14['p_diff'].loc[mout14['animal']=='Ruffio']),'df':dof})


#==============================================EXP4
###############
description15 = 'one-sample t-test on sum-singleton coef in logistic regression'
query15 = '''
        SELECT animal,session,augend as minuend,addend as subtrahend,singleton,chose_sum,
        trialset,trial,augend+addend-singleton as diff
        FROM behavioralstudy
        WHERE experiment='Subtraction' AND session>23
        ORDER BY animal,session,trial
'''

data15 = Helper.getData(cur,query15)

model15 = 'chose_sum ~ diff'
mout15 = Analyzer.logistic_regression(data15,model15,groupby=['animal','session'],standardize=True)

n15_x = len(mout15['animal'].loc[mout15['animal']=='Xavier'])
n15_r = len(mout15['animal'].loc[mout15['animal']=='Ruffio'])
t15_x = scipy.stats.ttest_1samp(mout15['b_diff'].loc[mout15['animal']=='Xavier'],0)
t15_r = scipy.stats.ttest_1samp(mout15['b_diff'].loc[mout15['animal']=='Ruffio'],0)

tests.append({'description':description15+'(Xavier)','p':t15_x.pvalue,'stat':t15_x.statistic,
              'mean':np.mean(mout15['b_diff'].loc[mout15['animal']=='Xavier']),'n':n15_x,'df':n15_x-1})
tests.append({'description':description15+'(Ruffio)','p':t15_r.pvalue,'stat':t15_r.statistic,
              'mean':np.mean(mout15['b_diff'].loc[mout15['animal']=='Ruffio']),'n':n15_r,'df':n15_r-1})

###############
description16 = 't-test on set 1 vs. set 2 accuracy for both animals'
query16 = '''
        SELECT animal,session,
        AVG(CASE WHEN trialset=1 THEN (chose_sum = ((augend+addend)>singleton)) ELSE NULL END) as set1perf,
        AVG(CASE WHEN trialset=2 THEN (chose_sum = ((augend+addend)>singleton)) ELSE NULL END) as set2perf
        FROM behavioralstudy
        WHERE experiment='Subtraction' AND session>23
        GROUP BY animal,session
        HAVING set2perf IS NOT NULL
        ORDER BY animal,session
'''
data16 = Helper.getData(cur,query16)
t16_x = scipy.stats.ttest_rel(data16['set1perf'].loc[data16['animal']=='Xavier'],
                             data16['set2perf'].loc[data16['animal']=='Xavier'])
n16_x = len(data16['set2perf'].loc[data16['animal']=='Xavier'])
t16_r = scipy.stats.ttest_rel(data16['set1perf'].loc[data16['animal']=='Ruffio'],
                             data16['set2perf'].loc[data16['animal']=='Ruffio'])
n16_r = len(data16['set2perf'].loc[data16['animal']=='Ruffio'])
tests.append({'description':description16+'(Xavier)','p':t16_x.pvalue,'stat':t16_x.statistic,
              'mean':(np.mean(data16['set1perf'].loc[data16['animal']=='Xavier']),np.mean(data16['set2perf'].loc[data16['animal']=='Xavier'])),'n':n16_x,'df':n16_x-1})
tests.append({'description':description16+'(Ruffio)','p':t16_r.pvalue,'stat':t16_r.statistic,
              'mean':(np.mean(data16['set1perf'].loc[data16['animal']=='Ruffio']),np.mean(data16['set2perf'].loc[data16['animal']=='Ruffio'])),'n':n16_r,'df':n16_r-1})


###############
description17 = 'one sample t-test on set 2 accuracy vs. chance'
data17 = data16
t17_x = scipy.stats.ttest_1samp(data17['set2perf'].loc[data17['animal']=='Xavier'],.5)
n17_x = n16_x
t17_r = scipy.stats.ttest_1samp(data17['set2perf'].loc[data17['animal']=='Ruffio'],.5)
n17_r = n16_r
tests.append({'description':description17+'(Xavier)','p':t17_x.pvalue,'stat':t17_x.statistic,
              'mean':np.mean(data17['set2perf'].loc[data17['animal']=='Xavier']),'n':n17_x,'df':n17_x-1})
tests.append({'description':description17+'(Ruffio)','p':t17_r.pvalue,'stat':t17_r.statistic,
              'mean':np.mean(data17['set2perf'].loc[data17['animal']=='Ruffio']),'n':n17_r,'df':n17_r-1})

###############
description18 = 'binomial test on set 2 accuracy on first day'
query18 = '''
        SELECT animal,session,(chose_sum = ((augend+addend)>singleton)) as correct
        FROM behavioralstudy
        WHERE trialset=2 AND experiment='Subtraction' AND
        ((animal='Xavier' AND session=(SELECT session FROM behavioralstudy WHERE trialset=2 AND experiment='Subtraction' AND animal='Xavier' ORDER BY session LIMIT 1))
        OR (animal='Ruffio' AND session=(SELECT session FROM behavioralstudy WHERE trialset=2 AND experiment='Subtraction' AND animal='Ruffio' ORDER BY session LIMIT 1)))
        ORDER BY animal,session
        '''
data18 = Helper.getData(cur,query18)

n18_x = len(data18['correct'].loc[data18['animal']=='Xavier'])
n18_r = len(data18['correct'].loc[data18['animal']=='Ruffio'])
mux = np.sum(data18['correct'].loc[data18['animal']=='Xavier'])
mur = np.sum(data18['correct'].loc[data18['animal']=='Ruffio'])
t18_x = scipy.stats.binom_test(mux,n18_x,.5)
t18_r = scipy.stats.binom_test(mur,n18_r,.5)
tests.append({'description':description18+'(Xavier)','p':t18_x,'stat':None,'mean':mux/n18_x,
              'n':n18_x,'df':None})
tests.append({'description':description18+'(Ruffio)','p':t18_r,'stat':None,'mean':mur/n18_r,
              'n':n18_r,'df':None})


###############
description19 = 'paired t-test on subtrahend and singleton coefficients'
data19 = data15

model19 = 'chose_sum ~ minuend+subtrahend+singleton'
mout19 = Analyzer.logistic_regression(data19,model19,groupby=['animal','session'],standardize=True)

t19_x = scipy.stats.ttest_rel(mout19['b_subtrahend'].loc[mout19['animal']=='Xavier'],
                              mout19['b_singleton'].loc[mout19['animal']=='Xavier'])
t19_r = scipy.stats.ttest_rel(mout19['b_subtrahend'].loc[mout19['animal']=='Ruffio'],
                              mout19['b_singleton'].loc[mout19['animal']=='Ruffio'])
n19_x = len(mout19['b_subtrahend'].loc[mout19['animal']=='Xavier'])
n19_r = len(mout19['b_subtrahend'].loc[mout19['animal']=='Ruffio'])

tests.append({'description':description19+'(Xavier)','p':t19_x.pvalue,'stat':t19_x.statistic,
              'mean':(np.mean(mout19['b_subtrahend'].loc[data19['animal']=='Xavier']),
              np.mean(mout19['b_singleton'].loc[data19['animal']=='Xavier'])),'n':n19_x,'df':n19_x-1})
tests.append({'description':description19+'(Ruffio)','p':t19_r.pvalue,'stat':t19_r.statistic,
              'mean':(np.mean(mout19['b_subtrahend'].loc[data19['animal']=='Ruffio']),
              np.mean(mout19['b_singleton'].loc[data19['animal']=='Ruffio'])),'n':n19_r,'df':n19_r-1})


###############
description20 = 'correlation between subtrahend-singleton and session #'
query20 = '''
        SELECT animal,session,chose_sum,
        augend as minuend,-addend as subtrahend,singleton as singleton
        FROM behavioralstudy
        WHERE experiment='Subtraction'
        ORDER BY animal,session
'''

data20 = Helper.getData(cur,query20) 


#recode binary variables as 1/-1 instead of 0/1
model20 = 'chose_sum ~ minuend + subtrahend + singleton'
mout20 = Analyzer.logistic_regression(df=data20,model=model20,groupby=['animal','session'])

t20_x = scipy.stats.pearsonr(mout20['b_subtrahend'].loc[mout20['animal']=='Xavier']
                             -mout20['b_singleton'].loc[mout20['animal']=='Xavier'],
                            mout20['session'].loc[mout20['animal']=='Xavier'])

t20_r = scipy.stats.pearsonr(mout20['b_subtrahend'].loc[mout20['animal']=='Ruffio']
                             -mout20['b_singleton'].loc[mout20['animal']=='Ruffio'],
                            mout20['session'].loc[mout20['animal']=='Ruffio'])

n20_x = len(mout20['b_subtrahend'].loc[mout20['animal']=='Xavier'])
n20_r = len(mout20['b_subtrahend'].loc[mout20['animal']=='Ruffio'])

tests.append({'description':description20+'(Xavier)','p':t20_x[1],'stat':t20_x[0],
              'mean':None,'n':n20_x,'df':n20_x-2})
tests.append({'description':description20+'(Ruffio)','p':t20_r[1],'stat':t20_r[0],
              'mean':None,'n':n20_r,'df':n20_r-2})
#%%
#Print it all out

testdict = pd.DataFrame.from_dict(tests)
with open('D:\\Bart\\dropbox\\thesis_stats.csv','w') as fout:
    testdict.to_csv(fout)
