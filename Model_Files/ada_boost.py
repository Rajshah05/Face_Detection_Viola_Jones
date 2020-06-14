import numpy as np
import math
import matplotlib.pyplot as plt
from random import sample
import operator
import os
import sys
from skimage.color import rgb2gray
from collections import defaultdict
from itertools import chain
import pickle


def normalize(wt_dict):
    sum_wts = sum(wt_dict[0]) + sum(wt_dict[1])
    for i in range(len(wt_dict[0])):
        wt_dict[0][i] = wt_dict[0][i]/sum_wts
    for i in range(len(wt_dict[1])):
        wt_dict[1][i] = wt_dict[1][i]/sum_wts
    #print(sum(wt_dict[0]) + sum(wt_dict[1]))
    return wt_dict


fts = np.load('fts.npy',allow_pickle='False')
numfeatures = fts.shape[0]

thr = np.load('thr.npy',allow_pickle='False')
print(thr.shape)

error = np.load('error.npy',allow_pickle='False')
print(min(error))
print(np.argmin(error))


len_pos = 7600
len_neg = fts.shape[1] - len_pos
print(len_pos, len_neg)


numimages = len_pos + len_neg
wts = {}
wts[0] = list(1/(2*len_neg)*np.ones(len_neg))
wts[1] = list(1/(2*len_pos)*np.ones(len_pos))


numimages = len_pos + len_neg

file2 = open("fid.pkl", "rb")
featureid = pickle.load(file2)


permerror = np.ones((numfeatures,1))


ftsp = fts[:,0:len_pos]
ftsn = fts[:,len_pos:]


weakclassifiers = []
T = 6061
for s_fts in range(0, 1501):
    wts = normalize(wts)
    print(s_fts)
    error_pos = None
    error_neg = None
    wt_p = np.asarray(np.copy(wts[1]))
    wt_n = np.asarray(np.copy(wts[0]))
    #print(wt_p)
    error_pos = np.vstack([wt_p]*numfeatures)
    error_neg = np.vstack([wt_n]*numfeatures)
    #print(type(error_pos[0][0]))
    #print(type(error_pos))
    error_pos[ftsp>thr] = 0
    
    error_neg[ftsn<thr] = 0
    errorp1 = np.sum(error_pos,axis = 1)+ np.sum(error_neg,axis = 1)

    error_pos = None
    error_neg = None

    for i in range(numfeatures):
        if permerror[i] == 5:
            errorp1[i] = 5 
            #print('bobo')
            
    
    errorp0 = 1-errorp1

    errorp0 = abs(errorp0)
    errorp1min = min(errorp1)
    bf1 = np.argmin(errorp1)
    errorp1 = None
    #errorp0 = np.where(permerror==5, 5, errorp0)
    errorp0min = min(errorp0)
    bf0 = np.argmin(errorp0)
    errorp0 = None

    if errorp1min < errorp0min:
        bf = bf1
        et = errorp1min
        parity = 1
    else:
        bf = bf0
        et = errorp0min
        parity = -1

    print("et", et)
    bt = et/(1-et)
    print(bf)
    #print(thr[bf])
    permerror[bf] = 5
    weakclassifiers.append((bf,bt,featureid[bf],parity,thr[bf]))
    for k,w in enumerate(wts[0]):
        if parity*ftsn[bf][k]<parity*thr[bf]:
            #print(wts[0][k])
            wts[0][k] = wts[0][k]*bt
            #print(wts[0][k])
            #print()
    for k,w in enumerate(wts[1]):
        if parity*ftsp[bf][k]>parity*thr[bf]:
            #print(wts[1][k])
            wts[1][k] = wts[1][k]*bt
            #print(wts[1][k])
            #print()
    print(featureid[bf])
#     print(parity)
#     if s_fts == 2000 or s_fts == 2500:
#       file5 = open(F"/content/gdrive/My Drive/wts_till3000.pkl", "wb")
#       pickle.dump(wts, file5)
#       file5.close()
#       file6 = open(F"/content/gdrive/My Drive/weakclassifiers_till3000.pkl", "wb")
#       pickle.dump(weakclassifiers, file6)
#       file6.close()
file7 = open("wts.pkl", "wb")
pickle.dump(wts, file7)
file7.close()

# file8 = open("thr_ada_2000.pkl", "wb")
# pickle.dump(thr, file8)
# file8.close()

file9 = open("strong_classifiers.pkl", "wb")
pickle.dump(weakclassifiers, file9)
file9.close()

file10 = open("permerror.pkl", "wb")
pickle.dump(permerror, file10)
file10.close()