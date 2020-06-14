import numpy as np
import scipy as sc

import math
import matplotlib.pyplot as plt
from random import sample
import operator
import os
from PIL import Image
from scipy import optimize
import sys
from skimage import data
from skimage.color import rgb2gray
from collections import defaultdict
from itertools import chain
import pickle
from scipy.optimize import minimize


fts = np.load('fts.npy',allow_pickle='False')
numfeatures = fts.shape[0]

len_pos = 7600
len_neg = fts.shape[1] - len_pos
print(len_pos, len_neg)


weakclassifiers = []
#list_pos = readtrain_imgset("train_pos")
#list_neg = readtrain_imgset("train_neg")
#trainingdata = np.asarray(list_pos + list_neg)
# len_pos = 4834
# len_neg = 7966
#numimages = len(list_pos) + len(list_neg)
#print(len(trainingdata))
wts = {}
wts[0] = list(1/(2*len_neg)*np.ones(len_neg))
wts[1] = list(1/(2*len_pos)*np.ones(len_pos))

# fts = getfeatures(trainingdata)


mmerror = []
for jj in range(numfeatures):
#for jj in range(500):
    print(jj)
    fts0p = fts[jj,0:len_pos]
    fts0n = fts[jj,len_pos:]
    medpos = np.nanmedian(fts0p)
    medneg = np.nanmedian(fts0n)
    wt_p = np.copy(np.asarray(wts[1]))
    wt_n = np.copy(np.asarray(wts[0]))
#     if medpos > medneg:
#         parity0 = 1
#     else:
#         parity0 = -1

    
    if abs(medneg - medpos) > 15:
        i = 25
    elif abs(medneg - medpos) > 5:
        i = 11
    elif abs(medneg - medpos) > 3:
        i = 5
    else:
        i = 3
    #print(abs(medneg - medpos))
    if medneg < medpos:
        rang = np.linspace(medneg,medpos,i)
    else:
        rang = np.linspace(medpos,medneg,i)
  
    def calcerror(thre):
        parity = 1
        wt_p = np.copy(np.asarray(wts[1]))
        wt_n = np.copy(np.asarray(wts[0]))
        wt_p[fts0p > thre] = 0
        wt_n[fts0n < thre] = 0
        error = sum(wt_p) + sum(wt_n)
        if error > 0.5:
            error = 1-error
            parity = -1
        return [error,thre,parity]
#     print(rang)
    if len(rang)!=0:
        errora = map(calcerror, rang)
        merror = min(errora)
    else:
        thret = medpos
        parityt = 1
        wt_p[fts0p > thret] = 0
        wt_n[fts0n < thret] = 0
        errort = sum(wt_p) + sum(wt_n)
        if errort > 0.5:
            errort = 1-errort
            parityt = -1
        merror = [errort, thret, parityt]
  
    #print(merror)
    mmerror.append(merror)
# abc = calcmmerror(fts[149945,:])
# mmerror = map(calcmmerror,ftsl)
print(min(mmerror))
file8 = open("mmerror.pkl", "wb")
pickle.dump(mmerror, file8)
file8.close()