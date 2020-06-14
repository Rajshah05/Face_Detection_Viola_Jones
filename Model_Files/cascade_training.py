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

fts = np.load('fts.npy',allow_pickle='False')
numfeatures = fts.shape[0]

global len_pos
global len_neg
global ftsp
global ftsn
global fni
global len_fn

len_pos = 7600
len_neg = fts.shape[1] - len_pos
print(len_pos, len_neg)


ftsp = fts[:,0:len_pos]
ftsn = fts[:,len_pos:]


fni = np.copy(ftsn)
print(fni.shape)


len_fn = fni.shape[1]

file = open("strong_classifier.pkl", "rb")
w_classifiers = pickle.load(file)

     
def eval_c(s_c):
    global len_pos
    global len_fn
    global ftsp
    global fni
    temp_n = np.zeros([len(s_c),len_fn])
    temp_p = np.zeros([len(s_c),len_pos])
    alphas = np.ones([len(s_c),1])
    
    for ii,vv in enumerate(s_c):
        temp_n[ii][vv[3]*fni[vv[0]] > vv[3]*vv[4]] = 1
        temp_p[ii][vv[3]*ftsp[vv[0]] > vv[3]*vv[4]] = 1
        alphas[ii][0] = np.log(1/vv[1])
    th = (np.sum(alphas, axis=0)*0.5)
    th1 = (np.sum(alphas, axis=0)*0.5)*np.ones([1,len_fn])
    FPR = np.sum(temp_n*alphas, axis=0)
    fp = np.ones([1,len_fn])
    fp[FPR<th1] = 0
    FP = np.sum(fp, axis=1)/len_fn
    
    th2 = (np.sum(alphas, axis=0)*0.5)*np.ones([1,len_pos])
    tp = np.ones([1,len_pos])
    TPR = np.sum(temp_p*alphas, axis=0)
    tp[TPR < th2] = 0
    TP = np.sum(tp, axis=1)/len_pos
    return (TP,FP,th)


def eval_c_tpr(s_c, th):
    global len_pos
    global len_fn
    global ftsp
    global fni
    temp_n = np.zeros([len(s_c),len_fn])
    temp_p = np.zeros([len(s_c),len_pos])
    alphas = np.ones([len(s_c),1])
    # th = np.sum(alphas, axis=0)/2
    for ii,vv in enumerate(s_c):
        temp_n[ii][vv[3]*fni[vv[0]] > vv[3]*vv[4]] = 1
        temp_p[ii][vv[3]*ftsp[vv[0]] > vv[3]*vv[4]] = 1
        alphas[ii][0] = np.log(1/vv[1])
    
    th1 = (th)*np.ones([1,len_fn])
    fp = np.ones([1,len_fn])
    FPR = np.sum(temp_n*alphas, axis=0)
    fp[FPR<th1] = 0
    FP = np.sum(fp, axis=1)/len_fn
    
    th2 = (th)*np.ones([1,len_pos])
    tp = np.ones([1,len_pos])
    TPR = np.sum(temp_p*alphas, axis=0)
    tp[TPR < th2] = 0
    TP = np.sum(tp, axis=1)/len_pos
    return (TP,FP)

def nbwc(strong_c):
#     global fni
#     global ftsp
#     global len_pos
#     global len_fn
    fp0 = 1
    for i in range(len(w_classifiers)):
#         st = np.zeros(len_fn)
#         r = w_classifiers[0]
#         th = w_classifiers[4]
#         p = w_classifiers[3]
#         st[p*fni[r] > p*th*np.ones(len_fn)] = 1
        strong_c.append(w_classifiers[i])
        (tp,fp,th) = eval_c(strong_c)
        if fp < fp0:
            fp0 = fp
            ind = i
        strong_c = strong_c[:-1]
    return ind


#def cascade(set_pos, set_neg, w_classifiers, fparity, fthr):

print(fni.shape)
Ftarget = 0.05
# Dtarget = 0.95
f = 0.51
d = 0.985
F = np.ones(2000, dtype = 'float64')
D = np.zeros(2000, dtype = 'float64')
F[0] = 1.0
D[0] = 1.0
i = 0
n = np.zeros(2000)
cascaded = []
wc_n0 = 0

while(F[i] > Ftarget):
#     cascaded[i] = []
    i += 1
    n[i] = 0
    F[i] = 1.0
    D[i] = 1.0
    strong_classifier = []
#     temp_strong_classifier = []
    while(F[i] > f) or (len(strong_classifier)>200):
        wc_n0 += 1
        n[i] += 1
#         nbc_ind = nbwc(strong_classifier)
        strong_classifier.append(w_classifiers[wc_n0])
#         del w_classifiers[nbc_ind]
        (TP,FP,thr) = eval_c(strong_classifier)
        F[i] = FP
        D[i] = TP
        #print("TP", TP, "FP", FP, thr)
        # thr = strong_classifier[wc_n0-1][4]
        count = 0
        while (D[i] < d):
            count+=1
            # if w_classifiers[wc_n0-1][3] == 1:
            thr -= 0.005
            # else:
                # thr += 1
            (TP1, FP1) = eval_c_tpr(strong_classifier, thr)
            D[i] = TP1
            F[i] = FP1
        print(i,"cur_TP", D[i],"required:",d,"cur_fp", F[i],"required:",f)
        #print("count", count)
        # cascaded[i].append((w_classifiers[wc_n0-1],thr))
#     cascaded[i-1] = [strong_classifier,thr]
    cascaded.append([strong_classifier,thr])
    if f<0.7:
        f = f+0.2
    else:
        f = 0.8
#     elif f<0.9:
#         f = f+0.10
#     if f >= 1:
#         f = 0.90
    print(cascaded[i-1])
    
    if F[i] >Ftarget:
        FN = np.zeros([len(cascaded),len_fn])
        FFN = np.ones(len_fn)
        evalu = np.ones([1,len_fn])
        for jj,vv in enumerate(cascaded):
            #print("vv0",vv[0])
            temp_n = np.zeros([len(vv[0]),len_fn])
            alphas = np.ones([len(vv[0]),1])
            for j,v in enumerate(vv[0]):                
                temp_n[j].reshape(1, len_fn)[(v[3]*fni[v[0]]).reshape(1,len_fn) > v[3]*v[4]*np.ones([1,len_fn])] = 1
                alphas[j][0] = np.log(1/v[1])
            eva = np.sum(temp_n*alphas, axis=0)
            #evalu[eva<vv[1]] = 0
            evalu = np.where(eva<vv[1], 0, evalu)
            FN[jj,:] = evalu
        FN = np.sum(FN, axis=0)
        #FFN[FN<len(cascaded)] = 0
        FFN = np.where(FN<len(cascaded), 0, FFN)
        fni = np.transpose(np.transpose(fni)[FFN==np.ones(len_fn)])
        len_fn = fni.shape[1]
        print("len_fn", len_fn)