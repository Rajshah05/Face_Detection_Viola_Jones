import numpy as np
import cv2
import math
#import matplotlib.pyplot as plt
from random import sample
import operator
import os
#from PIL import Image
#from scipy import optimize
import sys
#from skimage import data
#from skimage.color import rgb2gray
from collections import defaultdict
from itertools import chain
import pickle
#from scipy.optimize import minimize

def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        img = []
    else:    
        img = cv2.resize(img, (24, 24), interpolation = cv2.INTER_AREA)  #Resize
            
        # if not img.dtype == np.uint8:
        #     pass
        # if show:
        #     show_image(img)
    return img

def readtrain_imgset(folder):
    
    images = []
    # count = 0
    for fn in os.listdir(folder):
        # img = cv2.imread(folder +  '/' + fn, cv2.IMREAD_GRAYSCALE)
        img = read_image(folder +  '/' + fn)
        if len(img) != 0:
            img = varnorm(img)
            if np.isnan(img.any()) == True:
                continue
            # count+=1
            # if count>7600:
            #     break
            img = integ(img)
            images.append(img)
    return images

def varnorm(img):
    img = np.asarray(img)
    # mean = np.sum(np.sum(img))/576
    mean = np.mean(img)
    # std = np.sqrt(mean**2 - np.sum(np.sum(np.multiply(img,img)))/576)
    std = np.std(img)
    # print("std",std)
    img = ((img-mean)/float(std))

    return img


def integ(img):
    img = cv2.integral(img)
    #img = img[1:,1:]
    return img

def getfeatures(integimg):
    featureid = {}
    matrix = []
    id0 = 0 #ctwohor
    id1 = 1 #ctwover
    id2 = 2 #cthreehor
    id3 = 3 #cthreever
    id4 = 4 #cfour
    count = 0
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
#    for ir,vr in enumerate(integimg[:-1]):
    for ir in range(1,integimg.shape[1]):
        
        for ic in range(1,integimg.shape[2]):
            # for iir, vvr in enumerate(integimg[ir+1:]):
            for iir in range(ir,integimg.shape[1]):
                # for iic, vvc in enumerate(vvr[ic+1:]):
                for iic in range(ic,integimg.shape[2]):
                    if (iir-ir+1) % 2 == 0: #39077
                        # print(ir+((iir-ir)//2))
                        ctwohor = integimg[:,ir-1,ic-1] - integimg[:,ir-1,iic] + 2*integimg[:,ir+((iir-ir)//2),iic] - integimg[:,iir,iic] + integimg[:,iir,ic-1] - 2*integimg[:,ir+((iir-ir)//2),ic-1]
                        matrix.append(ctwohor)
                        featureid[count] = ((ir,ic),(iir,iic),id0)
                        count +=1
                        count1+=1
                    if (iic-ic+1) % 2 == 0: #39219
                        ctwover = -integimg[:,ir-1,ic-1] + 2*integimg[:,ir-1,ic+((iic-ic)//2)] - integimg[:,ir-1,iic] + integimg[:,iir,iic] - 2*integimg[:,iir,ic+((iic-ic)//2)] + integimg[:,iir,ic-1]
                        matrix.append(ctwover)
                        featureid[count] = ((ir,ic),(iir,iic),id1)
                        count +=1
                        count2+=1
                    if (iir-ir+1) % 3 == 0: #25060
                        cthreehor = -integimg[:,ir-1,ic-1] + 2*integimg[:,(ir+(iir-ir)//3),ic-1] - 2*integimg[:,(ir+2*((iir-ir)//3)+1),ic-1] + integimg[:,iir,ic-1] + integimg[:,ir-1,iic] - 2*integimg[:,(ir+(iir-ir)//3),iic] + 2*integimg[:,(ir+2*((iir-ir)//3)+1),iic] - integimg[:,iir,iic]
                        matrix.append(cthreehor)
                        featureid[count] = ((ir,ic),(iir,iic),id2)
                        count +=1
                        count3+=1
                    if (iic-ic+1) % 3 == 0: #25060
                        cthreever = -integimg[:,ir-1,ic-1] + 2*integimg[:,ir-1,(ic+((iic-ic)//3))] - 2*integimg[:,ir-1,ic+(2*(iic-ic)//3)+1] + integimg[:,ir-1,iic] + integimg[:,iir,ic-1] - 2*integimg[:,iir,(ic+((iic-ic)//3))] + 2*integimg[:,iir,ic+(2*((iic-ic)//3))+1] - integimg[:,iir,iic]
                        matrix.append(cthreever)
                        featureid[count] = ((ir,ic),(iir,iic),id3)
                        count +=1
                        count4+=1
                    if (iir-ir+1) % 2 == 0 and (iic-ic+1) % 2 == 0: #20736
                        cfour = -integimg[:,ir-1,ic-1] + 2*integimg[:,ir-1,ic+((iic-ic)//2)] - integimg[:,ir-1,iic] + 2*integimg[:,ir+((iir-ir)//2),ic-1] - 4*integimg[:,ir+((iir-ir)//2),ic+((iic-ic)//2)] +2*integimg[:,ir+((iir-ir)//2),iic] - integimg[:,iir,ic-1] + 2* integimg[:,iir,ic+((iic-ic)//2)] - integimg[:,iir,iic]
                       # cfour =       -J[ir-1,ic-1] +         2*J[ir-1,ic+((iic-ic)//2)] -             J[ir-1,iic] +         2*J[ir+((iir-ir)//2),ic-1] -     4*J[ir+((iir-ir)//2),ic+((iic-ic)//2)] +                2*J[ir+((iir-ir)//2),iic] -          J[iir,ic-1] + 2*           J[iir,ic+((iic-ic)//2)] -       J  [iir,iic]
                        matrix.append(cfour)
                        featureid[count] = ((ir,ic),(iir,iic),id4)
                        count +=1
                        count5+=1
                    print(count)
                
                    
                    
    print(count1, count2, count3, count4, count5)              

    #return np.transpose(twover + twohor + threever + four) 
    return (matrix,featureid)

def normalize(wt_dict):
    sum_wts = sum(wt_dict[0]) + sum(wt_dict[1])
    #print(sum_wts)
    for i in range(len(wt_dict[0])):
        #print(wt_dict[0][i])
        wt_dict[0][i] = wt_dict[0][i]/sum_wts
    for i in range(len(wt_dict[1])):
        wt_dict[1][i] = wt_dict[1][i]/sum_wts
    return wt_dict

weakclassifiers = []
list_pos = readtrain_imgset("train_posf")
list_neg = readtrain_imgset("train_negf")
trainingdata = np.asarray(list_pos + list_neg)
len_pos = len(list_pos)
len_neg = len(list_neg)
numimages = len(list_pos) + len(list_neg)
print(len(trainingdata))
print(len_pos)
print(len_neg)
wts = {}
wts[0] = list(1/(2*len_neg)*np.ones(len_neg))
wts[1] = list(1/(2*len_pos)*np.ones(len_pos))

output = getfeatures(trainingdata)
fts = np.asarray(output[0])
np.save("fts",fts)
file2 = open("fid.pkl", "wb")
pickle.dump(output[1], file2)
file2.close()