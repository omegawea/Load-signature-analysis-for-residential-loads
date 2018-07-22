# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 11:36:22 2018

@author: SF
"""

#%% X -> features, y -> label 
import pandas as pd 
import numpy as np 

from pylab import *
from numpy import *
import scipy.signal as signal
from scipy import signal
from scipy.signal import filter_design as fd
#%% High Pass Fitler & Frequencies (Top 5 Positions)
from pylab import *
from numpy import *
import scipy.signal as signal
from scipy import signal
from scipy.signal import filter_design as fd

from numpy import *
from numpy.fft import fft
import matplotlib.pyplot as plt
#from detect_peaks import *

import math
def getKey(item):
     return item[1]
#%% importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
#%% 
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
#%%
def get_minsignal(inputsignal, window = 40, threshold = 10):    
    temp = [0.0] * window
    minsignal = []
    for s in inputsignal:
        temp.pop(0)
        temp.append(s) 
        if max(temp) > 0 and min(temp) < 0:
            minsignal.append(0)
        elif(min(temp) < - threshold):            
            minsignal.append(max(temp))
        elif(max(temp) > threshold):   
            minsignal.append(min(temp))
        else:       
            minsignal.append(0)
    return minsignal  
#%%
def get_movingavg(signal, window = 10):
    result = []
    temp = [0.0] * window
    for s in signal:
        temp.pop(0)
        temp.append(s)  
        result.append(mean(temp))
    return result
#%%
def get_data(csv):      
    tfeatures = pd.read_csv(csv, sep=',')
    tfeatures['PF'] = tfeatures['PF'].fillna(tfeatures['P'] / tfeatures['Q'])
    temp = tfeatures['Q'].loc[tfeatures['PF'] < 0] * -1
    tfeatures['Q'].loc[temp.index] = temp
    tfeatures['dP'] = np.gradient(tfeatures['P'])
    tfeatures['dQ'] = np.gradient(tfeatures['Q'])
    return tfeatures
#%%
class dataset(object):
    
    def __init__(self):
        print ("initialize Class A dataset
        # AIRCON
        tfeatures = get_data('data/AIRCON2.csv')
        #Signal Filter
        tfeatures = tfeatures.ix[tfeatures['P'] >= 420]
        tfeatures = tfeatures.ix[tfeatures['Q'] <= 200]
        tfeatures = tfeatures.ix[tfeatures['Q'] >= -400]
        tfeatures = tfeatures.drop(['S', 'PF'], axis=1)
        #Export
        self.ac_feats = tfeatures
        self.ac_class = ['Air Conditioner' for x in range(len(tfeatures))]
        # BOILER
        tfeatures = get_data('data/BOILER.csv')
        #Signal Filter
        tfeatures = tfeatures.ix[tfeatures['P'] >= 1650]
        tfeatures = tfeatures.ix[tfeatures['Q'] <= 200]
        tfeatures = tfeatures.drop(['S', 'PF'], axis=1)
        #Export
        self.wb_feats = tfeatures
        self.wb_class = ['Water Boiler' for x in range(len(tfeatures))]
        # DRYER
        tfeatures = get_data('data/DRYER.csv')
        #Signal Filter
        tfeatures = tfeatures.ix[tfeatures['P'] >= 1000]
        #tfeatures = tfeatures.ix[tfeatures['Q'] <= 150]
        tfeatures = tfeatures.drop(['S', 'PF'], axis=1)
        #Export
        self.hd_feats = tfeatures
        self.hd_class = ['Hair Dryer' for x in range(len(tfeatures))]
        # COOKER
        tfeatures = get_data('data/COOKER.csv')
        #Signal Filter
        tfeatures = tfeatures.ix[tfeatures['P'] >= 800]
        #tfeatures = tfeatures.ix[tfeatures['Q'] <= 110]
        tfeatures = tfeatures.drop(['S', 'PF'], axis=1)
        #Export
        self.c_feats = tfeatures
        self.c_class = ['Electronic Cooker' for x in range(len(tfeatures))]
        # DEHUMID
        tfeatures = get_data('data/DEHUMID.csv')
        #Signal Filter
        tfeatures = tfeatures.ix[tfeatures['P'] >= 150]
        tfeatures = tfeatures.ix[tfeatures['Q'] <= 150]
        tfeatures = tfeatures.drop(['S', 'PF'], axis=1)
        #Export
        self.dh_feats = tfeatures
        self.dh_class = ['Dehumidifier' for x in range(len(tfeatures))]
        # FRIDGE
        tfeatures = get_data('data/FRIDGE.csv')
        #Signal Filter
        tfeatures = tfeatures.ix[tfeatures['P'] >= 90]
        tfeatures = tfeatures.ix[tfeatures['Q'] <= 200]
        tfeatures = tfeatures.drop(['S', 'PF'], axis=1)
        #Export
        self.f_feats = tfeatures
        self.f_class = ['Refrigerator' for x in range(len(tfeatures))]
        # HEATER
        tfeatures = get_data('data/HEATER.csv')
        #Signal Filter
        tfeatures = tfeatures.ix[tfeatures['P'] >= 600]
        #tfeatures = tfeatures.ix[tfeatures['Q'] <= 200]
        tfeatures = tfeatures.drop(['S', 'PF'], axis=1)
        #Export
        self.h_feats = tfeatures
        self.h_class = ['Ambient Heater' for x in range(len(tfeatures))]
#        # IRON
#        tfeatures = get_data('data/IRON.csv')
#        #Signal Filter
#        #tfeatures = tfeatures.ix[tfeatures['P'] >= 1000]
#        #tfeatures = tfeatures.ix[tfeatures['Q'] <= 150]
#        tfeatures = tfeatures.ix[tfeatures['S'] >= 30]
#        tfeatures = tfeatures.drop(['S', 'PF'], axis=1)
#        #Export
#        self.i_feats = tfeatures
#        self.i_class = ['IRON' for x in range(len(tfeatures))]
#        # RCOOK
#        tfeatures = get_data('data/RCOOK.csv')
#        #Signal Filter
#        #tfeatures = tfeatures.ix[tfeatures['P'] >= 325]
#        #tfeatures = tfeatures.ix[tfeatures['Q'] <= 32]
#        
#        #tfeatures = tfeatures.ix[(tfeatures['P'] > 325) | (tfeatures[5] > 50)]
#        tfeatures = tfeatures.drop(['S', 'PF'], axis=1)
#        #t1features = tfeatures.ix[tfeatures['P'] <= 700]
#        #t2features = tfeatures.ix[tfeatures['P'] >= 1800]s))]
#        #Export
#        self.rc_feats = tfeatures
#        self.rc_class = ['RCOOK' for x in range(len(tfeatures))]
        # TV
        tfeatures = get_data('data/TV.csv')
        #Signal Filter
        tfeatures = tfeatures.ix[tfeatures['P'] >= 55]
        tfeatures = tfeatures.ix[tfeatures['Q'] <= 32]
        tfeatures = tfeatures.drop(['S', 'PF'], axis=1)
        #Export
        self.tv_feats = tfeatures
        self.tv_class = ['Television' for x in range(len(tfeatures))]    
        # VENT
        tfeatures = get_data('data/VENT.csv')
        #Signal Filter
        tfeatures = tfeatures.ix[tfeatures['P'] >= 150]
        tfeatures = tfeatures.ix[tfeatures['Q'] <= 80]
        tfeatures = tfeatures.drop(['S', 'PF'], axis=1)
        #Export
        self.v_feats = tfeatures
        self.v_class = ['Vent Hood' for x in range(len(tfeatures))]
        
        # WASHER
        tfeatures = get_data('data/WASHER.csv')        
        tfeatures['P'] = get_minsignal(tfeatures['P'])
        tfeatures['Q'] = get_minsignal(tfeatures['Q'])
        tfeatures = tfeatures.ix[tfeatures['P'] >= 1800]
        tfeatures = tfeatures.drop(['S', 'PF'], axis=1) 
        #Export
        self.w_feats = tfeatures
        self.w_class = ['Washing Machine' for x in range(len(tfeatures))]
        
    def load(self):
        
        # Grouping
        self.classes = list()    
        self.features = pd.concat([self.ac_feats, self.wb_feats, self.hd_feats, self.c_feats, self.dh_feats, self.f_feats, self.h_feats, self.tv_feats, self.v_feats, self.w_feats], ignore_index=True)
        self.classes =             self.ac_class+ self.wb_class+ self.hd_class+ self.c_class+ self.dh_class+ self.f_class+ self.h_class+ self.tv_class+ self.v_class+ self.w_class
        
    #    features = features.drop(['dP', 'dQ', 0, 1, 2, 3, 4, 5], axis=1)
        self.features = self.features.drop(['dP', 'dQ'], axis=1)
        # dividing X, y into train and test data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.classes, random_state = 0)
        ##%%
        #fpeaks = pd.DataFrame({1: [], 2: []})
        #tdf = pd.DataFrame(columns=columns)    
        #features = pd.concat([features, tdf], axis=1, sort=False)    
        #
        #classes = list()
        #fpeaks = pd.concat([ac_fpeaks, wb_fpeaks, hd_fpeaks, c_fpeaks, dh_fpeaks, f_fpeaks, h_fpeaks, i_fpeaks, rc_fpeaks, tv_fpeaks, v_fpeaks, w_fpeaks], ignore_index=True)
        #fclasses = ac_fclass + wb_fclass + hd_fclass + c_fclass + dh_fclass + f_fclass + h_fclass + i_fclass + rc_fclass + tv_fclass + v_fclass + w_fclass
        ##%% dividing X, y into train and test data
        #Xf_train, Xf_test, yf_train, yf_test = train_test_split(fpeaks, fclasses, random_state = 0)
        return self.features, self.classes

#%% Plot and Show Data
    def plot(self, selection = ''):   
        plt.figure()
        plt.clf()
        if ('Air conditioner' in selection) or (selection is ''):
            plt.scatter(self.ac_feats['P'],self.ac_feats['Q'], c='y', label='Air Conditioner', edgecolors='none')
        if ('Hair Dryer' in selection) or (selection is ''):
            plt.scatter(self.hd_feats['P'],self.hd_feats['Q'], c='m', label='Hair Dryer', edgecolors='none')
        if ('Water Boiler' in selection) or (selection is ''):
            plt.scatter(self.wb_feats['P'],self.wb_feats['Q'], c='k', label='Water Boiler', edgecolors='none')
        if ('Electronic Cooker' in selection) or (selection is ''):
            plt.scatter(self.c_feats['P'] ,self.c_feats['Q'] , c='c', label='Electronic Cooker', edgecolors='none')
        if ('Dehumidifier' in selection) or (selection is ''):
            plt.scatter(self.dh_feats['P'],self.dh_feats['Q'], c='c', label='rc_data', edgecolors='m')
        if ('Refrigerator' in selection) or (selection is ''):
            plt.scatter(self.f_feats['P'] ,self.f_feats['Q'] , c='r', label='Refrigerator', edgecolors='none')
        if ('Ambient Heater' in selection) or (selection is ''):
            plt.scatter(self.h_feats['P'] ,self.h_feats['Q'] , c='k', label='Ambient Heater', edgecolors='m')
        if ('Television' in selection) or (selection is ''):
            plt.scatter(self.tv_feats['P'],self.tv_feats['Q'], c='c', label='Television', edgecolors='none')
        if ('Vent Hood' in selection) or (selection is ''):
            plt.scatter(self.v_feats['P'] ,self.v_feats['Q'] , c='g', label='Vent Hood', edgecolors='none')
        if ('Washing Machine' in selection) or (selection is ''):
            plt.scatter(self.w_feats['P'],self.w_feats['Q'], c='b', label='Washing Machine', edgecolors='y')
        plt.legend(loc='upper right', fontsize=16)
        
        axes = plt.gca()
        #axes.set_xlim([-50,200])
        #axes.set_ylim([-200,200])
        plt.title('Power Map', fontsize=18)
        plt.xlabel('P', fontsize=18)
        plt.ylabel('Q', fontsize=18)
        plt.show()
#%%
class composite(object):
    def __init__(self, filename):
        print ("initialize Class A composite signals...")        
        self.tsignal = get_data(filename)   
        
        plt.figure()
        plt.clf()  
        plt.plot(self.tsignal['S'])       
        axes = plt.gca()
        plt.title('Composite Signal', fontsize=18)
        plt.xlabel('time', fontsize=18)
        plt.ylabel('Apparent Power', fontsize=18)
        plt.show()
        
    def disaggrate(self, model):
        # Initialization                
        pth = 40
        
        mins = get_movingavg(get_minsignal((self.tsignal['S'])))
        minp = get_movingavg(get_minsignal((self.tsignal['P'])))
        minq = get_movingavg(get_minsignal((self.tsignal['Q'])))
                        
        
        nows = mins[0]
        nowp = minp[0]
        nowq = minq[0]
        
        base = nows
        lastp = nowp
        lastq = nowq
        
        baseline = pd.DataFrame({'P': [minp[0]],'Q': [minq[0]]})
        current = pd.DataFrame({'P': [minp[0]],'Q': [minq[0]]})
        
#        tbase = 0
        prediction = []
        for t in range(0, len(self.tsignal['S'])):    
            
            nowp = minp[t]
            nowq = minq[t]
            
            if abs(nowp - lastp) < 1 and  abs(nowq - lastq) < 1 :    # Buffer until stable
                nows = mins[t]
                if nows - base > pth: 
                    current['P'] = minp[t]
                    current['Q'] = minq[t]
                    appliance = model.predict(current - baseline)
#                    print ('at t =', t, 'Switched ON:', appliance)
                    
                    for time in reversed(range(t)):
                        if self.tsignal['S'][time] + pth < nows:
                            break
                        else:
                            prediction[time] = nows
                            
                    print ('at t =', time, 'Switched ON:', appliance)
                    
                    base = nows                  
                    baseline['P'] = minp[t]
                    baseline['Q'] = minq[t]
#                    lasts = nows 
                    
                elif base - nows > pth:
                    current['P']= minp[t]
                    current['Q']= minq[t]
                    appliance = model.predict(baseline - current)
#                    print ('at t =', t, 'Switched OFF:', appliance)
                    
                    for time in reversed(range(t)):
                        if self.tsignal['S'][time] > nows + pth:
                            break
                        else:
                            prediction[time] = nows
                            
                    print ('at t =', time, 'Switched OFF:', appliance)
                    
                    
                    base = nows                    
                    baseline['P'] = minp[t]  
                    baseline['Q'] = minq[t] 
#                    lasts = nows
                    
                    
            prediction.append(base)    
            lastp = nowp
            lastq = nowq

        plt.figure()
        plt.clf()
        plt.subplot(211)
        plt.plot(self.tsignal['S'])
        plt.ylabel('True Power')
        plt.xlabel('time')
        axes = plt.gca()
        
        plt.subplot(212)
        plt.plot(prediction)
        plt.ylabel('Predicted Power')
        plt.xlabel('time')
        axes = plt.gca()
        
        print ('r2 score:', r2_score(self.tsignal['S'], prediction))
        
#        print (len(self.tsignal['S']), len(prediction))
#        (self.tsignal['S'] - prediction)
##        f1_score(np.floor(self.tsignal['S']), np.floor(prediction), average='macro')  
##        f1_score(np.floor(self.tsignal['S']), np.floor(prediction), average='micro')  
#        f1_score(np.rint(self.tsignal['S']), np.rint(prediction), average='weighted', labels=np.unique(prediction))  
##        f1_score(np.floor(self.tsignal['S']), np.floor(prediction), average='None')  
            
#        plt.figure()
#        plt.clf()
#        plt.plot(prediction)
#        plt.ylabel('Prediction')
#        plt.xlabel('time')
#        axes = plt.gca()
        
#        plt.figure()
#        plt.clf()
#        plt.subplot(321)
#        plt.plot(self.tsignal['S'])
#        plt.ylabel('S')
#        plt.xlabel('time')
#        axes = plt.gca()
#        
#        plt.subplot(322)
#        plt.plot(mins)
#        plt.ylabel('mins')
#        plt.xlabel('time')
#        axes = plt.gca()
#        
#        plt.subplot(323)
#        plt.plot(self.tsignal['P'])
#        plt.ylabel('P')
#        plt.xlabel('time')
#        axes = plt.gca()
#        
#        plt.subplot(324)
#        plt.plot(minp)
#        plt.ylabel('minp')
#        plt.xlabel('time')
#        axes = plt.gca()
#        
#        plt.subplot(325)
#        plt.plot(self.tsignal['Q'])
#        plt.ylabel('Q')
#        plt.xlabel('time')
#        axes = plt.gca()
#        
#        
#        plt.subplot(326)
#        plt.plot(minq)
#        plt.ylabel('minq')
#        plt.xlabel('time')
#        axes = plt.gca()
