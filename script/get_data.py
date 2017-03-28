#coding=utf-8
import numpy as np
import sys, os
from random import randint
sys.path.append('.')
import feature, mylog 


#argv[1] is the filenamelist of imgsource like 'train.txt'
#line in filenamelist like '/home/database/snow.jpg 3'
#example : python get_feature.py ~/workspace/imgsource/test.txt

filenamelist = sys.argv[1]
log = mylog.mylog('caculate')
out = filenamelist.split('.')[1]
print '%s_res.npy'%out

with open(filenamelist) as f:
    trainData = np.array([])
    responses = np.array([])
   
    idx = 0
    for line in f:
        line = line.strip()
        path, labels = line.split()
        


        log.info('get features from %s'%(path)) 

        im = feature.feature(path)

        responses = np.append(responses, np.float32(labels))
        trainData = np.append(trainData, np.float32(im.get_features()))
    
        trainData = np.array(trainData, dtype = np.float32)
        responses = np.array(responses, dtype = np.float32)
        trainData.shape = (-1, 281)
        responses.shape = (1, -1)

        idx += 1 

        if idx%100 == 0:
            np.save('.%s.npy'%out, trainData)
            np.save('.%s_res.npy'%out, responses)
            #np.save('./weatherdata/features.npy', trainData)
            #np.save('./weatherdata/res_features.npy', responses)
            log.info('save result %s'%idx)

np.save('.%s.npy'%out, trainData)
np.save('.%s_res.npy'%out, responses)
#np.save('./weatherdata/features.npy', trainData)
#np.save('./weatherdata/res_features.npy', responses)

log.info('get_features from dir %s  finished'%filenamelist)
    






