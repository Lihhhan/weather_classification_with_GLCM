#coding=utf-8
import numpy as np
import sys, os
sys.path.append('.')
import feature, mylog 

#argv[1] is the path of imgsource like '~/workspace/imgsource'
#example : python get_feature.py ~/workspace/imgsource/test

path = sys.argv[1]
log = mylog.mylog('caculate')

count = 0
for root, dirs, files in os.walk(path):
    for d in dirs: 
        for rroot, ddirs, ffiles in os.walk('%s/%s'%(path, d)):
            count += len(ffiles)

    trainData = np.linspace(0.0, 0.0, 281*count)
    trainData.shape = count, 281
    responses = np.linspace(0.0, 0.0, count)
    responses.shape = count,1

    i = 0 
    for d in dirs:
        for rroot, ddirs, ffiles in os.walk('%s/%s'%(path, d)):
            for f in ffiles:
                responses[i] =  np.float32(d)
                log.info('get features from %s/%s/%s'%(path, d, f))
                im = feature.feature('%s/%s/%s' %(path, d, f))
                trainData[i] = np.float32(im.get_features())
                i += 1

    trainData = np.array(trainData, dtype = np.float32)
    responses = np.array(responses, dtype = np.float32)

    np.save('./data/featuress.npy', trainData)
    np.save('./data/res_featuress.npy', responses)
    log.info('get_features from dir %s  finished'%path)
    
    #这是一个坑，这个循环会进两次..第二次是空值会覆盖掉有效值，所以要break！
    break






