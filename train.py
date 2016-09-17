#coding=utf-8
import numpy as np
import cv2, os, feature, logging

logging.basicConfig(level=logging.DEBUG,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
    filename='train.log',
    filemode='a')


#训练
def train(c=500, g=5.383):

    logging.info('train start ...')
    svm_params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C = c, gamma=g )

    s = 0
    for root, dirs, files in os.walk('./images'):
        i = 0
        for d in dirs:
            for rroot, ddirs, ffiles in os.walk('./images/%s'%d):
                s += len(ffiles)
        
        trainData = np.linspace(0.0, 0.0, 169*s)
        trainData.shape = s, 169

        responses = np.linspace(0, 0.0, s)
        responses.shape = s,1

        for d in dirs:    
            for rrroot, dddirs, fffiles in os.walk('./images/%s'%d):
                for f in fffiles:
                    responses[i] =  np.float32(d)
                    logging.info('get features : ' + './images/%s/%s' %(d,f))
                    im = feature.feature('./images/%s/%s' %(d,f))
                    trainData[i] = np.float32(im.get_features())
                    i += 1

        #不重复进其他子目录
        break

    #这是个坑，只支持float32类型
    trainData = np.array(trainData, dtype = np.float32)
    responses = np.array(responses, dtype = np.float32)
    svm = cv2.SVM()
    svm.train(trainData, responses, params=svm_params)
    svm.save('svm_data.dat')
    logging.info('train end .data is saved')

train()
