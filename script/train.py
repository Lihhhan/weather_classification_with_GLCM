#coding=utf-8
import numpy as np
import sys
sys.path.append('.')
import cv2, os, feature, mylog 

#训练
def train(data='features.npy', c=500, g=5.383, kernel=cv2.ml.SVM_LINEAR, ttype=cv2.ml.SVM_C_SVC):
    
    logging = mylog.mylog('train')
    logging.info('train start ...')

    trainData = np.load('./data/%s' %data)
    responses = np.load('./data/res_%s' %data).ravel()
    responses = np.array(responses, dtype=np.int64) 

    svm = cv2.ml.SVM_create()
    svm.setGamma(g)
    svm.setC(c)
    svm.setKernel(kernel)
    svm.setType(ttype)



    svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    #svm.save('svm_data.dat')
    logging.info('train end .')
    #print svm.predict(trainData)


    log = mylog.mylog('predict')
    log.info('predict start ...')

    testData = np.load('./data/featuress.npy')
    responsess = np.load('./data/res_featuress.npy')

    label = responsess.ravel()
    
    r = svm.predict(testData)[1].ravel()
   
    
    error = (r!=label).mean()
    return [error, svm]

kernel = [cv2.ml.SVM_CUSTOM, cv2.ml.SVM_LINEAR, cv2.ml.SVM_POLY, cv2.ml.SVM_RBF, cv2.ml.SVM_SIGMOID, cv2.ml.SVM_CHI2, cv2.ml.SVM_INTER]
ttype = [cv2.ml.SVM_C_SVC, cv2.ml.SVM_NU_SVC, cv2.ml.SVM_ONE_CLASS, cv2.ml.SVM_EPS_SVR, cv2.ml.SVM_NU_SVR]
cc = np.linspace(0.1, 1000.0, 10000)
gg = np.linspace(0.1, 20.0, 200)

mmax = 0
for k in kernel:
    for t in ttype:
        for g in gg:
            for c in cc:
                res = train('features.npy', c, g, k, t)
                if res[0] > mmax:
                    res[1].save('max.dat')            
                    mmax = res[0]
                    print mmax





