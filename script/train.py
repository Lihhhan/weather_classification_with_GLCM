#coding=utf-8
import numpy as np
import sys, time
sys.path.append('.')
import cv2, os, feature, mylog 

#训练
def train(data='features.npy', test='test.npy', c=500, g=5.383, kernel=cv2.ml.SVM_LINEAR, ttype=cv2.ml.SVM_C_SVC):
    
    logging = mylog.mylog('train')
    logging.info('train start ...')

    trainData = np.load('./weatherdata/%s' %data)
    responses = np.load('./weatherdata/res_%s' %data).ravel()
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

    testData = np.load('./weatherdata/%s'%test)
    responsess = np.load('./weatherdata/res_%s'%test)

    label = responsess.ravel()
    
    r = svm.predict(testData)[1].ravel()
    
    error = (r!=label).mean()
    return [error, svm]
#kernel = [cv2.ml.SVM_CUSTOM, cv2.ml.SVM_LINEAR, cv2.ml.SVM_POLY, cv2.ml.SVM_RBF, cv2.ml.SVM_SIGMOID, cv2.ml.SVM_CHI2, cv2.ml.SVM_INTER]
kernel = [cv2.ml.SVM_CUSTOM, cv2.ml.SVM_LINEAR, cv2.ml.SVM_RBF, cv2.ml.SVM_SIGMOID, cv2.ml.SVM_CHI2, cv2.ml.SVM_INTER]
#kernel = [cv2.ml.SVM_CHI2, cv2.ml.SVM_INTER]
ttype = [cv2.ml.SVM_C_SVC, cv2.ml.SVM_NU_SVC, cv2.ml.SVM_ONE_CLASS, cv2.ml.SVM_EPS_SVR, cv2.ml.SVM_NU_SVR]
cc = np.linspace(0.1, 100.0, 1000)
gg = np.linspace(0.1, 20.0, 20)

mmin = 1

for g in gg:
    start_time = time.time()
    #print '%s kernel train start'%kernel.index(k)
    idx = 0
    #for t in ttype:
    for c in cc:
        for k in kernel:
            res = train('test.npy', 'validation.npy', c, g, k, cv2.ml.SVM_C_SVC)
            print 'the %s idx train start, acc is %s, min_acc is %s.'%(idx, res[0], mmin)
            idx += 1
            if res[0] < mmin:
                res[1].save('max.dat')
                mmin = res[0]
            elif res[0] - mmin > 0.3:
                print 'the %s idx train start, acc is %s, min_acc is %s break..'%(idx, res[0], mmin)
                kernel.pop(kernel.index(k))
                #break
    print '%s kernel train ended %s s spend.'%(kernel.index(k), time.time() - start_time)

'''

print ttype

mmin = 1 
for k in kernel:
    for t in ttype:
        print kernel.index(k), ttype.index(t)
        for g in gg:
            for c in cc:
                res = train('features.npy', c, g, k, t)
                if res[0] < mmin:
                    res[1].save('max.dat')            
                    mmin = res[0]
                    print mmin

'''


