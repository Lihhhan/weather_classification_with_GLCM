import cv2, sys
import numpy as np

svm = cv2.ml.SVM_create()
svm = cv2.ml.SVM_load('./min.dat')

f = sys.argv[1]

if sys.argv[2] == '1':
    testData = np.load('./weatherdata/%s.npy'%f)
    responsess = np.load('./weatherdata/res_%s.npy'%f)
else:
    testData = np.load('./file/data/%s.npy'%f)
    responsess = np.load('./file/data/res_%s.npy'%f)

label = responsess.ravel()

r = svm.predict(testData)[1].ravel()

s = [ 0.0 for i in xrange(4) ]
res = [ [ 0.0 for i in xrange(4) ] for i in xrange(4)]


m = ['rainy', 'cloudy', 'snowy', 'sunny']

for i in xrange(len(r)):
    s[int(label[i])] += 1
    res[int(label[i])][int(r[i])] += 1 

print res, s 
for i in xrange(len(s)):
    print 'the %s class accurity is :%s'%(m[i], res[i][i]/s[i])
    for j in xrange(len(s)):
        if i != j:
            print 'the %s class mis classify %s :%s'%(m[i], m[j], res[i][j]/s[i])
print 'sum acc is %s:' %( sum([res[0][0], res[1][1], res[2][2], res[3][3]])/sum(s) )


