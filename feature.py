#coding=utf-8
import cv2, math, random
import numpy as np

class feature:
    def __init__(self, arv):
        self.features = None
        if type(arv) == str:
            self.image = cv2.imread(arv)
            self.name = arv
        else:
            self.image = arv 

    #计算雨雪的纹理特征
    def Grain(self):
        #图像格式化
        im = cv2.resize(self.image, (512,512), interpolation=cv2.INTER_CUBIC)
        im = cv2.blur(im, (5,5))

        features = np.linspace(0.0, 0.0, 4*4*4) 
        features.shape = 16, 4

        #把图片分为4*4 的小图片分别计算纹理特征
        for i in range(4):
            for j in range(4):
                features[i*4+j] = feature.GLCM(im[i*128:(i+1)*128, j*128:(j+1)*128])
        features.shape = 64
        return features

    #计算特征向量
    #size 为图像取样大小
    #gray 为缩小的灰度值
    #random 图片上随机取块进行计算
    @staticmethod
    def GLCM(img, sizex=127, sizey=127, gray=16):

        #默认只计算四个方向的灰度共生矩阵
        Sym = np.linspace(0.0, 0.0, gray*gray*4)
        Sym.shape = gray, gray, 4
        
        #计算灰度图
        if len(img.shape) == 3:
            #Gray = img[:,:,0]/gray
            Gray = (img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114)/gray
        else:
            Gray = img/gray 
        
        #统计灰度共生矩阵
        for i in range(sizex):
            for j in range(sizey):
                #对四个方向采点，p[0]是原点
                p =  [Gray[i, j], Gray[i, j+1], Gray[i-1, j+1],Gray[i+1, j], Gray[i+1, j+1]]  

                Sym[p[0], p[1], 0] += 1 
                Sym[p[1], p[0], 0] += 1

                Sym[p[0], p[2], 1] += 1                  
                Sym[p[2], p[0], 1] += 1

                Sym[p[0], p[3], 2] += 1
                Sym[p[3], p[0], 2] += 1

                Sym[p[0], p[4], 3] += 1
                Sym[p[4], p[0], 3] += 1


        #归一化
        for i in range(4):
            Sum = sum(sum(Sym[:, :, i]))
            Sym[:,:,i] /= Sum
        
        #计算能量，熵，惯性矩，相关度四个参数，一共四个方向
        features = np.linspace(0.0, 0.0, 4*4)
        features.shape = 4, 4
        
        #相关性中间值
        Ux = np.linspace(0.0, 0.0, 4)
        Uy = np.linspace(0.0, 0.0, 4)
        deltaX = np.linspace(0.0, 0.0, 4)
        deltaY = np.linspace(0.0, 0.0, 4)
        
        for n in range(4):
            #能量
            features[0, n] = sum(sum(Sym[:, :, n]**2))
            for i in range(gray):
                for j in range(gray):
                    value = Sym[i, j ,n]
                    if value != 0:
                        #熵
                        features[1, n] -= value * math.log(value) 
                    features[2, n] += ((i-j)**2) * value
                    
                    Ux[n] += i * value
                    Uy[n] += j * value
        #计算相关性
        for n in range(4):
            for i in range(16):
                for j in range(16):
                    deltaX[n] += ((i-Ux[n])**2) * Sym[i, j ,n] 
                    deltaY[n] += ((j-Uy[n])**2) * Sym[i, j ,n]
                    features[3, n] += i*j*Sym[i, j, n]
            if deltaX[n] == 0 or deltaY[n] == 0:
                features[3, n] = 0
            else:
                features[3, n] = (features[3, n] - Ux[n]*Uy[n]) / deltaX[n]/deltaY[n]
        
        res = np.linspace(0.0, 0.0, 4)
        for i in range(4):
            res[i] = sum(features[i,:])/4
        #4个特征
        return res

    #色彩饱和度直方图，饱和度特征
    def HSVhistogram(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        #S饱和度取值0-256，饱和度直方图
        hist = cv2.calcHist([hsv], [1], None, [256], [0,256])
        pixels_num = float(self.image.shape[0]*self.image.shape[1])
        
        self.Contrast = np.linspace(256, 256, 20) 
        #统计像素百分比
        Count = [ sum(hist[:i])/pixels_num for i in range(len(hist))] 
        #print Count
        
        self.contrasttemp = sum(hist[i]*i/pixels_num for i in range(len(hist)))[0]
        #print self.contrasttemp

        i = 0
        for p in range(len(Count)):
            for q in range(i, int(Count[p]/0.05), 1):
                #print p,q,Count[p]
                self.Contrast[q] = p
                i = int(Count[p]/0.05)
        #占图像5%,10%..100%的饱和度的值,20个特征
        return self.Contrast


    #计算图片特征值
    def get_features(self):
        #16*4 + 20 + 85 一共169个特征
        self.features = np.linspace(0,0,105)
        self.features[0:20] = self.HSVhistogram()
        self.features[20:] = self.DarkChannel()
        self.features = np.array(self.features, dtype=np.float32)
        self.features.shape = 1,105
        return self.features 

    #用svm分类天气,只分晴雾
    def svmclassify(self):
        weather_map = ['fog','snow','sunny','rain']
        svm = cv2.SVM()
        svm.load('svm_data_fog2sunny.dat')
        return weather_map[int(svm.predict_all(self.get_features())[0][0])]

    @staticmethod
    def test(im):
        gray = im[:,:,0]*0.299 + im[:,:,1]*0.587 + im[:,:,2]*0.114
        print sum(sum(gray))/(im.shape[0]*im.shape[1])




