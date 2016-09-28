#coding=utf-8
import cv2, math, random
import numpy as np

class feature:
    def __init__(self, img, difference=None ):
        self.features = None
        if type(img) == str:
            self.image = cv2.imread(img)
            self.diff = self.image
            self.name = img 
        else:
            self.image = img
            self.diff = difference

    #计算纹理特征
    #size 为图像取样大小
    #gray 为缩小的灰度值
    #random 图片上随机取块进行计算
    @staticmethod
    def GLCM(img, sizex=127, sizey=127, gray=16):
        #print img.shape,sizex,sizey
        #默认只计算四个方向的灰度共生矩阵
        Sym = np.linspace(0.0, 0.0, gray*gray*4)
        Sym.shape = gray, gray, 4
        
        #计算灰度图
        if len(img.shape) == 3:
            #Gray = img[:,:,0]/gray
            Gray = (img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114)/gray
            Gray = np.array(Gray, np.int8)
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
    
    #分块计算前后帧纹理特征，如果是单张图片就计算图片分块后本身的纹理特征
    def Grain(self):
        res = np.linspace(0.0, 0.0, 8*4) 
        #print self.diff.shape
        x, y = self.diff.shape[:2]
        x=x/4
        y=y/4
        for i in range(4):
            for j in range(2):
                res[(i+j*4)*4:(i+j*4+1)*4] = self.GLCM(self.diff[x*i:x*(i+1), y*j:y*(j+1)], x-1, y-1)        
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

    #灰度
    def DarkChannel(self):
        im = cv2.resize(self.image, (512,512), interpolation=cv2.INTER_CUBIC)   

        #计算每个像素RGB中最小的通道
        DarkChannel = np.array([ np.array([ np.min(im[j, i, :]) for i in range(512) ])  for j in range(512) ])   
        #最小值滤波 
        for i in range(4, DarkChannel.shape[0]-4, 1):
            for j in range(4, DarkChannel.shape[1]-4, 1):
                DarkChannel[i,j] = np.min(DarkChannel[i-4:i+4, j-4:j+4])
        self.HazeMedian = np.linspace(0, 0, 85)
        self.HazeMedian[0] = np.median(DarkChannel)
        #cv2.imwrite('./DarkChannel/%s'%self.name, DarkChannel) 
        l = 1
        for n in [2,4,8]:
            for i in range(n):
                for j in range(n):
                    self.HazeMedian[i+j+l] = np.median(DarkChannel[512/n*i:512/n*(i+1),512/n*j:512/n*(j+1)])
            l += n**2
        
        #print self.HazeMedian[0],85个特征
        return self.HazeMedian

    #计算图片特征值
    def get_features(self):
        #8*4 + 20 + 85 一共137个特征
        self.features = np.linspace(0, 0, 137)
        self.features[0:20] = self.HSVhistogram()
        self.features[20:105] = self.DarkChannel()
        self.features[105:] = self.Grain()
        self.features = np.array(self.features, dtype=np.float32)
        self.features.shape = 1,137
        return self.features 

    #用svm分类天气
    def svmclassify(self):
        weather_map = ['fog','snow','sunny','rain']
        svm = cv2.SVM()
        svm.load('svm_data.dat')
        return weather_map[int(svm.predict_all(self.get_features())[0][0])]

