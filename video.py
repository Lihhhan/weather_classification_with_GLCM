#coding=utf-8
import numpy as np
import cv2, feature, Queue

#视频处理类，提取帧并提取feature做classify
class video:
    #输出字体
    font = cv2.FONT_HERSHEY_PLAIN

    #读取视频
    def __init__(self, name, size=51):
        self.cap = cv2.VideoCapture(name)
        self.capbak = cv2.VideoCapture(name)

        #雨雪取第一个结果，否则取第二个结果
        self.res = [ [0,0] for i in range(100)]
        
        #取前100帧,算出可能的结果再结合上下文挑选
        ret, frame = self.cap.read( )
        x, y, z = frame.shape
        self.x = y 
        self.y = x
        self.frames = np.linspace(0, 0, 100*size*size*3)
        self.frames.shape = 100, size, size, 3
        self.frames = np.array(self.frames, dtype=frame.dtype)
        #从中间位置取一个size*size的小块
        self.frames[0] = frame[x/2:x/2+size, y/2:y/2+size]

        im = feature.feature(frame)
        self.res[0] = [ im.svmclassify(),im.get_patch() ] 

        self.length = 1 
        while(self.length < 100 and self.cap.isOpened()):
            ret, frame = self.cap.read( )
            if ret == True:
                self.frames[self.length] = frame[x/2:x/2+size, y/2:y/2+size]

                im = feature.feature(frame)
                self.res[self.length] = [ im.svmclassify(),im.get_patch() ]
            #有效帧长度
            self.length += 1
            print self.length, self.res[self.length-1] 
        #当前帧
        self.cur = 0

        self.weathers = { 'fog':0, 'snow':0, 'rain':0, 'sunny':0}
        self.queue = Queue.Queue(10)
    
    def run(self, name):
    
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        print self.x, self.y
        out = cv2.VideoWriter(name, fourcc, 20.0, (self.x, self.y))

        while(self.length > 10):
            res = self.Handle_cur()
            ret, frame = self.capbak.read( )
            cv2.putText(frame, res, (0,50), self.font, 4, (255,255,255), 2)
            out.write(frame)
        out.release()

    def Handle_cur(self, N=50,size=51):
        K = np.array( np.linspace(1, 100, N), dtype=np.int)
        res = np.linspace(0, 0, N*size*size)
        res.shape = N, size, size
        count = 0 
        for k in K:
            #RGB
            temp = np.linspace(0, 0, size*size*3)
            temp.shape = size, size, 3

            #print range(self.cur, self.cur+self.length-k, 1), range(self.cur, self.cur+self.length, 1), k
            for i in range(3):
                ave = sum(self.frames[:,:,:,i])/float(self.length)
                #return ave
                up = np.linspace(0, 0, size*size)
                down = np.linspace(0, 0, size*size)
                up.shape = size, size
                down.shape = size, size
                for j in range(self.cur, self.cur+self.length-k, 1):
                    up += ( self.frames[(j+k)%self.length,:,:,i] - ave ) * ( self.frames[j%self.length,:,:,i] - ave )
                for j in range(self.cur, self.cur+self.length, 1):    
                    down += ( self.frames[j%self.length,:,:,i] - ave ) ** 2
                #print np.sum(down)/2601
                temp[:,:,i] = up/down

            res[count] = np.amax(temp, axis=2) 
            count += 1

        r = [ 0 for i in range(size**2)]
        count = 0
        for i in range(size):
            for j in range(size):
                a, b, r[count] = np.polyfit(K, res[:,i,j], 2)
                if np.isnan(r[count]) or r[count] == np.inf:
                    count -= 1
                    r[count] == 0
                count += 1
        print sum(r)/float(count), self.res[self.cur]
        if sum(r)/float(count) > 0.98:
            res = self.res[self.cur][0]
        else:
            res = self.res[self.cur][1]
            
        #读入下一帧
        ret, frame = self.cap.read( )
        if ret == True:
            x,y,z = frame.shape
            self.frames[self.cur] = frame[x/2:x/2+size, y/2:y/2+size]
   
            im = feature.feature(frame)
            self.res[self.cur] = [ im.svmclassify(),im.get_patch() ]
        else:
            self.length -= 1
        self.cur += 1
        self.cur %= 100
        return res

    #视频段中时间域特征,size 为选取区域大小， N为二项式拟合计算次数
    def Temporal_feature(self, size=51, N=10):
        #帧队列，计算自相关函数
        ret, frame = self.cap.read( )
        x, y, z = frame.shape
        frames = np.linspace(0, 0, 100*size*size*z)
        frames.shape = 100, size, size, z
        frames = np.array(frames, dtype=frame.dtype)
        frames[0] = frame[x/2:x/2+size, y/2:y/2+size]

        #init, 取100帧
        length = 1
        while(length < 100 and self.cap.isOpened()):
            try:
                frames[length] = self.cap.read()[1][x/2:x/2+size, y/2:y/2+size]
            except:
                break
            length += 1
            #cv2.imshow('frame', frames[length-1])
        
        X = np.linspace(1,100, N) 
        res = np.linspace(0, 0, size*size*N)
        res.shape = N, size, size
        count = 0
        for interval in X:
            #平均值矩阵
            ave = np.linspace(0, 0, size*size*z)
            ave.shape = size, size, z
            up = 0
            down = 0

            for i in range(size):
                for j in range(size):
                    temp = [ 0 for ii in range(z)]
                    for p in range(z):
                        ave[i, j, p] = float(sum(frames[:length, i, j, p]))/length
                         
                        up =0
                        down = 0
                        for m in range(int(interval)):
                            down += (frames[m, i, j, p] - ave[i, j, p])**2
                        for q in range(length-int(interval)):
                            up += (frames[q+int(interval), i, j, p] - ave[i, j, p]) * (frames[q, i, j, p] - ave[i, j, p])
                            down += (frames[q+int(interval), i, j, p] - ave[i, j, p])**2
                        if down != 0: 
                            temp[p] = float(up)/down
                        else: 
                            temp[p] = 1
                    
                    res[count, i, j] = max(temp)
            count += 1
            return ave[:,:,0]

        r = [ 0 for i in range(size**2)]
        count = 0
        for i in range(size):
            for j in range(size):
                a, b, r[count] = np.polyfit(X, res[:,i,j], 2)
                count += 1
        print sum(r)/float(size**2)
        return sum(r)/float(size**2)

    def r(self):
        ret, self.frame_pre = self.cap.read()
        self.frame_pre = cv2.blur(self.frame_pre, (5, 5))
        while(self.cap.isOpened()):
            ret, self.frame = self.cap.read()
            if ret == True:
                play = self.frame
                #图像平滑, 平均滤波器
                #self.frame = cv2.blur(self.frame, (5,5))
                
                #前后帧做差，前后帧相同就不重新检测
                #if not (self.frame == self.frame_pre).all():
                #    diff = self.frame - self.frame_pre
                #else:
                #    continue
                
                #计算特征
                features = feature.feature(self.frame)
                #res = features.classify()
                res = features.get_patch()

                if self.queue.full():
                    self.weathers[self.queue.get()] -= 1
                self.weathers[res] += 1
                self.queue.put(res) 
                #最近10次检测出现次数最多的为res 
                M = max([self.weathers[i] for i in self.weathers])
                for w in self.weathers:
                    if self.weathers[w] == M:
                        res = w
                        break
                print self.weathers, self.queue.qsize()

                cv2.putText(play, res, (0,50), self.font, 4, (255,255,255), 2)

                #显示
                #cv2.imshow('frame', play)
                #self.frame_pre = self.frame 
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
            else:
                break

