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

        self.weathers = { 'fog':0, 'snow':0, 'rain':0, 'sunny':0}
        self.queue = Queue.Queue(10)
    
    def r(self):
        ret, self.frame_pre = self.cap.read()
        self.frame_pre = cv2.blur(self.frame_pre, (5, 5))
        while(self.cap.isOpened()):
            ret, self.frame = self.cap.read()
            if ret == True:
                play = self.frame
                #图像平滑, 平均滤波器
                self.frame = cv2.blur(self.frame, (5,5))
                
                #前后帧做差，前后帧相同就不重新检测
                if not (self.frame == self.frame_pre).all():
                    diff = self.frame - self.frame_pre
                else:
                    continue
                
                #计算特征
                features = feature.feature(self.frame, diff)
                res = features.svmclassify()

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

