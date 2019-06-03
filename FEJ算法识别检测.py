

"""
从视屏中识别人脸，并实时标出面部特征点
"""

import dlib                     #人脸识别的库dlib
import numpy as np              #数据处理的库numpy
import cv2                      #图像处理的库OpenCv



class face_emotion():

    def __init__(self):
        # 使用特征提取器get_frontal_face_detector
        self.detector = dlib.get_frontal_face_detector()
        # dlib的68点模型，使用作者训练好的特征预测器
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        #建cv2摄像头对象，这里使用电脑自带摄像头，如果接了外部摄像头，则自动切换到外部摄像头
        self.cap = cv2.VideoCapture(0)
        # 设置视频参数，propId设置的视频参数，value设置的参数值
        self.cap.set(3, 480)
        # 截图screenshoot的计数器
        self.cnt = 0


    def learning_face(self):

        # 眉毛直线拟合数据缓冲
        line_brow_x = []
        line_brow_y = []

        # cap.isOpened（） 返回true/false 检查初始化是否成功
        while self.cap.isOpened():

            # cap.read()
            # 返回两个值：
            #    一个布尔值true/false，用来判断读取视频是否成功/是否到视频末尾
            #    图像对象，图像的三维矩阵
            flag, im_rd = self.cap.read()

            # 每帧数据延时1ms，延时为0读取的是静态帧
            k = cv2.waitKey(1)

            # 取灰度
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)

            # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数rects
            faces = self.detector(img_gray, 0)

            #print(faces)打印的为一个列表，其中记录着被识别出的人脸的位置——【【（左，上）（右，下）】】//对应于cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))

            # 待会要显示在屏幕上的字体
            font = cv2.FONT_HERSHEY_SIMPLEX

            # 如果检测到人脸
            if(len(faces)!=0):

                # 对每个人脸都标出68个特征点
                for i in range(len(faces)):
                    # enumerate方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
                    for k, d in enumerate(faces):
                        # 用红色矩形框出人脸
                        
                        cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))
                        # 计算人脸热别框边长
                        self.face_width = d.right() - d.left()
                        face_width = d.right() - d.left()
                        heigth = d.bottom() - d.top()

                        L = heigth
                        T = face_width

                        # 使用预测器得到68点数据的坐标
                        shape = self.predictor(im_rd, d)
                        # 圆圈显示每个特征点
                        for i in range(68):
                            cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
                            cv2.putText(im_rd, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                        (255, 255, 255))
                            #标记特征点的序号
                            #FEJ算法
                            Fe = ((shape.part(41).y-shape.part(37).y)/L+(shape.part(40).y-shape.part(38).y)/L)+((shape.part(47).y-shape.part(43).y)/L+(shape.part(46).y-shape.part(44).y)/L)
                            Fm = (shape.part(54).x-shape.part(48).x)/T+((shape.part(57).y-shape.part(51).y)/L+(shape.part(66).y-shape.part(62).y)/L)
                            Fo = ((shape.part(39).y-shape.part(21).y)/L+(shape.part(42).y-shape.part(22).y)/L)

          
                            x1 = shape.part(54).x-shape.part(66).x
                            y1 = shape.part(54).y-shape.part(66).y

                            x2 = shape.part(66).x-shape.part(48).x
                            y2 = shape.part(66).y-shape.part(48).y

                            k1 = ((y1)/(x1))*(-1)
                            k2 = ((y2)/(x2))

                            Fk = (k1+k2)
                            FEJ = (0.6)*Fe+(0.2)*Fm+(0.5)*Fo+(0.4*Fk)
                            FEJ = round(1000*FEJ,3)
                            #print('============'+str(FEJ)+'================')

                            if FEJ < 300:
                                cv2.putText(im_rd, "angry "+str(((300-FEJ)/300)*100)+'%', (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255), 2, 4)
                                #print(str((FEJ/300)*100)+'%')
                            elif 300 <= FEJ < 600:
                                cv2.putText(im_rd, "nature "+str(((FEJ-300)/300)*100)+'%', (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255), 2, 4)
                                #print(str(((FEJ-300)/300)*100)+'%')
                            elif 600 <= FEJ < 900:
                                cv2.putText(im_rd, "happy "+str(((FEJ-600)/300)*100)+'%', (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255), 2, 4)
                                #print(str(((FEJ-600)/300)*100)+'%')
                            elif 900 <= FEJ :
                                cv2.putText(im_rd, "amazing "+str(((FEJ-900)/300)*100)+'%', (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255), 2, 4)
                                #print(str(((FEJ-900)/300)*100)+'%')
                            

                            
                     

                # 标出人脸数
                cv2.putText(im_rd, "Faces: "+str(len(faces)), (20,50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                # 没有检测到人脸
                cv2.putText(im_rd, "No Face", (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

            # 添加说明
            im_rd = cv2.putText(im_rd, "S/s: screenshot", (20, 400), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            im_rd = cv2.putText(im_rd, "E/e: exit", (20, 450), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            # 窗口显示
            cv2.imshow("camera", im_rd)
            
            
            
            if cv2.waitKey(1) == ord('s') or cv2.waitKey(1) == ord('S'):
                cv2.imwrite('test.jpg',im_rd)
                print('拍摄成功！')
            if cv2.waitKey(1) == ord('e') or cv2.waitKey(1) == ord('E'):
                print('准备退出！')
                print('退出成功！')
                break

            

        # 释放摄像头
        self.cap.release()

        # 删除建立的窗口
        cv2.destroyAllWindows()


if __name__ == "__main__":
    my_face = face_emotion()
    my_face.learning_face()

