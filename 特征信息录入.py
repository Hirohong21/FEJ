import os
import cv2
import dlib
from skimage import io
import numpy as np
import openpyxl

path_faces_rd = "data_faces_from_camera/test/"

#创建表格

wb = openpyxl.Workbook()
#wb.active就是获取这个工作薄的活动表，通常就是第一个工作表。
sheet = wb.active
sheet.title = 'test'

sheet['A1'] ='姓名'     #加表头，给A1单元格赋值
sheet['B1'] ='人脸面积'  
sheet['C1'] ='嘴宽与框之比'   
sheet['D1'] ='嘴高与框之比'
sheet['E1'] ='眉毛斜率'
sheet['F1'] ='眉高与框之比'
sheet['G1'] ='眉间距与框之比'
sheet['H1'] ='眼开距与框之比'
sheet['I1'] ='判断情绪'


faces = os.listdir(path_faces_rd)
print(faces,end = '\n\n')

#print(faces[0])
# 使用特征提取器get_frontal_face_detector
detector = dlib.get_frontal_face_detector()
# dlib的68点模型，使用作者训练好的特征预测器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# 图片所在路径
for i in range(len(faces)):
    
    print('开始录入第'+str(i+1)+'张图片！')
    emotion = ''
    person_cho = i #int(input('请问你要录入第几张图片？'))
    img_in = faces[person_cho]
    print(img_in[:-4])
    img = io.imread(path_faces_rd+img_in)

    # 生成dlib的图像窗口
    #win = dlib.image_window()
    #win.clear_overlay()
    #win.set_image(img)


    line_brow_x = []
    line_brow_y = []

    # 特征提取器的实例化
    dets = detector(img, 1)

    print("人脸数：", len(dets))

    for k, d in enumerate(dets):
    
    
        print("第", k+1, "个人脸d的坐标：",
            "left:", d.left(),
            "right:", d.right(),
            "top:", d.top(),
            "bottom:", d.bottom())

        face_width = d.right() - d.left()
        heigth = d.bottom() - d.top()
        face_s = face_width*heigth
        print('人脸面积为：',(face_s))

        # 待会要显示在屏幕上的字体
        font = cv2.FONT_HERSHEY_SIMPLEX
        #人脸识别红框
        cv2.rectangle(img,(d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))
        #人脸的相应数据标注
        img = cv2.putText(img,"Num:"+str(k+1),(5, 10),font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        img = cv2.putText(img,"L:"+str(d.left()),(5, 25),font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        img = cv2.putText(img,"R:"+str(d.right()),(5, 40),font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        img = cv2.putText(img,"T:"+str(d.top()),(5, 55),font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        img = cv2.putText(img,"B:"+str(d.bottom()),(5, 70),font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        img = cv2.putText(img, 'Area:'+str((face_width*heigth)),(5, 85), font,0.4, (0, 0, 255), 1, cv2.LINE_AA)
        img = cv2.putText(img, 'S/s: screenshot',(5, 110), font,0.4, (0, 0, 255), 1, cv2.LINE_AA)
        img = cv2.putText(img, 'E/e: exit',(5, 120), font,0.4, (0, 0, 255), 1, cv2.LINE_AA)
        


    
    


        # 利用预测器预测
        shape = predictor(img, d)
        # 标出68个点的位置
        for i in range(68):
            cv2.circle(img, (shape.part(i).x, shape.part(i).y), 3, (0, 255, 0), -1, 8)
            cv2.putText(img, str(i), (shape.part(i).x, shape.part(i).y),cv2.LINE_AA, 0.25, (255, 255, 255))
            
        
        print(shape.part(1).x , shape.part(1).y)
        print(shape.part(0).x , shape.part(0).y)

        input('')
        #嘴巴特点
        mouth_width = (shape.part(54).x - shape.part(48).x) / face_width  # 嘴巴咧开程度
        mouth_higth = (shape.part(66).y - shape.part(62).y) / heigth  # 嘴巴张开程度
        print("嘴巴宽度与识别框宽度之比：",round(mouth_width,3))
        print("嘴巴高度与识别框高度之比：",round(mouth_higth,3))

        # 通过两个眉毛上的10个特征点，分析挑眉程度和皱眉程度
        brow_sum = 0  # 高度之和
        frown_sum = 0  # 两边眉毛距离之和
        for j in range(17, 21):
            brow_sum += (shape.part(j).y - d.top()) + (shape.part(j + 5).y - d.top())
            frown_sum += shape.part(j + 5).x - shape.part(j).x
            line_brow_x.append(shape.part(j).x)
            line_brow_y.append(shape.part(j).y)

        
        
        tempx = np.array(line_brow_x)
        tempy = np.array(line_brow_y)
        z1 = np.polyfit(tempx, tempy, 1)  # 拟合成一次直线

        print(z1)
        brow_k = -round(z1[0], 3)  # 拟合出曲线的斜率和实际眉毛的倾斜方向是相反的
        
        print('眉毛的斜率：',brow_k)

        brow_hight = (brow_sum / 10) / heigth  # 眉毛高度占比
        brow_width = (frown_sum / 5) / face_width  # 眉毛距离占比
        print("眉毛高度与识别框高度之比：",round(brow_hight,3))
        print("眉毛间距与识别框高度之比：",round(brow_width,3))

        # 眼睛睁开程度
        eye_sum = (shape.part(41).y - shape.part(37).y + shape.part(40).y - shape.part(38).y +
                                       shape.part(47).y - shape.part(43).y + shape.part(46).y - shape.part(44).y)
        eye_hight = (eye_sum / 4) / heigth
        print("眼睛睁开距离与识别框高度之比：",round(eye_hight,3))
        
        # 分情况讨论
        # 张嘴，可能是开心或者惊讶
        if float(mouth_higth) >= 0.03:
            if float(eye_hight) >= 0.056:
                emotiom = 'amazing'
                print('判断情绪为：'+emotion+'\n\n')
            elif float(eye_hight) < 0.056:
                emotion = 'happy'
                print('判断情绪为：'+emotion+'\n\n')
            else:
                emotion ='error'

        # 没有张嘴，可能是正常和生气
        elif float(mouth_higth) < 0.03:
            if float(brow_k) <= 0.2:
                emotion = 'angry'
                print('判断情绪为：'+emotion+'\n\n')
            elif float(brow_k) > 0.2:
                emotion = 'nature'
                print('判断情绪为：'+emotion+'\n\n')
            else:
                emotion ='error'
        
        
        #录入数据
        sheet.append([img_in[:-4],round(face_s,3),round(mouth_width,3),round(mouth_higth,3),round(brow_k,3),round(brow_hight,3),round(brow_width,3),round(eye_hight,3),emotion])

wb.save('Test.xlsx')     

    #while True:
        #cv2.imshow('face', img)
        #if cv2.waitKey(1) == ord('s') or cv2.waitKey(1) == ord('S'):
            #cv2.imwrite(str(img_in[:-4])+'_result'+img_in[-4:],img)
            #print('保存识别结果成功！')
        #if cv2.waitKey(1) == ord('e') or cv2.waitKey(1) == ord('E'):
            #print('准备退出！')
            #break







    
cv2.destroyAllWindows()
input('退出成功！输入回车键结束录入')
        

    
