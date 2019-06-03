import cv2
import dlib
from skimage import io
import numpy as np



# 使用特征提取器get_frontal_face_detector
detector = dlib.get_frontal_face_detector()
# dlib的68点模型，使用作者训练好的特征预测器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# 图片所在路径
img_in = input('需要测试的图片(带文件格式):')
img = io.imread(img_in)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
# 生成dlib的图像窗口
win = dlib.image_window()
win.clear_overlay()
win.set_image(img)

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
    
    face_width = d.right() - d.left()
    heigth = d.bottom() - d.top()

    L = heigth
    T = face_width

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
            cv2.putText(img, "angry "+str(((300-FEJ)/300)*100)+'%', (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255), 2, 4)
            #print(str((FEJ/300)*100)+'%')
        elif 300 <= FEJ < 600:
            cv2.putText(img, "nature "+str(((FEJ-300)/300)*100)+'%', (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255), 2, 4)
            #print(str(((FEJ-300)/300)*100)+'%')
        elif 600 <= FEJ < 900:
            cv2.putText(img, "happy "+str(((FEJ-600)/300)*100)+'%', (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255), 2, 4)
            #print(str(((FEJ-600)/300)*100)+'%')
        elif 900 <= FEJ :
            cv2.putText(img, "amazing "+str(((FEJ-900)/300)*100)+'%', (d.left(), d.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 255), 2, 4)
            #print(str(((FEJ-900)/300)*100)+'%')


while True:
    cv2.imshow('face', img)
    if cv2.waitKey(1) == ord('s') or cv2.waitKey(1) == ord('S'):
        cv2.imwrite(str(img_in[:-4])+'_result'+img_in[-4:],img)
        print('保存识别结果成功！')
    if cv2.waitKey(1) == ord('e') or cv2.waitKey(1) == ord('E'):
        print('准备退出！')
        break
    
cv2.destroyAllWindows()
input('退出成功！输入回车键结束测试')
