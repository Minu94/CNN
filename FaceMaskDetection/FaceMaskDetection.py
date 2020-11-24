from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
# cpaturing video frames
detector=MTCNN()
cap=cv2.VideoCapture(0)
if (cap.isOpened()==False):
    print("Error opening video stream or file")
while (cap.isOpened):
    ret,frame=cap.read()
    dt=detector.detect_faces(frame)
    for i in dt:
        box=i["box"]
        x,y,w,h=box[0],box[1],box[2],box[3]
        frame1=frame[y:y+h,x:x+w]
        frame1=cv2.resize(frame1,(96,96))
        frame1=np.expand_dims(frame1,axis=0)/255
        model=load_model("/home/user/Downloads/best_1.h5")
        rslt=model.predict(frame1)
        class_names=["with_mask","with_out_mask"]
        label=class_names[np.argmax(rslt)]
        cv2.putText(frame,label,(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        start_point=(x,y)
        end_point=(x+w,y+h)
        color=(255,0,0)
        frame=cv2.rectangle(frame,start_point,end_point,color)
        cv2.imshow("Video",frame)
    if cv2.waitKey(1) & 0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
           
