from data import read_filter
from data import get_width_height
from data import transformed_image

import cv2
import numpy as np
import mediapipe as mp
import os
import sys
import itertools
import matplotlib.pyplot as plt


mp_facemesh=mp.solutions.face_mesh
mp_pose=mp.solutions.pose
face_mesh_images=mp_facemesh.FaceMesh(static_image_mode=True,max_num_faces=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)
face_mesh_videos=mp_facemesh.FaceMesh(static_image_mode=False,max_num_faces=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

threshold=24.00

def mouth_check(threshold,landmark):
    y1=landmark[0]
    y2=landmark[1]
    if(y2[1]-y1[1])>threshold:
        return True
    else:
        return False
def get_size(image,face_landmark,Index):
    h,w,c=image.shape
    l=[]
    for i in Index: 
        l.append([int(face_landmark.landmark[i].x*w),int(face_landmark.landmark[i].y*h)])
    return l


camera_video = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    count=0
    while True:
        try:
            _,frame=camera_video.read()
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame=cv2.flip(frame,1)
            img_copy=frame
            results=face_mesh_images.process(img_copy)
            Lft_Index=list(set(itertools.chain(*mp_facemesh.FACEMESH_LEFT_EYE)))
            Rgt_Index=list(set(itertools.chain(*mp_facemesh.FACEMESH_RIGHT_EYE)))
            Index=list(itertools.chain(*mp_facemesh.FACEMESH_LIPS))
            tong=list(set(itertools.chain(*mp_facemesh.FACEMESH_LIPS)))
            h,w,_=img_copy.shape
            l=[]
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye=get_size(img_copy,face_landmarks,Lft_Index)
                    right_eye=get_size(img_copy,face_landmarks,Rgt_Index)
                    tongue=get_size(img_copy,face_landmarks,tong)
                    l.append([int(face_landmarks.landmark[Index[13]].x*w),int(face_landmarks.landmark[Index[13]].y*h)])
                    l.append([int(face_landmarks.landmark[Index[14]].x*w),int(face_landmarks.landmark[Index[14]].y*h)])

            #print(l[1][1]-l[0][1])
            width,heigh=get_width_height(left_eye)
            width2,heigh2=get_width_height(right_eye)
            wid_tong,heig_tong=get_width_height(tongue)
            heart_eye_path=r"C:\Users\user\Downloads\anime.png"
            heart_eye=read_filter(heart_eye_path)
            mouth_filter_path=r"C:\Users\user\Downloads\money.png"
            mouth_filter=read_filter(mouth_filter_path)
            
            img_tranform=transformed_image(heart_eye,img_copy,heigh,width,right_eye)
            img_tranform=transformed_image(heart_eye,img_tranform,heigh,width,left_eye)
            #img_tranform=transformed_image(mouth_filter,img_tranform,heig_tong,wid_tong,tongue)
            


            #print(h,w)
            if(mouth_check(threshold,l)==True):
                img_tranform=transformed_image(mouth_filter,img_tranform,heig_tong,wid_tong,tongue)
                #cv2.putText(img_tranform,"Mouth open ",(100,50),cv2.FONT_HERSHEY_SIMPLEX,0.9, (255,0,0), 2, cv2.LINE_AA)
            else:
                pass
                #cv2.putText(img_tranform,"Mouth close ",(100,50),cv2.FONT_HERSHEY_SIMPLEX,0.9, (255,0,0), 2, cv2.LINE_AA)
            cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Resized_Window",700,700)
            img_copy = cv2.cvtColor(img_tranform,cv2.COLOR_BGR2RGB)
            cv2.imshow("Resized_Window",img_copy)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(e)
            break
                
            

    camera_video.release()
    cv2.destroyAllWindows()
