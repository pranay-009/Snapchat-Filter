
import cv2
import os
import sys
import numpy as np

def read_filter(file_name):
    #read the filter image and convert it into rgb format
    image=cv2.imread(file_name)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image

def get_width_height(Index):
    #get height and width
    _,_,width,height=cv2.boundingRect(np.array(Index))
    return width,height

def transformed_image(filter_image,image,h,w,landmarks):

    ht,wt,_=filter_image.shape
    req_height=int(h*2.5)
    filter_image=cv2.resize(filter_image,(int(wt*(req_height/ht)),req_height))
    filter_image=cv2.cvtColor(filter_image,cv2.COLOR_BGR2RGB)
    #filter image size
    filter_img_height, filter_img_width, _  = filter_image.shape
    _, filter_img_mask = cv2.threshold(cv2.cvtColor(filter_image, cv2.COLOR_BGR2GRAY),25, 255, cv2.THRESH_BINARY_INV)
    
    landmarks=np.array(landmarks)
    #finding the centre region or middle point 
    center=landmarks.mean(axis=0).astype(dtype="int")
    location = (int(center[0]-filter_img_width/2), int(center[1]-filter_img_height/2))
    
    ROI=image[location[1]:location[1]+filter_img_height,location[0]:location[0]+filter_img_width]
    #bitwise operator removing the dark portion of the image
    resultant_image = cv2.bitwise_and(ROI, ROI, mask=filter_img_mask)
    
    resultant_image=cv2.add(resultant_image,filter_image)
    
    image[location[1]:location[1]+filter_img_height,location[0]:location[0]+filter_img_width]=resultant_image
    
    return image



