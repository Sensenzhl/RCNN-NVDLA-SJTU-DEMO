#!/home/abcd/anaconda2/bin/python2.7
import cv2
import scipy.io as scio
from google.protobuf import text_format as proto_text

mean = scio.loadmat('/home/abcd/rcnn/sjtu_demo/project/rcnn/mean/image_mean.mat')
#print(mean)

data = cv2.imread('/home/abcd/rcnn/sjtu_demo/project/rcnn/images/cropped_image_184.jpg')
data = data[:,:,[2,0,1]]
#
print(data[0,0:10,0])
print(data[0,0:10,1])
print(data[0,0:10,2]) 
