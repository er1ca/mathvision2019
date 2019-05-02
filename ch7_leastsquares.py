import os
import numpy as np
from PIL import Image
import cv2

img = cv2.imread('./hw10_sample.png',0) #(353, 438)
size = img.shape
img = img.flatten() #(154614,)

Matrix_A = np.zeros((154614,6),dtype=np.float32)

n = img.shape[0]
width = size[1]
x = 0
y = 0

# model I(x,y) = a*x*x + b*y*y + c*x*y + d*x + e*y + f
for i,m in enumerate(Matrix_A):
	y = int((i-x)/width)
	x = int(i - width*y)%width
	
	#print(i,x,y)
	m[0] = np.power(x,2)
	m[1] = np.power(y,2)
	m[2] = x * y
	m[3] = x
	m[4] = y
	m[5] = 1
#print(Matrix_A.shape) # (154614,6)
pinvA = np.linalg.pinv(Matrix_A) #print(pinv.shape) #(6, 154614)

coe = np.matmul(pinvA, img) # coefficients a,b,c,d,e,f 
#print(coe.shape) #(6,)

#최소 p
tmp= np.linalg.inv(np.matmul(Matrix_A.T,Matrix_A))
tmp2 = np.matmul(tmp,Matrix_A.T)
p = np.matmul(tmp2,img)
#print(p.shape)
app = np.matmul(Matrix_A,coe)
result = app-img
result = result.reshape(size)

ret,img_result1 = cv2.threshold(result,11,255,cv2.THRESH_BINARY)

cv2.imshow("t",result)
cv2.imshow("result",img_result1)

cv2.waitKey(0)

