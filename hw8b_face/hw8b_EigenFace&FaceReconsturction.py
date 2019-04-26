import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random 
from operator import add
from scipy import spatial
from sklearn.decomposition import PCA

def EigenFaces10(size,data):
	#mean, eig_vec = cv2.PCACompute(data[9:-1],mean=None, maxComponents=10)
	#averageFace = mean.reshape(size) 
	averageFace = data.mean(axis=0)
	pca = PCA(n_components=10)
	pca.fit(data)
	eig_vec = pca.components_
	eigenFaces = []
	
	######## visulizing EigenFaces #######
	for e in eig_vec:
		eigenFace = e.reshape(size)
		eigenFaces.append(eigenFace)

	rows = 2
	cols = 5
	plt.figure()
	for i in range(rows * cols):
		plt.subplot(rows, cols, i + 1)
		plt.imshow(eigenFaces[i], cmap=plt.cm.gray)
		plt.xticks(())
		plt.yticks(())

	plt.show()
	

def reconstruction(size,data):
	K = [1, 10, 100, 200]

	Person = random.sample(range(40),k=1)
	print(Person)
	
	newOrigin = []
	faceVector = 0
	nO = data[Person]-data.mean(axis=0)
	for i,k in enumerate(K):
		#v , ev = cv2.PCACompute(data, mean=None, maxComponents=k) 
		pca = PCA(n_components=k)
		pca.fit(data)
		ev = pca.components_
		
		weight = nO.dot(ev.T).flatten()
		print(weight.shape)
		faceVector = sum([weight[j]*ev[j] for j in range(len(ev))])

		newOrigin.append(faceVector)
	
	plt.figure()
	rows = 1
	cols = 5 

	ax1 = plt.subplot(rows,cols, 1)
	plt.imshow(data[Person].reshape(size),cmap='gray')	
	plt.xticks(())
	plt.yticks(())
	ax1.set_xlabel('Person'+str(Person[0]+1))
	for i,k in enumerate(K) :
		ax2 = plt.subplot(rows,cols, i+2)
		plt.imshow(newOrigin[i].reshape(size),cmap='gray')
		ax2.set_xlabel('k='+str(k))
		plt.xticks(())
		plt.yticks(())
		
	plt.show()

	#return newOrigin


if __name__ == '__main__':
	dirPath = "./att_faces"
	files = os.listdir(dirPath)

	imgList = []
	for i in range(40):
		i += 1
		plist = []
		for j in range(10):
			j += 1
			fileName = 's'+str(i)+'_'+str(j)+'.png'
			img = cv2.imread(dirPath+'/'+fileName,0)
			plist.append(img)	
		imgList.append(plist)

	numImages = 400
	size = [56, 46]

	data = np.zeros((numImages, size[0]*size[1]),dtype=np.float32)
	for i in range(40):
		for j in range(10):
			image = imgList[i][j].flatten()
			data[i,:] = image

	####### 10 Eigen Faces #######
	EigenFaces10(size,data)

	####### Reconstruction #######
	reconstruction(size,data)

	cv2.waitKey(0)
	cv2.destroyAllWindows()


