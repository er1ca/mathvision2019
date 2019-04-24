import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import mahalanobis

def PCA(data):
	#get covarience matrix
	cov_matrix = np.cov(data.T)
	
	#Eigendecomposition
	eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
	
	eig_pairs = [(eig_vals[i],eig_vecs[:,i]) for i in range(len(eig_vals))]
	eig_pairs.sort(reverse=True) #내림차순 정렬
	#print(eig_pairs)

	# 투영행렬 W : 변수를 2차원으로 축소시키는 투영행렬. eigen_pairs의 0,1 번째만 개의 에이겐 쌍으로만 차원축소
	mat_w = np.hstack((eig_pairs[0][1].reshape(4,1),eig_pairs[1][1].reshape(4,1))) #4x2

	return mat_w, cov_matrix

def MahalanobisDistance(data_conc,mat_w,mu_a,mu_b, cov_a,cov_b):
	f_t = open("test.txt","r")
	data_t = []
	for line in f_t:
		data_t.append([float(n) for n in line.strip().split(',')])
	data_t = np.asarray(data_t)
	
	pca_t = data_t.dot(mat_w)
	
	#test case 1의 Mahalanobis 거리 
	print('test 1:',mahalanobis(pca_t[0], mu_a, np.linalg.inv(cov_a)),mahalanobis(pca_t[0], mu_b, np.linalg.inv(cov_b)))
	#test case 2의 Mahalanobis 거리 
	print('test 2:',mahalanobis(pca_t[1], mu_a, np.linalg.inv(cov_a)),mahalanobis(pca_t[1], mu_b, np.linalg.inv(cov_b)))


	pca = data_conc.dot(mat_w)
	fig4 = plt.figure()
	ax1 = fig4.add_subplot(1,1,1)

	for k,pca2 in enumerate(pca):
		if k < 1000:
			ax1.scatter(pca2[0],pca2[1],c='red',alpha=0.1)
		else : 
			ax1.scatter(pca2[0],pca2[1],c='blue',alpha=0.1)

	ax1.scatter(pca_t[0][0],pca_t[0][1],c="black")
	ax1.scatter(pca_t[1][0],pca_t[1][1],c="purple")

	#plt.show()

	

if __name__ == "__main__":

	######## Initialization data_a and data_b #######
	f_a = open("data_a.txt","r")
	f_b = open("data_b.txt","r")
	data_a = []
	for line in f_a:
		data_a.append([float(n) for n in line.strip().split(',')])
	data_a = np.asarray(data_a)
	data_a = data_a.reshape(1000,4)
	#print(data_a)

	data_b = []
	for line in f_b:
		data_b.append([float(n) for n in line.strip().split(',')])
	data_b = np.asarray(data_b)
	data_b = data_b.reshape(500,4)

	data_conc = np.concatenate((data_a,data_b), axis = 0) # (4, 1500) 
	#print(data_conc)
	
	##################  PCA #####################
	mat_w, cov_matrix = PCA(data_conc)

	pca = data_conc.dot(mat_w)
	
	fig1 = plt.figure()
	ax1=fig1.add_subplot(1,1,1)

	for i,pca2 in enumerate(pca):
		# print(i)	
		if i < 1000:
			ax1.scatter(pca2[0],pca2[1],c='red',alpha=0.5)
		else : 
			ax1.scatter(pca2[0],pca2[1],c='blue',alpha=0.5)

	pca_a = pca[slice(0,1000)]
	pca_b = pca[slice(1000,1500)]

	##########  2D Gaussian Distribution ############

	x = np.linspace(np.min(pca_a),np.max(pca_a),100)
	y = np.linspace(np.min(pca_b),np.max(pca_b),100)


	x, y = np.meshgrid(x,y)

	mu_a = np.array([np.mean(pca_a[:,0]),np.mean(pca_a[:,1])])
	mu_b = np.array([np.mean(pca_b[:,0]),np.mean(pca_b[:,1])])
	cov_a = np.cov(pca_a.T)
	cov_b = np.cov(pca_b.T)


	# pack x and y into a single 3-dimensional array
	pos = np.empty(x.shape + (2,))
	pos[:,:,0] = x
	pos[:,:,1] = y

	z_a = multivariate_normal.pdf(pos,mu_a,[cov_a[0][0], cov_a[1][1]])
	z_b = multivariate_normal.pdf(pos,mu_b,[cov_b[0][0], cov_b[1][1]])

	fig2 = plt.figure()
	
	ax1 = fig2.gca(projection='3d')
	ax1.plot_surface(x,y,z_a,cmap='coolwarm',linewidth=0,antialiased=False)
	
	
	fig3 = plt.figure()
	ax2 = fig3.gca(projection='3d')
	ax2.plot_surface(x,y,z_b,cmap='coolwarm',linewidth=0,antialiased=False)
	
	#plt.show()
	
	############## Mahallanobis  Distance ################

	MahalanobisDistance(data_conc, mat_w, mu_a,mu_b,cov_a,cov_b)

	f_a.close()
	f_b.close()




