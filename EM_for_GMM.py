import matplotlib.pyplot as plt

import numpy as np

from numpy import *

from google.colab import drive
drive.mount('/content/gdrive')

from numpy import genfromtxt as gft



!pip install scipy

#https://riptutorial.com/numpy/example/22990/reading-csv-files

X_train= gft("/content/gdrive/My Drive/P2/Prob2_Xtrain.csv", delimiter=",")

Y_train= gft("/content/gdrive/My Drive/P2/Prob2_ytrain.csv", delimiter=",")

X_train

Y_train

X_test = gft("/content/gdrive/My Drive/P2/Prob2_Xtest.csv", delimiter=",")

Y_test = gft("/content/gdrive/My Drive/P2/Prob2_ytest.csv", delimiter=",")

#Splitting X_train into its two associated output classes based on Y_train

X_train_class0=[]

X_train_class1= []

for j in range(len(X_train)):
  if Y_train[j]==1:
    X_train_class1.append(X_train[j])
  elif Y_train[j]==0:
    X_train_class0.append(X_train[j])

mean_class0= np.mean(X_train_class0, axis=0)

mean_class1= np.mean(X_train_class1, axis=0)

#print(mean_class0)

covariance_mat_class0= np.cov(X_train_class0, rowvar=False )

covariance_mat_class1= np.cov(X_train_class1, rowvar=False )

print(covariance_class0)

covariance_class1

from math import log as lg

import scipy.stats as ss

def get_L(data, num_clusters, mean_li, cov_li, phi_matrix, pi_vect):
  L_func_value= 0
  #print(pi_vect)

  #print(phi_matrix)

  for i in range(len(data)):
    for j in range(num_clusters):
      L_func_value += phi_matrix[i][j] *  ( lg(pi_vect[j]) + lg(ss.multivariate_normal.pdf(data[i], mean_li[j], cov_li[j], allow_singular=True) ) ) 

  print(L_func_value)
  return L_func_value

#def randomly_sample_mean( mean_li, covariance_mat):
  # sampled_mean= np.random.multivariate_normal(mean= mean_li, cov= covariance_mat, size= 1)[0]

ss= scipy.stats.multivariate_normal

from collections import defaultdict

import copy

def run_EM(data, num_clusters, mean_li, cov_mat, iters):

  generated_means=[]

  for _ in range(num_clusters):
    generated_means.append( np.random.multivariate_normal(mean= mean_li, cov= cov_mat, size= 1)[0] )

    #Initialising pi distribution [per point] as a uniform distribution
  pi_vect= [1/num_clusters for _ in range(num_clusters)]


  all_cov_mat_li=[]
  for i in range(num_clusters):
    all_cov_mat_li.append(cov_mat)


  L=[]


  for _ in range(iters):

    #Updating our probability distribution vectors for each data point, E step

    phi = [] # N*K matrix where K= num_clusters, N= number of data points= len(data)

    for i in range(len(data)):
      n_k=0
      dist_over_clusters= []

      for k in range(num_clusters):

        #calculating the probability of the data point belonging to cluster k by plugging it into the pdf
        pr= scipy.stats.multivariate_normal( generated_means[k], all_cov_mat_li[k], allow_singular=True ).pdf(data[i])

        dist_over_clusters.append( pr* pi_vect[k])
        n_k= n_k + pr*pi_vect[k]
      
      dist_over_clusters_new = [] 
      for j in range(num_clusters):
        normed= dist_over_clusters[j]/ n_k 
        dist_over_clusters_new.append(normed)


      #print(len(dist_over_clusters))
    #Adding another point's distribution as a row to the phi matrix

      phi.append(dist_over_clusters_new)
      #print(len(phi) )
    print(phi)


    #Performing the M-step

    expectation_of_points_in_cluster_dict= defaultdict(int)

    for k in range(num_clusters):
      in_sum=0
      for i in range(len(data)):
        in_sum+=phi[i][k]
      expectation_of_points_in_cluster_dict[k]= in_sum

    #Update pi vectors for each cluster

    total_numofpoints= len(data)
    pi_vect= [expectation_of_points_in_cluster_dict[k]/ total_numofpoints for k in range(num_clusters) ]

    num_features= len(data[0])

    #Updating the mean_vect of gaussian for each cluster, using phi*each point across the dataset

    for k in range(num_clusters):

      new_mean_vect_kth_cluster= [0]* num_features


      for i in range(total_numofpoints):

        for j in range(num_features):
          new_mean_vect_kth_cluster[j]+= phi[i][k]* data[i][j] #extracting value of feature j of datapoint corresponing to row i
        
        #normalising by each feature mean by nk

        for p in range(num_features):
           
          new_mean_vect_kth_cluster[p]= new_mean_vect_kth_cluster[p]/expectation_of_points_in_cluster_dict[k]

        generated_means[k]= new_mean_vect_kth_cluster
        

      
      #Updating covariances, M step


      for k in range(num_clusters):

        #We will add a delta matrix iteratively to new_cov_kth_cluster
        update_cov = np.zeros( (num_features, num_features) )

        b = np.asarray(   [     generated_means[k] ]      )

        #print(b.shape)


        for i in range(total_numofpoints):
          a= np.asarray( [  data[i]     ]     )


          d= (a-b).transpose().dot( a-b)

          #print(delta.shape)
          d= phi[i][k] * d

          update_cov+= d

        update_cov= update_cov/ expectation_of_points_in_cluster_dict[k]

        all_cov_mat_li[k]= update_cov
      
    updated_means= copy.copy( generated_means )
    res=get_L(data, num_clusters, updated_means, all_cov_mat_li, phi, pi_vect) 
    L.append(res)

  return L, pi_vect, updated_means, all_cov_mat_li












          











        







      



  
    #print(pi_vect)

number_of_runs= 1

number_of_EM_iterations= 2



#PLOT func


def plot(L_li):
  #iter_li= list(range(5,number_of_EM_iterations+1))
  iter_li= [0,0]
  print(L_li)
  for run in range (number_of_runs):
    plt.plot(iter_li, L_li[run][:], label=run, color= "red")
  plt.legend()

#PREDICTING, binary decision-> class 0 i.e. class A OR  class 1 i.e. class B
def decide_class(point, mean_A, cov_A, mean_B, cov_B, pi_vect_A, pi_vect_B):

  # for class 0
  sum_class_A, sum_class_B= 0,0

  for i in range(len(pi_vect_A)):
    sum_class_A+= pi_vect_A[i]* scipy.stats.multivariate_normal.pdf(point, mean_A[i], cov_A[i], allow_singular=True)


  for i in range(len(pi_vect_B)):
    sum_class_B+= pi_vect_B[i]* scipy.stats.multivariate_normal.pdf(point, mean_B[i],cov_B[i], allow_singular=True )

  res = 1 if sum_class_A < sum_class_B else 0
  return res

def get_best_run(L_li, mean_li, pi_li, cov_li):
  min_L = float('-inf')
  chosen_pi_vect, chosen_means_vect, chosesn_covs_vect= None, None, None
  chosen={}
  for run in range(number_of_runs):
    #comparing with L value of last iteration of that run
    if(min_L<= L_li[run][-1]):
      min_L= L_li[run][-1]
      chosen_pi_vect= pi_li[run]
      chosen_means_vect= mean_li[run]
      chosen_covs_vect= cov_li[run]
    chosen['pis']=chosen_pi_vect
    chosen['means']= chosen_means_vect
    chosen['covs']=chosen_covs_vect
    return chosen

def run(num_clusters):

    L_A_li, L_B_li,  mean_A_li, mean_B_li = [],[],[],[]

    pi_A_li, pi_B_li= [], []

    covs_A_li, covs_B_li= [], []

    for i in range(number_of_runs):

      L_A, pi_A, mean_A, cov_A= run_EM(X_train_class0, num_clusters, mean_class0, covariance_mat_class0, number_of_EM_iterations)
      print(L_A)
      L_A_li.append(L_A)
      pi_A_li.append(pi_A)
      mean_A_li.append(mean_A)
      covs_A_li.append(cov_A)

      L_B, pi_B, mean_B, cov_B= run_EM(X_train_class1, num_clusters, mean_class1, covariance_mat_class1, number_of_EM_iterations)
      print(L_B)
      L_B_li.append(L_B)
      pi_B_li.append(pi_B)
      mean_B_li.append(mean_B)
      covs_B_li.append(cov_B)

      plot(L_A_li)

      plot(L_B_li)

      chosen_A= get_best_run(L_A_li, mean_A_li, pi_A_li, covs_A_li)

      chosen_B= get_best_run(L_B_li, mean_B_li, pi_B_li, covs_B_li)

      #Printing confusion matrix

      matrix= [ [0,0], [0,0]]


      for i in range(len(Y_test)):
        predicted_class= decide_class(X_test[i], chosen_A["means"], chosen_A["covs"], chosen_B["means"], chosen_B["covs"], chosen_A["pis"], chosen_B["pis"] )
        ground_truth= int( Y_test[i] )
        matrix[ground_truth][predicted_class]+=1
      print("NUmber of Clusters=")
      print(num_clusters)

      #Class A, aka, Class 0
      print(matrix[0])

      #Class B, aka, Class 1
      print(matrix[1])

      a= matrix[0][0] 
      b= matrix[1][1]
      N=len(X_test)

      acc=  (a+b)/ N

      print("Accuracy is: ")
      print(acc)

run(2)

run(1)

run(3)

run(4)



b= np.asarray([1,4,5,6])

b/b.sum()







