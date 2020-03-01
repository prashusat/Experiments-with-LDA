import numpy as np
import random
import csv
import scipy.io as sio
import matplotlib.pyplot as plt
random.seed(1)
import copy
import time
from sklearn.feature_extraction.text import CountVectorizer
import os

##Reading the file
def read_data():
    news_groups=[]
    artificial=[]
    labels_news_groups = []
    labels_artificial=[]
    for i in range(0,200):
        f=open("20newsgroups/"+str(i+1))
        for row in f:
            news_groups.append(row.strip().split(' '))
        f.close()
    f = open("20newsgroups/index.csv")
    for row in f:
        labels_news_groups.append(float(row.strip().split(',')[1]))
    f.close()
    
    for i in range(0,10):
        f=open("artificial/"+str(i+1))
        for row in f:
            artificial.append(row.strip().split(' '))
        f.close()
    f = open("artificial/index.csv")
    for row in f:
        labels_artificial.append(float(row.strip().split(',')[1]))
    f.close()
    return news_groups,artificial,labels_news_groups,labels_artificial
news_groups,artificial,labels_news_groups,labels_artificial=read_data()

beta=0.01
#utility fucntions
def generate_w_initial(data_list,a):
    ct=0
    w=[]
    for i in data_list:
        for j in i:
            w+=[a.index(j)]
            ct+=1
    return w,ct
def generate_d_initial(data_list):
    ct=0
    d=[]
    for i in data_list:
        for j in i:
            d+=[ct]
        ct+=1
    return d,ct
def generate_z_initial(data_list,K):
    z=[]
    for i in data_list:
        for j in i:
            z+=[random.randint(0,K-1)]
    return z

def random_permutation_generator(data_list):
    ct=0
    for i in data_list:
        for j in i:
            ct+=1
    return np.random.permutation(ct)
def calculate_V(data_list):
    a=[]
    for i in data_list:
        for j in i:
            if j not in a:
                a+=[j]
    return a,len(a)
            
start_time=time.time()
########################
data_list=news_groups
K=20
N_iters=500
alpha=5/K
########################
a,V=calculate_V(data_list)
w_indices,N_words=generate_w_initial(data_list,a)
d_indices,D=generate_d_initial(data_list)
z_indices=generate_z_initial(data_list,K)
pi_indices=random_permutation_generator(data_list)
perm=random_permutation_generator(data_list)
C_d = np.zeros((D,K))
C_t = np.zeros((K,V))
w_indices = np.array(w_indices)
d_indices = np.array(d_indices)
z_indices = np.array(z_indices)
for i in range(N_words):
        C_d[d_indices[i]][z_indices[i]] += 1
        C_t[z_indices[i]][w_indices[i]] += 1
P=np.zeros(K)
for i in range(0,N_iters):
    for n in range(0,N_words):
        word=w_indices[perm[n]]
        topic=z_indices[perm[n]]
        doc=d_indices[perm[n]]
        C_d[doc][topic]=C_d[doc][topic]-1
        C_t[topic][word]=C_t[topic][word]-1
        for k in range(0,K):
            temp_add1 = np.sum(C_t[k, :])
            temp_add2 = np.sum(C_d[doc, :])
            temp_1=((C_t[k][word]+beta)/(V*beta+temp_add1))
            temp_2=((C_d[doc][k]+alpha)/(K*alpha+temp_add2))
            P[k]=temp_1*temp_2
        P=np.divide(P,np.sum(P))
        topic_choices=[i for i in range(K)]
        topic=np.random.choice(topic_choices,p=P)
        z_indices[perm[n]]=topic
        C_d[doc][topic]=C_d[doc][topic]+1
        C_t[topic][word]=C_t[topic][word]+1
##temp_C_t=copy.deepcopy(C_t)
##t=1
##for i in temp_C_t:
##    print("Most occuring words in topic:",t)
##    t=t+1
##    print(a[np.argmax(i)])
##    i[np.argmax(i)]=-5
##    print(a[np.argmax(i)])
##    i[np.argmax(i)]=-5
##    print(a[np.argmax(i)])
##    i[np.argmax(i)]=-5
temp_C_t=copy.deepcopy(C_t)
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in temp_C_t:
        temp=[]
        temp+=[a[np.argmax(i)]]
        i[np.argmax(i)]=-5
        temp+=[a[np.argmax(i)]]
        i[np.argmax(i)]=-5
        temp+=[a[np.argmax(i)]]
        i[np.argmax(i)]=-5
        temp+=[a[np.argmax(i)]]
        i[np.argmax(i)]=-5
        temp+=[a[np.argmax(i)]]
        i[np.argmax(i)]=-5
        writer.writerow(temp)
print("time for gibbs",time.time()-start_time)       
#########################################################################################################Task 2##########################################################################################      

    
def build_topic_representation(K,data_list,alpha,C_d):
    topic_representation=[]
    for doc in range(len(data_list)):
        temp_sum=np.sum(C_d[doc,:])
        temp_vector=[]
        for k in range(K):
            temp_vector+=[(C_d[doc,k]+alpha)/(K*alpha+temp_sum)]
        topic_representation+=[temp_vector]

    with open("Logistic_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(topic_representation)
    return topic_representation
topic_representation=build_topic_representation(K,data_list,alpha,C_d)
    
####################################################
##
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA
import math
import time
def read_data(data_file,regression_values_file):
    df_data=pd.read_csv(data_file, sep=',',header=None)
    df_regression_values=pd.read_csv(regression_values_file, sep=',',header=None)
    return df_data,df_regression_values

#function to fetch the shape of data and the labels
def get_shape(dataframe_data,dataframe_values):
    return dataframe_data.shape,dataframe_values.shape

#function to convert pandas dataframe to numpy arrays
def convert_df_to_numpy(dataframe_data,dataframe_values):
 
    return dataframe_data.values,dataframe_values.values


#generates identity matrix based on the number of features present in the data
def identity_matrix_generator(number_of_features):
    return np.identity(number_of_features)

def add_wo_to_data(df):
    #df[len(df.columns)]=1
    df.insert(0,"wo",1)
    return df

#initialise w with 0's
def initialize_w(df):
    no_of_features=df.shape[1]
    return np.zeros(no_of_features).reshape(-1,1)


#fucntion to multiply w with data in order to predict labels
def predict_label_logistic(w,data_in_npy_format):
    predicted_labels=1/(1+np.exp(-np.dot(data_in_npy_format,w)))
    return predicted_labels

def predict_label_poisson(w,data_in_npy_format):
    predicted_labels=np.exp(np.dot(data_in_npy_format,w))
    return predicted_labels

def predict_test_poisson(w,data_in_npy_format):
    predicted_labels=np.floor(np.exp(np.dot(data_in_npy_format,w)))
    return predicted_labels
def predict_test_ordinal(w,data_in_npy_format,s):
    y0=0
    y1=1/(1+np.exp((-2-np.dot(data_in_npy_format,w))*(-s)))
    y2=1/(1+np.exp((-1-np.dot(data_in_npy_format,w))*(-s)))
    y3=1/(1+np.exp((0-np.dot(data_in_npy_format,w))*(-s)))
    y4=1/(1+np.exp((1-np.dot(data_in_npy_format,w))*(-s)))
    y5=1
    
    preds=np.array([np.transpose(y1-y0),np.transpose(y2-y1),np.transpose(y3-y2),np.transpose(y4-y3),np.transpose(y5-y4)])
    
    return np.argmax(preds[:,0,:],axis=0)+1
    
    

#calculates_d
def d_calculator(predicted_labels,labels):
    return labels-predicted_labels


def first_derivative(alpha,w,d,data):
    alpha_matrix=alpha*np.identity(w.shape[0])
    return np.dot(np.transpose(data),d)-np.dot(alpha_matrix,w)


def hessian(data,R,alpha,w):
    alpha_matrix=alpha*np.identity(w.shape[0])
    return np.linalg.inv(np.dot(np.dot(np.transpose(data),R),data)+np.dot(alpha_matrix,np.identity(w.shape[0])))

def calculate_R_logistic(y):
     R=np.multiply(y,1-y)
     return np.diag(R.reshape(1,-1)[0])
def calculate_R_poisson(y):
    return np.diag(y.reshape(1,-1)[0])


def y_i_j_calculator(data_npy,labels_npy,w,s):
    def phi_calc(x):
        if x==0:
            return -100
        elif x==1:
            return -2
        elif x==2:
            return -1
        elif x==3:
            return 0
        elif x==4:
            return 1
        elif x==5:
            return 100
    phi_j = np.array([phi_calc(xi[0]) for xi in labels_npy])
    
    phi_j=np.transpose(phi_j)
    
    phi_j=phi_j.reshape(-1,1)
    a=np.dot(data_npy,w)
    return 1/(1+np.exp(-(s*(np.subtract(phi_j,a)))))

def y_i_j_1_calculator(data_npy,labels_npy,w,s):
    def phi_calc(x):
        if x==1:
            return -100
        elif x==2:
            return -2
        elif x==3:
            return -1
        elif x==4:
            return 0
        elif x==5:
            return 1
    phi_j = np.array([phi_calc(xi[0]) for xi in labels_npy])
   
    phi_j=np.transpose(phi_j)
    phi_j=phi_j.reshape(-1,1)
    a=np.dot(data_npy,w)
    return 1/(1+np.exp(-(s*(np.subtract(phi_j,a)))))
    
def calculate_R_ordinal(data_npy,labels_npy,w,s):
    y_i_j=y_i_j_calculator(data_npy,labels_npy,w,s)
    y_i_j_1=y_i_j_1_calculator(data_npy,labels_npy,w,s)
    
    r_i=(np.multiply(y_i_j,1-y_i_j)+np.multiply(y_i_j_1,1-y_i_j_1))*s*s
    return np.diag(r_i.reshape(1,-1)[0])
    
    
def ordinal_deri1_calc(data_npy,labels_npy,w,s):
    y_i_j=y_i_j_calculator(data_npy,labels_npy,w,s)
    y_i_j_1=y_i_j_1_calculator(data_npy,labels_npy,w,s)
    return y_i_j+y_i_j_1-1





def update(alpha,dat_npy,labels_npy,dat_pd,regression_type):
    a=[]
    start_time=time.time()
    w=initialize_w(dat)
    a=a+[w]
    for i in range(100):
        if regression_type=="logistic":
            y=predict_label_logistic(w,dat_npy)
            d=d_calculator(y,labels_npy)
            deri_1=first_derivative(alpha,w,d,dat_npy)
            R=calculate_R_logistic(y)
            h=hessian(dat_npy,R,alpha,w)
        elif regression_type=="count":
            y=predict_label_poisson(w,dat_npy)
            d=d_calculator(y,labels_npy)
            deri_1=first_derivative(alpha,w,d,dat_npy)
            R=calculate_R_poisson(y)
            h=hessian(dat_npy,R,alpha,w)
        elif regression_type=="ordinal":
            d=ordinal_deri1_calc(dat_npy,labels_npy,w,1)
            deri_1=first_derivative(alpha,w,d,dat_npy)
            R=calculate_R_ordinal(dat_npy,labels_npy,w,1)
            h=hessian(dat_npy,R,alpha,w)
        w=np.add(w,np.dot(h,deri_1))
        a+=[w]
        if (LA.norm(a[i])!=0):
            if (LA.norm(a[i+1]-a[i])/LA.norm(a[i]))<0.001:
                return w,i,time.time()-start_time
    return w,i,time.time()-start_time

def split_into_test_and_train(data,labels):
    sampled_data=data.sample(frac=0.33)
    indi=sampled_data.index.values
    sampled_labels=labels.loc[indi,:]
    train_data=data.loc[~data.index.isin(indi)]
    train_labels=labels.loc[~labels.index.isin(indi)]
    return train_data,train_labels,sampled_data,sampled_labels
def predict_on_test(final_w,test_npy):
    predicted_test_labels=predict_label_logistic(final_w,test_npy)
    for i in range(len(predicted_test_labels)):
        if predicted_test_labels[i]>=0.5:
            predicted_test_labels[i]=1
        else:
            predicted_test_labels[i]=0
    return(predicted_test_labels)

def calculate_errors(true_labels,predicted_labels):
    return np.mean(np.absolute(true_labels-predicted_labels))

#fuctions to divide into fractions of train data

def divide_into_portions(fraction,train_data,train_labels):
    sampled_data=train_data.sample(frac=fraction)
    indi=sampled_data.index
    sampled_labels=train_labels.loc[indi,:]
    return sampled_data,sampled_labels
    
#Evaluating the implementation
def evaluation_function_logistic():
    regression_type="logistic"
    error=[[],[],[],[],[],[],[],[],[],[]]
    iterations=[[],[],[],[],[],[],[],[],[],[]]
    times=[[],[],[],[],[],[],[],[],[],[]]
    start_time=time.time()
    for i in range(0,30):
        train_data,train_labels,test_data,test_labels=split_into_test_and_train(dat,labels)
        
        for j in range(1,11):
            train_frac_data,train_frac_labels=divide_into_portions(j/10,train_data,train_labels)
            dat_npy,labels_npy=convert_df_to_numpy(train_frac_data,train_frac_labels)
            final_w,iteration,time_t=update(0.01,dat_npy,labels_npy,dat,regression_type)
            test_npy,labels_test_npy=convert_df_to_numpy(test_data,test_labels)
            predicted_test_labels=predict_on_test(final_w,test_npy)
            e=calculate_errors(labels_test_npy,predicted_test_labels)
            error[j-1]+=[e]
            times[j-1]+=[time_t]
            iterations[j-1]+=[iteration]
    final_errors=[]
    final_std=[]
    final_iterations=[]
    final_times=[]
    for i in range(0,10):
         #print(np.mean(error[i]))
        final_errors.append(np.mean(error[i]))
        final_iterations.append(sum(iterations[i])/len(iterations[i]))
        final_times.append(sum(times[i])/len(times[i]))
        final_std.append(np.std(error[i]))
##    print("no. of iterations for different sizes",final_iterations)
##    print("run times for different sizes",final_times)
    fin_acc=[]
    for i in final_errors:
        fin_acc+=[1-i]
    return fin_acc,final_std
 #   error_vs_size_plotter_logistic(fin_acc,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],final_std)


        
        
def error_vs_size_plotter(errors,size,y_error):
    plt.title("Mean absolute error vs Size of Data") 
    plt.xlabel("Training Set Portion") 
    plt.ylabel("Mean absolute error") 
    plt.plot(size,errors,color='grey')
    plt.errorbar(size, errors, yerr=y_error)
    plt.show()

def error_vs_size_plotter_logistic(errors,size,y_error):
    plt.title("error vs Size of Data") 
    plt.xlabel("Training Set Portion") 
    plt.ylabel("error") 
    plt.plot(size,errors,color='grey')
    plt.errorbar(size, errors, yerr=y_error)
    plt.show()


def bag_of_words_representation(data_list):
    m=0
    dic={}
    a=[]
    bow_representation=[]
    for i in data_list:
        for j in i:
            if j not in a:
                dic[j]=m
                m+=1
                a+=[j]
    for i in data_list:
        tmp=[0 for q in range(len(a))]
        for j in i:
            tmp[dic[j]]+=1
        bow_representation+=[tmp]
    with open("Logistic_data_bow.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(bow_representation)
    return bow_representation


##
bow_representation=bag_of_words_representation(data_list)
print("logistic regression in progress")
with open("20newsgroups/index.csv","r") as source:
    rdr= csv.reader( source )
    with open("20newsgroups/index2.csv","w") as result:
        wtr= csv.writer( result )
        for r in rdr:
            wtr.writerow((r[1]))
data,labels=read_data("Logistic_data.csv","20newsgroups/index2.csv")
dat=add_wo_to_data(data)
fin_acc_topic_rep,fin_std_topic_rep=evaluation_function_logistic()

data,labels=read_data("Logistic_data_bow.csv","20newsgroups/index2.csv")
dat=add_wo_to_data(data)
fin_acc_bow,fin_std_bow=evaluation_function_logistic()

plt.title("LDA VS BAG OF WORDS:Accuracy vs Size of Data") 
plt.xlabel("Training Set Portion") 
plt.ylabel("Accuracy") 
#plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],fin_acc_topic_rep,color='grey',label="Topic Representation")
plt.errorbar([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],fin_acc_topic_rep , yerr=fin_std_topic_rep,ecolor="yellow",color='blue',capsize=15,label="Topic Representation")
#plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],fin_acc_bow,color='red',label="Bag of Words")
plt.errorbar([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],fin_acc_bow , yerr=fin_std_bow,ecolor="red",capsize=15,color='black',label="Bag of Words")
plt.legend(loc="upper left")
plt.show()    
    
    
