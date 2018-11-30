import numpy as np
from itertools import product
from sklearn.cluster import KMeans
all_prob=[]
for i in product([0,1], repeat=15): 
    all_prob.append(i)
all_prob=np.array(all_prob)

kmeans = KMeans(n_clusters=225, random_state=0).fit(all_prob)
c_list=kmeans.cluster_centers_


zero=np.array([1,1,1,1,-1,1,1,-1,1,1,-1,1,1,1,1])
one=np.array([-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1])
two=np.array([1,1,1,-1,-1,1,1,1,1,1,-1,-1,1,1,1])
three=np.array([1,1,1,-1,-1,1,1,1,1,-1,-1,1,1,1,1])
four=np.array([1,-1,1,1,-1,1,1,1,1,-1,-1,1,-1,-1,1])
five=np.array([1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,1,1])
six=np.array([1,1,1,1,-1,-1,1,1,1,1,-1,1,1,1,1])
seven=np.array([1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1])
eight=np.array([1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,1])
nine=np.array([1,1,1,1,-1,1,1,1,1,-1,-1,1,1,1,1])

inputs=np.array([zero,one,two,three,four,five,six,seven,eight,nine])

def s(x,c):
    return np.exp(-np.linalg.norm(x-c)**2)



z_l=[]
for train in inputs:
    z_m=[]
    for c in c_list:
        z=s(train,c)
        z_m.append(z)
    z_l.append(z_m)

z_l=np.array(z_l)

i_list=[]
output_neurons=np.identity(10)
for i in output_neurons:
    j_list=[]
    for j in i:
        if j==0:
            j_list.append(-1)
        else:
            j_list.append(1)
    i_list.append(j_list)
    
output_neurons=np.array(i_list)    
bias=np.ones(10)
z_l=np.insert(z_l,225,bias,axis=1)
w_list=[]
for outputs in output_neurons:
    transpoze_mul=np.dot(z_l.T,z_l)
    gen_inverse=np.linalg.pinv(transpoze_mul)
    delta=np.dot(gen_inverse,z_l.T)
    w_star=np.dot(delta,outputs)
    w_list.append(w_star)

w_list=np.array(w_list)

for i in range(0,10):
    result=np.dot(z_l,w_list[i])
    
    print(str(i)+" "+str(result)+"\n")
    