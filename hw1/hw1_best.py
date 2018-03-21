import numpy as np
import math
import csv
import sys

# loading parameter
print('loading parameter...')
w = np.load('model1.npy')
text = open('r_parameter.txt' ,"r")
row = csv.reader(text , delimiter= ",")
r_mu = []
r_sigma = []
for i in row:
    r_mu.append(float(i[0]))
    r_sigma.append(float(i[1]))
text.close()

text = open('parameter.txt' ,"r")
row = csv.reader(text , delimiter= ",")
mu = []
sigma = []
for i in row:
    mu.append(float(i[0]))
    sigma.append(float(i[1]))
text.close()


# reading test data
print('reading test data...')
test_x = []
n_row = 0
filename = sys.argv[1]
text = open( filename,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])

    for i in range(2,11):
        if r[i] =="NR":
            test_x[n_row//18].append(0)
        elif float(r[i]) > 4*r_mu[n_row%18] :
            test_x[n_row//18].append(r_mu[n_row%18])
        elif float(r[i]) < 0 :
            test_x[n_row//18].append(r_mu[n_row%18])
        else:
            test_x[n_row//18].append(float(r[i]))
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

for i in range(test_x.shape[1]):
    if(i == 0):
        continue
    for j in range(test_x.shape[0]):
        test_x[j][i] = (test_x[j][i] - mu[i])/sigma[i]

test_x = np.concatenate((test_x,test_x**2), axis=1)
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

# output answer
print('Output predecition...')
ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()