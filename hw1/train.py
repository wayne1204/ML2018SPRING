import csv 
import numpy as np
from numpy.linalg import inv
import math


# ==== reading data ====
print('reading data...')
text = open('data/train.csv', 'r', encoding = 'big5')
rows = csv.reader(text, delimiter = ',')

raw_data = []
for i in range(18):
    raw_data.append([])

n_row = 0
for r in rows:
    if n_row != 0:
        for i in range (3, 27):
            if r[i] != "NR":
                raw_data[(n_row-1)%18].append(float(r[i]))
            else:
                raw_data[(n_row-1)%18].append(float(0))
    n_row +=1
text.close()

r_mu = np.mean(raw_data, axis = 1, dtype=np.float64)
r_sigma = np.std(raw_data, axis = 1, dtype=np.float64)

text = open("r_parameter.txt", "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
for i in range(len(r_mu)):
    s.writerow([r_mu[i], r_sigma[i]]) 
# s.writerow([y_mean, y_std])
text.close()


# ==== extracting feature =====
print('extracting feature...')
X = []
Y = []

for month in range(12):
    for hour in range(471):
        new_row = []
        for toxic in range(18):
            for i in range(9):
                if raw_data[toxic][480*month+hour+i] > 4*r_mu[toxic] :
                    new_row.append(r_mu[toxic])
                elif raw_data[toxic][480*month+hour+i] <0 :
                    new_row.append(r_mu[toxic])
                else:
                    new_row.append( raw_data[toxic][480*month + hour + i])
        if raw_data[9][480*month+hour+9] > 4*r_mu[9] :
            Y.append(r_mu[9])
        elif raw_data[9][480*month+hour+9] <0 :
            Y.append(r_mu[9])
        else:
            Y.append( raw_data[9][480*month+hour+9])
        X.append(new_row)

X = np.array(X)
Y = np.array(Y)   

# =======standardize data=======
mu = np.mean(X, axis = 0, dtype=np.float64)
sigma = np.std(X, axis = 0, dtype=np.float64)
for i in range(X.shape[1]):
    if i == 0:
        continue
    for j in range(X.shape[0]):
        X[j][i] = (X[j][i] - mu[i])/sigma[i]

y_mean = np.mean(Y, axis = 0, dtype=np.float64)
y_std = np.std(Y, axis = 0, dtype=np.float64)
# for i in range(Y.shape[0]):
#     Y[i] = (Y[i] - y_mean)/ y_std

text = open("parameter.txt", "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
for i in range(len(mu)):
    s.writerow([mu[i], sigma[i]]) 
text.close()


X = np.concatenate((X, X**2), axis=1)
X = np.concatenate( (np.ones((X.shape[0],1)), X) , axis=1)

# =======shuffling and splitting validation set=======
# all_size = X.shape[0]
# randomize = np.arange(all_size)
# np.random.shuffle(randomize)
# X,Y = X[randomize], Y[randomize]

# valid_size = int(math.floor(all_size * 0.9))
# X_train, Y_train = X[0:valid_size], Y[0:valid_size]
# X_valid, Y_valid = X[valid_size:], Y[valid_size:]
X_train, Y_train = X, Y

# start training
print('start training...')
w = np.zeros(len(X_train[0]))
l_rate = 0.01
repeat = 10000

x_t = X_train.transpose()
gra_sum = np.ones(len(X_train[0]))
movement = 0
beta1 = 0.9
beta2 = 0.999
first_moment = 0
second_moment = 0
regulation = .1
e = math.pow(10, -8)

for i in range(repeat):
    hypo = np.dot(X_train, w)
    loss = hypo - Y_train
    cost = np.sum(loss**2) / len(X_train)
    cost_a  = math.sqrt(cost) 
    gra = np.dot(x_t,loss) + regulation*w

    first_moment = beta1 * first_moment + (1 - beta1) * gra
    second_moment = beta2 * second_moment + (1 - beta2) * gra**2
    first_moment_b = first_moment / ( 1 - math.pow(beta1, i+1))
    second_moment_b = second_moment / ( 1 - math.pow(beta2, i+1))
    w = w - l_rate * first_moment / (np.sqrt(second_moment) + e)
    if i % 100 == 0:
        print ('iteration: %d | Cost: %f  ' % ( i,cost_a))

# for i in range(18):
#     print(np.sum(w[i*9: i*9+9]))

# hypo = np.dot(X_valid, w)
# loss = hypo - Y_valid
# cost = np.sum(loss**2)/len(Y_valid)
# cost_a = math.sqrt(cost) 
# print( 'validation set Cost: %f' % cost_a)

np.save('model.npy', w)

