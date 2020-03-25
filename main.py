import random,math
# Normailzing adding bias term
with open('Colon_Cancer_CNN_Features.csv','r') as fin:
    data = fin.readlines()
with open('dataset.csv','w') as f:
    for _,line in enumerate(data):
        f.write('1,')
        f.write(str(line))

# Shuffle the data
with open('dataset.csv','r') as source:
    data = [ (random.random(), line) for line in sourcew]
data.sort()
with open('dataset.csv','w') as target:
    for _, line in data:
        target.write(line)

with open('dataset.csv','r') as f: 
    dat = f.readlines()
    X1=[]
    Y1=[]
    for line in dat:
        Y1.append(line[-2])
    for line in dat:
        line = line.split(',')
        line.pop()
        X1.append(line)
X = X1[:5219]
Y = Y1[:5219]
testset = X1[:1305]
testval1 = Y1[:1305]
y = [[0 for i in range(4)] for j in range(len(Y))]
for i in range(len(Y)):
    y[i][int(Y[i])-1] = 1
testval = [[0 for i in range(4)] for j in range(len(testval1))]
for i in range(len(testval1)):
    testval[i][int(testval1[i])-1] = 1
def compress(val):
    return float(1.0/(1.0+( 2.71828 ** (-1.0 * float(val)))))
def sigmoid(list1):
    fin = list1    
    return list(map(compress, fin))

def ele_multi(A,B):
    rmatrix = [[0 for i in range(len(A[0]))] for j in range(len(A))]
    for k in range(len(A)):
        for l in range(len(A[0])):
            rmatrix[k][l] = A[k][l] * B[k][l]
    return rmatrix

def mat_sub(A,B):
    rmatrix = [[0 for i in range(len(A[0]))] for j in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            rmatrix[i][j] = A[i][j] - B[i][j]
    return rmatrix
def mat_add(A,B):
    rmatrix = [[0 for i in range(len(A[0]))] for j in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            rmatrix[i][j] = A[i][j] + B[i][j]
    return rmatrix

def sca_mul(A,c):
    result = A
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] *c
    return result
def sub_loss(A,B):
    rmatrix = [[0 for i in range(len(B[0]))] for j in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            rmatrix[i][j] = ((A[i][j] - B[i][j]) ** 2)/(2*5128.0)
    return rmatrix
def loss(A,B):
    return sum([sum(i) for i in sub_loss(A,B)])

def loss_cross(A,B):
    a = [[math.log(j) for j in i] for i in A]
    
    b2 =[[1-j for j in i] for i in B]
    a2 = [[1-j for j in i] for i in A]
    b = [[math.log(i) for j in i] for i in b2]
    sum = mat_add(ele_multi(B,a),ele_multi(b,a2))

    return sum

def transpose(X):
    result = [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]
    return result

def mat_mul(A,B):
    result= [[sum(float(a) * float(b) for a, b in zip(A_row, B_col)) 
						for B_col in zip(*B)] 
								for A_row in A]            
    return result

def mat_div(A,B):
    rmatrix = [[0 for i in range(len(A[0]))] for j in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            if (B[i][j] == 0):
                a=1
            else:
                a = B[i][j]    
            rmatrix[i][j] = 1.0* A[i][j]/a
    return rmatrix

hidden_neurons = 10
w1 = [[random.random()*0.01 for i in range(326)] for j in range(hidden_neurons)]
w2 = [[random.random() for i in range(hidden_neurons)] for j in range(4)]

result = mat_mul(X,transpose(w1))
Hidden_value = list(map(sigmoid,result))
for line in Hidden_value:
    line[0] = 1
result2 = mat_mul(Hidden_value,transpose(w2))
Final_value = list(map(sigmoid,result2))
print(Final_value)

eta = 0.01
epoch = 4
# BackPropogation

def misclassified(yk,y):
    miss = 0
    for  i in range(len(y)):
        index = 0
        max = 0 
        for j in range(len(y[0])):
            if (yk[i][j]>max):
                max = yk[i][j]
                index = j
        if (y[i][index] != 1):
            miss = miss+1
    return miss



def testing(input, w1,w2,y):
    result = mat_mul(input,transpose(w1))
    yj = list(map(sigmoid,result))
    for line in Hidden_value:
        line[0] = 1
    result2 = mat_mul(Hidden_value,transpose(w2))
    yk = list(map(sigmoid,result2))
    print('Mistakes : ',misclassified(yk,y))



testing(testset,w1,w2,testval)
def BackPropogation(y,yj,yk,w1,w2,input_layer,eta):
    delta = ele_multi((mat_sub(y,yk)),ele_multi(yk,[[1-j for j in i] for i in yk]))
    dw2 = sca_mul((mat_mul(transpose(delta),yj)),-1.0)
    w2 = mat_sub(w2 , sca_mul(dw2,eta))
    dw1 = mat_mul(transpose(ele_multi(mat_mul(transpose(delta), w2), yj)), input_layer)
    w1 = mat_sub(w1, sca_mul(dw1,eta))
    return w1,w2

def BackPropogation_cross(y,yj,yk,w1,w2,input_layer,eta):
    z = mat_sub(mat_div(y,yk),mat_div([[1-j for j in i] for i in y],[[1-j for j in i] for i in yk]))
    delta = ele_multi(sca_mul(z,-1.0) ,ele_multi(yk,[[1-j for j in i] for i in yk]))
    dw2 = sca_mul((mat_mul(transpose(delta),yj)),-1.0)
    w2 = mat_sub(w2 , sca_mul(dw2,eta))
    dw1 = mat_mul(transpose(ele_multi(mat_mul(transpose(delta), w2), yj)), input_layer)
    w1 = mat_sub(w1, sca_mul(dw1,eta))
    return w1,w2


def training(input_layer, y, w1,w2,epoch):
    for k in range(epoch):
        result = mat_mul(input_layer,transpose(w1))
        yj = list(map(sigmoid,result))
        for line in Hidden_value:
            line[0] = 1
        result2 = mat_mul(Hidden_value,transpose(w2))
        yk = list(map(sigmoid,result2))
        w1,w2 = BackPropogation(y,yj,yk,w1,w2,X,eta)
        print(loss(yk,y))
    return w1,w2,yk
w1,w2,yk = training(X,y,w1,w2,epoch)

def training_cross(input_layer, y, w1,w2,epoch):
    for k in range(epoch):
        result = mat_mul(input_layer,transpose(w1))
        yj = list(map(sigmoid,result))
        for line in Hidden_value:
            line[0] = 1
        result2 = mat_mul(Hidden_value,transpose(w2))
        yk = list(map(sigmoid,result2))
        w1,w2 = BackPropogation_cross(y,yj,yk,w1,w2,X,eta)
        print(loss(yk,y))
    return w1,w2,yk
w1,w2,yk = training(X,y,w1,w2,epoch)
print('Final w1:',w1)
print('Final w2:',w2)
print('Training Error: ',loss(yk,y),'\n\n\n')

w1c,w2c,ykc = training_cross(X,y,w1,w2,epoch)
print('Final w1:',w1c)
print('Final w2:',w2c)
testing(testset,w1,w2,testval)