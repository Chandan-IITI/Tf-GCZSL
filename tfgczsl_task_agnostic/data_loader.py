import numpy as np
import math
import scipy.io as sio

path = "../../../../Datasets/CUB_mat/"

# Read data from mat files:
matcontent = sio.loadmat(path + "res101.mat")
feature_mat = matcontent['features'].T
label_mat = matcontent['labels'].astype(int).squeeze() - 1

matcontent = sio.loadmat(path + 'att_splits.mat')
aux_data_mat = matcontent['att'].T



#some constants
total_tasks = 20
new_class_pt = 10
old_class_pt = 0

# Total data
Y = label_mat
Z = aux_data_mat
X = feature_mat

print ("shape of x is...", X.shape)
print ("shape of y is...", Y.shape)
print ("shape of z is...", Z.shape)
print (Y)

# Test and Train classes
fp = open(path + 'allclasses.txt', 'r')
allclasses = [x.split()[0] for x in fp.readlines()]
fp.close()

allLabels = {}
count = 0
for cl in allclasses:
    allLabels[cl] = count
    count = count + 1

testClasses = [allLabels[x.split()[0]] for x in open(path + 'testclasses.txt').readlines()]
trainClasses = [allLabels[x.split()[0]] for x in open(path + 'trainvalclasses.txt').readlines()]

print("the test class is...", len(testClasses))
print("the train class is...", len(trainClasses))

print ("the max value is...", max(Y))

# getting data into classes
X_c = [[] for jj in range(len(allclasses))]
Y_c = [[] for jj in range(len(allclasses))]
for jj in range(len(X)):
    X_c[Y[jj]].append(X[jj])
    Y_c[Y[jj]].append(Y[jj])

#X_c = np.array(X_c)
#Y_c = np.array(Y_c)
print ("the shape of X_C is...", np.array(X_c[0]).shape)
print ("the shape if Y_c is...", np.array(Y_c[0]).shape)
#print ("the value is...", np.array(X_c[0][0:(len(X_c[0])-1)/2]).shape)
#print ("the value is...", np.array(X_c[0][(len(X_c[0])-1)/2:]).shape)
print (len(allclasses))

#splitting data by classes for each task
X_t = [[] for gg in range(total_tasks)]
Y_t = [[] for gg in range(total_tasks)]
counter1 = new_class_pt
counter2 = 0
kk = 0
for jj in range(total_tasks):
    print ("jj is...", jj)
    print ("new_class_pt is...", counter1)
    print ("kk is...", kk)
    while True:
        #X_t[jj].append(X_c[kk])
        #Y_t[jj].append(Y_c[kk])
        X_t[jj] = X_t[jj] + X_c[kk]
        Y_t[jj] = Y_t[jj] + Y_c[kk]
        kk = kk + 1
        if kk >= counter1 or kk >= 200:
            counter1 = counter1 + new_class_pt
            break

print ("the shape is X_t is...", np.array(X_t[0]).shape)
print ("the shape is Y_t is...", np.array(Y_t[0]).shape)


#############################################################################

# checking whether there is overlapp of classes  between tasks
classes = [0]*total_tasks
for i in range(total_tasks):
    classes[i] = sorted(list(set(Y_t[i])))
print("classes are...", classes[4])
print(len(classes[4]))
dummy = 3
class_set1 = set(classes[dummy])
class_set2 = set(classes[dummy + 1])
testClasses_set = set(testClasses)

intersection = class_set1.intersection(class_set2)

intersection1 = class_set1.intersection(testClasses_set)
intersection2 = class_set2.intersection(testClasses_set)
#intersection = intersection1.intersection(intersection2)

print(intersection)
print(len(intersection))
a = [[] for g in range(total_tasks)]
print(a)

#storing data

trainDataX1 = [[] for g in range(total_tasks)]
trainDataLabels1 = [[] for g in range(total_tasks)]
trainDataAttrs1 = [[] for g in range(total_tasks)]

testDataX = [[] for g in range(total_tasks)]
testDataLabels = [[] for g in range(total_tasks)]
testDataAttrs = [[] for g in range(total_tasks)]

for jj in range(total_tasks):
    for ii in range(0,len(Y_t[jj])):
        trainDataX1[jj] = trainDataX1[jj] + [X_t[jj][ii]]
        trainDataLabels1[jj] = trainDataLabels1[jj] + [Y_t[jj][ii]]
        trainDataAttrs1[jj] = trainDataAttrs1[jj] + [Z[Y_t[jj][ii]]]


print("the length of dataX1 is...",len(trainDataX1[0]))

trainDataX = [[] for g in range(total_tasks)]
trainDataLabels = [[] for g in range(total_tasks)]
trainDataAttrs = [[] for g in range(total_tasks)]

'''testDataX = []
testDataLabels = [] 
testDataAttrs = []'''

# getting 20 percent test data from data set of each task
for kk in range(total_tasks):
    rndind = []
    train_size = len(trainDataX1[kk])
    #print(train_size)
    #print(noExs)
    num_train = int(math.ceil(train_size * 0.8))
    print(num_train)
    while True:
        r = np.random.randint(train_size)
        if r not in rndind: rndind.append(r)
        if len(rndind) == num_train:
            break
    print (len(list(set())))
    for jj in range(0,train_size):
        if(jj in rndind):
            trainDataX[kk] = trainDataX[kk] + [trainDataX1[kk][jj]]
            trainDataLabels[kk] = trainDataLabels[kk] + [trainDataLabels1[kk][jj]]
            trainDataAttrs[kk] = trainDataAttrs[kk] + [trainDataAttrs1[kk][jj]]
        else:
            testDataX[kk] = testDataX[kk] + [trainDataX1[kk][jj]]
            testDataLabels[kk] = testDataLabels[kk] + [trainDataLabels1[kk][jj]]
            testDataAttrs[kk] = testDataAttrs[kk] + [trainDataAttrs1[kk][jj]]

trainDataX = np.array(trainDataX)
trainDataLabels = np.array(trainDataLabels)
trainDataAttrs = np.array(trainDataAttrs)

print("the shape of train data is...", np.array(trainDataX[0]).shape)
print("the shape of train data is...", np.array(trainDataLabels[0]).shape)
print("the value of train data is...", trainDataLabels[0][90])


testDataX = np.array(testDataX)
testDataLabels = np.array(testDataLabels)
testDataAttrs = np.array(testDataAttrs)

print("the shape of test data is...", np.array(testDataX[0]).shape)
print("the shape of train data is...", np.array(testDataLabels[0]).shape)

np.save(open('trainData' , 'wb') , trainDataX)
np.save(open('trainLabels' , 'wb') , trainDataLabels)
np.save(open('trainAttributes' , 'wb') , trainDataAttrs)


np.save(open('testData' , 'wb') , testDataX)
np.save(open('testLabels' , 'wb') , testDataLabels)
np.save(open('testAttributes' , 'wb') , testDataAttrs)

np.save(open('dataAttributes', 'wb'), np.array(aux_data_mat))


