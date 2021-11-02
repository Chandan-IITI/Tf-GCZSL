import numpy as np
import math
import pickle
import scipy.io as sio

def unison_shuffled_copies(a, b):
    assert (len(a) == len(b))
    p = np.random.permutation(len(a))
    return a[p].tolist(), b[p].tolist()

# Load data from .mat file
# Some Constants

path = '../../../../Datasets/CUB_mat/'

# Read data from mat files:
matcontent = sio.loadmat(path + "res101.mat")
feature_mat = matcontent['features'].T
label_mat = matcontent['labels'].astype(int).squeeze() - 1

matcontent = sio.loadmat(path + 'att_splits.mat')
aux_data_mat = matcontent['att'].T



#some constants
total_tasks = 20
new_class_pt = 10
train_class_pt = 7
test_class_pt = 3

#old_class_pt = 20

# Total data
Y = label_mat
Z = aux_data_mat
X = feature_mat

print ("shape of x is...", X.shape)
print ("shape of y is...", Y.shape)
print ("shape of z is...", Z.shape)


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

# train classes
map_trainClasses = {}
inversemap_trainClasses = {}

counter1 = train_class_pt
kk = 0
ll = 0
dummy = 0

# train classes mapping
for jj in range(total_tasks):
    while True:
        if kk < counter1:
            map_trainClasses[trainClasses[ll]] = kk
            inversemap_trainClasses[kk] = trainClasses[ll]
            kk = kk + 1
            ll = ll + 1
        elif kk >= counter1 or ll >= train_class_pt:
            print ("jj is...", jj)
            print ("final kk is...", kk)
            print ("final ll is...", ll)
            if jj < 9:
                counter1 = counter1 + new_class_pt
                kk = kk + test_class_pt
            else:
                if dummy == 0:
                    counter1 = counter1 + (new_class_pt) + 1
                    kk = kk + test_class_pt
                    dummy = 1
                    break
                counter1 = counter1 + (new_class_pt)
                kk = kk + (test_class_pt - 1)
            break

print ("the train class is...", trainClasses)
print ("the map_trainClasses is...", sorted(map_trainClasses.items(), key= lambda mv:(mv[1], mv[0])))
print ("the inverse map trainClasses is...", sorted(inversemap_trainClasses.items()))

#print ("the value of map_trainClasses is...", map_trainClasses[trainClasses[1]])
#print ("the value of map_trainClasses is...", inversemap_trainClasses[5])
#print ("the value of trainClasses is...", trainClasses[5])

# test classes mapping
map_testClasses = {}
inversemap_testClasses = {}

counter1 = new_class_pt
kk = train_class_pt
ll = 0
dummy = 0

for jj in range(total_tasks):
    while True:
        if kk < counter1:
            map_testClasses[testClasses[ll]] = kk
            inversemap_testClasses[kk] = testClasses[ll]
            kk = kk + 1
            ll = ll + 1
        elif kk >= counter1 or ll >= test_class_pt:
            print ("jj is...", jj)
            print ("final kk is...", kk)
            print ("final ll is...", ll)
            if jj < 9:
                counter1 = counter1 + new_class_pt
                kk = kk + train_class_pt
            else:
                counter1 = counter1 + (new_class_pt)
                kk = kk + (train_class_pt + 1)
            break

print ("the map_testClasses is...", sorted(map_testClasses.items(), key= lambda mv:(mv[1], mv[0])))

print("the test class is...", len(testClasses))
print("the train class is...", len(trainClasses))

print ("the max value is...", max(Y))

# getting data into classes
X_c_train = {}
Y_c_train = {}

X_c_test = {}
Y_c_test = {}

for jj in trainClasses:
    X_c_train[map_trainClasses[jj]] = []
    Y_c_train[map_trainClasses[jj]] = []

for jj in testClasses:
    X_c_test[map_testClasses[jj]] = []
    Y_c_test[map_testClasses[jj]] = []

for jj in range(len(X)):
    if Y[jj] in trainClasses:
        Y[jj] = map_trainClasses[Y[jj]]
        X_c_train[Y[jj]].append(X[jj])
        Y_c_train[Y[jj]].append(Y[jj])
    elif Y[jj] in testClasses:
        Y[jj] = map_testClasses[Y[jj]]
        X_c_test[Y[jj]].append(X[jj])
        Y_c_test[Y[jj]].append(Y[jj])
    else:
        print ("error")

print ("Y_c_test is...", np.array(X_c_test[map_testClasses[testClasses[0]]]).shape)
#X_c = np.array(X_c)
#Y_c = np.array(Y_c)
print ("the shape of X_c_train is...", np.array(X_c_train[0]).shape)
print ("the shape if Y_c_train is...", np.array(Y_c_train[0]).shape)
print ("the shape of X_c_test is...", np.array(X_c_test[map_testClasses[testClasses[0]]]).shape)
print ("the shape if Y_c_test is...", np.array(Y_c_test[map_testClasses[testClasses[0]]]).shape)

print ("the value is...", np.array(X_c_train[0][0:int((len(X_c_train[0])-1)/2)]).shape)
print ("the value is...", np.array(X_c_train[0][int((len(X_c_train[0])-1)/2):]).shape)
print (len(allclasses))

#X_c, Y_c = unison_shuffled_copies(np.array(X_c), np.array(Y_c))



#splitting data by classes for each task
X_t = [[] for gg in range(total_tasks)]
Y_t = [[] for gg in range(total_tasks)]
X_t_test = [[] for gg in range(total_tasks)]
Y_t_test = [[] for gg in range(total_tasks)]
counter1 = train_class_pt
counter2 = test_class_pt
kk = 0
ll = 0
for jj in range(total_tasks):
    print ("jj is...", jj)
    print ("train_class_pt is...", counter1)
    print ("test_class_pt is...", counter2)
    print ("kk is...", kk)
    while True:
        if kk < counter1:
            #X_t[jj].append(X_c[kk])
            #Y_t[jj].append(Y_c[kk])
            X_t[jj] = X_t[jj] + X_c_train[map_trainClasses[trainClasses[kk]]]
            Y_t[jj] = Y_t[jj] + Y_c_train[map_trainClasses[trainClasses[kk]]]
            kk = kk + 1
        elif kk >= counter1 or kk >= train_class_pt:
            print ("final kk is...", kk)
            if jj < 9:
                counter1 = counter1 + train_class_pt
            else:
                counter1 = counter1 + (train_class_pt + 1)
            break

    while True:
        if ll < counter2:
            #X_t[jj].append(X_c[kk])
            #Y_t[jj].append(Y_c[kk])
            X_t_test[jj] = X_t_test[jj] + X_c_test[map_testClasses[testClasses[ll]]]
            Y_t_test[jj] = Y_t_test[jj] + Y_c_test[map_testClasses[testClasses[ll]]]
            ll = ll + 1
        elif ll >= counter2 or ll >= test_class_pt:
            print ("final ll is...", ll)
            if jj < 9:
                counter2 = counter2 + test_class_pt
            else:
                counter2 = counter2 + (test_class_pt - 1)
            break

print ("the shape is X_t is...", np.array(X_t[0]).shape)
print ("the shape is Y_t is...", np.array(Y_t[0]).shape)
print ("the shape is X_t_test is...", np.array(X_t_test[0]).shape)
print ("the shape is Y_t_test is...", np.array(Y_t_test[0]).shape)

#############################################################################

# checking whether there is overlapp of classes  between tasks
classes = [0]*total_tasks
classes_test = [0]*total_tasks
for i in range(total_tasks):
    classes[i] = sorted(list(set(Y_t[i])))
    classes_test[i] = sorted(list(set(Y_t_test[i])))
print ("classes are...", classes[0])
print ("test classes are...", classes_test[0])
print(len(classes[3]))
dummy = 2
class_set1 = set(classes[dummy])
class_set2 = set(classes[dummy + 1])
testClasses_set = set(testClasses)

intersection = class_set1.intersection(class_set2)

intersection1 = class_set1.intersection(testClasses_set)
intersection2 = class_set2.intersection(testClasses_set)
#intersection = intersection1.intersection(intersection2)

print (intersection)
print (len(intersection))
a = [[] for g in range(total_tasks)]
print (a)

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
        trainDataAttrs1[jj] = trainDataAttrs1[jj] + [Z[inversemap_trainClasses[Y_t[jj][ii]]]]

for jj in range(total_tasks):
    for ii in range(0,len(Y_t_test[jj])):
        testDataX[jj] = testDataX[jj] + [X_t_test[jj][ii]]
        testDataLabels[jj] = testDataLabels[jj] + [Y_t_test[jj][ii]]
        testDataAttrs[jj] = testDataAttrs[jj] + [Z[inversemap_testClasses[Y_t_test[jj][ii]]]]

print ("the shape of testData is...", np.array(testDataX[0]).shape)
print ("the shape of testLabels is...", np.array(testDataLabels[0]).shape)

print ("the length of dataX1 is...",len(trainDataX1[0]))

trainDataX = [[] for g in range(total_tasks)]
trainDataLabels = [[] for g in range(total_tasks)]
trainDataAttrs = [[] for g in range(total_tasks)]

'''testDataX = []
testDataLabels = [] 
testDataAttrs = []'''

testDataX_seen = [[] for g in range(total_tasks)]
testDataLabels_seen = [[] for g in range(total_tasks)]
testDataAttrs_seen = [[] for g in range(total_tasks)]

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
            testDataX_seen[kk] = testDataX_seen[kk] + [trainDataX1[kk][jj]]
            testDataLabels_seen[kk] = testDataLabels_seen[kk] + [trainDataLabels1[kk][jj]]
            testDataAttrs_seen[kk] = testDataAttrs_seen[kk] + [trainDataAttrs1[kk][jj]]

trainDataX = np.array(trainDataX)
trainDataLabels = np.array(trainDataLabels)
trainDataAttrs = np.array(trainDataAttrs)

print("the shape of train data is...", np.array(trainDataX[0]).shape)
print("the shape of train data is...", np.array(trainDataLabels[0]).shape)
print("the value of train data is...", trainDataLabels[0][120])


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

np.save(open('testData_seen' , 'wb') , testDataX_seen)
np.save(open('testLabels_seen' , 'wb') , testDataLabels_seen)
np.save(open('testAttributes_seen' , 'wb') , testDataAttrs_seen)

a_file = open("map_trainClasses.pkl", "wb")
pickle.dump(map_trainClasses, a_file)
a_file. close()

a_file = open("inversemap_trainClasses.pkl", "wb")
pickle.dump(inversemap_trainClasses, a_file)
a_file. close()

a_file = open("map_testClasses.pkl", "wb")
pickle.dump(map_testClasses, a_file)
a_file. close()

a_file = open("inversemap_testClasses.pkl", "wb")
pickle.dump(inversemap_testClasses, a_file)
a_file. close()

np.save(open('dataAttributes', 'wb'), np.array(aux_data_mat))
