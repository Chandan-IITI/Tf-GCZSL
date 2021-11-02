import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import math
import csv
from copy import deepcopy
import pickle

from sklearn.preprocessing import normalize

from tfgczsl.tfgczsl_model import TFGCZSL

#===================================================================#

# Some Constants

m = 50                                                       # Batch size
n_x = 2048                                                   # Feature size
n_y = 312                                                    # Attribute size
n_z = 50                                                     # Latent dimension size
path = '../../../../Datasets/CUB_mat/'                             # Path of the dataset
nTrain = 150                                                 # Total number of seen classes
nTest = 50                                                   # Total number of unseen classes

task_no = 20                                                 # Total number of tasks 
task_no_1 = 20                                               # Number of tasks to be concidered 
total_classes = 200                                          # Total number of classes

samples_per_class = 10                                       # Number of samples to be stored in memory per class
memory_batch_size = 100                                      # Batch size of the data to be selected from the replay memory
use_KD = False                                               # Whether to use Knowledge distillation
use_der = False                                               # Whether to use Dark Experience Replay

method_names = ["reservoir", "ring_buffer", "mof"]           # Type of memory that can be used
name = method_names[0]

# classifier parameters
classifier_lr = 0.001
classifier_wt_decay = 0.001 
classifier_epoch = 25
classifier_batch_size = 32                                       

# ============================================================ #
# defining the object for TFGCZSL class:

tfgczsl = TFGCZSL(m, n_x, n_y, n_z, nTrain, nTest, samples_per_class, total_classes, memory_batch_size, use_KD, use_der, name)

# =================================================================================#

# Non-linear classifier
class Classifier(nn.Module):
    def __init__(self, n_x, hidden, total_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(n_x, hidden)
        self.fc2 = nn.Linear(hidden, total_classes)

        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x))
        return x

# Linear classifier
class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction = nn.NLLLoss()

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)


classifier = LINEAR_LOGSOFTMAX(n_z, nclass=total_classes)
#classifier = Classifier(n_z, 64, total_classes)
# classifier.apply(init_weights)
classifier.cuda()
# classifier_criterion = nn.CrossEntropyLoss()
classifier_criterion = nn.NLLLoss()
classifier_optim = optim.Adam(classifier.parameters(), lr=classifier_lr, weight_decay=classifier_wt_decay)
#classifier_optim = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.5, 0.999))


# ============================================================ #
# Loading Data of all tasks

trainData1 = np.load(open('trainData' , 'rb'), allow_pickle = True, encoding='latin1')
trainLabels1 = np.load(open('trainLabels' , 'rb'), allow_pickle = True, encoding='latin1')
trainLabelVectors1 = np.load(open('trainAttributes' , 'rb'), allow_pickle = True, encoding='latin1')

testData1 = np.load(open('testData', 'rb'), allow_pickle=True, encoding='latin1')
testLabels1 = np.load(open('testLabels', 'rb'), allow_pickle=True, encoding='latin1')
testLabelVectors1 = np.load(open('testAttributes', 'rb'), allow_pickle=True, encoding='latin1')

ATTR = np.load(open('dataAttributes', 'rb'), encoding='latin1')


# Initializing accuracy metrics
seen_acc = []
unseen_acc = []
harmonic_mean = []
accuracy_matrix = [[] for kk in range(task_no)]
overall_acc = []

# initializing classes seen so far
replay_Classes = []
replay_TrainLabels = []

# ============================================================ #

for jj in range(task_no):
    # load train data for the current task
    trainData = np.array(trainData1[jj])
    trainLabelVectors = np.array(trainLabelVectors1[jj])
    trainLabels = np.array(trainLabels1[jj])

    X_train = np.concatenate([trainData , trainLabelVectors], axis=1)
    
    
    print('Fitting VAE Model...')
    
    if jj == 0:
        # defining the data loader for training
        vae_train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float),
                                       torch.tensor(trainData, dtype=torch.float),
                                       torch.tensor(trainLabelVectors, dtype=torch.float),
                                       torch.tensor(trainLabels, dtype=torch.float))

        vae_train_data_loader = DataLoader(vae_train_data, batch_size=m, shuffle=True)

        # training the vae
        losses = tfgczsl.train_vae(vae_train_data_loader, jj, model_prev = None)

        # Keep track of the classes seen till now
        replay_Classes = replay_Classes + sorted(list(set(trainLabels.tolist())))

    else:
        model_prev = deepcopy(tfgczsl)
        
        # classes seen until the previous task
        replay_TrainLabels = np.array(replay_Classes)
        
        # defining the data loader for training
        vae_train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float),
                                       torch.tensor(trainData, dtype=torch.float),
                                       torch.tensor(trainLabelVectors, dtype=torch.float),
                                       torch.tensor(trainLabels, dtype=torch.float))

        vae_train_data_loader = DataLoader(vae_train_data, batch_size=m, shuffle=True)

        # training the vae
        losses = tfgczsl.train_vae(vae_train_data_loader, jj, model_prev)

        # Keep track of the classes seen till now
        replay_Classes = replay_Classes + sorted(list(set(trainLabels.tolist())))
# =========================== UNSEEN CLASSES ======================================#

# TEST TRAIN SPLIT
   
    # initialize test data for seen classes
    testData_seen = []
    testLabels_seen = []
    testAttr_seen = []

    # initialize test data for unseen classes
    testLabels_unseen = []
    testData_unseen = []
    testAttr_unseen = []

    # test data of seen classes
    for ll in range(jj+1):
        testData_seen = testData_seen + list(testData1[ll])
        testLabels_seen = testLabels_seen + list(testLabels1[ll])
        testAttr_seen = testAttr_seen + list(testLabelVectors1[ll])
    
    # test data of unseen classes
    ll = task_no_1 - 1
    while True:
        if ll > jj:
            testData_unseen = testData_unseen + list(testData1[ll])
            testLabels_unseen = testLabels_unseen + list(testLabels1[ll])
            testAttr_unseen = testAttr_unseen + list(testLabelVectors1[ll])
            ll = ll - 1
        else:
            break

    print("****************")
    print("the length of test_data_seen.....", len(testData_seen))
    print("the length of test_data_unseen.....", len(testData_unseen))
    print("***************")

    trainClasses = sorted(list(set(testLabels_seen)))
    testClasses = sorted(list(set(testLabels_unseen)))
    
    # =============================================================================#

    # function to check whether the replay class tc is present in the memory
    def assert_cond(tfgczsl, tc):
        if name == method_names[0]:
            if tc in tfgczsl.er_mem.trainLabels_memory:
                return 1
            else:
                return 0
        else:
            if tfgczsl.er_mem.classes_filled[tc] != 0:
                return 1
            else:
                return 0
    
    # initialize the pseudo train data for the current task classes
    pseudoTrainData_trainClasses = []
    pseudoTrainLabels_trainClasses = []
    pseudoTrainAttr_trainClasses = []

    # initialize the pseudo train data for the previous task classes
    pseudoTrainData_replaytrainClasses = []
    pseudoTrainAttr_replaytrainClasses = []
    pseudoTrainLabels_replaytrainClasses = []

    # generate pseudo data for classes seen till now
    counter = 0
    for tc in trainClasses:
        if tc in sorted(list(set(list(replay_TrainLabels)))) and assert_cond(tfgczsl, tc):   
            # generate pseudo data for previous task classes using replay memory
            
            labels = [tc]
            labels = np.repeat(labels, tfgczsl.img_seen_samples, axis=0)
            pseudoTrainLabels_trainClasses = pseudoTrainLabels_trainClasses + labels.tolist()

            if name == method_names[0]:
                trainData_memories = np.array(tfgczsl.er_mem.trainData_memory)
                trainLabels_memories = np.array(tfgczsl.er_mem.trainLabels_memory)
            
                features_of_that_class = trainData_memories[trainLabels_memories == tc, :]
            else: 
                features_of_that_class = np.array(tfgczsl.er_mem.trainData_memory[tc])

            multiplier = math.ceil(np.maximum(1, tfgczsl.img_seen_samples / features_of_that_class.shape[0]))
            features_of_that_class = np.repeat(features_of_that_class, multiplier, axis=0)

            if counter == 0:
                features_to_return = torch.tensor(features_of_that_class[:tfgczsl.img_seen_samples, :], dtype=torch.float)
                counter = counter + 1
            else:
                # print("sizes are...")
                # print(features_to_return.size(), features_of_that_class.shape)
                features_to_return = torch.cat((features_to_return,
                                                torch.tensor(features_of_that_class[:tfgczsl.img_seen_samples, :],
                                                             dtype=torch.float)), dim=0)

        elif tc in sorted(list(set(list(trainLabels)))):
            # generate pseudo data for current task classes
            
            #print("train classes is...", tc)
            labels = [tc]
            labels = np.repeat(labels, tfgczsl.img_seen_samples, axis=0)
            pseudoTrainLabels_trainClasses = pseudoTrainLabels_trainClasses + labels.tolist()
            #print("pseudoTrainLabels_trainClasses is...", pseudoTrainLabels_trainClasses)

            features_of_that_class = trainData[trainLabels == tc, :]
            # print("the features of that class is...", features_of_that_class.shape[0])
            multiplier = math.ceil(np.maximum(1, tfgczsl.img_seen_samples / features_of_that_class.shape[0]))
            features_of_that_class = np.repeat(features_of_that_class, multiplier, axis=0)
            # print("the features of that class is...", features_of_that_class.shape[0])

            if counter == 0:
                features_to_return = torch.tensor(features_of_that_class[:tfgczsl.img_seen_samples, :], dtype=torch.float)
                counter = counter + 1
            else:
                # print("sizes are...")
                # print(features_to_return.size(), features_of_that_class.shape)
                features_to_return = torch.cat((features_to_return,
                                                torch.tensor(features_of_that_class[:tfgczsl.img_seen_samples, :],
                                                             dtype=torch.float)), dim=0)



    # encode the generated pseudo train data using the feature encoder (sampling is done based on mean and variance given by encoder)
    pseudoTrainData_trainClasses = tfgczsl.encode_feature(features_to_return.cuda())
    
    # initialize the train data for unseen classes
    pseudoTrainData_testClasses = []
    pseudoTrainLabels_testClasses = []
    pseudoTrainAttr_testClasses = []

    # generate pseudo data for unseen classes using the atribute information
    for tc in testClasses:
        labels = [tc]
        labels = np.repeat(labels, tfgczsl.att_unseen_samples, axis=0)

        attrs = [ATTR[tc]]
        attrs = np.repeat(attrs, tfgczsl.att_unseen_samples, axis=0)

        pseudoTrainAttr_testClasses = pseudoTrainAttr_testClasses + attrs.tolist()
        pseudoTrainLabels_testClasses = pseudoTrainLabels_testClasses + labels.tolist()

    if len(testClasses) != 0:
        # encode the pseudo data of unseen classes using the attribute encoder (sampling is done based on mean and variance given by encoder)
        
        pseudoTrainData_testClasses = tfgczsl.encode_attr(torch.tensor(pseudoTrainAttr_testClasses, dtype=torch.float).cuda())

        #print("pseudoTrainData_trainClasses is....", pseudoTrainData_testClasses.size())
        #print("pseudoTrainLabels_trainClasses is...", np.array(pseudoTrainLabels_testClasses).shape)

        # combine both seen and unseen classes pseudo train data
        pseudoTrainData = torch.cat((pseudoTrainData_trainClasses, pseudoTrainData_testClasses), dim=0)

        pseudoTrainLabels = np.concatenate((pseudoTrainLabels_trainClasses, pseudoTrainLabels_testClasses), axis=0)

    else:
        # when no unseen classes are there
        
        pseudoTrainData = pseudoTrainData_trainClasses
        pseudoTrainLabels = pseudoTrainLabels_trainClasses
        #print("the pseudoTrainData...", pseudoTrainData.size())
        #print("the pseudoTrainData...", pseudoTrainLabels.shape)


    # classifier:
    classifier_data = TensorDataset(torch.tensor(pseudoTrainData, dtype=torch.float),
                                    torch.tensor(pseudoTrainLabels, dtype=torch.long))
    classifier_data_loader = DataLoader(classifier_data, batch_size=classifier_batch_size, shuffle=True)
    print('Training Classifier')
    classifier.train()

    for epoch in range(classifier_epoch):
        correct = 0
        print("epoch_no = ", epoch)
        for x, targets in classifier_data_loader:
            x, targets = x.cuda(), targets.cuda()

            classifier_optim.zero_grad()
            pred = classifier(x)
            loss = classifier_criterion(pred, targets)
            loss.backward()
            classifier_optim.step()

            with torch.no_grad():
                output = torch.argmax(pred, dim=1).cpu().detach().numpy()
                correct += np.array(output == targets.cpu().detach().numpy()).astype(float).sum()
                # print("the correct is...", correct)

        # print('the size of pseudotrain attributes is...', pseudoTrainData.shape)
        print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch + 1, 10, loss.item(),
                                                                   correct / len(pseudoTrainLabels)))

    print('Predicting...')
    # To predict seen class accuracy
    classifier.eval()

    # get the test data of seen classes of the current task
    testData_seen = np.array(testData_seen)
    
    # encode the test data using the feature attribute (no sampling is done)
    testData_seen = tfgczsl.encode_test_data(torch.tensor(testData_seen, dtype=torch.float).cuda())
    
    # use the trained classifier to make predictions for class labels
    pred_s = classifier(torch.tensor(testData_seen, dtype=torch.float).cuda())
    pred_s = torch.argmax(pred_s, dim=1).cpu().detach().numpy()

        
    # initializing the variables to compute accuracy metrics
    dict_correct = {}
    dict_total = {}

    for ii in range(total_classes):
        dict_total[ii] = 0
        dict_correct[ii] = 0

    # computing seen class accuracy for the current task
    allSeenClasses = sorted(list(set(testLabels_seen)))
    
    for ii in range(0, np.array(testLabels_seen).shape[0]):
        if (testLabels_seen[ii] == pred_s[ii]):
            dict_correct[testLabels_seen[ii]] = dict_correct[testLabels_seen[ii]] + 1
        dict_total[testLabels_seen[ii]] = dict_total[testLabels_seen[ii]] + 1

    avgAcc1 = 0.0
    num_seen = 0.0
    
    for ii in allSeenClasses:
        avgAcc1 = avgAcc1 + (dict_correct[ii] * 1.0) / (dict_total[ii])
        num_seen = num_seen + 1

    avgAcc1 = avgAcc1 / num_seen
    seen_acc.append(avgAcc1)

    # To predict unseen class accuracy
    if jj != task_no - 1:
        # load the test data of unseen classes of the current task
        testData_unseen = np.array(testData_unseen)
        
        # encode the test data using the feature attribute (no sampling is done) 
        testData_unseen = tfgczsl.encode_test_data(torch.tensor(testData_unseen, dtype=torch.float).cuda())

        # use the trained classifier to make predictions for class labels
        pred_us = classifier(torch.tensor(testData_unseen, dtype=torch.float).cuda())
        pred_us = torch.argmax(pred_us, dim=1).cpu().detach().numpy()
        
        # computing unseen class accuracy for the current task
        allUnseenClasses = sorted(list(set(testLabels_unseen)))
        
        avgAcc2 = 0.0
        num_unseen = 0.0
        
        for ii in range(0, np.array(testLabels_unseen).shape[0]):
            if (testLabels_unseen[ii] == pred_us[ii]):
                dict_correct[testLabels_unseen[ii]] = dict_correct[testLabels_unseen[ii]] + 1
            dict_total[testLabels_unseen[ii]] = dict_total[testLabels_unseen[ii]] + 1

        for ii in allUnseenClasses:
            avgAcc2 = avgAcc2 + (dict_correct[ii] * 1.0) / (dict_total[ii])
            num_unseen = num_unseen + 1

        avgAcc2 = avgAcc2 / num_unseen
        unseen_acc.append(avgAcc2)

    # computing overall accuracy (joint accuracy):
    if jj != task_no-1:
        testData_total = torch.cat((testData_seen, testData_unseen), dim=0)
    elif jj < task_no_1-1:
        testData_unseen = np.array(testData_unseen)
        testData_unseen = tfgczsl.encode_test_data(torch.tensor(testData_unseen, dtype=torch.float).cuda())
        testData_total = torch.cat((testData_seen, testData_unseen), dim=0)
    else:
        testData_total = testData_seen

    targets = testLabels_seen + testLabels_unseen
    targets_classes = sorted(list(set(targets)))

    pred_ov = classifier(torch.tensor(testData_total, dtype=torch.float).cuda())
    pred_ov = torch.argmax(pred_ov, dim=1).cpu().detach().numpy()

    dict_correct_oacc = {}
    dict_total_oacc = {}

    for ii in targets_classes:
        dict_total_oacc[ii] = 0
        dict_correct_oacc[ii] = 0

    for ii in range(0, np.array(targets).shape[0]):
        if (targets[ii] == pred_ov[ii]):
            dict_correct_oacc[targets[ii]] = dict_correct_oacc[targets[ii]] + 1
        dict_total_oacc[targets[ii]] = dict_total_oacc[targets[ii]] + 1

    avgAcc_ov = 0.0
    num_seen_ov = 0.0

    for ii in targets_classes:
        avgAcc_ov = avgAcc_ov + (dict_correct_oacc[ii] * 1.0) / (dict_total_oacc[ii])
        num_seen_ov = num_seen_ov + 1

    avgAcc_ov = avgAcc_ov / num_seen_ov

    overall_acc.append(avgAcc_ov)

    #########################################################################
    # To compute accuracy matrix:
    for kk in range(task_no):
        testData_tw = np.array(testData1[kk])
        testLabels_tw = np.array(testLabels1[kk])
        testAttr_tw = np.array(testLabelVectors1[kk])
        testLabels_tw_classes = sorted(list(set(testLabels_tw.tolist())))

        testData_tw = tfgczsl.encode_test_data(torch.tensor(testData_tw, dtype=torch.float).cuda())
        pred_tw = classifier(torch.tensor(testData_tw, dtype=torch.float).cuda())
        pred_tw = torch.argmax(pred_tw, dim=1).cpu().detach().numpy()

        dict_correct_tw = {}
        dict_total_tw = {}

        for ii in testLabels_tw_classes:
            dict_total_tw[ii] = 0
            dict_correct_tw[ii] = 0

        for ii in range(0, np.array(testLabels_tw).shape[0]):
            if (testLabels_tw[ii] == pred_tw[ii]):
                dict_correct_tw[testLabels_tw[ii]] = dict_correct_tw[testLabels_tw[ii]] + 1
            dict_total_tw[testLabels_tw[ii]] = dict_total_tw[testLabels_tw[ii]] + 1

        avgAcc_tw = 0.0
        num_seen_tw = 0.0

        for ii in testLabels_tw_classes:
            avgAcc_tw = avgAcc_tw + (dict_correct_tw[ii] * 1.0) / (dict_total_tw[ii])
            num_seen_tw = num_seen_tw + 1

        avgAcc_tw = avgAcc_tw / num_seen_tw

        accuracy_matrix[jj].append(avgAcc_tw)

#########################################################################

# calculating the harmonic mean
for jj in range(task_no - 1):
    hm = (2 * (seen_acc[jj] * unseen_acc[jj])) / (seen_acc[jj] + unseen_acc[jj])
    harmonic_mean.append(hm)
print("the harmonic mean is...", np.mean(np.array(harmonic_mean)))
print("the seen acc mean is...", np.mean(np.array(seen_acc)))
print("the unseen acc mean is...", np.mean(np.array(unseen_acc)))

# calculating forgetting measure:
accuracy_matrix = np.array(accuracy_matrix)
forgetting_measure = []
for after_task_idx in range(1, task_no):
    after_task_num = after_task_idx + 1
    prev_acc = accuracy_matrix[:after_task_num - 1, :after_task_num - 1]
    forgettings = prev_acc.max(axis=0) - accuracy_matrix[after_task_num - 1, :after_task_num - 1]
    forgetting_measure.append(np.mean(forgettings).item())

#print("the forgetting measure is...", np.mean(np.array(forgetting_measure)))

# calculating joint accuracy:
mean_joint_acc = np.mean(np.array(overall_acc))

# saving data:
with open('results_tfgczsl_reservoir.csv', 'a') as file:
    writer = csv.writer(file)
    writer.writerow(harmonic_mean + [0, np.mean(np.array(harmonic_mean)).item()])
    writer.writerow(seen_acc + [np.mean(np.array(seen_acc)).item()])
    writer.writerow(unseen_acc + [0, np.mean(np.array(unseen_acc)).item()])
    #writer.writerow(overall_acc + [mean_joint_acc])
    #writer.writerow(forgetting_measure + [0, np.mean(np.array(forgetting_measure)).item()])











