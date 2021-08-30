import numpy as np
import torch
import random

class ER_MEM():
    def __init__(self, name, samples_per_class, total_classes, memory_batch_size):
        # call the corresponding initalization function based on the selected method
        method_to_call_mem_initialize = getattr(self, name + "_initialize")
        method_to_call_mem_initialize(samples_per_class, total_classes, memory_batch_size)

    
    #======================================================================================================#
        
    # Reservoir memory (changed for dark er):
    
    # function to initialize the reservoir memory
    def reservoir_initialize(self, samples_per_class, total_classes, memory_batch_size):
        self.X_train_memory = []
        self.trainLabelVectors_memory = []
        self.trainData_memory = []
        self.trainLabels_memory = []
        self.softLabels_mean_feat_memory = []
        self.softLabels_var_feat_memory = []
        self.softLabels_mean_attr_memory = []
        self.softLabels_var_attr_memory = []
        self.cls_softLabels = []
        self.mem_size = samples_per_class * total_classes
        self.mem_batch_size = memory_batch_size
        self.mem_counter = 0

    # function to select data from the reservoir memory
    def reservoir_mem_select_data(self, X_train, trainData, attributes):
        
        # when the memory is empty return the train data as it is
        if len(self.X_train_memory) == 0:
            return X_train, trainData, attributes, None, None, None, None, None, None, None

        # when the memory size is less than mem_batch size use all the data in memory
        elif len(self.X_train_memory) < self.mem_batch_size:
            X_train_memory_batch = torch.tensor(self.X_train_memory, dtype=torch.float32)
            trainLabelVectors_memory_batch = torch.tensor(self.trainLabelVectors_memory, dtype=torch.float32)
            trainData_memory_batch = torch.tensor(self.trainData_memory, dtype=torch.float32)

            # Dark er
            soft_mean_feat_batch = torch.tensor(self.softLabels_mean_feat_memory, dtype=torch.float32)
            soft_var_feat_batch = torch.tensor(self.softLabels_var_feat_memory, dtype=torch.float32)
            soft_mean_attr_batch = torch.tensor(self.softLabels_mean_attr_memory, dtype=torch.float32)
            soft_var_attr_batch = torch.tensor(self.softLabels_var_attr_memory, dtype=torch.float32)

            X_train = torch.cat((X_train, X_train_memory_batch), dim=0)
            attributes = torch.cat((attributes, trainLabelVectors_memory_batch), dim=0)
            trainData = torch.cat((trainData, trainData_memory_batch), dim=0)

            return X_train, trainData, attributes, X_train_memory_batch, trainData_memory_batch, \
                   trainLabelVectors_memory_batch, soft_mean_feat_batch, soft_var_feat_batch, soft_mean_attr_batch, \
                   soft_var_attr_batch

        # when the memory size is greater than mem_batch size randomly select data of size mem_batch
        elif len(self.X_train_memory) >= self.mem_batch_size:
            mem_index = np.random.choice(len(self.X_train_memory), self.mem_batch_size, replace=False)
            X_train_memory_batch = torch.tensor(np.array(self.X_train_memory)[mem_index], dtype=torch.float32)
            trainLabelVectors_memory_batch = torch.tensor(np.array(self.trainLabelVectors_memory)[mem_index],
                                                          dtype=torch.float32)
            trainData_memory_batch = torch.tensor(np.array(self.trainData_memory)[mem_index], dtype=torch.float32)

            # Dark er
            soft_mean_feat_batch = torch.tensor(np.array(self.softLabels_mean_feat_memory)[mem_index], dtype=torch.float32)
            soft_var_feat_batch = torch.tensor(np.array(self.softLabels_var_feat_memory)[mem_index], dtype=torch.float32)
            soft_mean_attr_batch = torch.tensor(np.array(self.softLabels_mean_attr_memory)[mem_index], dtype=torch.float32)
            soft_var_attr_batch = torch.tensor(np.array(self.softLabels_var_attr_memory)[mem_index], dtype=torch.float32)

            X_train = torch.cat((X_train, X_train_memory_batch), dim=0)
            attributes = torch.cat((attributes, trainLabelVectors_memory_batch), dim=0)
            trainData = torch.cat((trainData, trainData_memory_batch), dim=0)

            return X_train, trainData, attributes, X_train_memory_batch, trainData_memory_batch, \
                   trainLabelVectors_memory_batch, soft_mean_feat_batch, soft_var_feat_batch, soft_mean_attr_batch, \
                   soft_var_attr_batch



    # function to update the reservoir memory
    def reservoir_mem_update(self, X_train_copy, trainData_copy, attributes_copy, trainLabels, encoder_feats, encoder_attrs, classifier=None):
        mean_feat, var_feat = encoder_feats(torch.tensor(trainData_copy, dtype=torch.float, device='cuda'))
        mean_feat, var_feat = mean_feat.detach().cpu().numpy(), var_feat.detach().cpu().numpy()
        mean_attr, var_attr = encoder_attrs(torch.tensor(attributes_copy, dtype=torch.float, device='cuda'))
        mean_attr, var_attr = mean_attr.detach().cpu().numpy(), var_attr.detach().cpu().numpy()
        # class_softLabels = classifier(mean_feat)

        for ll in range(len(X_train_copy)):
            # when the memory is not full directly add the data to memory
            if len(self.X_train_memory) < self.mem_size:
                self.X_train_memory.append(X_train_copy[ll])
                self.trainLabelVectors_memory.append(attributes_copy[ll])
                self.trainData_memory.append(trainData_copy[ll])
                self.trainLabels_memory.append(trainLabels[ll])

                # Dark er
                self.softLabels_mean_feat_memory.append(mean_feat[ll])
                self.softLabels_var_feat_memory.append(var_feat[ll])
                self.softLabels_mean_attr_memory.append(mean_attr[ll])
                self.softLabels_var_attr_memory.append(var_attr[ll])

            # When the memory is full update the memory based on reservoir strategy
            else:
                rand_num = np.random.randint(0, self.mem_counter)
                if rand_num < self.mem_size:
                    self.X_train_memory[rand_num] = X_train_copy[ll]
                    self.trainLabelVectors_memory[rand_num] = attributes_copy[ll]
                    self.trainData_memory[rand_num] = trainData_copy[ll]
                    self.trainLabels_memory[rand_num] = trainLabels[ll]

                    # Dark er
                    self.softLabels_mean_feat_memory[rand_num] = mean_feat[ll]
                    self.softLabels_var_feat_memory[rand_num] = var_feat[ll]
                    self.softLabels_mean_attr_memory[rand_num] = mean_attr[ll]
                    self.softLabels_var_attr_memory[rand_num] = var_attr[ll]
            self.mem_counter = self.mem_counter + 1

    #======================================================================================================#
    
    # Ring Buffer memory:
    
    # function to initialize the ring buffer memory
    def ring_buffer_initialize(self, samples_per_class, total_classes, memory_batch_size):
        self.samples_pc = samples_per_class
        self.total_classes = total_classes
        self.mem_batch_size = memory_batch_size

        self.X_train_memory = [[] for _ in range(self.total_classes)]
        self.trainLabelVectors_memory = [[] for _ in range(self.total_classes)]
        self.trainData_memory = [[] for _ in range(self.total_classes)]
        self.trainLabels_memory = [[] for _ in range(self.total_classes)]
        self.classes_filled = [0] * self.total_classes

    
    # function to select data from the ring buffer memory
    def ring_buffer_mem_select_data(self, X_train, trainData, attributes):
        X_train_mem = []
        trainLabelVectors_mem = []
        trainData_mem = []

        # when the memory is empty return the train data as it is
        if sum(self.classes_filled) == 0:
            return X_train, trainData, attributes

        # when the memory size is less than mem_batch size use all the data in memory
        elif sum(self.classes_filled) < self.mem_batch_size:
            mem_filled = [idx for idx, val in enumerate(self.classes_filled) if val != 0]
            for mm in mem_filled:
                X_train_mem += self.X_train_memory[mm]
                trainLabelVectors_mem += self.trainLabelVectors_memory[mm]
                trainData_mem += self.trainData_memory[mm]

            X_train_memory_batch = torch.tensor(X_train_mem, dtype=torch.float32)
            trainLabelVectors_memory_batch = torch.tensor(trainLabelVectors_mem, dtype=torch.float32)
            trainData_memory_batch = torch.tensor(trainData_mem, dtype=torch.float32)

            X_train = torch.cat((X_train, X_train_memory_batch), dim=0)
            attributes = torch.cat((attributes, trainLabelVectors_memory_batch), dim=0)
            trainData = torch.cat((trainData, trainData_memory_batch), dim=0)

            return X_train, trainData, attributes

        # when the memory size is greater than mem_batch size randomly select data of size mem_batch
        elif sum(self.classes_filled) >= self.mem_batch_size:
            mem_filled = [idx for idx, val in enumerate(self.classes_filled) if val != 0]
            for mm in range(self.mem_batch_size):
                mem_index = random.choice(mem_filled)
                temp_index = random.randint(0, self.classes_filled[mem_index] - 1)

                # print("the temp_index is...", temp_index, self.classes_filled[mem_index])
                # print("the classes_filled is...", self.classes_filled)
                # print("X_train_memory is...", [self.X_train_memory[mem_index][temp_index]])

                X_train_mem.append(self.X_train_memory[mem_index][temp_index])
                trainLabelVectors_mem.append(self.trainLabelVectors_memory[mem_index][temp_index])
                trainData_mem.append(self.trainData_memory[mem_index][temp_index])

            X_train_memory_batch = torch.tensor(X_train_mem, dtype=torch.float32)
            trainLabelVectors_memory_batch = torch.tensor(trainLabelVectors_mem, dtype=torch.float32)
            trainData_memory_batch = torch.tensor(trainData_mem, dtype=torch.float32)

            X_train = torch.cat((X_train, X_train_memory_batch), dim=0)
            attributes = torch.cat((attributes, trainLabelVectors_memory_batch), dim=0)
            trainData = torch.cat((trainData, trainData_memory_batch), dim=0)

            return X_train, trainData, attributes

    # function to update the ring buffer memory
    def ring_buffer_mem_update(self, X_train_copy, trainData_copy, attributes_copy, trainLabels):
        for ll in range(len(X_train_copy)):
            # when the memory is not full directly add the data to memory
            if len(self.X_train_memory[trainLabels[ll]]) >= self.samples_pc:
                self.X_train_memory[trainLabels[ll]].pop(0)
                self.X_train_memory[trainLabels[ll]].append(X_train_copy[ll])

                self.trainLabelVectors_memory[trainLabels[ll]].pop(0)
                self.trainLabelVectors_memory[trainLabels[ll]].append(attributes_copy[ll])

                self.trainData_memory[trainLabels[ll]].pop(0)
                self.trainData_memory[trainLabels[ll]].append(trainData_copy[ll])

                self.trainLabels_memory[trainLabels[ll]].pop(0)
                self.trainLabels_memory[trainLabels[ll]].append(trainLabels[ll])

            # When the memory is full update the memory based on reservoir strategy
            else:
                self.X_train_memory[trainLabels[ll]].append(X_train_copy[ll])
                self.trainLabelVectors_memory[trainLabels[ll]].append(attributes_copy[ll])
                self.trainData_memory[trainLabels[ll]].append(trainData_copy[ll])
                self.trainLabels_memory[trainLabels[ll]].append(trainLabels[ll])

            assert len(self.X_train_memory[trainLabels[ll]]) <= self.samples_pc

            self.classes_filled[trainLabels[ll]] = min(self.samples_pc, self.classes_filled[trainLabels[ll]] + 1)

    
    #======================================================================================================#
    
    # Mean of Features memory
    
    # function to initialize the mof memory
    def mof_initialize(self, samples_per_class, total_classes, memory_batch_size):
        self.samples_pc = samples_per_class
        self.total_classes = total_classes
        self.mem_batch_size = memory_batch_size

        self.X_train_memory = [[] for _ in range(self.total_classes)]
        self.trainLabelVectors_memory = [[] for _ in range(self.total_classes)]
        self.trainData_memory = [[] for _ in range(self.total_classes)]
        self.trainLabels_memory = [[] for _ in range(self.total_classes)]
        self.classes_filled = [0] * self.total_classes
        self.average_features = [[] for _ in range(self.total_classes)]
        self.distance_values = [[] for _ in range(self.total_classes)]
        self.avg_alpha = 0.7

    # function to select data from mof memory
    def mof_mem_select_data(self, X_train, trainData, attributes):
        X_train_mem = []
        trainLabelVectors_mem = []
        trainData_mem = []

        # when the memory is empty return the train data as it is
        if sum(self.classes_filled) == 0:
            return X_train, trainData, attributes

        # when the memory size is less than mem_batch size use all the data in memory
        elif sum(self.classes_filled) < self.mem_batch_size:
            mem_filled = [idx for idx, val in enumerate(self.classes_filled) if val != 0]
            for mm in mem_filled:
                X_train_mem += self.X_train_memory[mm]
                trainLabelVectors_mem += self.trainLabelVectors_memory[mm]
                trainData_mem += self.trainData_memory[mm]

            X_train_memory_batch = torch.tensor(X_train_mem, dtype=torch.float32)
            trainLabelVectors_memory_batch = torch.tensor(trainLabelVectors_mem, dtype=torch.float32)
            trainData_memory_batch = torch.tensor(trainData_mem, dtype=torch.float32)

            X_train = torch.cat((X_train, X_train_memory_batch), dim=0)
            attributes = torch.cat((attributes, trainLabelVectors_memory_batch), dim=0)
            trainData = torch.cat((trainData, trainData_memory_batch), dim=0)

            return X_train, trainData, attributes

        # when the memory size is greater than mem_batch size randomly select data of size mem_batch
        elif sum(self.classes_filled) >= self.mem_batch_size:
            mem_filled = [idx for idx, val in enumerate(self.classes_filled) if val != 0]
            for mm in range(self.mem_batch_size):
                mem_index = random.choice(mem_filled)
                temp_index = random.randint(0, self.classes_filled[mem_index] - 1)

                # print("the temp_index is...", temp_index, self.classes_filled[mem_index])
                # print("the classes_filled is...", self.classes_filled)
                # print("X_train_memory is...", [self.X_train_memory[mem_index][temp_index]])

                X_train_mem.append(self.X_train_memory[mem_index][temp_index])
                trainLabelVectors_mem.append(self.trainLabelVectors_memory[mem_index][temp_index])
                trainData_mem.append(self.trainData_memory[mem_index][temp_index])

            X_train_memory_batch = torch.tensor(X_train_mem, dtype=torch.float32)
            trainLabelVectors_memory_batch = torch.tensor(trainLabelVectors_mem, dtype=torch.float32)
            trainData_memory_batch = torch.tensor(trainData_mem, dtype=torch.float32)

            X_train = torch.cat((X_train, X_train_memory_batch), dim=0)
            attributes = torch.cat((attributes, trainLabelVectors_memory_batch), dim=0)
            trainData = torch.cat((trainData, trainData_memory_batch), dim=0)

            return X_train, trainData, attributes

    # function to update the mof memory
    def mof_mem_update(self, X_train_copy, trainData_copy, attributes_copy, trainLabels):
        for ll in range(len(X_train_copy)):
            
            # when the memory is not full directly add the data to memory
            if len(self.X_train_memory[trainLabels[ll]]) >= self.samples_pc:
                self.average_features[trainLabels[ll]] = self.avg_alpha * self.average_features[trainLabels[ll]] + (
                            1 - self.avg_alpha) * X_train_copy[ll]

                distance = np.linalg.norm((X_train_copy[ll] - self.average_features[trainLabels[ll]]))
                max_distance = max(self.distance_values[trainLabels[ll]])

                if distance < max_distance:
                    f = lambda i: self.distance_values[trainLabels[ll]][i]
                    index_max_distance = max(range(len(self.distance_values[trainLabels[ll]])), key=f)

                    self.X_train_memory[trainLabels[ll]][index_max_distance] = (X_train_copy[ll])

                    self.trainLabelVectors_memory[trainLabels[ll]][index_max_distance] = (attributes_copy[ll])

                    self.trainData_memory[trainLabels[ll]][index_max_distance] = (trainData_copy[ll])

                    self.trainLabels_memory[trainLabels[ll]][index_max_distance] = (trainLabels[ll])

                    self.distance_values[trainLabels[ll]][index_max_distance] = distance

            # When the memory is full update the memory based on reservoir strategy
            else:
                if self.classes_filled[trainLabels[ll]] == 0:
                    self.average_features[trainLabels[ll]] = X_train_copy[ll]
                else:
                    self.average_features[trainLabels[ll]] = self.avg_alpha * self.average_features[trainLabels[ll]] + (1 - self.avg_alpha) * X_train_copy[ll]

                self.X_train_memory[trainLabels[ll]].append(X_train_copy[ll])
                self.trainLabelVectors_memory[trainLabels[ll]].append(attributes_copy[ll])
                self.trainData_memory[trainLabels[ll]].append(trainData_copy[ll])
                self.trainLabels_memory[trainLabels[ll]].append(trainLabels[ll])
                distance = np.linalg.norm((X_train_copy[ll] - self.average_features[trainLabels[ll]]))
                self.distance_values[trainLabels[ll]].append(distance)

            assert len(self.X_train_memory[trainLabels[ll]]) <= self.samples_pc

            self.classes_filled[trainLabels[ll]] = min(self.samples_pc, self.classes_filled[trainLabels[ll]] + 1)








