import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tfcgzsl import models
from tfcgzsl.er_memory import ER_MEM

class TFCGZSL(nn.Module):

    def __init__(self, m, n_x, n_y, n_z, nTrain, nTest, samples_per_class, total_classes, memory_batch_size, use_KD, use_der, name):
        super(TFCGZSL, self).__init__()
        
        # create an object for selecting and storing data in the replay memory
        self.er_mem = ER_MEM(name, samples_per_class, total_classes, memory_batch_size)
                
        # defining the function names for initializing, selecting data and updating replay memory based on the chosen method
        self.name_mem_initialize = name + "_initialize"
        self.name_mem_select_data = name + "_mem_select_data"
        self.name_mem_update = name + "_mem_update"

        self.device = 'cuda'
        self.all_data_sources = ['resnet_features', 'attributes']
        
        self.latent_size = n_z
        self.batch_size = m
        self.hidden_size_rule = {'resnet_features': (1560, 1660), 'attributes': (1450, 665), 'sentences': (1450, 665)}

        self.warmup = {'beta': {'factor': 0.25, 'end_epoch': 93, 'start_epoch': 0},
                       'cross_reconstruction': {'factor': 2.37, 'end_epoch': 75, 'start_epoch': 21},
                       'distance': {'factor': 8.13, 'end_epoch': 22, 'start_epoch': 6}}

        self.generalized = True
        self.img_seen_samples = 200
        self.att_seen_samples = 0
        self.att_unseen_samples = 400
        self.img_unseen_samples = 0

        self.reco_loss_function = 'l1'
        self.nepoch = 100
        self.cross_reconstruction = True
        self.use_KD = use_KD
        self.use_der = use_der
        
        feature_dimensions = [n_x, n_y]

        # Here, the encoders and decoders for all modalities are created and put into dict

        self.encoder = {}

        for datatype, dim in zip(self.all_data_sources, feature_dimensions):

            self.encoder[datatype] = models.encoder_template(dim, self.latent_size, self.hidden_size_rule[datatype],
                                                             self.device)
            print(str(datatype) + ' ' + str(dim))

        self.decoder = {}

        for datatype, dim in zip(self.all_data_sources, feature_dimensions):
            self.decoder[datatype] = models.decoder_template(self.latent_size, dim, self.hidden_size_rule[datatype],
                                                             self.device)

        # An optimizer for all encoders and decoders is defined here
        parameters_to_optimize = list(self.parameters())
        for datatype in self.all_data_sources:
            parameters_to_optimize += list(self.encoder[datatype].parameters())
            parameters_to_optimize += list(self.decoder[datatype].parameters())

        self.optimizer = optim.Adam(parameters_to_optimize, lr=0.00015, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                    amsgrad=True)

        if self.reco_loss_function == 'l2':
            self.reconstruction_criterion = nn.MSELoss(size_average=False)

        elif self.reco_loss_function == 'l1':
            self.reconstruction_criterion = nn.L1Loss(size_average=False)

        

    # function for sampling
    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            # sample based on mean and variance
            sigma = torch.exp(logvar)
            eps = torch.cuda.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
            eps = eps.expand(sigma.size())
            return mu + sigma * eps
        else:
            # no sampling is done
            return mu

    
    # function for performing a single gradient descent step after computing the respective losses 
    def trainstep(self, img, att, trainData_mem = None, attributes_mem = None,
                  soft_mean_feat = None, soft_var_feat = None, soft_mean_attr = None, soft_var_attr = None,
                  feature_mean = None, feature_variance = None, attr_mean = None, attr_variance = None, use_KD = False, use_der = False):

        ##############################################
        # Encode image features and additional
        # features
        ##############################################

        mu_img, logvar_img = self.encoder['resnet_features'](img)
        z_from_img = self.reparameterize(mu_img, logvar_img)

        mu_att, logvar_att = self.encoder['attributes'](att)
        z_from_att = self.reparameterize(mu_att, logvar_att)

        ##############################################
        # Reconstruct inputs
        ##############################################

        #z_from_img_attr = torch.cat((z_from_img, att), dim=1)
        img_from_img = self.decoder['resnet_features'](z_from_img)
        att_from_att = self.decoder['attributes'](z_from_att)

        reconstruction_loss = self.reconstruction_criterion(img_from_img, img) \
                              + self.reconstruction_criterion(att_from_att, att)

        ##############################################
        # Cross Reconstruction Loss
        ##############################################
        #z_from_att_attr = torch.cat((z_from_att, att), dim=1)
        img_from_att = self.decoder['resnet_features'](z_from_att)
        att_from_img = self.decoder['attributes'](z_from_img)

        cross_reconstruction_loss = self.reconstruction_criterion(img_from_att, img) \
                                    + self.reconstruction_criterion(att_from_img, att)

        if use_KD:
            ##############################################
            # Knowledge Distillation
            ##############################################
            KD_loss = nn.L1Loss(size_average=False)
            KD_feature = KD_loss(mu_img, feature_mean) + KD_loss(logvar_img, feature_variance)

            KD_att = KD_loss(mu_att, attr_mean) + KD_loss(logvar_att, attr_variance)


        ##############################################
        # KL-Divergence
        ##############################################

        KLD = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) \
              + (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))

        ##############################################
        # Distribution Alignment
        ##############################################
        distance = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) + \
                              torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1))

        distance = distance.sum()

        if use_der:
            ##############################################
            # Der
            ##############################################
            mu_img_current, logvar_img_current = self.encoder['resnet_features'](trainData_mem)
            mu_att_current, logvar_att_current = self.encoder['attributes'](attributes_mem)
            #der_loss_criteria = torch.nn.MSELoss(reduction='sum')
            der_loss_criteria = torch.nn.L1Loss(reduction='sum')
            der_loss = der_loss_criteria(mu_img_current, soft_mean_feat) + der_loss_criteria(logvar_img_current, soft_var_feat) + der_loss_criteria(mu_att_current, soft_mean_attr) + der_loss_criteria(logvar_att_current, soft_var_attr)
            der_loss = 1 * der_loss

        ##############################################
        # scale the loss terms according to the warmup
        # schedule
        ##############################################

        # factor for cross-reconstruction loss
        f1 = 1.0 * (self.current_epoch - self.warmup['cross_reconstruction']['start_epoch']) / (1.0 * (
                self.warmup['cross_reconstruction']['end_epoch'] - self.warmup['cross_reconstruction'][
            'start_epoch']))
        f1 = f1 * (1.0 * self.warmup['cross_reconstruction']['factor'])
        cross_reconstruction_factor = torch.cuda.FloatTensor(
            [min(max(f1, 0), self.warmup['cross_reconstruction']['factor'])])

        # factor for KL Divergence loss
        f2 = 1.0 * (self.current_epoch - self.warmup['beta']['start_epoch']) / (
                1.0 * (self.warmup['beta']['end_epoch'] - self.warmup['beta']['start_epoch']))
        f2 = f2 * (1.0 * self.warmup['beta']['factor'])
        beta = torch.cuda.FloatTensor([min(max(f2, 0), self.warmup['beta']['factor'])])

        # factor for distribution alignment
        f3 = 1.0 * (self.current_epoch - self.warmup['distance']['start_epoch']) / (
                1.0 * (self.warmup['distance']['end_epoch'] - self.warmup['distance']['start_epoch']))
        f3 = f3 * (1.0 * self.warmup['distance']['factor'])
        distance_factor = torch.cuda.FloatTensor([min(max(f3, 0), self.warmup['distance']['factor'])])

        ##############################################
        # Put the loss together and call the optimizer
        ##############################################

        self.optimizer.zero_grad()

        if use_KD:
            loss = reconstruction_loss - beta * KLD + KD_feature + KD_att
        else:
            loss = 0
        
        if use_der:
            loss += reconstruction_loss - beta * KLD + der_loss
        else:
            loss += reconstruction_loss - beta * KLD

        if cross_reconstruction_loss > 0:
            loss += cross_reconstruction_factor * cross_reconstruction_loss
        if distance_factor > 0:
            loss += distance_factor * distance

        loss.backward()

        self.optimizer.step()

        return loss.item()
    
    # function for training the TFCGZSL
    def train_vae(self, dataset_vae, jj, model_prev, classifier=None):

        losses = []

        self.train()
        self.reparameterize_with_noise = True

        print('train for reconstruction')
        for epoch in range(0, self.nepoch):
            self.current_epoch = epoch
            i = -1
            for iters, (X_train, trainData, attributes, trainLabels) in enumerate(dataset_vae):
                i += 1

                # storing data separately for upadatin the replay memory
                X_train_copy = np.array(X_train.numpy())
                trainData_copy = np.array(trainData.numpy())
                attributes_copy = np.array(attributes.numpy())
                trainLabels = np.array(trainLabels.numpy(), dtype=np.int32)

                # calling the function to append the current data with replay data
                method_to_call_mem_select_data = getattr(self.er_mem, self.name_mem_select_data)
                X_train, trainData, attributes, X_train_mem, trainData_mem, attributes_mem, \
                   soft_mean_feat, soft_var_feat, soft_mean_attr, soft_var_attr = method_to_call_mem_select_data(X_train, trainData, attributes)

                if jj == 0:
                    # for first task replay memory is not used for training
                    X_train, trainData, attributes = torch.tensor(X_train_copy, dtype=torch.float).cuda(), torch.tensor \
                        (trainData_copy, dtype=torch.float).cuda(), torch.tensor(attributes_copy, dtype=torch.float).cuda()
                    
                    loss = self.trainstep(trainData, attributes)
                else:
                    # replay memory is used from second task onwards for training
                    X_train, trainData, attributes = X_train.cuda(), trainData.cuda(), attributes.cuda()
                    # X_train, trainData, attributes = torch.tensor(X_train_copy, dtype=torch.float).cuda(), torch.tensor \
                    #     (trainData_copy, dtype=torch.float).cuda(), torch.tensor(attributes_copy, dtype=torch.float).cuda()
                    
                    X_train_mem, trainData_mem, attributes_mem = X_train_mem.cuda(), trainData_mem.cuda(), attributes_mem.cuda()
                    soft_mean_feat, soft_var_feat, soft_mean_attr, soft_var_attr = soft_mean_feat.cuda(), soft_var_feat.cuda(), \
                                                                                   soft_mean_attr.cuda(), soft_var_attr.cuda()

                    mean_feature, logvar_feature = model_prev.encoder['resnet_features'](trainData)
                    mean_att, logvar_att = model_prev.encoder['attributes'](attributes)
                
                    loss = self.trainstep(trainData, attributes, trainData_mem, attributes_mem,
                                          soft_mean_feat, soft_var_feat, soft_mean_attr, soft_var_attr,
                                          mean_feature, logvar_feature, mean_att, logvar_att, self.use_KD, self.use_der)

                # calling the function to update the replay memory
                method_to_call_update_mem = getattr(self.er_mem, self.name_mem_update)
                method_to_call_update_mem(X_train_copy, trainData_copy, attributes_copy, trainLabels,
                                          self.encoder['resnet_features'], self.encoder['attributes'], classifier=None)

                if i % 50 == 0:
                    print('epoch ' + str(epoch) + ' | iter ' + str(i) + '\t' +
                          ' | loss ' + str(loss)[:5])

                if i % 50 == 0 and i > 0:
                    losses.append(loss)

        # turn into evaluation mode:
        for key, value in self.encoder.items():
            self.encoder[key].eval()
        for key, value in self.decoder.items():
            self.decoder[key].eval()

        self.reparameterize_with_noise = False
        return losses

    # function for encoding the attributes using the attribute encoder (used for generating pseudo data) 
    def encode_attr(self, attr):
        self.reparameterize_with_noise = True

        mu_attr, logvar_attr = self.encoder['attributes'](attr)
        z_from_attr = self.reparameterize(mu_attr, logvar_attr)

        self.reparameterize_with_noise = False
        return z_from_attr

    # function for encoding the features using the feature encoder (used for generating pseudo data) 
    def encode_feature(self, feat):
        self.reparameterize_with_noise = True

        mu_feat, logvar_feat = self.encoder['resnet_features'](feat)
        z_from_feat = self.reparameterize(mu_feat, logvar_feat)

        self.reparameterize_with_noise = False
        return z_from_feat

    # function for encoding the test data
    def encode_test_data(self, feat):
        self.reparameterize_with_noise = False

        mu_feat, logvar_feat = self.encoder['resnet_features'](feat)

        return mu_feat
    
