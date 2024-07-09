import torch
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from collections import defaultdict, Counter
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
import optuna
from torchmetrics.classification import F1Score
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")
import random
from sklearn.model_selection import ShuffleSplit
from sklearn.cluster import KMeans,DBSCAN,Birch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, v_measure_score

sys.path.insert(1, '/home/wangc90/Data_integration/MOCSS/mocss/code/')
from critic import LinearCritic
# from Supcon import SupConLoss
from contrastive_loss import InstanceLoss, ClusterLoss
import evaluation
from sklearn import metrics
from Data_prep import DataSet_Prep, DataSet_construction
random.seed(2023)
torch.manual_seed(2023)


### prepare the dataset for the model
from torch.utils.data import Dataset
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

np.random.seed(2023)

### create DATASET for training
class DataSet_Prep():
    def __init__(self, data1, data2, label, training_prop=None):
        self.data1 = data1
        self.data2 = data2

        self.label = label
        self.training_prop = training_prop

    def get_train_test_keys(self):
        np.random.seed(42)

        group1_index = []
        group2_index = []
        group3_index = []
        group4_index = []
        group5_index = []
        group6_index = []

        for i, j in enumerate(self.label):
            if j == 'Group1':
                group1_index.append(i)
            elif j == 'Group2':
                group2_index.append(i)
            elif j == 'Group3':
                group3_index.append(i)
            elif j == 'Group4':
                group4_index.append(i)
            elif j == 'Group5':
                group5_index.append(i)
            elif j == 'Group6':
                group6_index.append(i)

        group1_select = np.random.choice(np.array(group1_index),
                                         size=round(len(group1_index) * self.training_prop), replace=False)
        group2_select = np.random.choice(np.array(group2_index),
                                         size=round(len(group2_index) * self.training_prop), replace=False)
        group3_select = np.random.choice(np.array(group3_index),
                                         size=round(len(group3_index) * self.training_prop), replace=False)
        group4_select = np.random.choice(np.array(group4_index),
                                         size=round(len(group4_index) * self.training_prop), replace=False)
        group5_select = np.random.choice(np.array(group5_index),
                                         size=round(len(group5_index) * self.training_prop), replace=False)
        group6_select = np.random.choice(np.array(group6_index),
                                         size=round(len(group6_index) * self.training_prop), replace=False)

        training_index = np.concatenate([group1_select, group2_select, group3_select,
                                         group4_select, group5_select, group6_select])

        testing_index = set(self.label.index.values).difference(training_index)

        return training_index, testing_index

    def to_tensor(self, data_keys):
        '''
            construct the torch tensor based on the input keys
        '''

        feature1 = np.array(self.data1[self.data1.index.isin(data_keys)])

        feature2 = np.array(self.data2[self.data2.index.isin(data_keys)])

        labels = np.array(self.label[self.label.index.isin(data_keys)])

        ### do the data preprocessing with MinMaxScaler as in MOSS paper
        ### the downloaded data seems already scaled
        scaler = MinMaxScaler()

        feature1 = scaler.fit_transform(feature1)
        feature2 = scaler.fit_transform(feature2)
        print('feature1 and feature2 are being scaled with MinMaxScaler')

        feature1_tensors = torch.tensor(feature1)
        feature2_tensors = torch.tensor(feature2)

        labels_dict = {'Group1': 0, 'Group2': 1,
                       'Group3': 2, 'Group4': 3,
                       'Group5': 4, 'Group6': 5}
        num_labels = np.array([labels_dict[i] for i in labels])

        label_tensors = torch.tensor(num_labels)

        return feature1_tensors, feature2_tensors, label_tensors


### Dataset preparation
class DataSet_construction(Dataset):
    def __init__(self, feature1_tensors, feature2_tensors, label_tensors):
        # construction of the map-style datasets
        # data loading

        self.x1 = feature1_tensors
        self.x2 = feature2_tensors

        self.y = label_tensors

        self.n_samples = feature1_tensors.size()[0]

    def __getitem__(self, index):
        # dataset[0]

        return self.x1[index], self.x2[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples


# print(type(hyper_dict['l1_s12_out_dim']))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# setup_seed(3)


class SharedAndSpecificLoss(nn.Module):
    def __init__(self, ):
        super(SharedAndSpecificLoss, self).__init__()

    ### The orthogonal loss defined here for shared and specific embeddings are
    ### essentially the dot product of each correspoinding features between
    ## shared embedding and specific embedding followed by taking the average

    ### shared: 100 X 1024
    ### specific: 100 X 1024
    ### torch.mul(shared, specific) results in 100 X 1024 (element-wise product between these two matrix)
    ### is the same as the dot product of embedding from shared and specific for each row (1024)
    ### same as the shared X specific^t then take the average of the diagnol entries the the resulting 100 X 100 matrix
    @staticmethod
    def orthogonal_loss(shared, specific):
        #         shared = shared - shared.mean()
        #         specific = specific - specific.mean()
        shared = F.normalize(shared, p=2, dim=1)
        specific = F.normalize(specific, p=2, dim=1)
        correlation_matrix = torch.mul(shared, specific)
        cost = correlation_matrix.mean()
        return cost

    @staticmethod
    def contrastive_loss(shared_1, shared_2, temperature, batch_size):
        assert (shared_1.dim() == 2)
        assert (shared_2.dim() == 2)
        #         shared_1 = shared_1 - shared_1.mean()
        #         shared_2 = shared_2 - shared_2.mean()
        shared_1 = F.normalize(shared_1, p=2, dim=1)
        shared_2 = F.normalize(shared_2, p=2, dim=1)

        # Contrastive loss version1
        criterion_instance = InstanceLoss(batch_size=batch_size, temperature=temperature)
        loss = criterion_instance(shared_1, shared_2)
        return loss

    @staticmethod
    def reconstruction_loss(rec, ori):
        assert (rec.dim() == 2)
        assert (ori.dim() == 2)
        #         rec = rec - rec.mean()
        #         ori = ori - ori.mean()
        rec = F.normalize(rec, p=2, dim=1)
        ori = F.normalize(ori, p=2, dim=1)

        ## this is the forbenius norm of the normalized difference between
        ## the reconstructed input and the original input
        loss = torch.linalg.matrix_norm(rec - ori)
        return loss

    def forward(self, shared1_output, shared2_output, specific1_output, specific2_output,
                shared1_rec, shared2_rec, specific1_rec, specific2_rec,
                ori1, ori2, shared1_mlp, shared2_mlp, temperature, batch_size):
        # orthogonal restrict
        orthogonal_loss1 = self.orthogonal_loss(shared1_output, specific1_output)
        orthogonal_loss2 = self.orthogonal_loss(shared2_output, specific2_output)
        orthogonal_loss_all = orthogonal_loss1 + orthogonal_loss2

        # Contrastive Loss
        contrastive_loss1 = self.contrastive_loss(shared1_mlp, shared2_mlp, temperature, batch_size)
        contrastive_loss_all = contrastive_loss1
        # print(contrastive_loss_all)

        # reconstruction Loss
        reconstruction_loss1 = self.reconstruction_loss(shared1_rec, ori1) + self.reconstruction_loss(specific1_rec,
                                                                                                      ori1)
        reconstruction_loss2 = self.reconstruction_loss(shared2_rec, ori2) + self.reconstruction_loss(specific2_rec,
                                                                                                      ori2)
        reconstruction_loss_all = reconstruction_loss1 + reconstruction_loss2
        # print(reconstruction_loss_all)

        ###################
        # the reconstruction loss is weigthed by 0.7

        ###################

        return orthogonal_loss_all, contrastive_loss_all, reconstruction_loss_all


# print(CNC_AE())


def MOCSS_AE_retraining(model, model_folder_path, RNA_df_path, miRNA_df_path, hyper_dict):

    RNA_df = pd.read_csv(RNA_df_path, sep='\t').T
    miRNA_df = pd.read_csv(miRNA_df_path, sep='\t').T

    print(f'The index are aligned: {np.alltrue(RNA_df.index == miRNA_df.index)}')

    if np.alltrue(RNA_df.index == miRNA_df.index):
        labels = pd.DataFrame([i.split('.')[0] for i in miRNA_df.index])[0]
        RNA_df_ = RNA_df.reset_index().drop(columns=['index'])
        miRNA_df_ = miRNA_df.reset_index().drop(columns=['index'])

        dataset_prep = DataSet_Prep(data1=RNA_df_, data2=miRNA_df_, label=labels, training_prop=0.8)
        train_key, test_key = dataset_prep.get_train_test_keys()
        feature1_tensors, feature2_tensors, label_tensors = dataset_prep.to_tensor(train_key)
        train_dataset = DataSet_construction(feature1_tensors, feature2_tensors, label_tensors)

        print(len(train_dataset))

        train_recon_loss_ = []

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        batch_size = hyper_dict['batch_size']

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        #     print(len(train_loader))

        model = model().to(device=device)
        print(model)

        optimizer_name = 'Adam'
        lr = hyper_dict['lr']
        l2_lambda = hyper_dict['l2_lambda']
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=l2_lambda)

        epochs = hyper_dict['epoch']  ### reduce the epochs from 150 to 100 to reduce the potential overfitting
        ortho_multiplier = hyper_dict['ortho_multiplier']
        for epoch in range(epochs):
            #         print(f"I'am in the epoch {epoch}")
            model.train()
            # record the training loss
            total_recon_loss = 0.0
            total_train = 0.0

            ## deal with different number of features in different dataset with star* notation
            for view1_train_data, view2_train_data, train_labels in train_loader:
                ### this line is just for nn.CrossEntropy loss otherwise can be safely removed
                view1_train_data = view1_train_data.type(torch.float32).to(device)
                view2_train_data = view2_train_data.type(torch.float32).to(device)
                train_labels = train_labels.type(torch.LongTensor).to(device)

                view1_specific_em, view1_shared_em, view2_specific_em, view2_shared_em, \
                view1_specific_rec, view1_shared_rec, view2_specific_rec, view2_shared_rec, \
                view1_shared_mlp, view2_shared_mlp, train_labels = model(view1_train_data, view2_train_data,
                                                                         train_labels)

                train_size = view1_specific_em.size()[0]

                loss_function = SharedAndSpecificLoss()
                ortho_loss, contrastive_loss, recon_loss = loss_function(shared1_output=view1_shared_em, \
                                                                         shared2_output=view2_shared_em, \
                                                                         specific1_output=view1_specific_em, \
                                                                         specific2_output=view2_specific_em, \
                                                                         shared1_rec=view1_shared_rec, \
                                                                         specific1_rec=view1_specific_rec, \
                                                                         shared2_rec=view2_shared_rec, \
                                                                         specific2_rec=view2_specific_rec, \
                                                                         ori1=view1_train_data, \
                                                                         ori2=view2_train_data, \
                                                                         shared1_mlp=view1_shared_mlp, \
                                                                         shared2_mlp=view2_shared_mlp, \
                                                                         batch_size=view1_shared_em.shape[0], \
                                                                         temperature=0.4)

                loss = ortho_loss + contrastive_loss + (ortho_multiplier * recon_loss)

                # backward pass
                optimizer.zero_grad()  # empty the gradient from last round

                # calculate the gradient
                loss.backward()
                # update the parameters
                optimizer.step()

                total_train += train_size

                total_recon_loss += recon_loss.item()

            train_recon_loss_.append(total_recon_loss / total_train)

            if (epoch + 1) % 10 == 0:
                print(f'finished retraining on epoch: {epoch}')
        # save the model at the end of 150 epochs
        model_path = f"{model_folder_path}/retrained_model_{epoch}.pt"

        torch.save(model, model_path)

        return train_recon_loss_

    else:
        print('Data are not aligned')


if __name__ == '__main__':
    ### loop each simulation dataset
    ### create corresponding folder to save the result
    for group in [2, 3, 4, 5]:
        for prop_diff in [0.2, 0.4, 0.6, 0.8, 1]:

            ### read in the optimal hyperparameter set into dict and fill up the hyperparameter in model and training
            print('Read the corresponding optimal hyperparameter set')
            hyper_dict = {}
            with open(f'/home/wangc90/Data_integration/simulation_model_outputs/model_selection_outputs/MOCSS_AE/optuna/optuna_{group}_groups_{prop_diff}/optuna.txt') as f:
                for _ in range(7):
                    next(f)
                for line in f:
                    value_char = line.split(':')[1].strip()
                    if '.' in value_char:
                        value = float(value_char)
                    else:
                        value = int(value_char)
                    hyper_dict[line.split(':')[0]] = value

            print('Load the optimal hyperparameter set for the model')


            class SharedAndSpecificEmbedding(nn.Module):

                def __init__(self):
                    ### embeding layers have the same dimensions for both shared and specific AE for all views

                    view_size = [20531, 1046]

                    n_units_1 = [0, 0, 0, 0]
                    n_units_1[0] = hyper_dict['unit_1_first']
                    n_units_1[1] = hyper_dict['unit_1_second']
                    n_units_1[2] = hyper_dict['unit_1_third']
                    n_units_1[3] = hyper_dict['unit_1_fourth']

                    n_units_2 = n_units_1.copy()

                    mlp_size = [0, 0, 0, 0]
                    mlp_size[0] = hyper_dict['mlp_first']
                    mlp_size[1] = hyper_dict['mlp_second']

                    super(SharedAndSpecificEmbedding, self).__init__()
                    # View1
                    self.shared1_l1 = nn.Linear(view_size[0], n_units_1[0])

                    self.shared1_l2 = nn.Linear(n_units_1[0], n_units_1[1])
                    self.shared1_l3 = nn.Linear(n_units_1[1], n_units_1[2])
                    self.shared1_l4 = nn.Linear(n_units_1[2], n_units_1[3])

                    self.shared1_l3_ = nn.Linear(n_units_1[3], n_units_1[2])
                    self.shared1_l2_ = nn.Linear(n_units_1[2], n_units_1[1])
                    self.shared1_l1_ = nn.Linear(n_units_1[1], n_units_1[0])

                    self.shared1_rec = nn.Linear(n_units_1[0], view_size[0])

                    self.specific1_l1 = nn.Linear(view_size[0], n_units_1[0])
                    self.specific1_l2 = nn.Linear(n_units_1[0], n_units_1[1])
                    self.specific1_l3 = nn.Linear(n_units_1[1], n_units_1[2])
                    self.specific1_l4 = nn.Linear(n_units_1[2], n_units_1[3])

                    self.specific1_l3_ = nn.Linear(n_units_1[3], n_units_1[2])
                    self.specific1_l2_ = nn.Linear(n_units_1[2], n_units_1[1])
                    self.specific1_l1_ = nn.Linear(n_units_1[1], n_units_1[0])
                    self.specific1_rec = nn.Linear(n_units_1[0], view_size[0])

                    self.view1_mlp1 = nn.Linear(n_units_1[3], mlp_size[0])

                    self.view1_mlp2 = nn.Linear(mlp_size[0], mlp_size[1])

                    # View2
                    self.shared2_l1 = nn.Linear(view_size[1], n_units_2[0])
                    self.shared2_l2 = nn.Linear(n_units_2[0], n_units_2[1])
                    self.shared2_l3 = nn.Linear(n_units_2[1], n_units_2[2])
                    self.shared2_l4 = nn.Linear(n_units_2[2], n_units_2[3])

                    self.shared2_l3_ = nn.Linear(n_units_2[3], n_units_2[2])
                    self.shared2_l2_ = nn.Linear(n_units_2[2], n_units_2[1])
                    self.shared2_l1_ = nn.Linear(n_units_2[1], n_units_2[0])
                    self.shared2_rec = nn.Linear(n_units_2[0], view_size[1])

                    self.specific2_l1 = nn.Linear(view_size[1], n_units_2[0])
                    self.specific2_l2 = nn.Linear(n_units_2[0], n_units_2[1])
                    self.specific2_l3 = nn.Linear(n_units_2[1], n_units_2[2])
                    self.specific2_l4 = nn.Linear(n_units_2[2], n_units_2[3])

                    self.specific2_l3_ = nn.Linear(n_units_2[3], n_units_2[2])
                    self.specific2_l2_ = nn.Linear(n_units_2[2], n_units_2[1])
                    self.specific2_l1_ = nn.Linear(n_units_2[1], n_units_2[0])
                    self.specific2_rec = nn.Linear(n_units_2[0], view_size[1])

                    self.view2_mlp1 = nn.Linear(n_units_2[3], mlp_size[0])
                    self.view2_mlp2 = nn.Linear(mlp_size[0], mlp_size[1])

                def forward(self, view1_input, view2_input, label):
                    # View1
                    view1_specific = F.tanh(self.specific1_l1(view1_input))
                    view1_specific = F.tanh(self.specific1_l2(view1_specific))
                    view1_specific = F.tanh(self.specific1_l3(view1_specific))
                    view1_specific_em = F.tanh(self.specific1_l4(view1_specific))

                    view1_specific = F.tanh(self.specific1_l3_(view1_specific_em))
                    view1_specific = F.tanh(self.specific1_l2_(view1_specific))
                    view1_specific = F.tanh(self.specific1_l1_(view1_specific))
                    view1_specific_rec = torch.sigmoid(self.specific1_rec(view1_specific))

                    view1_shared = F.tanh(self.shared1_l1(view1_input))
                    view1_shared = F.tanh(self.shared1_l2(view1_shared))
                    view1_shared = F.tanh(self.shared1_l3(view1_shared))
                    view1_shared_em = F.tanh(self.shared1_l4(view1_shared))

                    view1_shared = F.tanh(self.shared1_l3_(view1_shared_em))
                    view1_shared = F.tanh(self.shared1_l2_(view1_shared))
                    view1_shared = F.tanh(self.shared1_l1_(view1_shared))
                    view1_shared_rec = torch.sigmoid(self.shared1_rec(view1_shared))

                    view1_shared_mlp = F.tanh(self.view1_mlp1(view1_shared_em))
                    view1_shared_mlp = F.tanh(self.view1_mlp2(view1_shared_mlp))

                    # View2
                    view2_specific = F.tanh(self.specific2_l1(view2_input))
                    view2_specific = F.tanh(self.specific2_l2(view2_specific))
                    view2_specific = F.tanh(self.specific2_l3(view2_specific))
                    view2_specific_em = F.tanh(self.specific2_l4(view2_specific))
                    view2_specific = F.tanh(self.specific2_l3_(view2_specific_em))
                    view2_specific = F.tanh(self.specific2_l2_(view2_specific))
                    view2_specific = F.tanh(self.specific2_l1_(view2_specific))
                    view2_specific_rec = torch.sigmoid(self.specific2_rec(view2_specific))

                    view2_shared = F.tanh(self.shared2_l1(view2_input))
                    view2_shared = F.tanh(self.shared2_l2(view2_shared))
                    view2_shared = F.tanh(self.shared2_l3(view2_shared))
                    view2_shared_em = F.tanh(self.shared2_l4(view2_shared))
                    view2_shared = F.tanh(self.shared2_l3_(view2_shared_em))
                    view2_shared = F.tanh(self.shared2_l2_(view2_shared))
                    view2_shared = F.tanh(self.shared2_l1_(view2_shared))
                    view2_shared_rec = torch.sigmoid(self.shared2_rec(view2_shared))

                    view2_shared_mlp = F.tanh(self.view2_mlp1(view2_shared_em))
                    view2_shared_mlp = F.tanh(self.view2_mlp2(view2_shared_mlp))

                    return view1_specific_em, view1_shared_em, \
                           view2_specific_em, view2_shared_em, \
                           view1_specific_rec, view1_shared_rec, \
                           view2_specific_rec, view2_shared_rec, \
                           view1_shared_mlp, view2_shared_mlp, label

            print(SharedAndSpecificEmbedding())

            simulation_RNA_path = f'/home/wangc90/Data_integration/simulation_data/RNA_seq_{group}_groups_{prop_diff}_diff.csv'
            simulation_miRNA_path = f'/home/wangc90/Data_integration/simulation_data/miRNA_seq_{group}_groups_{prop_diff}_diff.csv'

            print('load the data')
            print(simulation_RNA_path, simulation_miRNA_path)

            model_folder_path = f'/home/wangc90/Data_integration/simulation_model_outputs/model_retraining_outputs/MOCSS_AE/models/retrained_{group}_groups_{prop_diff}_diff'
            if not os.path.exists(model_folder_path):
                print('Create folder to store retrained models')
                os.makedirs(model_folder_path)

            MOCSS_AE_retraining(model=SharedAndSpecificEmbedding,
                           model_folder_path=model_folder_path,
                           RNA_df_path=simulation_RNA_path,
                           miRNA_df_path=simulation_miRNA_path,
                           hyper_dict=hyper_dict)
