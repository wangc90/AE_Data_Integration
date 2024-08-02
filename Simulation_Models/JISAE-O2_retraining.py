import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import sys
import warnings
warnings.filterwarnings("ignore")
import random
import matplotlib.pyplot as plt
sys.path.insert(1, '/home/wangc90/Data_integration/MOCSS/mocss/code/')
import evaluation
from Data_prep import DataSet_Prep, DataSet_construction
from tsn_visulization import tsn_data, tsn_plot
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


# print(CNC_AE())

def CustomLoss(s1, s2, s1_out, s2_out,
               z1, z2, z12, labels):
    """
        Ortho_AE2
    """

    ### normalize the feature vector with length 1
    s1_out = F.normalize(s1_out, p=2, dim=1)
    s2_out = F.normalize(s2_out, p=2, dim=1)

    s1 = F.normalize(s1, p=2, dim=1)
    s2 = F.normalize(s2, p=2, dim=1)

    recon_loss = torch.linalg.matrix_norm(s1_out - s1) + torch.linalg.matrix_norm(s2_out - s2)

    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    z12 = F.normalize(z12, p=2, dim=1)

    z1_t = torch.t(z1)
    z2_t = torch.t(z2)

    z12_z1_t = torch.matmul(z12, z1_t)
    z12_z2_t = torch.matmul(z12, z2_t)
    #     z2_z1_t = torch.matmul(z2, z1_t)

    #### shared and specific
    z12_z1_t_diag = torch.diagonal(z12_z1_t, 0)  ## get the main diagnol
    z12_z1_t_diag_square_sum = torch.sum(
        torch.square(z12_z1_t_diag))  ## get squared term to make it close to 0 in magnitude

    z12_z2_t_diag = torch.diagonal(z12_z2_t, 0)  ## get the main diagnol
    z12_z2_t_diag_square_sum = torch.sum(
        torch.square(z12_z2_t_diag))  ## get squared term to make it close to 0 in magnitude

    ### between two specific
    #     z2_z1_t_diag = torch.diagonal(z2_z1_t, 0) ## get the main diagnol
    #     z2_z1_t_diag_square_sum = torch.sum(torch.square(z2_z1_t_diag)) ## get squared term to make it close to 0 in magnitude

    ortho_loss = z12_z1_t_diag_square_sum + z12_z2_t_diag_square_sum  # + z2_z1_t_diag_square_sum

    return recon_loss, ortho_loss


def Ortho_AE2_retraining(model, model_folder_path, RNA_df_path, miRNA_df_path, hyper_dict):

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

                z1, z2, z12, s1_out, s2_out, labels = \
                    model(view1_train_data, view2_train_data, train_labels)

                train_size = z12.size()[0]

                recon_loss, ortho_loss = CustomLoss(s1=view1_train_data, \
                                                    s2=view2_train_data, \
                                                    s1_out=s1_out, \
                                                    s2_out=s2_out, \
                                                    z1=z1, \
                                                    z2=z2, \
                                                    z12=z12, \
                                                    labels=train_labels)

                loss = recon_loss + (ortho_loss * ortho_multiplier)

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
            with open(f'/home/wangc90/Data_integration/simulation_model_outputs/model_selection_outputs/Ortho_AE2/optuna/optuna_{group}_groups_{prop_diff}/optuna.txt') as f:
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


            class SSO2_Encoder(nn.Module):
                """
                    takes in 3 omic data type measurements for the same set of subjects
                """

                def __init__(self):
                    ### input dimension for omic 1, omic 2 and omic 3
                    self.s1_input_dim = 20531
                    self.s2_input_dim = 1046

                    self.l1_s1_out_dim = hyper_dict['l1_s1_out_dim']
                    self.l1_s2_out_dim = hyper_dict['l1_s2_out_dim']
                    self.l1_s12_out_dim = hyper_dict['l1_s12_out_dim']

                    self.l2_s1_out_dim = hyper_dict['l2_s1_out_dim']
                    self.l2_s2_out_dim = hyper_dict['l2_s2_out_dim']
                    self.l2_s12_out_dim = hyper_dict['l2_s12_out_dim']

                    self.l3_s1_out_dim = hyper_dict['l3_s1_out_dim']
                    self.l3_s2_out_dim = hyper_dict['l3_s2_out_dim']
                    self.l3_s12_out_dim = hyper_dict['l3_s12_out_dim']

                    ### embedding for z1, z2 and z12 have to have the same dimension for the
                    ### orthogonal losss based on MOCSS to work
                    self.embed_s1_out_dim = hyper_dict['embed_s1_out_dim']
                    self.embed_s2_out_dim = self.embed_s1_out_dim
                    self.embed_s12_out_dim = self.embed_s1_out_dim

                    super(SSO2_Encoder, self).__init__()

                    ### encoder structure:

                    ######################################################################################
                    self.l1_s1 = nn.Linear(self.s1_input_dim, self.l1_s1_out_dim)
                    self.l1_s1_bn = nn.BatchNorm1d(self.l1_s1_out_dim)
                    l1_s1_drop_rate = hyper_dict['l1_s1_drop_rate']
                    self.drop_l1_s1 = nn.Dropout(p=l1_s1_drop_rate)

                    self.l2_s1 = nn.Linear(self.l1_s1_out_dim, self.l2_s1_out_dim)
                    self.l2_s1_bn = nn.BatchNorm1d(self.l2_s1_out_dim)
                    l2_s1_drop_rate = hyper_dict['l2_s1_drop_rate']
                    self.drop_l2_s1 = nn.Dropout(p=l2_s1_drop_rate)

                    self.l3_s1 = nn.Linear(self.l2_s1_out_dim, self.l3_s1_out_dim)
                    self.l3_s1_bn = nn.BatchNorm1d(self.l3_s1_out_dim)
                    l3_s1_drop_rate = hyper_dict['l3_s1_drop_rate']
                    self.drop_l3_s1 = nn.Dropout(p=l3_s1_drop_rate)

                    self.embed_s1 = nn.Linear(self.l3_s1_out_dim, self.embed_s1_out_dim)
                    self.embed_s1_bn = nn.BatchNorm1d(self.embed_s1_out_dim)
                    embed_s1_drop_rate = hyper_dict['embed_s1_drop_rate']
                    self.drop_embed_s1 = nn.Dropout(p=embed_s1_drop_rate)

                    ###########################################################################################
                    self.l1_s2 = nn.Linear(self.s2_input_dim, self.l1_s2_out_dim)
                    self.l1_s2_bn = nn.BatchNorm1d(self.l1_s2_out_dim)
                    l1_s2_drop_rate = hyper_dict['l1_s2_drop_rate']
                    self.drop_l1_s2 = nn.Dropout(p=l1_s2_drop_rate)

                    self.l2_s2 = nn.Linear(self.l1_s2_out_dim, self.l2_s2_out_dim)
                    self.l2_s2_bn = nn.BatchNorm1d(self.l2_s2_out_dim)
                    l2_s2_drop_rate = hyper_dict['l2_s2_drop_rate']
                    self.drop_l2_s2 = nn.Dropout(p=l2_s2_drop_rate)

                    self.l3_s2 = nn.Linear(self.l2_s2_out_dim, self.l3_s2_out_dim)
                    self.l3_s2_bn = nn.BatchNorm1d(self.l3_s2_out_dim)
                    l3_s2_drop_rate = hyper_dict['l3_s2_drop_rate']
                    self.drop_l3_s2 = nn.Dropout(p=l3_s2_drop_rate)

                    self.embed_s2 = nn.Linear(self.l3_s2_out_dim, self.embed_s2_out_dim)
                    self.embed_s2_bn = nn.BatchNorm1d(self.embed_s2_out_dim)
                    embed_s2_drop_rate = hyper_dict['embed_s2_drop_rate']
                    self.drop_embed_s2 = nn.Dropout(p=embed_s2_drop_rate)

                    ##########################################################################################

                    self.l1_s12 = nn.Linear(self.s1_input_dim + self.s2_input_dim,
                                            self.l1_s12_out_dim)
                    self.l1_s12_bn = nn.BatchNorm1d(self.l1_s12_out_dim)
                    l1_s12_drop_rate = hyper_dict['l1_s12_drop_rate']
                    self.drop_l1_s12 = nn.Dropout(p=l1_s12_drop_rate)

                    self.l2_s12 = nn.Linear(self.l1_s12_out_dim, self.l2_s12_out_dim)
                    self.l2_s12_bn = nn.BatchNorm1d(self.l2_s12_out_dim)
                    l2_s12_drop_rate = hyper_dict['l2_s12_drop_rate']
                    self.drop_l2_s12 = nn.Dropout(p=l2_s12_drop_rate)

                    self.l3_s12 = nn.Linear(self.l2_s12_out_dim, self.l3_s12_out_dim)
                    self.l3_s12_bn = nn.BatchNorm1d(self.l3_s12_out_dim)
                    l3_s12_drop_rate = hyper_dict['l3_s12_drop_rate']
                    self.drop_l3_s12 = nn.Dropout(p=l3_s12_drop_rate)

                    self.embed_s12 = nn.Linear(self.l3_s12_out_dim, self.embed_s12_out_dim)
                    self.embed_s12_bn = nn.BatchNorm1d(self.embed_s12_out_dim)
                    embed_s12_drop_rate = hyper_dict['embed_s12_drop_rate']
                    self.drop_embed_s12 = nn.Dropout(p=embed_s12_drop_rate)

                def forward(self, s1, s2, labels=None):
                    #############################################################
                    s1_ = self.drop_l1_s1(self.l1_s1_bn(F.relu(self.l1_s1(s1))))
                    s1_ = self.drop_l2_s1(self.l2_s1_bn(F.relu(self.l2_s1(s1_))))
                    s1_ = self.drop_l3_s1(self.l3_s1_bn(F.relu(self.l3_s1(s1_))))
                    z1 = self.drop_embed_s1(self.embed_s1_bn(F.relu(self.embed_s1(s1_))))

                    s2_ = self.drop_l1_s2(self.l1_s2_bn(F.relu(self.l1_s2(s2))))
                    s2_ = self.drop_l2_s2(self.l2_s2_bn(F.relu(self.l2_s2(s2_))))
                    s2_ = self.drop_l3_s2(self.l3_s2_bn(F.relu(self.l3_s2(s2_))))
                    z2 = self.drop_embed_s2(self.embed_s2_bn(F.relu(self.embed_s2(s2_))))

                    ### concatenate s1, s2 together for the joint embedding
                    s12 = torch.cat((s1, s2), dim=1)
                    s12_ = self.drop_l1_s12(self.l1_s12_bn(F.relu(self.l1_s12(s12))))
                    s12_ = self.drop_l2_s12(self.l2_s12_bn(F.relu(self.l2_s12(s12_))))
                    s12_ = self.drop_l3_s12(self.l3_s12_bn(F.relu(self.l3_s12(s12_))))
                    z12 = self.drop_embed_s12(self.embed_s12_bn(F.relu(self.embed_s12(s12_))))

                    return z1, z2, z12, labels


            class SSO2_Decoder(nn.Module):

                ### decoder: construct s1 and s2  based on the concatenated z12 z1 and z2
                ### and calculate the reconstruction loss separately for s1 and s2

                def __init__(self):
                    self.s1_input_dim = SSO2_Encoder().s1_input_dim
                    self.s2_input_dim = SSO2_Encoder().s2_input_dim

                    self.s1_embed_dim = SSO2_Encoder().embed_s1_out_dim
                    self.s2_embed_dim = SSO2_Encoder().embed_s2_out_dim
                    self.s12_embed_dim = SSO2_Encoder().embed_s12_out_dim

                    self._embed_s1_out_dim = hyper_dict['_embed_s1_out_dim']
                    self._l3_s1_out_dim = hyper_dict['_l3_s1_out_dim']
                    self._l2_s1_out_dim = hyper_dict['_l2_s1_out_dim']
                    self._l1_s1_out_dim = self.s1_input_dim

                    self._embed_s2_out_dim = hyper_dict['_embed_s2_out_dim']
                    self._l3_s2_out_dim = hyper_dict['_l3_s2_out_dim']
                    self._l2_s2_out_dim = hyper_dict['_l2_s2_out_dim']
                    self._l1_s2_out_dim = self.s2_input_dim

                    super(SSO2_Decoder, self).__init__()

                    self._embed_s1 = nn.Linear(self.s1_embed_dim + self.s2_embed_dim + self.s12_embed_dim, \
                                               self._embed_s1_out_dim)

                    self._embed_s1_bn = nn.BatchNorm1d(self._embed_s1_out_dim)
                    _embed_s1_drop_rate = hyper_dict['_embed_s1_drop_rate']
                    self._drop_embed_s1 = nn.Dropout(p=_embed_s1_drop_rate)

                    self._l3_s1 = nn.Linear(self._embed_s1_out_dim, self._l3_s1_out_dim)
                    self._l3_s1_bn = nn.BatchNorm1d(self._l3_s1_out_dim)
                    _l3_s1_drop_rate = hyper_dict['_l3_s1_drop_rate']
                    self._drop_l3_s1 = nn.Dropout(p=_l3_s1_drop_rate)

                    self._l2_s1 = nn.Linear(self._l3_s1_out_dim, self._l2_s1_out_dim)
                    self._l2_s1_bn = nn.BatchNorm1d(self._l2_s1_out_dim)
                    _l2_s1_drop_rate = hyper_dict['_l2_s1_drop_rate']
                    self._drop_l2_s1 = nn.Dropout(p=_l2_s1_drop_rate)

                    self._l1_s1 = nn.Linear(self._l2_s1_out_dim, self._l1_s1_out_dim)
                    self._l1_s1_bn = nn.BatchNorm1d(self._l1_s1_out_dim)
                    _l1_s1_drop_rate = hyper_dict['_l1_s1_drop_rate']
                    self._drop_l1_s1 = nn.Dropout(p=_l1_s1_drop_rate)

                    #############################################################################

                    self._embed_s2 = nn.Linear(self.s1_embed_dim + self.s2_embed_dim + self.s12_embed_dim, \
                                               self._embed_s2_out_dim)

                    self._embed_s2_bn = nn.BatchNorm1d(self._embed_s2_out_dim)
                    _embed_s2_drop_rate = hyper_dict['_embed_s2_drop_rate']
                    self._drop_embed_s2 = nn.Dropout(p=_embed_s2_drop_rate)

                    self._l3_s2 = nn.Linear(self._embed_s2_out_dim, self._l3_s2_out_dim)
                    self._l3_s2_bn = nn.BatchNorm1d(self._l3_s2_out_dim)
                    _l3_s2_drop_rate = hyper_dict['_l3_s2_drop_rate']
                    self._drop_l3_s2 = nn.Dropout(p=_l3_s2_drop_rate)

                    self._l2_s2 = nn.Linear(self._l3_s2_out_dim, self._l2_s2_out_dim)
                    self._l2_s2_bn = nn.BatchNorm1d(self._l2_s2_out_dim)
                    _l2_s2_drop_rate = hyper_dict['_l2_s2_drop_rate']
                    self._drop_l2_s2 = nn.Dropout(p=_l2_s2_drop_rate)

                    self._l1_s2 = nn.Linear(self._l2_s2_out_dim, self._l1_s2_out_dim)
                    self._l1_s2_bn = nn.BatchNorm1d(self._l1_s2_out_dim)
                    _l1_s2_drop_rate = hyper_dict['_l1_s2_drop_rate']
                    self._drop_l1_s2 = nn.Dropout(p=_l1_s2_drop_rate)

                def forward(self, z1, z2, z12):
                    z_all = torch.cat((z1, z2, z12), dim=1)

                    s1_ = self._drop_embed_s1(self._embed_s1_bn(F.relu(self._embed_s1(z_all))))
                    s1_ = self._drop_l3_s1(self._l3_s1_bn(F.relu(self._l3_s1(s1_))))
                    s1_ = self._drop_l2_s1(self._l2_s1_bn(F.relu(self._l2_s1(s1_))))
                    s1_ = self._drop_l1_s1(self._l1_s1_bn(F.relu(self._l1_s1(s1_))))

                    s1_out = torch.sigmoid(s1_)

                    s2_ = self._drop_embed_s2(self._embed_s2_bn(F.relu(self._embed_s2(z_all))))

                    s2_ = self._drop_l3_s2(self._l3_s2_bn(F.relu(self._l3_s2(s2_))))
                    s2_ = self._drop_l2_s2(self._l2_s2_bn(F.relu(self._l2_s2(s2_))))
                    s2_ = self._drop_l1_s2(self._l1_s2_bn(F.relu(self._l1_s2(s2_))))

                    s2_out = torch.sigmoid(s2_)

                    return s1_out, s2_out


            class SSO2_AE(nn.Module):
                def __init__(self):
                    super(SSO2_AE, self).__init__()

                    self.encoder = SSO2_Encoder()
                    self.decoder = SSO2_Decoder()

                def forward(self, s1, s2, labels):
                    ### encoder ouput for embeddings
                    z1, z2, z12, labels = self.encoder(s1, s2, labels)

                    ### decoder output for reconstructed input
                    s1_out, s2_out = self.decoder(z1, z2, z12)

                    return z1, z2, z12, s1_out, s2_out, labels

            # print(CNC_AE())

            simulation_RNA_path = f'/home/wangc90/Data_integration/simulation_data/RNA_seq_{group}_groups_{prop_diff}_diff.csv'
            simulation_miRNA_path = f'/home/wangc90/Data_integration/simulation_data/miRNA_seq_{group}_groups_{prop_diff}_diff.csv'

            print('load the data')
            print(simulation_RNA_path, simulation_miRNA_path)

            model_folder_path = f'/home/wangc90/Data_integration/simulation_model_outputs/model_retraining_outputs/Ortho_AE2/models/retrained_{group}_groups_{prop_diff}_diff'
            if not os.path.exists(model_folder_path):
                print('Create folder to store retrained models')
                os.makedirs(model_folder_path)

            Ortho_AE2_retraining(model=SSO2_AE,
                           model_folder_path=model_folder_path,
                           RNA_df_path=simulation_RNA_path,
                           miRNA_df_path=simulation_miRNA_path,
                           hyper_dict=hyper_dict)
