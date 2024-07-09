import torch
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
random.seed(2023)
torch.manual_seed(2023)

### This python script is used to run the simulation dataset

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


class SharedAndSpecificEmbedding(nn.Module):

    def __init__(self, trial):
        ### embeding layers have the same dimensions for both shared and specific AE for all views

        view_size = [20531, 1046]

        possible_nodes = [32, 64, 128, 256, 512, 1024]
        possible_dropout = [0, 0.1, 0.2, 0.4, 0.6]

        n_units_1 = [0, 0, 0, 0]
        n_units_1[0] = trial.suggest_categorical("unit_1_first", possible_nodes)
        n_units_1[1] = trial.suggest_categorical("unit_1_second", possible_nodes)
        n_units_1[2] = trial.suggest_categorical("unit_1_third", possible_nodes)
        n_units_1[3] = trial.suggest_categorical("unit_1_fourth", possible_nodes)

        n_units_2 = n_units_1.copy()

        mlp_size = [0, 0, 0, 0]
        #         mlp_size[0] = 64
        mlp_size[0] = trial.suggest_categorical("mlp_first", possible_nodes)
        #         mlp_size[1] = 16
        mlp_size[1] = trial.suggest_categorical("mlp_second", possible_nodes)

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

    def forward(self, view1_input, view2_input):
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
               view1_shared_mlp, view2_shared_mlp


def Objective(device, trial, fold, model, optimizer,
              epochs, train_loader, val_loader, ortho_multiplier):
    for epoch in range(epochs):
        #         print(f"I'am in the epoch {epoch}")
        model.train()
        # record the training loss
        total_recon_loss = 0.0
        total_ortho_loss = 0.0
        total_train = 0.0

        for iteration_index, train_batch in enumerate(train_loader):
            view1_train_data, view2_train_data, train_labels = train_batch

            view1_train_data = view1_train_data.type(torch.float32).to(device)
            view2_train_data = view2_train_data.type(torch.float32).to(device)
            train_labels = train_labels.type(torch.LongTensor).to(device)

            view1_specific_em, view1_shared_em, view2_specific_em, view2_shared_em, \
            view1_specific_rec, view1_shared_rec, view2_specific_rec, view2_shared_rec, \
            view1_shared_mlp, view2_shared_mlp = model(view1_train_data, view2_train_data)

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

            #             print(view1_shared_em.shape, view1_specific_em.shape)

            # backward pass
            optimizer.zero_grad()  # empty the gradient from last round

            # calculate the gradient
            loss.backward()
            # update the parameters
            optimizer.step()

            total_train += train_size
            total_recon_loss += recon_loss.item()
            total_ortho_loss += ortho_loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'fold {fold + 1} epoch {epoch + 1}')
            print(f'average train recon loss is: {total_recon_loss / total_train}')
            print(f'average train ortho loss is: {total_ortho_loss / total_train}')

    #### model evaluation on the validation set at the last epoch and return acc
    model.eval()

    with torch.no_grad():
        # validation metrics
        total_val_loss = 0.0
        total_val = 0.0

        total_recon_loss = 0.0
        total_val = 0.0

        for iteration_index, val_batch in enumerate(val_loader):
            #                     print('val loop', iteration_index)
            view1_val_data, view2_val_data, val_labels = val_batch

            view1_val_data = view1_val_data.type(torch.float32).to(device)
            view2_val_data = view2_val_data.type(torch.float32).to(device)
            val_labels = val_labels.type(torch.LongTensor).to(device)

            view1_specific_em_val, view1_shared_em_val, view2_specific_em_val, view2_shared_em_val, \
            view1_specific_rec_val, view1_shared_rec_val, view2_specific_rec_val, view2_shared_rec_val, \
            view1_shared_mlp_val, view2_shared_mlp_val = model(view1_val_data, view2_val_data)

            val_size = view1_specific_em_val.size()[0]

            loss_function = SharedAndSpecificLoss()
            ortho_loss, contrastive_loss, recon_loss = loss_function(shared1_output=view1_shared_em_val, \
                                                                     shared2_output=view2_shared_em_val, \
                                                                     specific1_output=view1_specific_em_val, \
                                                                     specific2_output=view2_specific_em_val, \
                                                                     shared1_rec=view1_shared_rec_val, \
                                                                     specific1_rec=view1_specific_rec_val, \
                                                                     shared2_rec=view2_shared_rec_val, \
                                                                     specific2_rec=view2_specific_rec_val, \
                                                                     ori1=view1_val_data, \
                                                                     ori2=view2_val_data, \
                                                                     shared1_mlp=view1_shared_mlp_val, \
                                                                     shared2_mlp=view2_shared_mlp_val, \
                                                                     batch_size=view1_shared_em_val.shape[0], \
                                                                     temperature=0.4)

            total_val += val_size

            total_recon_loss += recon_loss.item()
        avg_recon_loss = total_recon_loss / total_val

    return avg_recon_loss


class Objective_CV:

    def __init__(self, cv, model, dataset, val_loss_folder):
        self.cv = cv  ## number of CV
        self.model = model  ## pass the corresponding model
        self.dataset = dataset  ## the corresponding dataset object
        self.val_loss_folder = val_loss_folder  ## folder to store the cross_validation accuracy

    def __call__(self, trial):
        ### just use the sequence feature for now
        device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        #         lr = 0.001
        l2_lambda = trial.suggest_float("l2_lambda", 1e-8, 1e-5, log=True)
        #         l2_lambda = 0
        ### fix and use the maximal allowed batch size
        #         batch_size = trial.suggest_categorical("batch_size", [24])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])

        ### optimize epoch number
        epochs = trial.suggest_categorical("epoch", [30, 60, 90, 120, 150])

        ortho_multiplier = trial.suggest_categorical("ortho_multiplier", [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])

        ## choose the optimizer
        #         optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop","Adagrad"])
        optimizer_name = 'Adam'

        kfold = KFold(n_splits=self.cv, shuffle=True)

        setup_seed(21)

        val_fold_loss = []

        for fold, (train_index, val_index) in enumerate(kfold.split(np.arange(len(self.dataset)))):
            ### get the train and val loader
            train_subset = torch.utils.data.Subset(self.dataset, train_index)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

            val_subset = torch.utils.data.Subset(self.dataset, val_index)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)

            ## model should be initilized here for each fold to have a new model with same hyperparameters

            ### for the model the process the concatenated upper and lower the is_rcm is always False
            Model = self.model(trial).to(device=device)
            #             print(Model)
            optimizer = getattr(optim, optimizer_name)(Model.parameters(), lr=lr, weight_decay=l2_lambda)

            val_loss = Objective(device, trial, fold=fold, model=Model, optimizer=optimizer,
                                 epochs=epochs, train_loader=train_loader, val_loader=val_loader,
                                 ortho_multiplier=ortho_multiplier)

            val_fold_loss.append(val_loss)

        avg_val_loss = np.mean(val_fold_loss)

        val_loss_path = f"{self.val_loss_folder}/val_loss.csv"

        val_loss_str = '\t'.join([str(i) for i in val_fold_loss])
        with open(val_loss_path, 'a') as f:
            f.write('trial' + str(trial.number) + '\t' + val_loss_str + '\n')

        return avg_val_loss


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


def MOCSS_AE_model_selection(num_trial, val_loss_folder_path, optuna_folder_path, RNA_df_path, miRNA_df_path):
    ### where to save the 3-fold CV validation acc
    val_loss_folder = val_loss_folder_path
    ### wehre to save the detailed optuna results
    optuna_folder = optuna_folder_path

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

        study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
                                    direction='minimize')

        study.optimize(Objective_CV(cv=5, model=SharedAndSpecificEmbedding,
                                    dataset=train_dataset,
                                    val_loss_folder=val_loss_folder),
                       n_trials=num_trial, gc_after_trial=True)

        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        with open(optuna_folder + '/optuna.txt', 'a') as f:
            f.write("Study statistics: \n")
            f.write(f"Number of finished trials: {len(study.trials)}\n")
            f.write(f"Number of pruned trials: {len(pruned_trials)}\n")
            f.write(f"Number of complete trials: {len(complete_trials)}\n")

            f.write("Best trial:\n")
            trial = study.best_trial
            f.write(f"Value: {trial.value}\n")
            f.write("Params:\n")
            for key, value in trial.params.items():
                f.write(f"{key}:{value}\n")

        df = study.trials_dataframe().drop(['state', 'datetime_start', 'datetime_complete', 'duration', 'number'], axis=1)
        df.to_csv(optuna_folder + '/optuna.csv', sep='\t', index=None)
    else:
        print('Data are not aligned')


if __name__ == '__main__':
    ### loop each simulation dataset
    ### create corresponding folder to save the result
    for group in [2, 3, 4, 5]:
        for prop_diff in [0.2, 0.4, 0.6, 0.8, 1]:
            simulation_RNA_path = f'/home/wangc90/Data_integration/simulation_data/RNA_seq_{group}_groups_{prop_diff}_diff.csv'
            simulation_miRNA_path = f'/home/wangc90/Data_integration/simulation_data/miRNA_seq_{group}_groups_{prop_diff}_diff.csv'

            print('load the data')
            print(simulation_RNA_path, simulation_miRNA_path)

            val_loss_folder_path = f'/home/wangc90/Data_integration/simulation_model_outputs/model_selection_outputs/MOCSS_AE/val_loss/val_{group}_groups_{prop_diff}_loss'
            optuna_folder_path = f'/home/wangc90/Data_integration/simulation_model_outputs/model_selection_outputs/MOCSS_AE/optuna/optuna_{group}_groups_{prop_diff}'

            if not os.path.exists(val_loss_folder_path):
                print('Create folder for val loss')
                os.makedirs(val_loss_folder_path)

            if not os.path.exists(optuna_folder_path):
                print('Create folder for Optuna')
                os.makedirs(optuna_folder_path)

            MOCSS_AE_model_selection(num_trial=50, val_loss_folder_path=val_loss_folder_path,
                                   optuna_folder_path=optuna_folder_path,
                                   RNA_df_path=simulation_RNA_path,
                                   miRNA_df_path=simulation_miRNA_path)


