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

class MM_AE_Encoder(nn.Module):

    def __init__(self, trial):
        possible_nodes = [32, 64, 128, 256, 512, 1024]
        possible_dropout = [0, 0.1, 0.2, 0.4, 0.6]

        ### input dimension for omic 1, omic 2 and omic 3
        self.s1_input_dim = 20531
        self.s2_input_dim = 1046

        ### layer 1 output dimension for omic 1, omic 2, omic 3 and omic 123
        self.l1_s1_out_dim = trial.suggest_categorical("l1_s1_out_dim", possible_nodes)
        self.l1_s2_out_dim = trial.suggest_categorical("l1_s2_out_dim", possible_nodes)

        self.l2_s12_out_dim = trial.suggest_categorical("l2_s12_out_dim", possible_nodes)
        self.l2_s21_out_dim = trial.suggest_categorical("l2_s21_out_dim", possible_nodes)

        self.l3_ss_out_dim = trial.suggest_categorical("l3_ss_out_dim", possible_nodes)
        ### output dimension for common embedding dimension
        self.common_embed_dim = trial.suggest_categorical("common_embed_dim", possible_nodes)
        super(MM_AE_Encoder, self).__init__()

        ### encoder structure:
        ### first layer
        self.l1_s1 = nn.Linear(self.s1_input_dim, self.l1_s1_out_dim)
        self.l1_s1_bn = nn.BatchNorm1d(self.l1_s1_out_dim)
        l1_s1_drop_rate = trial.suggest_categorical("l1_s1_drop_rate", possible_dropout)
        self.drop_l1_s1 = nn.Dropout(p=l1_s1_drop_rate)

        self.l1_s2 = nn.Linear(self.s2_input_dim, self.l1_s2_out_dim)
        self.l1_s2_bn = nn.BatchNorm1d(self.l1_s2_out_dim)
        l1_s2_drop_rate = trial.suggest_categorical("l1_s2_drop_rate", possible_dropout)
        self.drop_l1_s2 = nn.Dropout(p=l1_s2_drop_rate)

        self.l2_s12 = nn.Linear(self.l1_s1_out_dim + self.l1_s2_out_dim, self.l2_s12_out_dim)
        self.l2_s12_bn = nn.BatchNorm1d(self.l2_s12_out_dim)
        l2_s12_drop_rate = trial.suggest_categorical("l2_s12_drop_rate", possible_dropout)
        self.drop_l2_s12 = nn.Dropout(p=l2_s12_drop_rate)

        self.l2_s21 = nn.Linear(self.l1_s1_out_dim + self.l1_s2_out_dim, self.l2_s21_out_dim)
        self.l2_s21_bn = nn.BatchNorm1d(self.l2_s21_out_dim)
        l2_s21_drop_rate = trial.suggest_categorical("l2_s21_drop_rate", possible_dropout)
        self.drop_l2_s21 = nn.Dropout(p=l2_s21_drop_rate)

        self.l3_ss = nn.Linear(self.l2_s12_out_dim + self.l2_s21_out_dim,
                               self.l3_ss_out_dim)
        self.l3_ss_bn = nn.BatchNorm1d(self.l3_ss_out_dim)
        l3_ss_drop_rate = trial.suggest_categorical("l3_ss_drop_rate", possible_dropout)
        self.drop_l3_ss = nn.Dropout(p=l3_ss_drop_rate)

        self.embed_ss = nn.Linear(self.l3_ss_out_dim, self.common_embed_dim)
        self.embed_ss_bn = nn.BatchNorm1d(self.common_embed_dim)
        embed_ss_drop_rate = trial.suggest_categorical("embed_ss_drop_rate", possible_dropout)
        self.drop_embed_ss = nn.Dropout(p=embed_ss_drop_rate)

    def forward(self, s1, s2, labels=None):
        s1_ = self.drop_l1_s1(self.l1_s1_bn(F.relu(self.l1_s1(s1))))
        s2_ = self.drop_l1_s2(self.l1_s2_bn(F.relu(self.l1_s2(s2))))

        s12_ = torch.cat((s1_, s2_), dim=1)
        s12_ = self.drop_l2_s12(self.l2_s12_bn(F.relu(self.l2_s12(s12_))))

        s21_ = torch.cat((s2_, s1_), dim=1)
        s21_ = self.drop_l2_s21(self.l2_s21_bn(F.relu(self.l2_s21(s21_))))

        s12__ = torch.cat((s12_, s21_), dim=1)
        s12__ = self.drop_l3_ss(self.l3_ss_bn(F.relu(self.l3_ss(s12__))))

        z12 = self.drop_embed_ss(self.embed_ss_bn(F.relu(self.embed_ss(s12__))))

        return z12, labels


class MM_AE_Decoder(nn.Module):

    def __init__(self, trial):
        self.s1_input_dim = MM_AE_Encoder(trial).s1_input_dim
        self.s2_input_dim = MM_AE_Encoder(trial).s2_input_dim
        self.common_embed_dim = MM_AE_Encoder(trial).common_embed_dim

        possible_nodes = [32, 64, 128, 256, 512, 1024]
        possible_dropout = [0, 0.1, 0.2, 0.4, 0.6]

        self._embed_s1_out_dim = trial.suggest_categorical("_embed_s1_out_dim", possible_nodes)
        self._l3_s1_out_dim = trial.suggest_categorical("_l3_s1_out_dim", possible_nodes)
        self._l2_s1_out_dim = trial.suggest_categorical("_l2_s1_out_dim", possible_nodes)
        self._l1_s1_out_dim = self.s1_input_dim

        self._embed_s2_out_dim = trial.suggest_categorical("_embed_s2_out_dim", possible_nodes)
        self._l3_s2_out_dim = trial.suggest_categorical("_l3_s2_out_dim", possible_nodes)
        self._l2_s2_out_dim = trial.suggest_categorical("_l2_s2_out_dim", possible_nodes)
        self._l1_s2_out_dim = self.s2_input_dim

        super(MM_AE_Decoder, self).__init__()

        self._embed_s1 = nn.Linear(self.common_embed_dim, self._embed_s1_out_dim)
        self._embed_s1_bn = nn.BatchNorm1d(self._embed_s1_out_dim)
        _embed_s1_drop_rate = trial.suggest_categorical("_embed_s1_drop_rate", possible_dropout)
        self._drop_embed_s1 = nn.Dropout(p=_embed_s1_drop_rate)

        self._l3_s1 = nn.Linear(self._embed_s1_out_dim, self._l3_s1_out_dim)
        self._l3_s1_bn = nn.BatchNorm1d(self._l3_s1_out_dim)
        _l3_s1_drop_rate = trial.suggest_categorical("_l3_s1_drop_rate", possible_dropout)
        self._drop_l3_s1 = nn.Dropout(p=_l3_s1_drop_rate)

        self._l2_s1 = nn.Linear(self._l3_s1_out_dim, self._l2_s1_out_dim)
        self._l2_s1_bn = nn.BatchNorm1d(self._l2_s1_out_dim)
        _l2_s1_drop_rate = trial.suggest_categorical("_l2_s1_drop_rate", possible_dropout)
        self._drop_l2_s1 = nn.Dropout(p=_l2_s1_drop_rate)

        self._l1_s1 = nn.Linear(self._l2_s1_out_dim, self._l1_s1_out_dim)
        self._l1_s1_bn = nn.BatchNorm1d(self._l1_s1_out_dim)
        _l1_s1_drop_rate = trial.suggest_categorical("_l1_s1_drop_rate", possible_dropout)
        self._drop_l1_s1 = nn.Dropout(p=_l1_s1_drop_rate)

        #############################################################################

        self._embed_s2 = nn.Linear(self.common_embed_dim, self._embed_s2_out_dim)
        self._embed_s2_bn = nn.BatchNorm1d(self._embed_s2_out_dim)
        _embed_s2_drop_rate = trial.suggest_categorical("_embed_s2_drop_rate", possible_dropout)
        self._drop_embed_s2 = nn.Dropout(p=_embed_s2_drop_rate)

        self._l3_s2 = nn.Linear(self._embed_s2_out_dim, self._l3_s2_out_dim)
        self._l3_s2_bn = nn.BatchNorm1d(self._l3_s2_out_dim)
        _l3_s2_drop_rate = trial.suggest_categorical("_l3_s2_drop_rate", possible_dropout)
        self._drop_l3_s2 = nn.Dropout(p=_l3_s2_drop_rate)

        self._l2_s2 = nn.Linear(self._l3_s2_out_dim, self._l2_s2_out_dim)
        self._l2_s2_bn = nn.BatchNorm1d(self._l2_s2_out_dim)
        _l2_s2_drop_rate = trial.suggest_categorical("_l2_s2_drop_rate", possible_dropout)
        self._drop_l2_s2 = nn.Dropout(p=_l2_s2_drop_rate)

        self._l1_s2 = nn.Linear(self._l2_s2_out_dim, self._l1_s2_out_dim)
        self._l1_s2_bn = nn.BatchNorm1d(self._l1_s2_out_dim)
        _l1_s2_drop_rate = trial.suggest_categorical("_l1_s2_drop_rate", possible_dropout)
        self._drop_l1_s2 = nn.Dropout(p=_l1_s2_drop_rate)

    def forward(self, z12):
        s1_ = self._drop_embed_s1(self._embed_s1_bn(F.relu(self._embed_s1(z12))))
        s1_ = self._drop_l3_s1(self._l3_s1_bn(F.relu(self._l3_s1(s1_))))
        s1_ = self._drop_l2_s1(self._l2_s1_bn(F.relu(self._l2_s1(s1_))))
        s1_ = self._drop_l1_s1(self._l1_s1_bn(F.relu(self._l1_s1(s1_))))

        s1_out = torch.sigmoid(s1_)

        s2_ = self._drop_embed_s2(self._embed_s2_bn(F.relu(self._embed_s2(z12))))
        s2_ = self._drop_l3_s2(self._l3_s2_bn(F.relu(self._l3_s2(s2_))))
        s2_ = self._drop_l2_s2(self._l2_s2_bn(F.relu(self._l2_s2(s2_))))
        s2_ = self._drop_l1_s2(self._l1_s2_bn(F.relu(self._l1_s2(s2_))))

        s2_out = torch.sigmoid(s2_)

        return s1_out, s2_out


class MM_AE(nn.Module):
    def __init__(self, trial):
        super(MM_AE, self).__init__()

        self.encoder = MM_AE_Encoder(trial)

        self.decoder = MM_AE_Decoder(trial)

    def forward(self, s1, s2, labels):
        ### encoder ouput for embeddings
        z12, labels = self.encoder(s1, s2, labels)
        ### decoder output for reconstructed input
        s1_out, s2_out = self.decoder(z12)
        return z12, s1_out, s2_out, labels

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def CustomLoss(s1, s2, s1_out, s2_out,
               z12, labels, z1=None, z2=None):
    #         self.alpha = alpha

    ### subtract the mean

    #     s1_out = s1_out - s1_out.mean()
    #     s2_out = s2_out - s2_out.mean()

    ### normalize the feature vector with length 1
    s1_out = F.normalize(s1_out, p=2, dim=1)
    s2_out = F.normalize(s2_out, p=2, dim=1)

    #     s1 = s1 - s1.mean()
    #     s2 = s2 - s2.mean()
    s1 = F.normalize(s1, p=2, dim=1)
    s2 = F.normalize(s2, p=2, dim=1)

    recon_loss = torch.linalg.matrix_norm(s1_out - s1) + torch.linalg.matrix_norm(s2_out - s2)

    return recon_loss


### create a training function that is seamless for all different models
def Objective(device, trial, fold, model, optimizer,
              epochs, train_loader, val_loader):
    #### model training on the training set

    for epoch in range(epochs):

        model.train()
        # record the training loss
        total_recon_loss = 0.0
        total_train = 0.0

        for iteration_index, train_batch in enumerate(train_loader):
            view1_train_data, view2_train_data, train_labels = train_batch

            view1_train_data = view1_train_data.type(torch.float32).to(device)
            view2_train_data = view2_train_data.type(torch.float32).to(device)
            train_labels = train_labels.type(torch.LongTensor).to(device)

            z12, s1_out, s2_out, labels = \
                model(view1_train_data, view2_train_data, train_labels)

            #             print(z123.shape, s1_out.shape, s2_out.shape, s3_out.shape)
            train_size = z12.size()[0]

            recon_loss = CustomLoss(s1=view1_train_data, \
                                    s2=view2_train_data, \
                                    s1_out=s1_out, \
                                    s2_out=s2_out, \
                                    z12=z12, \
                                    labels=train_labels)

            #             print(clustering_loss)

            loss = recon_loss
            # backward pass
            optimizer.zero_grad()  # empty the gradient from last round

            # calculate the gradient
            loss.backward()

            #             print(model.weight.grad)
            # update the parameters
            optimizer.step()

            total_train += train_size
            #             total_clustering_loss += (clustering_loss.item()*train_size)
            total_recon_loss += recon_loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'fold {fold + 1} epoch {epoch + 1}')
            print(f'average train recon loss is: {total_recon_loss / total_train}')

    #### model evaluation on the validation set at the last epoch and return val_loss

    model.eval()
    with torch.no_grad():

        total_recon_loss = 0.0
        total_val = 0.0

        ### collect the embeddings of all the validation set for subsequent clustering measurement
        for iteration_index, val_batch in enumerate(val_loader):
            view1_val_data, view2_val_data, val_labels = val_batch

            view1_val_data = view1_val_data.type(torch.float32).to(device)
            view2_val_data = view2_val_data.type(torch.float32).to(device)
            val_labels = val_labels.type(torch.LongTensor).to(device)

            z12_val, s1_out_val, s2_out_val, labels_val = \
                model(view1_val_data, view2_val_data, val_labels)

            val_size = z12.size()[0]

            recon_loss = CustomLoss(s1=view1_val_data, \
                                    s2=view2_val_data, \
                                    s1_out=s1_out_val, \
                                    s2_out=s2_out_val, \
                                    z12=z12_val, \
                                    labels=labels_val)

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
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
        #         batch_size = 100

        ### optimize epoch number
        epochs = trial.suggest_categorical("epoch", [30, 60, 90, 120, 150])

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
                                 epochs=epochs, train_loader=train_loader, val_loader=val_loader)

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


def MM_AE_model_selection(num_trial, val_loss_folder_path, optuna_folder_path, RNA_df_path, miRNA_df_path):
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

        study.optimize(Objective_CV(cv=5, model=MM_AE,
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

            val_loss_folder_path = f'/home/wangc90/Data_integration/simulation_model_outputs/model_selection_outputs/MM_AE/val_loss/val_{group}_groups_{prop_diff}_loss'
            optuna_folder_path = f'/home/wangc90/Data_integration/simulation_model_outputs/model_selection_outputs/MM_AE/optuna/optuna_{group}_groups_{prop_diff}'

            if not os.path.exists(val_loss_folder_path):
                print('Create folder for val loss')
                os.makedirs(val_loss_folder_path)

            if not os.path.exists(optuna_folder_path):
                print('Create folder for Optuna')
                os.makedirs(optuna_folder_path)

            MM_AE_model_selection(num_trial=50, val_loss_folder_path=val_loss_folder_path,
                                   optuna_folder_path=optuna_folder_path,
                                   RNA_df_path=simulation_RNA_path,
                                   miRNA_df_path=simulation_miRNA_path)


