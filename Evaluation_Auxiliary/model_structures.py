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


### variationalAE with concatenated inputs (CNC-VAE)
### In order to train the variational autoencoder, we only need to add the auxillary loss in our training algorithm

class CNC_Encoder(nn.Module):
    """
        concatenate the s1, s2 directly for a joint embedding
    """

    def __init__(self):
        ### input dimension for omic 1, omic 2 and omic 3
        self.s1_input_dim = 20531
        self.s2_input_dim = 1046

        ### layer 1 output dimension for omic 12
        self.l1_s12_out_dim = 256

        ### layer 2 output dimension for omic 12
        self.l2_s12_out_dim = 128

        ### layer 3 output dimension for omic 12
        self.l3_s12_out_dim = 1024

        ### output dimension for common embedding dimension
        self.common_embed_dim = 32

        super(CNC_Encoder, self).__init__()

        ### encoder structure:
        ### first layer
        self.l1_s12 = nn.Linear(self.s1_input_dim + self.s2_input_dim,
                                self.l1_s12_out_dim)
        ### corresponding bn layer
        self.l1_s12_bn = nn.BatchNorm1d(self.l1_s12_out_dim)
        ### corresponding dropout layer
        l1_s12_drop_rate = 0.1
        self.drop_l1_s12 = nn.Dropout(p=l1_s12_drop_rate)

        ### second layer
        self.l2_s12 = nn.Linear(self.l1_s12_out_dim, self.l2_s12_out_dim)
        ### corresponding bn layer
        self.l2_s12_bn = nn.BatchNorm1d(self.l2_s12_out_dim)
        ### corresponding dropout layer
        l2_s12_drop_rate = 0.4
        self.drop_l2_s12 = nn.Dropout(p=l2_s12_drop_rate)

        ### third layer
        self.l3_s12 = nn.Linear(self.l2_s12_out_dim, self.l3_s12_out_dim)
        ### corresponding bn layer
        self.l3_s12_bn = nn.BatchNorm1d(self.l3_s12_out_dim)
        ### corresponding dropout layer
        l3_s12_drop_rate = 0.4
        self.drop_l3_s12 = nn.Dropout(p=l3_s12_drop_rate)

        ### embedding layers
        self.embed_s12 = nn.Linear(self.l3_s12_out_dim, self.common_embed_dim)
        self.embed_s12_bn = nn.BatchNorm1d(self.common_embed_dim)
        ### corresponding dropout layer
        embed_s12_drop_rate = 0.1
        self.drop_embed_s12 = nn.Dropout(p=embed_s12_drop_rate)

    def forward(self, s1, s2, labels=None):
        s12 = torch.cat((s1, s2), dim=1)
        s12_ = self.drop_l1_s12(self.l1_s12_bn(F.relu(self.l1_s12(s12))))
        s12_ = self.drop_l2_s12(self.l2_s12_bn(F.relu(self.l2_s12(s12_))))
        s12_ = self.drop_l3_s12(self.l3_s12_bn(F.relu(self.l3_s12(s12_))))
        z12 = self.drop_embed_s12(self.embed_s12_bn(F.relu(self.embed_s12(s12_))))

        return z12, labels


class CNC_Decoder(nn.Module):

    def __init__(self):
        self.s1_input_dim = CNC_Encoder().s1_input_dim
        self.s2_input_dim = CNC_Encoder().s2_input_dim

        self.common_embed_dim = CNC_Encoder().common_embed_dim

        self._embed_s1_out_dim = 256
        self._l3_s1_out_dim = 256
        self._l2_s1_out_dim = 512
        self._l1_s1_out_dim = self.s1_input_dim

        self._embed_s2_out_dim = 32
        self._l3_s2_out_dim = 256
        self._l2_s2_out_dim = 512
        self._l1_s2_out_dim = self.s2_input_dim

        super(CNC_Decoder, self).__init__()

        #############################################################################
        self._embed_s1 = nn.Linear(self.common_embed_dim, self._embed_s1_out_dim)
        self._embed_s1_bn = nn.BatchNorm1d(self._embed_s1_out_dim)
        _embed_s1_drop_rate = 0.1
        self._drop_embed_s1 = nn.Dropout(p=_embed_s1_drop_rate)

        self._l3_s1 = nn.Linear(self._embed_s1_out_dim, self._l3_s1_out_dim)
        self._l3_s1_bn = nn.BatchNorm1d(self._l3_s1_out_dim)
        _l3_s1_drop_rate = 0.1
        self._drop_l3_s1 = nn.Dropout(p=_l3_s1_drop_rate)

        self._l2_s1 = nn.Linear(self._l3_s1_out_dim, self._l2_s1_out_dim)
        self._l2_s1_bn = nn.BatchNorm1d(self._l2_s1_out_dim)
        _l2_s1_drop_rate = 0.2
        self._drop_l2_s1 = nn.Dropout(p=_l2_s1_drop_rate)

        self._l1_s1 = nn.Linear(self._l2_s1_out_dim, self._l1_s1_out_dim)
        self._l1_s1_bn = nn.BatchNorm1d(self._l1_s1_out_dim)
        _l1_s1_drop_rate = 0
        self._drop_l1_s1 = nn.Dropout(p=_l1_s1_drop_rate)

        #############################################################################

        self._embed_s2 = nn.Linear(self.common_embed_dim, self._embed_s2_out_dim)
        self._embed_s2_bn = nn.BatchNorm1d(self._embed_s2_out_dim)
        _embed_s2_drop_rate = 0.6
        self._drop_embed_s2 = nn.Dropout(p=_embed_s2_drop_rate)

        self._l3_s2 = nn.Linear(self._embed_s2_out_dim, self._l3_s2_out_dim)
        self._l3_s2_bn = nn.BatchNorm1d(self._l3_s2_out_dim)
        _l3_s2_drop_rate = 0.6
        self._drop_l3_s2 = nn.Dropout(p=_l3_s2_drop_rate)

        self._l2_s2 = nn.Linear(self._l3_s2_out_dim, self._l2_s2_out_dim)
        self._l2_s2_bn = nn.BatchNorm1d(self._l2_s2_out_dim)
        _l2_s2_drop_rate = 0.2
        self._drop_l2_s2 = nn.Dropout(p=_l2_s2_drop_rate)

        self._l1_s2 = nn.Linear(self._l2_s2_out_dim, self._l1_s2_out_dim)
        self._l1_s2_bn = nn.BatchNorm1d(self._l1_s2_out_dim)
        _l1_s2_drop_rate = 0.1
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


class CNC_AE(nn.Module):
    def __init__(self):
        super(CNC_AE, self).__init__()

        self.encoder = CNC_Encoder()

        self.decoder = CNC_Decoder()

    def forward(self, s1, s2, labels):
        ### encoder ouput for embeddings
        z12, labels = self.encoder(s1, s2, labels)
        ### decoder output for reconstructed input
        s1_out, s2_out = self.decoder(z12)

        return z12, s1_out, s2_out, labels


### variationalAE with concatenated inputs (X-VAE)
### In order to train the variational autoencoder, we only
### need to add the auxillary loss in our training algorithm

class X_AE_Encoder(nn.Module):

    def __init__(self):
        ### input dimension for omic 1, omic 2 and omic 3
        self.s1_input_dim = 20531
        self.s2_input_dim = 1046

        ### layer 1 output dimension for omic 1, omic 2, omic 3 and omic 123
        self.l1_s1_out_dim = 128
        self.l1_s2_out_dim = 1024

        self.l2_s12_out_dim = 128
        self.l3_s12_out_dim = 1024

        self.common_embed_dim = 256

        super(X_AE_Encoder, self).__init__()

        ### encoder structure:

        ### first layer
        self.l1_s1 = nn.Linear(self.s1_input_dim, self.l1_s1_out_dim)
        self.l1_s1_bn = nn.BatchNorm1d(self.l1_s1_out_dim)
        l1_s1_drop_rate = 0
        self.drop_l1_s1 = nn.Dropout(p=l1_s1_drop_rate)

        self.l1_s2 = nn.Linear(self.s2_input_dim, self.l1_s2_out_dim)
        self.l1_s2_bn = nn.BatchNorm1d(self.l1_s2_out_dim)
        l1_s2_drop_rate = 0
        self.drop_l1_s2 = nn.Dropout(p=l1_s2_drop_rate)

        self.l2_s12 = nn.Linear(self.l1_s1_out_dim + self.l1_s2_out_dim,
                                self.l2_s12_out_dim)
        self.l2_s12_bn = nn.BatchNorm1d(self.l2_s12_out_dim)
        l2_s12_drop_rate = 0.2
        self.drop_l2_s12 = nn.Dropout(p=l2_s12_drop_rate)

        self.l3_s12 = nn.Linear(self.l2_s12_out_dim, self.l3_s12_out_dim)
        self.l3_s12_bn = nn.BatchNorm1d(self.l3_s12_out_dim)
        l3_s12_drop_rate = 0.4
        self.drop_l3_s12 = nn.Dropout(p=l3_s12_drop_rate)

        ## embedding layer
        self.embed_s12 = nn.Linear(self.l3_s12_out_dim, self.common_embed_dim)
        self.embed_s12_bn = nn.BatchNorm1d(self.common_embed_dim)
        embed_s12_drop_rate = 0
        self.drop_embed_s12 = nn.Dropout(p=embed_s12_drop_rate)

    def forward(self, s1, s2, labels=None):
        s1_ = self.drop_l1_s1(self.l1_s1_bn(F.relu(self.l1_s1(s1))))
        s2_ = self.drop_l1_s2(self.l1_s2_bn(F.relu(self.l1_s2(s2))))

        s12_ = torch.cat((s1_, s2_), dim=1)

        s12_ = self.drop_l2_s12(self.l2_s12_bn(F.relu(self.l2_s12(s12_))))
        s12_ = self.drop_l3_s12(self.l3_s12_bn(F.relu(self.l3_s12(s12_))))

        z12 = self.drop_embed_s12(self.embed_s12_bn(F.relu(self.embed_s12(s12_))))

        return z12, labels


class X_AE_Decoder(nn.Module):

    def __init__(self):
        self.s1_input_dim = X_AE_Encoder().s1_input_dim
        self.s2_input_dim = X_AE_Encoder().s2_input_dim
        self.common_embed_dim = X_AE_Encoder().common_embed_dim

        self._embed_s1_out_dim = 64
        self._l3_s1_out_dim = 128
        self._l2_s1_out_dim = 1024
        self._l1_s1_out_dim = self.s1_input_dim

        self._embed_s2_out_dim = 256
        self._l3_s2_out_dim = 1024
        self._l2_s2_out_dim = 32
        self._l1_s2_out_dim = self.s2_input_dim

        super(X_AE_Decoder, self).__init__()

        self._embed_s1 = nn.Linear(self.common_embed_dim, self._embed_s1_out_dim)
        self._embed_s1_bn = nn.BatchNorm1d(self._embed_s1_out_dim)
        _embed_s1_drop_rate = 0.2
        self._drop_embed_s1 = nn.Dropout(p=_embed_s1_drop_rate)

        self._l3_s1 = nn.Linear(self._embed_s1_out_dim, self._l3_s1_out_dim)
        self._l3_s1_bn = nn.BatchNorm1d(self._l3_s1_out_dim)
        _l3_s1_drop_rate = 0.6
        self._drop_l3_s1 = nn.Dropout(p=_l3_s1_drop_rate)

        self._l2_s1 = nn.Linear(self._l3_s1_out_dim, self._l2_s1_out_dim)
        self._l2_s1_bn = nn.BatchNorm1d(self._l2_s1_out_dim)
        _l2_s1_drop_rate = 0
        self._drop_l2_s1 = nn.Dropout(p=_l2_s1_drop_rate)

        self._l1_s1 = nn.Linear(self._l2_s1_out_dim, self._l1_s1_out_dim)
        self._l1_s1_bn = nn.BatchNorm1d(self._l1_s1_out_dim)
        _l1_s1_drop_rate = 0
        self._drop_l1_s1 = nn.Dropout(p=_l1_s1_drop_rate)

        #############################################################################

        self._embed_s2 = nn.Linear(self.common_embed_dim, self._embed_s2_out_dim)
        self._embed_s2_bn = nn.BatchNorm1d(self._embed_s2_out_dim)
        _embed_s2_drop_rate = 0.1
        self._drop_embed_s2 = nn.Dropout(p=_embed_s2_drop_rate)

        self._l3_s2 = nn.Linear(self._embed_s2_out_dim, self._l3_s2_out_dim)
        self._l3_s2_bn = nn.BatchNorm1d(self._l3_s2_out_dim)
        _l3_s2_drop_rate = 0.6
        self._drop_l3_s2 = nn.Dropout(p=_l3_s2_drop_rate)

        self._l2_s2 = nn.Linear(self._l3_s2_out_dim, self._l2_s2_out_dim)
        self._l2_s2_bn = nn.BatchNorm1d(self._l2_s2_out_dim)
        _l2_s2_drop_rate = 0.4
        self._drop_l2_s2 = nn.Dropout(p=_l2_s2_drop_rate)

        self._l1_s2 = nn.Linear(self._l2_s2_out_dim, self._l1_s2_out_dim)
        self._l1_s2_bn = nn.BatchNorm1d(self._l1_s2_out_dim)
        _l1_s2_drop_rate = 0.1
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


class X_AE(nn.Module):
    def __init__(self):
        super(X_AE, self).__init__()

        self.encoder = X_AE_Encoder()

        self.decoder = X_AE_Decoder()

    def forward(self, s1, s2, labels):
        ### encoder ouput for embeddings
        z12, labels = self.encoder(s1, s2, labels)
        ### decoder output for reconstructed input
        s1_out, s2_out = self.decoder(z12)
        return z12, s1_out, s2_out, labels


### In order to train the variational autoencoder, we only
### need to add the auxillary loss in our training algorithm

class MM_AE_Encoder(nn.Module):

    def __init__(self):
        ### input dimension for omic 1, omic 2 and omic 3
        self.s1_input_dim = 20531
        self.s2_input_dim = 1046

        ### layer 1 output dimension for omic 1, omic 2, omic 3 and omic 123
        self.l1_s1_out_dim = 64
        self.l1_s2_out_dim = 32

        self.l2_s12_out_dim = 256
        self.l2_s21_out_dim = 32

        self.l3_ss_out_dim = 1024
        ### output dimension for common embedding dimension
        self.common_embed_dim = 512
        super(MM_AE_Encoder, self).__init__()

        ### encoder structure:
        ### first layer
        self.l1_s1 = nn.Linear(self.s1_input_dim, self.l1_s1_out_dim)
        self.l1_s1_bn = nn.BatchNorm1d(self.l1_s1_out_dim)
        l1_s1_drop_rate = 0.2
        self.drop_l1_s1 = nn.Dropout(p=l1_s1_drop_rate)

        self.l1_s2 = nn.Linear(self.s2_input_dim, self.l1_s2_out_dim)
        self.l1_s2_bn = nn.BatchNorm1d(self.l1_s2_out_dim)
        l1_s2_drop_rate = 0.6
        self.drop_l1_s2 = nn.Dropout(p=l1_s2_drop_rate)

        self.l2_s12 = nn.Linear(self.l1_s1_out_dim + self.l1_s2_out_dim, self.l2_s12_out_dim)
        self.l2_s12_bn = nn.BatchNorm1d(self.l2_s12_out_dim)
        l2_s12_drop_rate = 0.6
        self.drop_l2_s12 = nn.Dropout(p=l2_s12_drop_rate)

        self.l2_s21 = nn.Linear(self.l1_s1_out_dim + self.l1_s2_out_dim, self.l2_s21_out_dim)
        self.l2_s21_bn = nn.BatchNorm1d(self.l2_s21_out_dim)
        l2_s21_drop_rate = 0.1
        self.drop_l2_s21 = nn.Dropout(p=l2_s21_drop_rate)

        self.l3_ss = nn.Linear(self.l2_s12_out_dim + self.l2_s21_out_dim,
                               self.l3_ss_out_dim)
        self.l3_ss_bn = nn.BatchNorm1d(self.l3_ss_out_dim)
        l3_ss_drop_rate = 0.6
        self.drop_l3_ss = nn.Dropout(p=l3_ss_drop_rate)

        self.embed_ss = nn.Linear(self.l3_ss_out_dim, self.common_embed_dim)
        self.embed_ss_bn = nn.BatchNorm1d(self.common_embed_dim)
        embed_ss_drop_rate = 0.2
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

    def __init__(self):
        self.s1_input_dim = MM_AE_Encoder().s1_input_dim
        self.s2_input_dim = MM_AE_Encoder().s2_input_dim
        self.common_embed_dim = MM_AE_Encoder().common_embed_dim

        self._embed_s1_out_dim = 64
        self._l3_s1_out_dim = 512
        self._l2_s1_out_dim = 1024
        self._l1_s1_out_dim = self.s1_input_dim

        self._embed_s2_out_dim = 32
        self._l3_s2_out_dim = 256
        self._l2_s2_out_dim = 1024
        self._l1_s2_out_dim = self.s2_input_dim

        super(MM_AE_Decoder, self).__init__()

        self._embed_s1 = nn.Linear(self.common_embed_dim, self._embed_s1_out_dim)
        self._embed_s1_bn = nn.BatchNorm1d(self._embed_s1_out_dim)
        _embed_s1_drop_rate = 0
        self._drop_embed_s1 = nn.Dropout(p=_embed_s1_drop_rate)

        self._l3_s1 = nn.Linear(self._embed_s1_out_dim, self._l3_s1_out_dim)
        self._l3_s1_bn = nn.BatchNorm1d(self._l3_s1_out_dim)
        _l3_s1_drop_rate = 0.2
        self._drop_l3_s1 = nn.Dropout(p=_l3_s1_drop_rate)

        self._l2_s1 = nn.Linear(self._l3_s1_out_dim, self._l2_s1_out_dim)
        self._l2_s1_bn = nn.BatchNorm1d(self._l2_s1_out_dim)
        _l2_s1_drop_rate = 0.2
        self._drop_l2_s1 = nn.Dropout(p=_l2_s1_drop_rate)

        self._l1_s1 = nn.Linear(self._l2_s1_out_dim, self._l1_s1_out_dim)
        self._l1_s1_bn = nn.BatchNorm1d(self._l1_s1_out_dim)
        _l1_s1_drop_rate = 0.1
        self._drop_l1_s1 = nn.Dropout(p=_l1_s1_drop_rate)

        #############################################################################

        self._embed_s2 = nn.Linear(self.common_embed_dim, self._embed_s2_out_dim)
        self._embed_s2_bn = nn.BatchNorm1d(self._embed_s2_out_dim)
        _embed_s2_drop_rate = 0
        self._drop_embed_s2 = nn.Dropout(p=_embed_s2_drop_rate)

        self._l3_s2 = nn.Linear(self._embed_s2_out_dim, self._l3_s2_out_dim)
        self._l3_s2_bn = nn.BatchNorm1d(self._l3_s2_out_dim)
        _l3_s2_drop_rate = 0.4
        self._drop_l3_s2 = nn.Dropout(p=_l3_s2_drop_rate)

        self._l2_s2 = nn.Linear(self._l3_s2_out_dim, self._l2_s2_out_dim)
        self._l2_s2_bn = nn.BatchNorm1d(self._l2_s2_out_dim)
        _l2_s2_drop_rate = 0
        self._drop_l2_s2 = nn.Dropout(p=_l2_s2_drop_rate)

        self._l1_s2 = nn.Linear(self._l2_s2_out_dim, self._l1_s2_out_dim)
        self._l1_s2_bn = nn.BatchNorm1d(self._l1_s2_out_dim)
        _l1_s2_drop_rate = 0.1
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
    def __init__(self):
        super(MM_AE, self).__init__()

        self.encoder = MM_AE_Encoder()

        self.decoder = MM_AE_Decoder()

    def forward(self, s1, s2, labels):
        ### encoder ouput for embeddings
        z12, labels = self.encoder(s1, s2, labels)
        ### decoder output for reconstructed input
        s1_out, s2_out = self.decoder(z12)
        return z12, s1_out, s2_out, labels


class SS_Encoder(nn.Module):
    """
        takes in 3 omic data type measurements for the same set of subjects
    """

    def __init__(self):
        ### input dimension for omic 1, omic 2 and omic 3
        self.s1_input_dim = 20531
        self.s2_input_dim = 1046

        self.l1_s1_out_dim = 128
        self.l1_s2_out_dim = 512
        self.l1_s12_out_dim = 64

        self.l2_s1_out_dim = 1024
        self.l2_s2_out_dim = 32
        self.l2_s12_out_dim = 1024

        self.l3_s1_out_dim = 512
        self.l3_s2_out_dim = 1024
        self.l3_s12_out_dim = 256

        ### embedding for z1, z2 and z12 have to have the same dimension for the
        ### orthogonal losss based on MOCSS to work
        self.embed_s1_out_dim = 512
        self.embed_s2_out_dim = self.embed_s1_out_dim
        self.embed_s12_out_dim = self.embed_s1_out_dim

        super(SS_Encoder, self).__init__()

        ### encoder structure:

        ######################################################################################
        self.l1_s1 = nn.Linear(self.s1_input_dim, self.l1_s1_out_dim)
        self.l1_s1_bn = nn.BatchNorm1d(self.l1_s1_out_dim)
        l1_s1_drop_rate = 0.4
        self.drop_l1_s1 = nn.Dropout(p=l1_s1_drop_rate)

        self.l2_s1 = nn.Linear(self.l1_s1_out_dim, self.l2_s1_out_dim)
        self.l2_s1_bn = nn.BatchNorm1d(self.l2_s1_out_dim)
        l2_s1_drop_rate = 0.6
        self.drop_l2_s1 = nn.Dropout(p=l2_s1_drop_rate)

        self.l3_s1 = nn.Linear(self.l2_s1_out_dim, self.l3_s1_out_dim)
        self.l3_s1_bn = nn.BatchNorm1d(self.l3_s1_out_dim)
        l3_s1_drop_rate = 0.2
        self.drop_l3_s1 = nn.Dropout(p=l3_s1_drop_rate)

        self.embed_s1 = nn.Linear(self.l3_s1_out_dim, self.embed_s1_out_dim)
        self.embed_s1_bn = nn.BatchNorm1d(self.embed_s1_out_dim)
        embed_s1_drop_rate = 0.6
        self.drop_embed_s1 = nn.Dropout(p=embed_s1_drop_rate)

        ###########################################################################################
        self.l1_s2 = nn.Linear(self.s2_input_dim, self.l1_s2_out_dim)
        self.l1_s2_bn = nn.BatchNorm1d(self.l1_s2_out_dim)
        l1_s2_drop_rate = 0
        self.drop_l1_s2 = nn.Dropout(p=l1_s2_drop_rate)

        self.l2_s2 = nn.Linear(self.l1_s2_out_dim, self.l2_s2_out_dim)
        self.l2_s2_bn = nn.BatchNorm1d(self.l2_s2_out_dim)
        l2_s2_drop_rate = 0
        self.drop_l2_s2 = nn.Dropout(p=l2_s2_drop_rate)

        self.l3_s2 = nn.Linear(self.l2_s2_out_dim, self.l3_s2_out_dim)
        self.l3_s2_bn = nn.BatchNorm1d(self.l3_s2_out_dim)
        l3_s2_drop_rate = 0.1
        self.drop_l3_s2 = nn.Dropout(p=l3_s2_drop_rate)

        self.embed_s2 = nn.Linear(self.l3_s2_out_dim, self.embed_s2_out_dim)
        self.embed_s2_bn = nn.BatchNorm1d(self.embed_s2_out_dim)
        embed_s2_drop_rate = 0.6
        self.drop_embed_s2 = nn.Dropout(p=embed_s2_drop_rate)

        ##########################################################################################

        self.l1_s12 = nn.Linear(self.s1_input_dim + self.s2_input_dim,
                                self.l1_s12_out_dim)
        self.l1_s12_bn = nn.BatchNorm1d(self.l1_s12_out_dim)
        l1_s12_drop_rate = 0
        self.drop_l1_s12 = nn.Dropout(p=l1_s12_drop_rate)

        self.l2_s12 = nn.Linear(self.l1_s12_out_dim, self.l2_s12_out_dim)
        self.l2_s12_bn = nn.BatchNorm1d(self.l2_s12_out_dim)
        l2_s12_drop_rate = 0.1
        self.drop_l2_s12 = nn.Dropout(p=l2_s12_drop_rate)

        self.l3_s12 = nn.Linear(self.l2_s12_out_dim, self.l3_s12_out_dim)
        self.l3_s12_bn = nn.BatchNorm1d(self.l3_s12_out_dim)
        l3_s12_drop_rate = 0.2
        self.drop_l3_s12 = nn.Dropout(p=l3_s12_drop_rate)

        self.embed_s12 = nn.Linear(self.l3_s12_out_dim, self.embed_s12_out_dim)
        self.embed_s12_bn = nn.BatchNorm1d(self.embed_s12_out_dim)
        embed_s12_drop_rate = 0.2
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


class SS_Decoder(nn.Module):

    ### decoder: construct s1 and s2  based on the concatenated z12 z1 and z2
    ### and calculate the reconstruction loss separately for s1 and s2

    def __init__(self):
        self.s1_input_dim = SS_Encoder().s1_input_dim
        self.s2_input_dim = SS_Encoder().s2_input_dim

        self.s1_embed_dim = SS_Encoder().embed_s1_out_dim
        self.s2_embed_dim = SS_Encoder().embed_s2_out_dim
        self.s12_embed_dim = SS_Encoder().embed_s12_out_dim

        self._embed_s1_out_dim = 32
        self._l3_s1_out_dim = 128
        self._l2_s1_out_dim = 64
        self._l1_s1_out_dim = self.s1_input_dim

        self._embed_s2_out_dim = 512
        self._l3_s2_out_dim = 512
        self._l2_s2_out_dim = 256
        self._l1_s2_out_dim = self.s2_input_dim

        super(SS_Decoder, self).__init__()

        self._embed_s1 = nn.Linear(self.s1_embed_dim + self.s2_embed_dim + self.s12_embed_dim, \
                                   self._embed_s1_out_dim)

        self._embed_s1_bn = nn.BatchNorm1d(self._embed_s1_out_dim)
        _embed_s1_drop_rate = 0.1
        self._drop_embed_s1 = nn.Dropout(p=_embed_s1_drop_rate)

        self._l3_s1 = nn.Linear(self._embed_s1_out_dim, self._l3_s1_out_dim)
        self._l3_s1_bn = nn.BatchNorm1d(self._l3_s1_out_dim)
        _l3_s1_drop_rate = 0.1
        self._drop_l3_s1 = nn.Dropout(p=_l3_s1_drop_rate)

        self._l2_s1 = nn.Linear(self._l3_s1_out_dim, self._l2_s1_out_dim)
        self._l2_s1_bn = nn.BatchNorm1d(self._l2_s1_out_dim)
        _l2_s1_drop_rate = 0.1
        self._drop_l2_s1 = nn.Dropout(p=_l2_s1_drop_rate)

        self._l1_s1 = nn.Linear(self._l2_s1_out_dim, self._l1_s1_out_dim)
        self._l1_s1_bn = nn.BatchNorm1d(self._l1_s1_out_dim)
        _l1_s1_drop_rate = 0
        self._drop_l1_s1 = nn.Dropout(p=_l1_s1_drop_rate)

        #############################################################################

        self._embed_s2 = nn.Linear(self.s1_embed_dim + self.s2_embed_dim + self.s12_embed_dim, \
                                   self._embed_s2_out_dim)

        self._embed_s2_bn = nn.BatchNorm1d(self._embed_s2_out_dim)
        _embed_s2_drop_rate = 0.1
        self._drop_embed_s2 = nn.Dropout(p=_embed_s2_drop_rate)

        self._l3_s2 = nn.Linear(self._embed_s2_out_dim, self._l3_s2_out_dim)
        self._l3_s2_bn = nn.BatchNorm1d(self._l3_s2_out_dim)
        _l3_s2_drop_rate = 0.1
        self._drop_l3_s2 = nn.Dropout(p=_l3_s2_drop_rate)

        self._l2_s2 = nn.Linear(self._l3_s2_out_dim, self._l2_s2_out_dim)
        self._l2_s2_bn = nn.BatchNorm1d(self._l2_s2_out_dim)
        _l2_s2_drop_rate = 0.1
        self._drop_l2_s2 = nn.Dropout(p=_l2_s2_drop_rate)

        self._l1_s2 = nn.Linear(self._l2_s2_out_dim, self._l1_s2_out_dim)
        self._l1_s2_bn = nn.BatchNorm1d(self._l1_s2_out_dim)
        _l1_s2_drop_rate = 0.1
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


class SS_AE(nn.Module):
    def __init__(self):
        super(SS_AE, self).__init__()

        self.encoder = SS_Encoder()
        self.decoder = SS_Decoder()

    def forward(self, s1, s2, labels):
        ### encoder ouput for embeddings
        z1, z2, z12, labels = self.encoder(s1, s2, labels)

        ### decoder output for reconstructed input
        s1_out, s2_out = self.decoder(z1, z2, z12)

        return z1, z2, z12, s1_out, s2_out, labels


class SSO_Encoder(nn.Module):
    """
        takes in 3 omic data type measurements for the same set of subjects
    """

    def __init__(self):
        ### input dimension for omic 1, omic 2 and omic 3
        self.s1_input_dim = 20531
        self.s2_input_dim = 1046

        self.l1_s1_out_dim = 1024
        self.l1_s2_out_dim = 32
        self.l1_s12_out_dim = 256

        self.l2_s1_out_dim = 1024
        self.l2_s2_out_dim = 512
        self.l2_s12_out_dim = 32

        self.l3_s1_out_dim = 128
        self.l3_s2_out_dim = 32
        self.l3_s12_out_dim = 256

        ### embedding for z1, z2 and z12 have to have the same dimension for the
        ### orthogonal losss based on MOCSS to work
        self.embed_s1_out_dim = 1024
        self.embed_s2_out_dim = self.embed_s1_out_dim
        self.embed_s12_out_dim = self.embed_s1_out_dim

        super(SSO_Encoder, self).__init__()

        ### encoder structure:

        ######################################################################################
        self.l1_s1 = nn.Linear(self.s1_input_dim, self.l1_s1_out_dim)
        self.l1_s1_bn = nn.BatchNorm1d(self.l1_s1_out_dim)
        l1_s1_drop_rate = 0
        self.drop_l1_s1 = nn.Dropout(p=l1_s1_drop_rate)

        self.l2_s1 = nn.Linear(self.l1_s1_out_dim, self.l2_s1_out_dim)
        self.l2_s1_bn = nn.BatchNorm1d(self.l2_s1_out_dim)
        l2_s1_drop_rate = 0.2
        self.drop_l2_s1 = nn.Dropout(p=l2_s1_drop_rate)

        self.l3_s1 = nn.Linear(self.l2_s1_out_dim, self.l3_s1_out_dim)
        self.l3_s1_bn = nn.BatchNorm1d(self.l3_s1_out_dim)
        l3_s1_drop_rate = 0
        self.drop_l3_s1 = nn.Dropout(p=l3_s1_drop_rate)

        self.embed_s1 = nn.Linear(self.l3_s1_out_dim, self.embed_s1_out_dim)
        self.embed_s1_bn = nn.BatchNorm1d(self.embed_s1_out_dim)
        embed_s1_drop_rate = 0.1
        self.drop_embed_s1 = nn.Dropout(p=embed_s1_drop_rate)

        ###########################################################################################
        self.l1_s2 = nn.Linear(self.s2_input_dim, self.l1_s2_out_dim)
        self.l1_s2_bn = nn.BatchNorm1d(self.l1_s2_out_dim)
        l1_s2_drop_rate = 0.2
        self.drop_l1_s2 = nn.Dropout(p=l1_s2_drop_rate)

        self.l2_s2 = nn.Linear(self.l1_s2_out_dim, self.l2_s2_out_dim)
        self.l2_s2_bn = nn.BatchNorm1d(self.l2_s2_out_dim)
        l2_s2_drop_rate = 0.4
        self.drop_l2_s2 = nn.Dropout(p=l2_s2_drop_rate)

        self.l3_s2 = nn.Linear(self.l2_s2_out_dim, self.l3_s2_out_dim)
        self.l3_s2_bn = nn.BatchNorm1d(self.l3_s2_out_dim)
        l3_s2_drop_rate = 0.6
        self.drop_l3_s2 = nn.Dropout(p=l3_s2_drop_rate)

        self.embed_s2 = nn.Linear(self.l3_s2_out_dim, self.embed_s2_out_dim)
        self.embed_s2_bn = nn.BatchNorm1d(self.embed_s2_out_dim)
        embed_s2_drop_rate = 0.6
        self.drop_embed_s2 = nn.Dropout(p=embed_s2_drop_rate)

        ##########################################################################################

        self.l1_s12 = nn.Linear(self.s1_input_dim + self.s2_input_dim,
                                self.l1_s12_out_dim)
        self.l1_s12_bn = nn.BatchNorm1d(self.l1_s12_out_dim)
        l1_s12_drop_rate = 0.2
        self.drop_l1_s12 = nn.Dropout(p=l1_s12_drop_rate)

        self.l2_s12 = nn.Linear(self.l1_s12_out_dim, self.l2_s12_out_dim)
        self.l2_s12_bn = nn.BatchNorm1d(self.l2_s12_out_dim)
        l2_s12_drop_rate = 0.1
        self.drop_l2_s12 = nn.Dropout(p=l2_s12_drop_rate)

        self.l3_s12 = nn.Linear(self.l2_s12_out_dim, self.l3_s12_out_dim)
        self.l3_s12_bn = nn.BatchNorm1d(self.l3_s12_out_dim)
        l3_s12_drop_rate = 0
        self.drop_l3_s12 = nn.Dropout(p=l3_s12_drop_rate)

        self.embed_s12 = nn.Linear(self.l3_s12_out_dim, self.embed_s12_out_dim)
        self.embed_s12_bn = nn.BatchNorm1d(self.embed_s12_out_dim)
        embed_s12_drop_rate = 0.2
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


class SSO_Decoder(nn.Module):

    ### decoder: construct s1 and s2  based on the concatenated z12 z1 and z2
    ### and calculate the reconstruction loss separately for s1 and s2

    def __init__(self):
        self.s1_input_dim = SSO_Encoder().s1_input_dim
        self.s2_input_dim = SSO_Encoder().s2_input_dim

        self.s1_embed_dim = SSO_Encoder().embed_s1_out_dim
        self.s2_embed_dim = SSO_Encoder().embed_s2_out_dim
        self.s12_embed_dim = SSO_Encoder().embed_s12_out_dim

        self._embed_s1_out_dim = 512
        self._l3_s1_out_dim = 32
        self._l2_s1_out_dim = 256
        self._l1_s1_out_dim = self.s1_input_dim

        self._embed_s2_out_dim = 128
        self._l3_s2_out_dim = 512
        self._l2_s2_out_dim = 1024
        self._l1_s2_out_dim = self.s2_input_dim

        super(SSO_Decoder, self).__init__()

        self._embed_s1 = nn.Linear(self.s1_embed_dim + self.s2_embed_dim + self.s12_embed_dim, \
                                   self._embed_s1_out_dim)

        self._embed_s1_bn = nn.BatchNorm1d(self._embed_s1_out_dim)
        _embed_s1_drop_rate = 0.6
        self._drop_embed_s1 = nn.Dropout(p=_embed_s1_drop_rate)

        self._l3_s1 = nn.Linear(self._embed_s1_out_dim, self._l3_s1_out_dim)
        self._l3_s1_bn = nn.BatchNorm1d(self._l3_s1_out_dim)
        _l3_s1_drop_rate = 0.4
        self._drop_l3_s1 = nn.Dropout(p=_l3_s1_drop_rate)

        self._l2_s1 = nn.Linear(self._l3_s1_out_dim, self._l2_s1_out_dim)
        self._l2_s1_bn = nn.BatchNorm1d(self._l2_s1_out_dim)
        _l2_s1_drop_rate = 0
        self._drop_l2_s1 = nn.Dropout(p=_l2_s1_drop_rate)

        self._l1_s1 = nn.Linear(self._l2_s1_out_dim, self._l1_s1_out_dim)
        self._l1_s1_bn = nn.BatchNorm1d(self._l1_s1_out_dim)
        _l1_s1_drop_rate = 0
        self._drop_l1_s1 = nn.Dropout(p=_l1_s1_drop_rate)

        #############################################################################

        self._embed_s2 = nn.Linear(self.s1_embed_dim + self.s2_embed_dim + self.s12_embed_dim, \
                                   self._embed_s2_out_dim)

        self._embed_s2_bn = nn.BatchNorm1d(self._embed_s2_out_dim)
        _embed_s2_drop_rate = 0.4
        self._drop_embed_s2 = nn.Dropout(p=_embed_s2_drop_rate)

        self._l3_s2 = nn.Linear(self._embed_s2_out_dim, self._l3_s2_out_dim)
        self._l3_s2_bn = nn.BatchNorm1d(self._l3_s2_out_dim)
        _l3_s2_drop_rate = 0
        self._drop_l3_s2 = nn.Dropout(p=_l3_s2_drop_rate)

        self._l2_s2 = nn.Linear(self._l3_s2_out_dim, self._l2_s2_out_dim)
        self._l2_s2_bn = nn.BatchNorm1d(self._l2_s2_out_dim)
        _l2_s2_drop_rate = 0.1
        self._drop_l2_s2 = nn.Dropout(p=_l2_s2_drop_rate)

        self._l1_s2 = nn.Linear(self._l2_s2_out_dim, self._l1_s2_out_dim)
        self._l1_s2_bn = nn.BatchNorm1d(self._l1_s2_out_dim)
        _l1_s2_drop_rate = 0
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


class SSO_AE(nn.Module):
    def __init__(self):
        super(SSO_AE, self).__init__()

        self.encoder = SSO_Encoder()
        self.decoder = SSO_Decoder()

    def forward(self, s1, s2, labels):
        ### encoder ouput for embeddings
        z1, z2, z12, labels = self.encoder(s1, s2, labels)

        ### decoder output for reconstructed input
        s1_out, s2_out = self.decoder(z1, z2, z12)

        return z1, z2, z12, s1_out, s2_out, labels


class SSO2_Encoder(nn.Module):
    """
        takes in 3 omic data type measurements for the same set of subjects
    """

    def __init__(self):
        ### input dimension for omic 1, omic 2 and omic 3
        self.s1_input_dim = 20531
        self.s2_input_dim = 1046

        self.l1_s1_out_dim = 1024
        self.l1_s2_out_dim = 1024
        self.l1_s12_out_dim = 512

        self.l2_s1_out_dim = 256
        self.l2_s2_out_dim = 128
        self.l2_s12_out_dim = 32

        self.l3_s1_out_dim = 1024
        self.l3_s2_out_dim = 256
        self.l3_s12_out_dim = 32

        ### embedding for z1, z2 and z12 have to have the same dimension for the
        ### orthogonal losss based on MOCSS to work
        self.embed_s1_out_dim = 1024
        self.embed_s2_out_dim = self.embed_s1_out_dim
        self.embed_s12_out_dim = self.embed_s1_out_dim

        super(SSO2_Encoder, self).__init__()

        ### encoder structure:

        ######################################################################################
        self.l1_s1 = nn.Linear(self.s1_input_dim, self.l1_s1_out_dim)
        self.l1_s1_bn = nn.BatchNorm1d(self.l1_s1_out_dim)
        l1_s1_drop_rate = 0.2
        self.drop_l1_s1 = nn.Dropout(p=l1_s1_drop_rate)

        self.l2_s1 = nn.Linear(self.l1_s1_out_dim, self.l2_s1_out_dim)
        self.l2_s1_bn = nn.BatchNorm1d(self.l2_s1_out_dim)
        l2_s1_drop_rate = 0
        self.drop_l2_s1 = nn.Dropout(p=l2_s1_drop_rate)

        self.l3_s1 = nn.Linear(self.l2_s1_out_dim, self.l3_s1_out_dim)
        self.l3_s1_bn = nn.BatchNorm1d(self.l3_s1_out_dim)
        l3_s1_drop_rate = 0
        self.drop_l3_s1 = nn.Dropout(p=l3_s1_drop_rate)

        self.embed_s1 = nn.Linear(self.l3_s1_out_dim, self.embed_s1_out_dim)
        self.embed_s1_bn = nn.BatchNorm1d(self.embed_s1_out_dim)
        embed_s1_drop_rate = 0
        self.drop_embed_s1 = nn.Dropout(p=embed_s1_drop_rate)

        ###########################################################################################
        self.l1_s2 = nn.Linear(self.s2_input_dim, self.l1_s2_out_dim)
        self.l1_s2_bn = nn.BatchNorm1d(self.l1_s2_out_dim)
        l1_s2_drop_rate = 0.4
        self.drop_l1_s2 = nn.Dropout(p=l1_s2_drop_rate)

        self.l2_s2 = nn.Linear(self.l1_s2_out_dim, self.l2_s2_out_dim)
        self.l2_s2_bn = nn.BatchNorm1d(self.l2_s2_out_dim)
        l2_s2_drop_rate = 0.6
        self.drop_l2_s2 = nn.Dropout(p=l2_s2_drop_rate)

        self.l3_s2 = nn.Linear(self.l2_s2_out_dim, self.l3_s2_out_dim)
        self.l3_s2_bn = nn.BatchNorm1d(self.l3_s2_out_dim)
        l3_s2_drop_rate = 0.2
        self.drop_l3_s2 = nn.Dropout(p=l3_s2_drop_rate)

        self.embed_s2 = nn.Linear(self.l3_s2_out_dim, self.embed_s2_out_dim)
        self.embed_s2_bn = nn.BatchNorm1d(self.embed_s2_out_dim)
        embed_s2_drop_rate = 0
        self.drop_embed_s2 = nn.Dropout(p=embed_s2_drop_rate)

        ##########################################################################################

        self.l1_s12 = nn.Linear(self.s1_input_dim + self.s2_input_dim,
                                self.l1_s12_out_dim)
        self.l1_s12_bn = nn.BatchNorm1d(self.l1_s12_out_dim)
        l1_s12_drop_rate = 0
        self.drop_l1_s12 = nn.Dropout(p=l1_s12_drop_rate)

        self.l2_s12 = nn.Linear(self.l1_s12_out_dim, self.l2_s12_out_dim)
        self.l2_s12_bn = nn.BatchNorm1d(self.l2_s12_out_dim)
        l2_s12_drop_rate = 0.4
        self.drop_l2_s12 = nn.Dropout(p=l2_s12_drop_rate)

        self.l3_s12 = nn.Linear(self.l2_s12_out_dim, self.l3_s12_out_dim)
        self.l3_s12_bn = nn.BatchNorm1d(self.l3_s12_out_dim)
        l3_s12_drop_rate = 0.6
        self.drop_l3_s12 = nn.Dropout(p=l3_s12_drop_rate)

        self.embed_s12 = nn.Linear(self.l3_s12_out_dim, self.embed_s12_out_dim)
        self.embed_s12_bn = nn.BatchNorm1d(self.embed_s12_out_dim)
        embed_s12_drop_rate = 0
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

        self._embed_s1_out_dim = 512
        self._l3_s1_out_dim = 64
        self._l2_s1_out_dim = 128
        self._l1_s1_out_dim = self.s1_input_dim

        self._embed_s2_out_dim = 512
        self._l3_s2_out_dim = 256
        self._l2_s2_out_dim = 64
        self._l1_s2_out_dim = self.s2_input_dim

        super(SSO2_Decoder, self).__init__()

        self._embed_s1 = nn.Linear(self.s1_embed_dim + self.s2_embed_dim + self.s12_embed_dim, \
                                   self._embed_s1_out_dim)

        self._embed_s1_bn = nn.BatchNorm1d(self._embed_s1_out_dim)
        _embed_s1_drop_rate = 0
        self._drop_embed_s1 = nn.Dropout(p=_embed_s1_drop_rate)

        self._l3_s1 = nn.Linear(self._embed_s1_out_dim, self._l3_s1_out_dim)
        self._l3_s1_bn = nn.BatchNorm1d(self._l3_s1_out_dim)
        _l3_s1_drop_rate = 0.1
        self._drop_l3_s1 = nn.Dropout(p=_l3_s1_drop_rate)

        self._l2_s1 = nn.Linear(self._l3_s1_out_dim, self._l2_s1_out_dim)
        self._l2_s1_bn = nn.BatchNorm1d(self._l2_s1_out_dim)
        _l2_s1_drop_rate = 0.2
        self._drop_l2_s1 = nn.Dropout(p=_l2_s1_drop_rate)

        self._l1_s1 = nn.Linear(self._l2_s1_out_dim, self._l1_s1_out_dim)
        self._l1_s1_bn = nn.BatchNorm1d(self._l1_s1_out_dim)
        _l1_s1_drop_rate = 0
        self._drop_l1_s1 = nn.Dropout(p=_l1_s1_drop_rate)

        #############################################################################

        self._embed_s2 = nn.Linear(self.s1_embed_dim + self.s2_embed_dim + self.s12_embed_dim, \
                                   self._embed_s2_out_dim)

        self._embed_s2_bn = nn.BatchNorm1d(self._embed_s2_out_dim)
        _embed_s2_drop_rate = 0.6
        self._drop_embed_s2 = nn.Dropout(p=_embed_s2_drop_rate)

        self._l3_s2 = nn.Linear(self._embed_s2_out_dim, self._l3_s2_out_dim)
        self._l3_s2_bn = nn.BatchNorm1d(self._l3_s2_out_dim)
        _l3_s2_drop_rate = 0
        self._drop_l3_s2 = nn.Dropout(p=_l3_s2_drop_rate)

        self._l2_s2 = nn.Linear(self._l3_s2_out_dim, self._l2_s2_out_dim)
        self._l2_s2_bn = nn.BatchNorm1d(self._l2_s2_out_dim)
        _l2_s2_drop_rate = 0.1
        self._drop_l2_s2 = nn.Dropout(p=_l2_s2_drop_rate)

        self._l1_s2 = nn.Linear(self._l2_s2_out_dim, self._l1_s2_out_dim)
        self._l1_s2_bn = nn.BatchNorm1d(self._l1_s2_out_dim)
        _l1_s2_drop_rate = 0.1
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


class SSO3_Encoder(nn.Module):
    """
        takes in 3 omic data type measurements for the same set of subjects
    """

    def __init__(self):
        ### input dimension for omic 1, omic 2 and omic 3
        self.s1_input_dim = 20531
        self.s2_input_dim = 1046

        self.l1_s1_out_dim = 512
        self.l1_s2_out_dim = 512
        self.l1_s12_out_dim = 128

        self.l2_s1_out_dim = 512
        self.l2_s2_out_dim = 256
        self.l2_s12_out_dim = 128

        self.l3_s1_out_dim = 128
        self.l3_s2_out_dim = 512
        self.l3_s12_out_dim = 1024

        ### embedding for z1, z2 and z12 have to have the same dimension for the
        ### orthogonal losss based on MOCSS to work
        self.embed_s1_out_dim = 32
        self.embed_s2_out_dim = self.embed_s1_out_dim
        self.embed_s12_out_dim = self.embed_s1_out_dim

        super(SSO3_Encoder, self).__init__()

        ### encoder structure:

        ######################################################################################
        self.l1_s1 = nn.Linear(self.s1_input_dim, self.l1_s1_out_dim)
        self.l1_s1_bn = nn.BatchNorm1d(self.l1_s1_out_dim)
        l1_s1_drop_rate = 0.2
        self.drop_l1_s1 = nn.Dropout(p=l1_s1_drop_rate)

        self.l2_s1 = nn.Linear(self.l1_s1_out_dim, self.l2_s1_out_dim)
        self.l2_s1_bn = nn.BatchNorm1d(self.l2_s1_out_dim)
        l2_s1_drop_rate = 0.6
        self.drop_l2_s1 = nn.Dropout(p=l2_s1_drop_rate)

        self.l3_s1 = nn.Linear(self.l2_s1_out_dim, self.l3_s1_out_dim)
        self.l3_s1_bn = nn.BatchNorm1d(self.l3_s1_out_dim)
        l3_s1_drop_rate = 0
        self.drop_l3_s1 = nn.Dropout(p=l3_s1_drop_rate)

        self.embed_s1 = nn.Linear(self.l3_s1_out_dim, self.embed_s1_out_dim)
        self.embed_s1_bn = nn.BatchNorm1d(self.embed_s1_out_dim)
        embed_s1_drop_rate = 0
        self.drop_embed_s1 = nn.Dropout(p=embed_s1_drop_rate)

        ###########################################################################################
        self.l1_s2 = nn.Linear(self.s2_input_dim, self.l1_s2_out_dim)
        self.l1_s2_bn = nn.BatchNorm1d(self.l1_s2_out_dim)
        l1_s2_drop_rate = 0.6
        self.drop_l1_s2 = nn.Dropout(p=l1_s2_drop_rate)

        self.l2_s2 = nn.Linear(self.l1_s2_out_dim, self.l2_s2_out_dim)
        self.l2_s2_bn = nn.BatchNorm1d(self.l2_s2_out_dim)
        l2_s2_drop_rate = 0.1
        self.drop_l2_s2 = nn.Dropout(p=l2_s2_drop_rate)

        self.l3_s2 = nn.Linear(self.l2_s2_out_dim, self.l3_s2_out_dim)
        self.l3_s2_bn = nn.BatchNorm1d(self.l3_s2_out_dim)
        l3_s2_drop_rate = 0
        self.drop_l3_s2 = nn.Dropout(p=l3_s2_drop_rate)

        self.embed_s2 = nn.Linear(self.l3_s2_out_dim, self.embed_s2_out_dim)
        self.embed_s2_bn = nn.BatchNorm1d(self.embed_s2_out_dim)
        embed_s2_drop_rate = 0.1
        self.drop_embed_s2 = nn.Dropout(p=embed_s2_drop_rate)

        ##########################################################################################

        self.l1_s12 = nn.Linear(self.s1_input_dim + self.s2_input_dim,
                                self.l1_s12_out_dim)
        self.l1_s12_bn = nn.BatchNorm1d(self.l1_s12_out_dim)
        l1_s12_drop_rate = 0.2
        self.drop_l1_s12 = nn.Dropout(p=l1_s12_drop_rate)

        self.l2_s12 = nn.Linear(self.l1_s12_out_dim, self.l2_s12_out_dim)
        self.l2_s12_bn = nn.BatchNorm1d(self.l2_s12_out_dim)
        l2_s12_drop_rate = 0.4
        self.drop_l2_s12 = nn.Dropout(p=l2_s12_drop_rate)

        self.l3_s12 = nn.Linear(self.l2_s12_out_dim, self.l3_s12_out_dim)
        self.l3_s12_bn = nn.BatchNorm1d(self.l3_s12_out_dim)
        l3_s12_drop_rate = 0.6
        self.drop_l3_s12 = nn.Dropout(p=l3_s12_drop_rate)

        self.embed_s12 = nn.Linear(self.l3_s12_out_dim, self.embed_s12_out_dim)
        self.embed_s12_bn = nn.BatchNorm1d(self.embed_s12_out_dim)
        embed_s12_drop_rate = 0
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


class SSO3_Decoder(nn.Module):

    ### decoder: construct s1 and s2  based on the concatenated z12 z1 and z2
    ### and calculate the reconstruction loss separately for s1 and s2

    def __init__(self):
        self.s1_input_dim = SSO3_Encoder().s1_input_dim
        self.s2_input_dim = SSO3_Encoder().s2_input_dim

        self.s1_embed_dim = SSO3_Encoder().embed_s1_out_dim
        self.s2_embed_dim = SSO3_Encoder().embed_s2_out_dim
        self.s12_embed_dim = SSO3_Encoder().embed_s12_out_dim

        self._embed_s1_out_dim = 512
        self._l3_s1_out_dim = 128
        self._l2_s1_out_dim = 256
        self._l1_s1_out_dim = self.s1_input_dim

        self._embed_s2_out_dim = 1024
        self._l3_s2_out_dim = 64
        self._l2_s2_out_dim = 512
        self._l1_s2_out_dim = self.s2_input_dim

        super(SSO3_Decoder, self).__init__()

        self._embed_s1 = nn.Linear(self.s1_embed_dim + self.s2_embed_dim + self.s12_embed_dim, \
                                   self._embed_s1_out_dim)

        self._embed_s1_bn = nn.BatchNorm1d(self._embed_s1_out_dim)
        _embed_s1_drop_rate = 0.2
        self._drop_embed_s1 = nn.Dropout(p=_embed_s1_drop_rate)

        self._l3_s1 = nn.Linear(self._embed_s1_out_dim, self._l3_s1_out_dim)
        self._l3_s1_bn = nn.BatchNorm1d(self._l3_s1_out_dim)
        _l3_s1_drop_rate = 0.1
        self._drop_l3_s1 = nn.Dropout(p=_l3_s1_drop_rate)

        self._l2_s1 = nn.Linear(self._l3_s1_out_dim, self._l2_s1_out_dim)
        self._l2_s1_bn = nn.BatchNorm1d(self._l2_s1_out_dim)
        _l2_s1_drop_rate = 0.4
        self._drop_l2_s1 = nn.Dropout(p=_l2_s1_drop_rate)

        self._l1_s1 = nn.Linear(self._l2_s1_out_dim, self._l1_s1_out_dim)
        self._l1_s1_bn = nn.BatchNorm1d(self._l1_s1_out_dim)
        _l1_s1_drop_rate = 0
        self._drop_l1_s1 = nn.Dropout(p=_l1_s1_drop_rate)

        #############################################################################

        self._embed_s2 = nn.Linear(self.s1_embed_dim + self.s2_embed_dim + self.s12_embed_dim, \
                                   self._embed_s2_out_dim)

        self._embed_s2_bn = nn.BatchNorm1d(self._embed_s2_out_dim)
        _embed_s2_drop_rate = 0.2
        self._drop_embed_s2 = nn.Dropout(p=_embed_s2_drop_rate)

        self._l3_s2 = nn.Linear(self._embed_s2_out_dim, self._l3_s2_out_dim)
        self._l3_s2_bn = nn.BatchNorm1d(self._l3_s2_out_dim)
        _l3_s2_drop_rate = 0.1
        self._drop_l3_s2 = nn.Dropout(p=_l3_s2_drop_rate)

        self._l2_s2 = nn.Linear(self._l3_s2_out_dim, self._l2_s2_out_dim)
        self._l2_s2_bn = nn.BatchNorm1d(self._l2_s2_out_dim)
        _l2_s2_drop_rate = 0.4
        self._drop_l2_s2 = nn.Dropout(p=_l2_s2_drop_rate)

        self._l1_s2 = nn.Linear(self._l2_s2_out_dim, self._l1_s2_out_dim)
        self._l1_s2_bn = nn.BatchNorm1d(self._l1_s2_out_dim)
        _l1_s2_drop_rate = 0
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


class SSO3_AE(nn.Module):
    def __init__(self):
        super(SSO3_AE, self).__init__()

        self.encoder = SSO3_Encoder()
        self.decoder = SSO3_Decoder()

    def forward(self, s1, s2, labels):
        ### encoder ouput for embeddings
        z1, z2, z12, labels = self.encoder(s1, s2, labels)

        ### decoder output for reconstructed input
        s1_out, s2_out = self.decoder(z1, z2, z12)

        return z1, z2, z12, s1_out, s2_out, labels


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

    def __init__(self):
        ### embeding layers have the same dimensions for both shared and specific AE for all views

        view_size = [20531, 1046]

        n_units_1 = [0, 0, 0, 0]
        n_units_1[0] = 1024
        n_units_1[1] = 1024
        n_units_1[2] = 32
        n_units_1[3] = 32

        n_units_2 = n_units_1.copy()

        mlp_size = [0, 0, 0, 0]
        mlp_size[0] = 64
        mlp_size[1] = 512

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

    #         # Init weight
    #         self.init_weights()

    #     def init_weights(self):
    #         init.kaiming_normal_(self.shared1_l1.weight)
    #         init.kaiming_normal_(self.shared1_l2.weight)
    #         init.kaiming_normal_(self.shared1_l3.weight)
    #         init.kaiming_normal_(self.shared1_l4.weight)

    #         init.kaiming_normal_(self.shared1_l3_.weight)
    #         init.kaiming_normal_(self.shared1_l2_.weight)
    #         init.kaiming_normal_(self.shared1_l1_.weight)
    #         init.kaiming_normal_(self.shared1_rec.weight)

    #         init.kaiming_normal_(self.specific1_l1.weight)
    #         init.kaiming_normal_(self.specific1_l2.weight)
    #         init.kaiming_normal_(self.specific1_l3.weight)
    #         init.kaiming_normal_(self.specific1_l4.weight)

    #         init.kaiming_normal_(self.specific1_l3_.weight)
    #         init.kaiming_normal_(self.specific1_l2_.weight)
    #         init.kaiming_normal_(self.specific1_l1_.weight)
    #         init.kaiming_normal_(self.specific1_rec.weight)

    #         init.kaiming_normal_(self.shared2_l1.weight)
    #         init.kaiming_normal_(self.shared2_l2.weight)
    #         init.kaiming_normal_(self.shared2_l3.weight)
    #         init.kaiming_normal_(self.shared2_l4.weight)

    #         init.kaiming_normal_(self.shared2_l3_.weight)
    #         init.kaiming_normal_(self.shared2_l2_.weight)
    #         init.kaiming_normal_(self.shared2_l1_.weight)
    #         init.kaiming_normal_(self.shared2_rec.weight)

    #         init.kaiming_normal_(self.specific2_l1.weight)
    #         init.kaiming_normal_(self.specific2_l2.weight)
    #         init.kaiming_normal_(self.specific2_l3.weight)
    #         init.kaiming_normal_(self.specific2_l4.weight)

    #         init.kaiming_normal_(self.specific2_l3_.weight)
    #         init.kaiming_normal_(self.specific2_l2_.weight)
    #         init.kaiming_normal_(self.specific2_l1_.weight)
    #         init.kaiming_normal_(self.specific2_rec.weight)

    #         init.kaiming_normal_(self.view1_mlp1.weight)
    #         init.kaiming_normal_(self.view1_mlp2.weight)
    #         init.kaiming_normal_(self.view2_mlp1.weight)
    #         init.kaiming_normal_(self.view2_mlp2.weight)

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

