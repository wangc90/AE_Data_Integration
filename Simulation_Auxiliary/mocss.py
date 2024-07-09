import numpy as np
import pandas as pd
# from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import torch as pt
from torch import nn
import argparse
import scipy.io as scio
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim
import torch.nn.init as init
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit
from sklearn.cluster import KMeans,DBSCAN,Birch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, v_measure_score
from critic import LinearCritic
from Supcon import SupConLoss
from contrastive_loss import InstanceLoss, ClusterLoss
import evaluation


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
# setup_seed(3)


class SharedAndSpecificLoss(nn.Module):
    def __init__(self, ):
        super(SharedAndSpecificLoss, self).__init__()

    @staticmethod
    def orthogonal_loss(shared, specific):
        shared = shared - shared.mean()
        specific = specific - specific.mean()
        shared = F.normalize(shared, p=2, dim=1)
        specific = F.normalize(specific, p=2, dim=1)
        correlation_matrix = torch.mul(shared, specific)
        cost = correlation_matrix.mean()
        return cost
        # correlation_matrix = torch.mm(shared.T, specific)
        # cost = torch.linalg.matrix_norm(correlation_matrix)
        # return cost * cost

    @staticmethod
    def contrastive_loss(shared_1, shared_2, temperature, batch_size):
        assert (shared_1.dim() == 2)
        assert (shared_2.dim() == 2)
        shared_1 = shared_1 - shared_1.mean()
        shared_2 = shared_2 - shared_2.mean()
        shared_1 = F.normalize(shared_1, p=2, dim=1)
        shared_2 = F.normalize(shared_2, p=2, dim=1)

        #Contrastive loss version1
        criterion_instance = InstanceLoss(batch_size=batch_size, temperature=temperature)
        loss = criterion_instance(shared_1, shared_2)
        # label = torch.tensor(label)
        # # print(label.shape)
        # criterion = SupConLoss(temperature=temperature)
        # features = torch.cat([shared_1.unsqueeze(1), shared_2.unsqueeze(1)], dim=1)
        # # print(features.shape)
        # loss = criterion(features, label)
        return loss

    @staticmethod
    def reconstruction_loss(rec, ori):
        assert (rec.dim() == 2)
        assert (ori.dim() == 2)
        rec = rec - rec.mean()
        ori = ori - ori.mean()
        rec = F.normalize(rec, p=2, dim=1)
        ori = F.normalize(ori, p=2, dim=1)
        loss = torch.linalg.matrix_norm(rec-ori)
        return loss

        # Contrastive loss version2
        # # [2*B, D]
        # out = torch.cat([shared_1, shared_2], dim=0)
        # # [2*B, 2*B]
        # sim_matrix = torch.exp(torch.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0),dim=-1) / temperature)
        # mask = (torch.ones_like(sim_matrix) - torch.eye(sim_matrix.shape[0], device=sim_matrix.device)).bool()
        # # [2*B, 2*B-1]
        # sim_matrix = sim_matrix.masked_select(mask).view(sim_matrix.shape[0], -1)
        #
        # # compute loss
        # pos_sim = torch.exp(torch.sum(shared_1 * shared_2, dim=-1) / temperature)
        # # [2*B]
        # pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        # loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

        # Contrastive loss version3
        # label = torch.tensor(label)
        # # print(label.shape)
        # criterion = SupConLoss(temperature=temperature)
        # features = torch.cat([shared_1.unsqueeze(1), shared_2.unsqueeze(1)], dim=1)
        # # print(features.shape)
        # loss = criterion(features, label)

    # @staticmethod
    # def clustering_loss(self, shared_specific_con):
    #     # Clustering loss
    #     pass

    def forward(self, shared1_output, shared2_output, shared3_output, specific1_output, specific2_output, specific3_output,
                shared1_rec, shared2_rec, shared3_rec, specific1_rec, specific2_rec, specific3_rec, ori1, ori2, ori3, shared1_mlp,shared2_mlp,shared3_mlp,temperature, batch_size):
        # orthogonal restrict
        orthogonal_loss1 = self.orthogonal_loss(shared1_output, specific1_output)
        orthogonal_loss2 = self.orthogonal_loss(shared2_output, specific2_output)
        orthogonal_loss3 = self.orthogonal_loss(shared3_output, specific3_output)
        orthogonal_loss_all =  orthogonal_loss1 +   orthogonal_loss2 +  orthogonal_loss3
        # print(orthogonal_loss_all)

        # Contrastive Loss
        contrastive_loss1 = self.contrastive_loss(shared1_mlp, shared2_mlp, temperature, batch_size)
        contrastive_loss2 = self.contrastive_loss(shared1_mlp, shared3_mlp, temperature, batch_size)
        contrastive_loss3 = self.contrastive_loss(shared2_mlp, shared3_mlp, temperature, batch_size)
        contrastive_loss_all =  contrastive_loss1 +  contrastive_loss2 +   contrastive_loss3
        # print(contrastive_loss_all)

        # reconstruction Loss
        reconstruction_loss1 = self.reconstruction_loss(shared1_rec, ori1) + self.reconstruction_loss(specific1_rec, ori1)
        reconstruction_loss2 = self.reconstruction_loss(shared2_rec, ori2) + self.reconstruction_loss(specific2_rec, ori2)
        reconstruction_loss3 = self.reconstruction_loss(shared3_rec, ori3) + self.reconstruction_loss(specific3_rec, ori3)
        reconstruction_loss_all =  reconstruction_loss1 +  reconstruction_loss2 +  reconstruction_loss3
        # print(reconstruction_loss_all)

        loss_total = orthogonal_loss_all + contrastive_loss_all +  0.7 * reconstruction_loss_all

        return loss_total


class SharedAndSpecificEmbedding(nn.Module):
    def __init__(self, view_size=[1000, 1000, 503], n_units_1=[512, 256, 128, 64], n_units_2=[512, 256, 128, 64], n_units_3=[512, 256, 128, 64],mlp_size=[64,16]):
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

        # View3
        self.shared3_l1 = nn.Linear(view_size[2], n_units_3[0])
        self.shared3_l2 = nn.Linear(n_units_3[0], n_units_3[1])
        self.shared3_l3 = nn.Linear(n_units_3[1], n_units_3[2])
        self.shared3_l4 = nn.Linear(n_units_3[2], n_units_3[3])
        self.shared3_l3_ = nn.Linear(n_units_3[3], n_units_3[2])
        self.shared3_l2_ = nn.Linear(n_units_3[2], n_units_3[1])
        self.shared3_l1_ = nn.Linear(n_units_3[1], n_units_3[0])
        self.shared3_rec = nn.Linear(n_units_3[0], view_size[2])

        self.specific3_l1 = nn.Linear(view_size[2], n_units_3[0])
        self.specific3_l2 = nn.Linear(n_units_3[0], n_units_3[1])
        self.specific3_l3 = nn.Linear(n_units_3[1], n_units_3[2])
        self.specific3_l4 = nn.Linear(n_units_3[2], n_units_3[3])
        self.specific3_l3_ = nn.Linear(n_units_3[3], n_units_3[2])
        self.specific3_l2_ = nn.Linear(n_units_3[2], n_units_3[1])
        self.specific3_l1_ = nn.Linear(n_units_3[1], n_units_3[0])
        self.specific3_rec = nn.Linear(n_units_3[0], view_size[2])

        self.view3_mlp1 = nn.Linear(n_units_3[3], mlp_size[0])
        self.view3_mlp2 = nn.Linear(mlp_size[0], mlp_size[1])

        # # Classification
        # self.classification_l1 = nn.Linear(out_size * 4, c_n_units[0])
        # self.classification_l2 = nn.Linear(c_n_units[0], c_n_units[1])
        # self.classification_l3 = nn.Linear(c_n_units[1], 2)

        # Init weight
        self.init_weights()

    def init_weights(self):
        init.kaiming_normal_(self.shared1_l1.weight)
        init.kaiming_normal_(self.shared1_l2.weight)
        init.kaiming_normal_(self.shared1_l3.weight)
        init.kaiming_normal_(self.shared1_l4.weight)
        init.kaiming_normal_(self.shared1_l3_.weight)
        init.kaiming_normal_(self.shared1_l2_.weight)
        init.kaiming_normal_(self.shared1_l1_.weight)
        init.kaiming_normal_(self.shared1_rec.weight)

        init.kaiming_normal_(self.specific1_l1.weight)
        init.kaiming_normal_(self.specific1_l2.weight)
        init.kaiming_normal_(self.specific1_l3.weight)
        init.kaiming_normal_(self.specific1_l4.weight)
        init.kaiming_normal_(self.specific1_l3_.weight)
        init.kaiming_normal_(self.specific1_l2_.weight)
        init.kaiming_normal_(self.specific1_l1_.weight)
        init.kaiming_normal_(self.specific1_rec.weight)

        init.kaiming_normal_(self.shared2_l1.weight)
        init.kaiming_normal_(self.shared2_l2.weight)
        init.kaiming_normal_(self.shared2_l3.weight)
        init.kaiming_normal_(self.shared2_l4.weight)
        init.kaiming_normal_(self.shared2_l3_.weight)
        init.kaiming_normal_(self.shared2_l2_.weight)
        init.kaiming_normal_(self.shared2_l1_.weight)
        init.kaiming_normal_(self.shared2_rec.weight)

        init.kaiming_normal_(self.specific2_l1.weight)
        init.kaiming_normal_(self.specific2_l2.weight)
        init.kaiming_normal_(self.specific2_l3.weight)
        init.kaiming_normal_(self.specific2_l4.weight)
        init.kaiming_normal_(self.specific2_l3_.weight)
        init.kaiming_normal_(self.specific2_l2_.weight)
        init.kaiming_normal_(self.specific2_l1_.weight)
        init.kaiming_normal_(self.specific2_rec.weight)

        init.kaiming_normal_(self.shared3_l1.weight)
        init.kaiming_normal_(self.shared3_l2.weight)
        init.kaiming_normal_(self.shared3_l3.weight)
        init.kaiming_normal_(self.shared3_l4.weight)
        init.kaiming_normal_(self.shared3_l3_.weight)
        init.kaiming_normal_(self.shared3_l2_.weight)
        init.kaiming_normal_(self.shared3_l1_.weight)
        init.kaiming_normal_(self.shared3_rec.weight)

        init.kaiming_normal_(self.specific3_l1.weight)
        init.kaiming_normal_(self.specific3_l2.weight)
        init.kaiming_normal_(self.specific3_l3.weight)
        init.kaiming_normal_(self.specific3_l4.weight)
        init.kaiming_normal_(self.specific3_l3_.weight)
        init.kaiming_normal_(self.specific3_l2_.weight)
        init.kaiming_normal_(self.specific3_l1_.weight)
        init.kaiming_normal_(self.specific3_rec.weight)

        init.kaiming_normal_(self.view1_mlp1.weight)
        init.kaiming_normal_(self.view1_mlp2.weight)
        init.kaiming_normal_(self.view2_mlp1.weight)
        init.kaiming_normal_(self.view2_mlp2.weight)
        init.kaiming_normal_(self.view3_mlp1.weight)
        init.kaiming_normal_(self.view3_mlp2.weight)

        # init.kaiming_uniform_(self.classification_l1.weight)
        # init.kaiming_uniform_(self.classification_l2.weight)
        # init.kaiming_uniform_(self.classification_l3.weight)

    def forward(self, view1_input, view2_input, view3_input):
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

        # View3
        view3_specific = F.tanh(self.specific3_l1(view3_input))
        view3_specific = F.tanh(self.specific3_l2(view3_specific))
        view3_specific = F.tanh(self.specific3_l3(view3_specific))
        view3_specific_em = F.tanh(self.specific3_l4(view3_specific))
        view3_specific = F.tanh(self.specific3_l3_(view3_specific_em))
        view3_specific = F.tanh(self.specific3_l2_(view3_specific))
        view3_specific = F.tanh(self.specific3_l1_(view3_specific))
        view3_specific_rec = torch.sigmoid(self.specific3_rec(view3_specific))

        view3_shared = F.tanh(self.shared3_l1(view3_input))
        view3_shared = F.tanh(self.shared3_l2(view3_shared))
        view3_shared = F.tanh(self.shared3_l3(view3_shared))
        view3_shared_em = F.tanh(self.shared3_l4(view3_shared))
        view3_shared = F.tanh(self.shared3_l3_(view3_shared_em))
        view3_shared = F.tanh(self.shared3_l2_(view3_shared))
        view3_shared = F.tanh(self.shared3_l1_(view3_shared))
        view3_shared_rec = torch.sigmoid(self.shared3_rec(view3_shared))

        view3_shared_mlp = F.tanh(self.view3_mlp1(view3_shared_em))
        view3_shared_mlp = F.tanh(self.view3_mlp2(view3_shared_mlp))

        # # Classification
        # classification_input = torch.cat([view1_specific, view1_shared, view2_shared, view2_specific], dim=1)
        # classification_output = F.tanh(self.classification_l1(F.dropout(classification_input)))
        # classification_output = F.tanh(self.classification_l2(F.dropout(classification_output)))
        # classification_output = self.classification_l3(classification_output)

        return view1_specific_em, view1_shared_em, view2_specific_em, view2_shared_em, view3_specific_em, view3_shared_em, \
               view1_specific_rec, view1_shared_rec, view2_specific_rec, view2_shared_rec, view3_specific_rec, view3_shared_rec,\
               view1_shared_mlp, view2_shared_mlp, view3_shared_mlp


def main():
    # Hyper Parameters
    EPOCH = 1000  # train the training data n times
    BATCH_SIZE = 32
    USE_GPU = False
    # number_split = 10
    n_clusters = 3
    temperature = 0.4
    setup_seed(21)  #4, 8, 11, 21
    print(pt.__version__)
    # Load BRCA_data
    view1_data = pd.read_csv('../../data/LUAD/1_all.csv', header=None)
    view2_data = pd.read_csv('../../data/LUAD/2_all.csv', header=None)
    view3_data = pd.read_csv('../../data/LUAD/3_all.csv', header=None)
    label = pd.read_csv('../../data/LUAD/labels_all.csv', header=None)

    # # Load 2V_MNIST_USPS
    # data = scio.loadmat('../../data/BRCA/2V_MNIST_USPS.mat')
    # view1_data = data['X1']
    # view2_data = data['X2']
    # label = data['Y']
    # view1_data = view1_data.reshape(5000, 784)
    # view2_data = view2_data.reshape(5000, 784)
    # label = label[0].reshape(5000,1)

    # # Load NmRm
    # view1_data = np.load('../../data/BRCA/NmRmXn.npy')
    # view2_data = np.load('../../data/BRCA/NmRmXr.npy')
    # label = np.load('../../data/BRCA/NmRmY.npy')
    # view1_data = view1_data.reshape(70000, 784)
    # view2_data = view2_data.reshape(70000, 784)
    # label = label.reshape(-1, 1)

    # # Load svhn/cifar10
    # view1_data = np.load('../../data/BRCA/cifar_view1.npy')
    # view2_data = np.load('../../data/BRCA/cifar_view2.npy')
    # label = np.load('../../data/BRCA/cifar_label.npy')
    # label = label.reshape(-1, 1)
    # print(view1_data.shape)  #[n,1024]
    # print(view2_data.shape)  #[n,1024]
    # print(label.shape)

    #数据归一化
    scaler = preprocessing.MinMaxScaler()
    view1_data = scaler.fit_transform(view1_data)
    view2_data = scaler.fit_transform(view2_data)
    view3_data = scaler.fit_transform(view3_data)

    Nmi_test = []
    Ari_test = []
    km = KMeans(n_clusters=n_clusters,init='k-means++')
    dbscan = DBSCAN()
    birch = Birch(n_clusters=n_clusters)
    nmi = normalized_mutual_info_score
    ari = adjusted_rand_score


    view_train_concatenate = np.concatenate((view1_data, view2_data, view3_data), axis=1)
    print(view_train_concatenate.shape)
    view_train_concatenate = np.concatenate((view_train_concatenate, label), axis=1)
    print(view_train_concatenate.shape)
    y_true = view_train_concatenate[:,-1]
    print(y_true.shape)

    # Build Model
    # model = SharedAndSpecificEmbedding(view_size=[1000, 1000, 503], n_units_1=[512, 256, 128, 32], n_units_2=[512, 256, 128, 32], n_units_3=[256,128,64,32],mlp_size=[32,8])
    # model = SharedAndSpecificEmbedding(view_size=[6000, 534, 5000], n_units_1=[512, 256, 128, 32], n_units_2=[512, 256, 128, 32], n_units_3=[512, 128, 64, 32], mlp_size=[32, 8])
    model = SharedAndSpecificEmbedding(view_size=[6000, 554, 6000], n_units_1=[512, 256, 128, 32], n_units_2=[512, 256, 128, 32], n_units_3=[512, 256, 128, 32], mlp_size=[32, 8])
    # model = SharedAndSpecificEmbedding(view_size=[6000, 820, 5000], n_units_1=[512, 256, 128, 32], n_units_2=[512, 256, 128, 32], n_units_3=[512, 256, 128, 32], mlp_size=[32, 8])
    # model = SharedAndSpecificEmbedding(view_size=[6000, 519, 6000], n_units_1=[512, 256, 128, 32], n_units_2=[512, 256, 128, 32], n_units_3=[512, 256, 128, 32], mlp_size=[32, 8])

    # print(model)
    if USE_GPU:
        model = model.cuda()

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0003)
    loss_function = SharedAndSpecificLoss()

    print("Training...")

    train_loss_ = []
    # train_acc_ = []
    # test_acc_ = []

    # Data Loader for easy mini-batch return in training
    train_loader = torch.utils.data.DataLoader(dataset=view_train_concatenate, batch_size=BATCH_SIZE, shuffle=True)

    nmi_max = 0.0
    ari_max = 0.0
    f_max = 0.0
    acc_max = 0.0

    for epoch in range(EPOCH):

        # training epoch
        # model.train()
        # total_acc = 0.0
        Nmi_train = []
        Ari_train = []
        total_loss = 0.0
        total = 0.0
        for iteration_index, train_batch in enumerate(train_loader):
            train_data = train_batch
            # train_data = pd.DataFrame(train_data)
            # print(train_data.shape)
            view1_train_data = train_data[:, :6000]
            # print(view1_train_data.shape)
            view1_train_data = torch.tensor(view1_train_data).clone().detach()
            # print(view1_train_data.shape)
            view2_train_data = train_data[:, 6000:6554]
            # print(view2_train_data.shape)
            view2_train_data = torch.tensor(view2_train_data).clone().detach()
            # print(view2_train_data.shape)
            view3_train_data = train_data[:, 6554:12554]
            # print(view3_train_data.shape)
            view3_train_data = torch.tensor(view3_train_data).clone().detach()
            # print(view3_train_data.shape)

            train_labels = train_data[:, -1]
            # print(train_labels.shape)
            train_labels = torch.squeeze(train_labels)
            # print(train_labels.shape)

            if USE_GPU:
                view1_train_data, view2_train_data, view3_train_data = Variable(view1_train_data.cuda()), Variable(
                    view2_train_data.cuda()), Variable(view3_train_data.cuda())
            else:
                view1_train_data = Variable(view1_train_data).type(torch.FloatTensor)
                view2_train_data = Variable(view2_train_data).type(torch.FloatTensor)
                view3_train_data = Variable(view3_train_data).type(torch.FloatTensor)


            view1_specific_em, view1_shared_em, view2_specific_em, view2_shared_em, view3_specific_em, view3_shared_em, \
                view1_specific_rec, view1_shared_rec, view2_specific_rec, view2_shared_rec, view3_specific_rec, view3_shared_rec,\
                  view1_shared_mlp, view2_shared_mlp, view3_shared_mlp = \
                model(view1_train_data, view2_train_data, view3_train_data)

            loss = loss_function(shared1_output=view1_shared_em, shared2_output=view2_shared_em, shared3_output=view3_shared_em,
                                 specific1_output = view1_specific_em, specific2_output = view2_specific_em, specific3_output=view3_specific_em,
                                 shared1_rec = view1_shared_rec, specific1_rec = view1_specific_rec,
                                 shared2_rec=view2_shared_rec, specific2_rec=view2_specific_rec,
                                 shared3_rec=view3_shared_rec, specific3_rec=view3_specific_rec,
                                 ori1 = view1_train_data, ori2 = view2_train_data, ori3 = view3_train_data,
                                 shared1_mlp=view1_shared_mlp,shared2_mlp=view2_shared_mlp,shared3_mlp=view3_shared_mlp,
                                 batch_size = view1_shared_em.shape[0],
                                 temperature = temperature,
                                 # label = train_labels,
                                 )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calc training acc
            # _, predicted = torch.max(classification_output.data, 1)
            # total_acc += (predicted == train_labels.data).sum()
            total += len(train_data)
            total_loss += loss.item()
        train_loss_.append(total_loss / total)
        #train metrics
        view1_all = torch.tensor(view_train_concatenate[:, :6000]).clone().detach().float()
        view2_all = torch.tensor(view_train_concatenate[:, 6000:6554]).clone().detach().float()
        view3_all = torch.tensor(view_train_concatenate[:, 6554:12554]).clone().detach().float()
        view1_specific_em_train, view1_shared_em_train, view2_specific_em_train, view2_shared_em_train, view3_specific_em_train, view3_shared_em_train, \
             view1_specific_rec_train, view1_shared_rec_train, view2_specific_rec_train, view2_shared_rec_train, view3_specific_rec_train, view3_shared_rec_train,\
                view1_shared_mlp_train, view2_shared_mlp_train,view3_shared_mlp_train\
              = model(view1_all, view2_all, view3_all)
        y_true_all = np.array(y_true).flatten()
        view_shared_common_train = (view1_shared_em_train + view2_shared_em_train + view3_shared_em_train) / 3
        final_embedding_train = torch.cat((view1_specific_em_train, view2_specific_em_train, view3_specific_em_train, view_shared_common_train), dim=1)
        # final_embedding_train = view_shared_common_train
        final_embedding_train = final_embedding_train.detach().numpy()
        y_pred_train = km.fit_predict(final_embedding_train)
        nmi, ari, f_score, acc = evaluation.evaluate(y_true_all, y_pred_train)
        if epoch>600 and nmi>nmi_max and ari>ari_max and f_score>f_max and acc>ari_max:
            nmi_max = nmi
            ari_max = ari
            f_max = f_score
            acc_max = acc
        # print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f  <==|' % (nmi(y_true_all, y_pred_train), ari(y_true_all, y_pred_train)))
        # print('[Epoch: %3d/%3d] Training Loss: %f' % (epoch + 1, EPOCH, train_loss_[epoch]), '|==>  nmi: %.4f,  ari: %.4f  <==|' % (nmi(y_true_all, y_pred_train), ari(y_true_all, y_pred_train)))
        print('[Epoch: %3d/%3d] Training Loss: %f' % (epoch + 1, EPOCH, train_loss_[epoch]), '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f  <==|' % (nmi, ari, f_score, acc))
        # train_acc_.append(total_acc.numpy() / total)
    print('\n' + ' ' * 8 + '|==>  nmi_max: %.4f,  ari_max: %.4f,  f_score_max: %.4f,  acc_max: %.4f  <==|' % (nmi_max, ari_max, f_max, acc_max))
    # testing
    # view1_test_data = view_test_concatenate[:, :1000]
    # # print(view1_train_data.shape)
    # view1_test_data = torch.tensor(view1_test_data).clone().detach().float()
    # print(view1_test_data.shape)
    # view2_test_data = view_test_concatenate[:, 1000:2000]
    # view2_test_data = torch.tensor(view2_test_data).clone().detach().float()
    # print(view2_test_data.shape)
    # view3_test_data = view_test_concatenate[:, 2000:2503]
    # view3_test_data = torch.tensor(view3_test_data).clone().detach().float()
    # print(view3_test_data.shape)
    # view1_specific, view1_shared, view2_specific, view2_shared, view1_contrastive, view2_contrastive = \
    #     model(view1_test_data, view2_test_data)
    # y_true = np.array(y_true).flatten()

    #clustering
    view1_data_test = torch.tensor(view1_data).clone().detach().float()
    view2_data_test = torch.tensor(view2_data).clone().detach().float()
    view3_data_test = torch.tensor(view3_data).clone().detach().float()
    view1_specific_em_test, view1_shared_em_test, view2_specific_em_test, view2_shared_em_test, view3_specific_em_test, view3_shared_em_test, \
       view1_specific_rec_test, view1_shared_rec_test, view2_specific_rec_test, view2_shared_rec_test, view3_specific_rec_test, view3_shared_rec_test,\
         view1_shared_mlp_test, view2_shared_mlp_test,view3_shared_mlp_test\
         = model(view1_data_test, view2_data_test, view3_data_test)
    view_shared_common = (view1_shared_em_test + view2_shared_em_test + view3_shared_em_test) / 3
    #transfer tensor to numpy
    # view1_specific_test = view1_specific.detach().numpy()
    # view2_specific_test = view2_specific.detach().numpy()
    # view_shared_common_test = view_shared_common.detach().numpy()
    # final_embedding = np.concatenate(view_shared_common, view1_specific_test, view2_specific_test, axis=1)
    # final_embedding = torch.cat((view1_specific_em_test, view2_specific_em_test, view3_specific_em_test, view_shared_common), dim=1)
    # final_embedding1 = view_shared_common
    # final_embedding1 = final_embedding1.detach().numpy()
    # # np.savetxt('../../data/SARC/final_embedding_sarc.txt', final_embedding)
    # y_pred1 = km.fit_predict(final_embedding1)
    # nmi_, ari_, f_score_, acc_ = evaluation.evaluate(y_true, y_pred1)
    # print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f  <==|' % (nmi_, ari_, f_score_, acc_))
    #
    # final_embedding2 = view1_specific_em_test
    # final_embedding2 = final_embedding2.detach().numpy()
    # # np.savetxt('../../data/SARC/final_embedding_sarc.txt', final_embedding)
    # y_pred2 = km.fit_predict(final_embedding2)
    # nmi_, ari_, f_score_, acc_ = evaluation.evaluate(y_true, y_pred2)
    # print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f  <==|' % (nmi_, ari_, f_score_, acc_))
    #
    # final_embedding3 = view2_specific_em_test
    # final_embedding3 = final_embedding3.detach().numpy()
    # # np.savetxt('../../data/SARC/final_embedding_sarc.txt', final_embedding)
    # y_pred3 = km.fit_predict(final_embedding3)
    # nmi_, ari_, f_score_, acc_ = evaluation.evaluate(y_true, y_pred3)
    # print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f  <==|' % (nmi_, ari_, f_score_, acc_))
    #
    # final_embedding4 = view3_specific_em_test
    # final_embedding4 = final_embedding4.detach().numpy()
    # # np.savetxt('../../data/SARC/final_embedding_sarc.txt', final_embedding)
    # y_pred4 = km.fit_predict(final_embedding4)
    # nmi_, ari_, f_score_, acc_ = evaluation.evaluate(y_true, y_pred4)
    # print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f  <==|' % (nmi_, ari_, f_score_, acc_))

    final_embedding5 = torch.cat((view1_specific_em_test, view2_specific_em_test, view3_specific_em_test, view_shared_common), dim=1)
    final_embedding5 = final_embedding5.detach().numpy()
    # np.savetxt('../../data/SARC/final_embedding_sarc.txt', final_embedding5)
    y_pred5 = km.fit_predict(final_embedding5)
    nmi_, ari_, f_score_, acc_ = evaluation.evaluate(y_true, y_pred5)
    print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f  <==|' % (nmi_, ari_, f_score_, acc_))
    # print('------------------------------------')
    # Nmi_test.append(nmi(y_true, y_pred1)), Ari_test.append(ari(y_true, y_pred1))
    # print('\n' + ' ' * 8 + '|==>  nmi_avg: %.4f,  ari_avg: %.4f  <==|' % (np.mean(Nmi_test), np.mean(Ari_test)))
    # KMeans
    # view_test_data = np.concatenate((view1_test_data, view2_test_data), axis=1)
    # y_pred_ori = km.fit_predict(view_test_data)
    # print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f  <==|' % (nmi(y_true, y_pred_ori), ari(y_true, y_pred_ori)))
    # testing epoch ========================================================================================
    # total_acc = 0.0
    # total = 0.0
    # model.eval()
    # for iter, testdata in enumerate(test_loader):
    #     test_page_inputs, test_link_inputs, test_labels = testdata
    #     test_labels = torch.squeeze(test_labels)
    #
    #     if USE_GPU:
    #         test_page_inputs, test_link_inputs, test_labels = Variable(test_page_inputs.cuda()), \
    #                                                           Variable(test_link_inputs.cuda()), test_labels.cuda()
    #     else:
    #         test_page_inputs = Variable(test_page_inputs).type(torch.FloatTensor)
    #         test_link_inputs = Variable(test_link_inputs).type(torch.FloatTensor)
    #         test_labels = Variable(test_labels).type(torch.LongTensor)
    #
    #     view1_specific, view1_shared, view2_specific, view2_shared, classification_output = \
    #         model(test_page_inputs, test_link_inputs)
    #
    #     # calc testing acc
    #     _, predicted = torch.max(classification_output.data, 1)
    #     total_acc += (predicted == test_labels.data).sum()
    #     total += len(test_labels)
    # test_acc_.append(total_acc.numpy() / total)
    #
    # print('[Epoch: %3d/%3d] Training Loss: %.3f, Training Acc: %.3f, Testing Acc: %.4f'
    #       % (epoch, EPOCH, train_loss_[epoch], train_acc_[epoch], test_acc_[epoch] * 100))


if __name__ == "__main__":
    main()
