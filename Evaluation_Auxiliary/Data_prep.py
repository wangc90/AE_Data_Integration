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

        breast_index = []
        lung_index = []
        melanoma_index = []
        liver_index = []
        sarcoma_index = []
        kidney_index = []

        for i, j in enumerate(self.label):
            if j == 'Primary Tumor*breast':
                breast_index.append(i)
            elif j == 'Primary Tumor*lung':
                lung_index.append(i)
            elif j == 'Primary Tumor*melanoma':
                melanoma_index.append(i)
            elif j == 'Primary Tumor*liver':
                liver_index.append(i)
            elif j == 'Primary Tumor*sarcoma':
                sarcoma_index.append(i)
            elif j == 'Primary Tumor*kidney':
                kidney_index.append(i)

        breast_select = np.random.choice(np.array(breast_index),
                                         size=round(len(breast_index) * self.training_prop), replace=False)
        lung_select = np.random.choice(np.array(lung_index),
                                       size=round(len(lung_index) * self.training_prop), replace=False)
        melanoma_select = np.random.choice(np.array(melanoma_index),
                                           size=round(len(melanoma_index) * self.training_prop), replace=False)
        liver_select = np.random.choice(np.array(liver_index),
                                        size=round(len(liver_index) * self.training_prop), replace=False)
        sarcoma_select = np.random.choice(np.array(sarcoma_index),
                                          size=round(len(sarcoma_index) * self.training_prop), replace=False)
        kidney_select = np.random.choice(np.array(kidney_index),
                                         size=round(len(kidney_index) * self.training_prop), replace=False)

        training_index = np.concatenate([breast_select, lung_select, melanoma_select,
                                         liver_select, sarcoma_select, kidney_select])

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

        ### convert labels to numbers
        ### Primary Tumor*breast': 0; Primary Tumor*lung: 1;
        ### Primary Tumor*melanoma: 2; Primary Tumor*liver:3 ;
        ### Primary Tumor*sarcoma: 4; Primary Tumor*kidney: 5

        labels_dict = {'Primary Tumor*breast': 0, 'Primary Tumor*lung': 1,
                       'Primary Tumor*melanoma': 2, 'Primary Tumor*liver': 3,
                       'Primary Tumor*sarcoma': 4, 'Primary Tumor*kidney': 5}
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