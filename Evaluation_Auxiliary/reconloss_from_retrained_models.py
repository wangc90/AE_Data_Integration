import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
import numpy as np
import torch.nn.functional as F



def recon_loss(model_path, data_set, is_SS, is_MOCSS):
    '''
    ### load the saved model and apply it to the data_set to extract the embeddings for further analysis

    '''
    saved_model = torch.load(model_path).to('cuda')

    saved_model.eval()

    # test_data_set = torch.utils.data.Subset(BS_LS_test_dataset)

    data_loader = DataLoader(data_set, batch_size=len(data_set))

    print(len(data_set))

    with torch.no_grad():

        for view1_data, view2_data, labels in data_loader:
            view1_data = view1_data.type(torch.float32).to('cuda')
            view2_data = view2_data.type(torch.float32).to('cuda')
            labels = labels.type(torch.LongTensor).to('cuda')

            if is_SS:## if SS model, expect z1, z2, z12
                z1, z2, z12, s1_out, s2_out, labels = \
                    saved_model(view1_data, view2_data, labels)
                
                s1_out = F.normalize(s1_out, p=2, dim=1)
                s2_out = F.normalize(s2_out, p=2, dim=1)
                
                
                s1 = F.normalize(view1_data, p=2, dim=1)
                s2 = F.normalize(view2_data, p=2, dim=1)

                ### this is the old recon loss just for average
                # recon_loss = torch.linalg.matrix_norm(s1_out-s1) + torch.linalg.matrix_norm(s2_out-s2)

                ### this is the new recon loss for each subject: the sum of the l2 normal for each data source
                recon_loss = (s1_out-s1).pow(2).sum(dim=1).sqrt() + (s2_out-s2).pow(2).sum(dim=1).sqrt()


            elif is_MOCSS: ### if MOCSS
                view1_specific_em, view1_shared_em, view2_specific_em, view2_shared_em, \
                view1_specific_rec, view1_shared_rec, view2_specific_rec, view2_shared_rec, \
                view1_shared_mlp, view2_shared_mlp, train_labels = saved_model(view1_data, view2_data, labels)
                
                view1_specific_rec = F.normalize(view1_specific_rec, p=2, dim=1)
                view1_shared_rec = F.normalize(view1_shared_rec, p=2, dim=1)
                
                view2_specific_rec = F.normalize(view2_specific_rec, p=2, dim=1)
                view2_shared_rec = F.normalize(view2_shared_rec, p=2, dim=1)
                
                s1 = F.normalize(view1_data, p=2, dim=1)
                s2 = F.normalize(view2_data, p=2, dim=1)

                ### this is the old recon loss just for average
                # recon_loss = torch.linalg.matrix_norm(view1_specific_rec-s1) +\
                #                 torch.linalg.matrix_norm(view1_shared_rec-s1) + \
                #                 torch.linalg.matrix_norm(view2_specific_rec-s2) + \
                #                 torch.linalg.matrix_norm(view2_shared_rec-s2)

                ### this is the new recon loss for each subject: the sum of the l2 normal for each data source
                recon_loss = (view1_specific_rec-s1).pow(2).sum(dim=1).sqrt() +\
                             (view1_shared_rec-s1).pow(2).sum(dim=1).sqrt() + \
                             (view2_specific_rec-s2).pow(2).sum(dim=1).sqrt() + \
                             (view2_shared_rec-s2).pow(2).sum(dim=1).sqrt()

            else:
                z12, s1_out, s2_out, labels = \
                    saved_model(view1_data, view2_data, labels)

                s1_out = F.normalize(s1_out, p=2, dim=1)
                s2_out = F.normalize(s2_out, p=2, dim=1)
                
                
                s1 = F.normalize(view1_data, p=2, dim=1)
                s2 = F.normalize(view2_data, p=2, dim=1)

                ### this is the old recon loss just for average
                # recon_loss = torch.linalg.matrix_norm(s1_out-s1) + torch.linalg.matrix_norm(s2_out-s2)

                ### this is the new recon loss for each subject: the sum of the l2 normal for each data source
                recon_loss = (s1_out-s1).pow(2).sum(dim=1).sqrt() + (s2_out-s2).pow(2).sum(dim=1).sqrt()

    ### this is the old recon loss just for average
    # return recon_loss.item()/len(data_set)
    ### this is the new recon loss for each subject: the sum of the l2 normal for each data source
    ### this one should have the length of the dataset
    return np.array(recon_loss.cpu()).tolist()