import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
import numpy as np



def embedding_collection(model_path, data_set, is_SS, is_MOCSS):
    '''
    ### load the saved model and apply it to the data_set to extract the embeddings for further analysis

    '''
    saved_model = torch.load(model_path).to('cuda')

    saved_model.eval()

    # test_data_set = torch.utils.data.Subset(BS_LS_test_dataset)

    data_loader = DataLoader(data_set, batch_size=100)

    with torch.no_grad():
        ### collect the embeddings for subsequent clustering measurement
        z1_list = []
        z2_list = []
        z12_list = []
        labels_list = []
        view1_shared_list = []
        view2_shared_list = []
        view1_specific_list = []
        view2_specific_list = []

        for view1_data, view2_data, labels in data_loader:
            view1_data = view1_data.type(torch.float32).to('cuda')
            view2_data = view2_data.type(torch.float32).to('cuda')
            labels = labels.type(torch.LongTensor).to('cuda')

            if is_SS:## if SS model, expect z1, z2, z12
                z1, z2, z12, s1_out, s2_out, labels = \
                    saved_model(view1_data, view2_data, labels)

                z1_list.append(z1.cpu().detach().numpy())
                z2_list.append(z2.cpu().detach().numpy())

                z12_list.append(z12.cpu().detach().numpy())

                labels_list.append(labels.cpu().detach().numpy())

            elif is_MOCSS: ### if MOCSS
                view1_specific_em, view1_shared_em, view2_specific_em, view2_shared_em, \
                view1_specific_rec, view1_shared_rec, view2_specific_rec, view2_shared_rec, \
                view1_shared_mlp, view2_shared_mlp, train_labels = saved_model(view1_data, view2_data, labels)

                view1_shared_list.append(view1_shared_em.cpu().detach().numpy())
                view2_shared_list.append(view2_shared_em.cpu().detach().numpy())
                view1_specific_list.append(view1_specific_em.cpu().detach().numpy())
                view2_specific_list.append(view2_specific_em.cpu().detach().numpy())
                labels_list.append(labels.cpu().detach().numpy())
            else:
                z12, s1_out, s2_out, labels = \
                    saved_model(view1_data, view2_data, labels)

                z12_list.append(z12.cpu().detach().numpy())

                labels_list.append(labels.cpu().detach().numpy())

        if is_SS:
            z1_all = np.concatenate(z1_list)
            z2_all = np.concatenate(z2_list)
            z12_all = np.concatenate(z12_list)
            labels_all = np.concatenate(labels_list)

            final_embedding = np.concatenate((z1_all, z2_all, z12_all), axis=1)

        elif is_MOCSS:
            view1_shared_em_all = np.concatenate(view1_shared_list)
            view2_shared_em_all = np.concatenate(view2_shared_list)

            view1_specific_em_all = np.concatenate(view1_specific_list)
            view2_specific_em_all = np.concatenate(view2_specific_list)
            labels_all = np.concatenate(labels_list)

            view_shared_common = (view1_shared_em_all + view2_shared_em_all) / 2

            final_embedding = np.concatenate((view1_specific_em_all, view2_specific_em_all, \
                                              view_shared_common), axis=1)

        else:
            z12_all = np.concatenate(z12_list)
            labels_all = np.concatenate(labels_list)

            final_embedding = z12_all

    return final_embedding, labels_all