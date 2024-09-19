from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def tsn_data(features, label):
    tsne = TSNE(n_components=2, verbose=1, init='pca')
    tsne_results = tsne.fit_transform(features)

    df = pd.DataFrame(features).copy()

    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]

    labels_dict = {'Primary Tumor*breast': 0, 'Primary Tumor*lung': 1,
                   'Primary Tumor*melanoma': 2, 'Primary Tumor*liver': 3,
                   'Primary Tumor*sarcoma': 4, 'Primary Tumor*kidney': 5}

    labels_dict_reverse = {v: k.replace('Primary Tumor*', '') for k, v in labels_dict.items()}

    ### add labels
    df['y'] = pd.Series(label)
    df['Primary Tumor'] = df['y'].apply(lambda X: labels_dict_reverse[X])

    return df


def tsn_plot(tsn_data):
    plt.figure(figsize=(16, 10))

    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="Primary Tumor",
        #         palette=sns.color_palette("Spectral", as_cmap=True),
        data=tsn_data,
        legend="full"
    )
    
    plt.xlabel('tsne-2d-one', fontsize=20)  # Adjust fontsize as needed
    plt.ylabel('tsne-2d-two', fontsize=20)
    plt.legend(fontsize=20)
    plt.show()