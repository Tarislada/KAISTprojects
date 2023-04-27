import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics as mt
import matplotlib.pyplot as plt

# needed files - visualization_sup, y_labels
# for checking # of superclusters: if unique<4, exclude
# col 1 = sample id
# col 2,3 = embedding 9x,y),
# col 4,5 = supercluster id?
# treat y_label as cluster label and silhouette away

# path = 'C:/Users/endyd/OneDrive/Onedrive-CK/OneDrive/Gradschool/Kaist/mingle-20230426T103005Z-001/mingle'

path = 'C:/Users/endyd/OneDrive/Onedrive-CK/OneDrive/Gradschool/Kaist/mingle_s-20230426T135558Z-001/mingle_s'
dfs = {}

# Loop through all the subdirectories in the specified path
for subdir, dirs, files in os.walk(path):

    # For the files in a mouse
    for dir in dirs:
        # Load human labels
        hlabels = pd.read_csv(os.path.join(subdir, dir,'ylabels.csv'),header=None,names=['human_label'])
        # If human labels does not contain all four labels, skip
        if hlabels['human_label'].nunique() < 4:
            print(f"No mount or No attack. Skipping file for {dir}.")
            continue
        # If all labels are present, load coordinates and concatenate with human labels
        coord = pd.read_csv(os.path.join(subdir, dir,'embeddings.csv'),header=None,names=['x','y'])
        df = pd.concat([coord,hlabels],axis=1)

        # Assign an index to the dataframe with the mouse number
        idx = os.path.basename(dir)

        # Add the dataframe to the dictionary using the mouse number as the key
        dfs[idx] = df

print(f'Done extracting attack & mount cases. {dfs.keys()} have both attack and mount')

# Replace labels
label_dict = {'other': 0, 'investigation': 1, 'mount': 2, 'attack': 3}

# Silhouette score calculation
mean_sil_scores = {}
sil_score = {}
figures = {}

# For all mouse data
for key, df in dfs.items():
    # Change label to integers
    df['label'] = df['human_label'].map(label_dict)
    # Calculate silhouette scores
    meansil = mt.silhouette_score(df[['x', 'y']], df['label'])
    samplesil = mt.silhouette_samples(df[['x', 'y']], df['label'])
    mean_sil_scores[key] = meansil
    sil_score[key] = samplesil

    # Convert the silhouette samples and labels to a dataframe
    tmp_silhouette_df = pd.DataFrame({'silhouette_score': samplesil,
                                  'label': df['label']})

# Create a silhouette plot
    silhouette_val_list = tmp_silhouette_df['silhouette_score']
    labels = tmp_silhouette_df['label']
    uniq_labels = np.unique(labels)
    sorted_cluster_svl = []
    rearr_labels = []
    for ul in uniq_labels:
        labels_idx = np.where(labels == ul)[0]
        target_svl = silhouette_val_list[labels_idx]
        sorted_cluster_svl += sorted(target_svl)
        rearr_labels += [ul] * len(target_svl)

    colors = sns.color_palette('hls', len(uniq_labels))
    color_labels = [colors[i] for i in rearr_labels]

    fig = plt.figure(figsize=(6, 10))
    fig.set_facecolor('white')
    plt.barh(range(len(sorted_cluster_svl)), sorted_cluster_svl, color=color_labels)
    plt.ylabel('Data Index')
    plt.xlabel('Silhouette Value')
    figures[key] = plt
dfs
