import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


INPUT_DIR = os.path.join(os.getcwd(), 'data', 'interim')
OUTPUT_DIR = os.path.join(os.getcwd(), 'data', 'processed')
COLUMNS = ('rile', 'markeco', 'welfare', 'intpeace')
RESCALE = True
N_CLUSTERS = 4


def main():
    # read data for clustering
    fname = os.path.join(INPUT_DIR, 'all_manifestos.csv')
    df = pd.read_csv(fname, quotechar='"', encoding='utf-8')
    X = np.array(df.loc[:, COLUMNS])
    if RESCALE:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # classify X with kmeans
    model = KMeans(n_clusters=N_CLUSTERS, init='k-means++')
    clusters = model.fit_predict(X)
    centers = model.cluster_centers_
    if RESCALE:
        centers = scaler.inverse_transform(centers)

    # construct DataFrame of cluster centers
    df_clusters = pd.DataFrame(data=centers, columns=COLUMNS)
    counts = [(clusters == i).sum() for i in range(N_CLUSTERS)]
    df_clusters['count'] = counts
    print(df_clusters)

    # describe clusters
    print('\nParties in different clusters:')
    describe_clusters(clusters, df['party'])

    # save cluster numbers to a separate file
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    df_out = pd.DataFrame({'cluster': clusters}, index=df['id_perm'])
    df_out.to_csv(os.path.join(OUTPUT_DIR, 'clusters.csv'))


def describe_clusters(labels, party_ids):
    parties_db_path = os.path.join(
        os.getcwd(),
        'data',
        'parties_MPDataset_MPDS2021a.csv'
    )
    df = pd.read_csv(parties_db_path)

    for label in sorted(np.unique(labels)):
        print(f'Cluster #{label}')
        counts = {}
        for party_id in party_ids[labels == label]:
            row = df[df['party'] == party_id]
            assert row.shape[0] == 1
            country = row['countryname'].values[0]
            party_name = row['name_english'].values[0]
            name = country + ': ' + party_name
            counts[name] = counts.get(name, 0) + 1

        for name in sorted(counts.keys()):
            print(f'{name} -- {counts[name]} manifestos')
        print(f'Total -- {sum(counts.values())} manifestos')
        print()


if __name__ == '__main__':
    main()
