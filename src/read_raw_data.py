import os
import pandas as pd


DATASET_PATH = os.path.join(os.getcwd(), 'data', 'MPDataset_MPDS2021a.csv')
INPUT_PATH = os.path.join(os.getcwd(), 'data', 'raw')
OUTPUT_PATH = os.path.join(os.getcwd(), 'data', 'interim')


def main():
    # open "main" dataset to get information about every manuscript
    df_main = pd.read_csv(DATASET_PATH)

    # create directories for output path if needed
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # manifestos are grouped by country
    for country in os.listdir(INPUT_PATH):
        cur_path = os.path.join(INPUT_PATH, country)

        out_fname = os.path.join(OUTPUT_PATH, country + '.csv')
        with open(out_fname, 'w', encoding='utf-8') as outfile:
            # header
            columns = ('date', 'party', 'id_perm', 'rile',
                       'markeco', 'welfare', 'intpeace', 'text')
            outfile.write(','.join(columns) + '\n')

            # iterate over documents
            for fname in os.listdir(cur_path):
                if not fname.endswith('.csv'):
                    continue
                party_id, date = map(int, fname[:-4].split('_'))
                full_path = os.path.join(cur_path, fname)
                df = pd.read_csv(full_path, quotechar='"', encoding='utf-8')
                manifesto = ' '.join(df['text'])
                manifesto = manifesto.replace('"', '""')
                manifesto = manifesto.replace('\n', ' ')

                # find entry in the main dataset
                row = df_main[(df_main['date'] == date) &
                              (df_main['party'] == party_id)]
                assert row.shape[0] == 1

                # write to new csv file
                values = [str(row[col].values[0]) for col in columns[:-1]]
                outfile.write(','.join(values))
                outfile.write(',"' + manifesto + '"\n')


if __name__ == '__main__':
    main()
