import os
import re
from string import punctuation
import pandas as pd
from nltk.stem import SnowballStemmer


INPUT_DIR = os.path.join(os.getcwd(), 'data', 'interim')
OUTPUT_DIR = os.path.join(os.getcwd(), 'data', 'processed')


def main():
    fname = os.path.join(INPUT_DIR, 'all_manifestos.csv')
    df = pd.read_csv(fname)

    with open(os.path.join(OUTPUT_DIR, 'manifestos.csv'), 'w') as outfile:
        outfile.write('id_perm,text\n')
        for _, row in df.iterrows():
            outfile.write(str(row['id_perm']) + ',')
            manifesto = process_text(row['text'])
            manifesto = manifesto.replace('"', '""')
            outfile.write('"' + manifesto + '"\n')


def process_text(s: str) -> str:
    # make lowercase
    s = s.lower()

    # remove numbers
    s = re.sub(r'[0-9]+(?:\.[0-9]+)?', '', s)

    # remove punctuation
    for ch in punctuation:
        s = s.replace(ch, '')

    # change encoding to ascii and remove unicode characters
    # s = s.encode(encoding='ascii', errors='ignore').decode()

    # remove tabs and multiple spaces
    s = re.sub(r'\s+', ' ', s)  # done in stemmer

    stemmer = SnowballStemmer('english', ignore_stopwords=False)
    s = ' '.join(stemmer.stem(word) for word in re.split(r'\s+', s))

    return s


if __name__ == '__main__':
    main()
