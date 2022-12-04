import csv
import sys
import numpy as np

VECTOR_LEN = 300   # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map


def process(infile, word2vec_map, outfile_path):
    with open(outfile_path, "w") as outfile:
        for line in infile:
            label, sentence = line[0], line[1]
            feature_sum = np.zeros((1, 300))
            word_count = 0
            for word in sentence.split(" "):
                if word in word2vec_map.keys():
                    feature_sum += word2vec_map[word]
                    word_count += 1
                if word_count == 0:
                    continue
            feature_vec = (feature_sum / word_count)
            outline = "{:.6f}".format(label)
            for feature in feature_vec[0]:
                outline += "\t"
                outline += str("{:.6f}".format(feature))
            outline += "\n"
            outfile.write(outline)


train_input = sys.argv[1]
val_input = sys.argv[2]
test_input = sys.argv[3]
wordmap = sys.argv[4]
train_out = sys.argv[5]
val_out = sys.argv[6]
test_out = sys.argv[7]

train_file = load_tsv_dataset(train_input)
val_file = load_tsv_dataset(val_input)
test_file = load_tsv_dataset(test_input)
word_embedding_map = load_feature_dictionary(wordmap)
process(train_file, word_embedding_map, train_out)
process(val_file, word_embedding_map, val_out)
process(test_file, word_embedding_map, test_out)


