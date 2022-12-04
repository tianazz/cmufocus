import argparse
import numpy as np


def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()
    
    Where above the arguments have the following types:

        train_data --> A list of training examples, where each training example is a list
            of tuples train_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        init_out --> A file path to which you should write your initial probabilities

        emit_out --> A file path to which you should write your emission probabilities

        trans_out --> A file path to which you should write your transition probabilities
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmmprior", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)

    args = parser.parse_args()

    train_data = list()
    with open(args.train_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            train_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    return train_data, words_to_indices, tags_to_indices, args.hmmprior, args.hmmemit, args.hmmtrans


if __name__ == "__main__":
    # Collect the input data
    train_data, words_to_indices, tags_to_indices, hmmprior, hmmemit, hmmtrans = get_inputs()
    # get the number of words and tags in the dataset
    num_word = len(words_to_indices)
    num_tag = len(tags_to_indices)

    # Initialize the initial, emission, and transition matrices
    init = np.zeros(num_tag)
    transition = np.zeros((num_tag, num_tag))
    emission = np.zeros((num_tag, num_word))
    # Increment the matrices
    for seq in train_data[:1000]:
        init[tags_to_indices.get(seq[0][1])] += 1
        for i in range(len(seq)):
            if i < len(seq) - 1:
                transition[tags_to_indices.get(seq[i][1]), tags_to_indices.get(seq[i+1][1])] += 1
            emission[tags_to_indices.get(seq[i][1]), words_to_indices.get(seq[i][0])] += 1
    # Add a pseudocount
    init += 1
    transition += 1
    emission += 1
    # normalize
    initi = init / np.sum(init)
    transition /= np.sum(transition, axis=1)[:, None]
    emission /= np.sum(emission, axis=1)[:, None]

    # Save your matrices to the output files --- the reference solution uses 
    # np.savetxt (specify delimiter="\t" for the matrices)
    # change to =' '
    np.savetxt(hmmprior, initi, fmt='%.18e', delimiter=' ')
    np.savetxt(hmmtrans,transition, fmt='%.18e', delimiter=' ')
    np.savetxt(hmmemit, emission, fmt='%.18e', delimiter=' ')
    pass



