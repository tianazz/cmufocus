import argparse
import numpy as np

def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = parse_args()

    Where above the arguments have the following types:

        validation_data --> A list of validation examples, where each element is a list:
            validation_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        hmminit --> A np.ndarray matrix representing the initial probabilities

        hmmemit --> A np.ndarray matrix representing the emission probabilities

        hmmtrans --> A np.ndarray matrix representing the transition probabilities

        predicted_file --> A file path (string) to which you should write your predictions

        metric_file --> A file path (string) to which you should write your metrics
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("validation_data", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmminit", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)
    parser.add_argument("predicted_file", type=str)
    parser.add_argument("metric_file", type=str)

    args = parser.parse_args()

    validation_data = list()
    with open(args.validation_data, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            validation_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    hmminit = np.loadtxt(args.hmminit, dtype=float, delimiter=" ")
    hmmemit = np.loadtxt(args.hmmemit, dtype=float, delimiter=" ")
    hmmtrans = np.loadtxt(args.hmmtrans, dtype=float, delimiter=" ")

    return validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, args.predicted_file, args.metric_file

# You should implement a logsumexp function that takes in either a vector or matrix
# and performs the log-sum-exp trick on the vector, or on the rows of the matrix

# perform the log-sum-exp trick
# on the vector
def logsumexp(vector):
    m = np.max(vector)
    logsumexp_value = np.log(np.sum(np.exp(vector - m)))
    return logsumexp_value + m
# on the matrix:
def m_logsumexp(matrix):
    m = np.max(matrix, axis= 0)
    logsumexp_value = np.log(np.sum(np.exp(matrix - m[None, :]), axis=0))
    return logsumexp_value + m


def forwardbackward(seq, loginit, logtrans, logemit, tag_dict, word_index):
    """
    Your implementation of the forward-backward algorithm.

        seq is an input sequence, a list of words (represented as strings)

        loginit is a np.ndarray matrix containing the log of the initial matrix

        logtrans is a np.ndarray matrix containing the log of the transition matrix

        logemit is a np.ndarray matrix containing the log of the emission matrix
    
    You should compute the log-alpha and log-beta values and predict the tags for this sequence.
    """
    L = len(seq)
    M = len(loginit)

    # Initialize log_alpha and fill it in
    log_alpha = np.empty((M, L))
    log_alpha[:, 0] = loginit + logemit[:, word_index.get(seq[0][0])]
    for i in range(1, L):
        log_alpha[:, i] = logemit[:, word_index.get(seq[i][0])] + m_logsumexp(logtrans + log_alpha[:, i-1][:, None])

    # Initialize log_beta and fill it in
    log_beta = np.empty((M, L))
    # BT = 1
    log_beta[:, -1] = 0
    for i in range(L-2, -1, -1):
        log_beta[:, i] = m_logsumexp(logemit[:, word_index.get(seq[i+1][0])][:, None] + log_beta[:, i+1][:, None] + logtrans.T)

    # Compute the predicted tags for the sequence
    log_tag = log_alpha + log_beta
    tag_index = np.argmax(log_tag, axis=0)
    pred_tags = []
    for i in tag_index:
        for k, v in tag_dict.items():
            if v == i:
                pred_tags.append(k)

    # Compute the log-probability of the sequence
    log_prob = logsumexp(log_alpha[:, -1])
    # Return the predicted tags and the log-probability
    return pred_tags, log_prob
    pass
    

    
    
if __name__ == "__main__":
    # Get the input data
    validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = get_inputs()
    # log the matrices
    loginit = np.log(hmminit)
    logemit = np.log(hmmemit)
    logtrans = np.log(hmmtrans)
    # For each sequence, run forward_backward to get the predicted tags and 
    # the log-probability of that sequence.

    org_tags = []
    total_tags = []
    total_log_prob = []
    for i in validation_data:
        seq_tag, seq_lob_prob = forwardbackward(i,loginit,logtrans,logemit,tags_to_indices,words_to_indices)
        total_tags.append(seq_tag)
        total_log_prob.append(seq_lob_prob)
        for j in i:
            org_tags.append(j[1])


    # Compute the average log-likelihood and the accuracy. The average log-likelihood 
    # is just the average of the log-likelihood over all sequences. The accuracy is 
    # the total number of correct tags across all sequences divided by the total number 
    # of tags across all sequences.

    avg_log_prob = np.mean(total_log_prob)
    final_tags = []
    for i in total_tags:
        for j in i:
            final_tags.append(j)
    correct = 0
    for i in range(len(final_tags)):
        if final_tags[i] == org_tags[i]:
            correct += 1

    accuracy = correct/len(org_tags)


    # write output
    with open(predicted_file, "w") as f:
        for i in range(len(validation_data)):
            for j in range(len(validation_data[i])):
                f.write(validation_data[i][j][0] + "\t" + total_tags[i][j] + "\n")
            f.write("\n")

    with open(metric_file, "w") as f:
        f.write("Average Log-Likelihood: {:.16f}\nAccuracy: {:.16f}".format(avg_log_prob, accuracy))
    pass


