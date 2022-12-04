
import numpy as np
import sys


def load_tsv_dataset(file):
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8')
    y = dataset[:, [0]]
    x = dataset[:, 1:]
    return y, x


def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(X, y, num_epoch, learning_rate):
    # TODO: Implement `train` using vectorization
    theta = np.zeros((X.shape[1] + 1, 1))
    num = y.size
    fold_x = np.concatenate((np.ones((1, num)), X.T), axis=0, dtype=float)
    adjust_y = y.T
    for e in range(int(num_epoch)):
        for i in range(fold_x.shape[1]):
            x_i = fold_x[:, [i]]
            y_i = adjust_y[:, [i]]
            prob_i = sigmoid(np.dot(theta.T, x_i))
            w = (prob_i - y_i)*x_i
            theta -= float(learning_rate) * w
    return theta


def predict(theta, X):
    # TODO: Implement `predict` using vectorization
    num = X.shape[0]
    x = np.concatenate((np.ones((1, num)), X.T), axis=0, dtype=float)
    prob_y = sigmoid(np.dot(theta.T, x))
    prob_y[prob_y >= 0.5] = 1
    prob_y[prob_y < 0.5] = 0
    return prob_y.T


def compute_error(y_pred, y):
    # TODO: Implement `compute_error` using vectorization
    num = y.size
    error = np.sum(y_pred != y) / num
    return "{:.6f}".format(error)


def write_output(outfile_path, y_pred):
    with open(outfile_path, "w") as outfile:
        for i in y_pred:
            outfile.write(str(int(i[0])))
            outfile.write('\n')


def write_rate(output_file, etrain, etest):
    with open(output_file, 'w') as f_out:
        f_out.write("error(train): " + str(etrain))
        f_out.write("\n")
        f_out.write("error(test): " + str(etest))


train_input = sys.argv[1]
val_input = sys.argv[2]
test_input = sys.argv[3]
train_out = sys.argv[4]
test_out = sys.argv[5]
metrics_out = sys.argv[6]
epoch = sys.argv[7]
alpha = sys.argv[8]

format_train_y, format_train_X = load_tsv_dataset(train_input)
theta_train = train(format_train_X, format_train_y, epoch, alpha)

format_test_y, format_test_X = load_tsv_dataset(test_input)
predict_train = predict(theta_train, format_train_X)
write_output(train_out, predict_train)
predict_test = predict(theta_train, format_test_X)
write_output(test_out, predict_test)

train_error = compute_error(predict_train, format_train_y)
test_error = compute_error(predict_test, format_test_y)

write_rate(metrics_out, train_error, test_error)


