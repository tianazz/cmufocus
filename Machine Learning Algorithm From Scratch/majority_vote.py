import statistics
import sys
import numpy as np


def read_input(file):
    content = np.genfromtxt(file, dtype=str, delimiter="\t")
    return content


def write_output(output_file, nrows, value):
    with open(output_file, 'w') as f_out:
        for i in range(nrows):
            f_out.write(value)
            f_out.write("\n")





# Get all arguments from command line. Parse every argument
train_input = sys.argv[1]
test_input = sys.argv[2]
train_output = sys.argv[3]
test_output = sys.argv[4]
metrics = sys.argv[5]

dtrain = read_input(train_input)
dtest = read_input(test_input)

# Train data without the titles
data_train = dtrain[1:]
# Test data without the titles
data_test = dtest[1:]


# Get the actual labels of data
def label_list(data):
    d = data.tolist()
    labels = []
    for n in d:
        labels.append(n[-1])
    return labels


train_label_list = label_list(data_train)
test_label_list = label_list(data_test)


# Function of finding the majority value
def train(input_list):
    ans = statistics.multimode(input_list)
    # citation https://www.delftstack.com/howto/python/mode-of-list-in-python/
    ans.sort(reverse=True)
    return ans[0]


# Get the majority value of the train data
majority_value_train = train(train_label_list)


# Apply the majority value to the test data
def test(input_list):
    nrow = len(input_list)
    prediction = []
    for i in range(nrow):
        prediction.append(majority_value_train)
    return prediction


train_prediction = test(train_label_list)
test_prediction = test(test_label_list)
test_row = len(test_prediction)
train_row = len(train_prediction)


def error_rate(label, value):
    nerror = 0
    for l in label:
        if l != value:
            nerror += 1

    rate = nerror / len(label)

    return rate


train_error = error_rate(train_label_list, majority_value_train)
test_error = error_rate(test_label_list, majority_value_train)

write_output(test_output, test_row, majority_value_train)
write_output(train_output, train_row, majority_value_train)
write_rate(metrics, train_error, test_error)
