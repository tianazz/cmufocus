import statistics
import sys
import numpy as np


class Node:
    """
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    """
    data = []

    def __init__(self, data, attr):
        self.left = None
        self.right = None
        self.attr = attr
        self.data = data


if __name__ == '__main__':
    pass


def majority_value(data):
    input_list = [i[-1] for i in data]
    ans = statistics.multimode(input_list)
    ans.sort(reverse=True)
    return ans[0]


# calculate the entropy of the label
def cal_entropy(dataset):
    label = [i[-1] for i in dataset]
    freq = {}
    entropy = 0
    for i in label:
        if i in freq:
            freq[i] += 1
        else:
            freq[i] = 1
    for key in freq:
        entropy -= freq[key] / len(label) * np.log2(freq[key] / len(label))
    return entropy


# calculate the mutual info
def mutual_info(dataset, attr_index):
    # H(Y)
    entro = cal_entropy(dataset)

    # the attribute list
    attr = [i[attr_index] for i in dataset]
    label = [i[-1] for i in dataset]
    # distinct values from the attribute
    attr_value = list(set(attr))
    # will have 1/0 values

    cnt_label = len(label)
    conditional_entro = 0
    for i in attr_value:
        # get the H(Y]A = ...)
        data_hy_a = []
        for j in range(len(label)):
            if attr[j] == i:
                data_hy_a.append(dataset[j])
        cnt_hy_a = len(data_hy_a)
        conditional_entro = conditional_entro + (cnt_hy_a / cnt_label) * cal_entropy(data_hy_a)

    mutual_information = entro - conditional_entro
    return mutual_information


res = []


# split the data by the attribute chosen by the highest information gain
def split_data(data):
    label = [i[-1] for i in data]
    if label.count(label[0]) == len(label):
        return None
    info_gain = []

    for i in range(len(data[0]) - 1):
        info_gain.append(mutual_info(data, i))
    if len(info_gain) == 0:
        return None
    max_attr = max(info_gain)

    if max_attr > 0:
        attr_index = info_gain.index(max_attr)
        data = np.array(data)
        data1 = data[data[:, attr_index] == '1']
        data_0 = data[data[:, attr_index] == '0']
        data1 = np.delete(data1, attr_index, axis=1)
        data0 = np.delete(data_0, attr_index, axis=1)

        return attr_index, data1, data0
    else:
        return None


def tree_recursive(node, maxd, d, attr):
    if len(node.data) == 0:
        return
    # the case where max-depth is zero (i.e. a majority vote classifier)
    if maxd == 0:
        node.attr = majority_value(node.data)
        return
    # the case where information gain not > 0
    else:
        if split_data(node.data) is None or d + 1 > maxd:
            node.attr = majority_value(node.data)
            res.append(node.attr)
            return
        else:
            attr_index, data_1, data_0 = split_data(node.data)
            data_0 = data_0.tolist()
            data_1 = data_1.tolist()
            node.attr = attr[attr_index]
            res.append(node.attr)
            d += 1
            new_attr = [i for i in attr]
            new_attr.pop(attr_index)
            node.left = Node(data_1, None)
            tree_recursive(node.left, maxd, d, new_attr)
            node.right = Node(data_0, None)
            tree_recursive(node.right, maxd, d, new_attr)
            return


# Predict part1
def traverse_tree(row, node, attr):
    if node.left is None:
        return node.attr
    index = attr.index(node.attr)
    if row[index] == '1':
        return traverse_tree(row, node.left, attr)
    return traverse_tree(row, node.right, attr)


# Predict part2
# attr: the attributes list
def predict(data, node, attr):
    predict_label = []
    for i in data:
        predict_label.append(traverse_tree(i, node, attr))

    return predict_label


# Calculate the error rate
def error(label, pred_label):
    error_cnt = 0
    for i in range(len(label)):
        if label[i] != pred_label[i]:
            error_cnt = error_cnt + 1

    return error_cnt / len(label)


# read the file
def read_input(file):
    content = np.genfromtxt(file, dtype=str, delimiter="\t")
    return content


train_input = sys.argv[1]
test_input = sys.argv[2]
max_depth = sys.argv[3]
train_output = sys.argv[4]
test_output = sys.argv[5]
metrics = sys.argv[6]

ftrain = read_input(train_input)
ftest = read_input(test_input)

# Train data without the titles
data_train = ftrain[1:].tolist()

# Test data without the titles
data_test = ftest[1:].tolist()
# Train data attribute list
attr_train = ftrain[0].tolist()
# Test data attribute list
attr_test = ftest[0].tolist()
# Train data label list
train_label = [i[-1] for i in data_train]  # list
# Test data label list
test_label = [i[-1] for i in data_test]

# Grow the tree
mynode = Node(data_train, attr_train)
tree_recursive(mynode, int(max_depth), 0, attr_train)

# Predict the train data
predict_train_label = predict(data_train, mynode, attr_train)
# Predict the test data
predict_test_label = predict(data_test, mynode, attr_test)


def write_output(output_file, output_list):
    with open(output_file, 'w') as f_out:
        for i in output_list:
            f_out.write(i)
            f_out.write("\n")


def write_rate(output_file, etrain, etest):
    with open(output_file, 'w') as f_out:
        f_out.write("error(train): " + str(etrain))
        f_out.write("\n")
        f_out.write("error(test): " + str(etest))


print(res)

train_error = error(train_label, predict_train_label)
test_error = error(test_label, predict_test_label)
write_output(train_output, predict_train_label)
write_output(test_output, predict_test_label)
write_rate(metrics, train_error, test_error)


