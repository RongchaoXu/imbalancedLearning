import random
from sklearn.linear_model import LogisticRegression
from dataset import get_dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import *
from imblearn.under_sampling import *


def get_model(solver='newton-cg', max_iter=1000):
    return LogisticRegression(solver=solver, max_iter=max_iter, class_weight={1: 1, 2: 8})


def train_model(x_train, y_train, x_test, y_test, model):
    # add strategy
    # sm = NearMiss()
    # x_train, y_train = sm.fit_resample(x_train, y_train)
    # print(x_train.shape, y_train.shape)
    model.fit(x_train, y_train)
    pred_labels = model.predict(x_test)
    acc = accuracy_score(y_test, pred_labels)
    f1 = f1_score(y_test, pred_labels, average='macro')
    recall = recall_score(y_test, pred_labels, pos_label=2)
    return acc, f1, recall


def unbalanced_learning(dataset, model):
    x_train = dataset[0]
    y_train = dataset[1]
    x_test = dataset[2]
    y_test = dataset[3]
    return train_model(x_train, y_train, x_test, y_test, model)


if __name__ == '__main__':
    dataset = get_dataset('./Data for Problem 3/X_train_digits.mat', './Data for Problem 3/X_test_digits.mat',
                          './Data for Problem 3/y_train_digits.mat', './Data for Problem 3/y_test_digits.mat')
    # train_model(datasets[0], datasets[1], datasets[2], datasets[3], get_model())
    result = unbalanced_learning(dataset, get_model())
    print(result)
