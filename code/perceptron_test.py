import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Perceptron import Perceptron


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy

def converter(y):
    y_ = np.array([1 if i=='satisfied' else 0 for i in y])
    return y_

def value_converter(csv):
    csv['Gender'] = csv['Gender'].replace(['Male', 'Female'],[1, 0])
    csv['Customer Type'] = csv['Customer Type'].replace(['Loyal Customer','disloyal Customer'],[1, 0])
    csv['Type of Travel'] = csv['Type of Travel'].replace(['Business travel', 'Personal Travel'],[1, 0])
    csv['Class'] = csv['Class'].replace(['Business', 'Eco Plus', 'Eco'],[2, 1, 0])
    csv_ = csv
    return csv_



train_csv = pd.read_csv('./data/train.csv')
y_train = train_csv['satisfaction']
test_csv = pd.read_csv('./data/test.csv')
y_test = test_csv['satisfaction']

train_csv = value_converter(train_csv)
train_csv.rename(columns=train_csv.iloc[0]).drop(train_csv.index[0])
train_csv.drop(train_csv.columns[0], axis=1, inplace=True)
train_csv.drop(train_csv.columns[len(train_csv.columns)-1], axis=1, inplace=True)
X_train = train_csv.to_numpy()
# print(X_train)

test_csv = value_converter(test_csv)
test_csv.rename(columns=test_csv.iloc[0]).drop(test_csv.index[0])
test_csv.drop(test_csv.columns[0], axis=1, inplace=True)
test_csv.drop(test_csv.columns[len(test_csv.columns)-1], axis=1, inplace=True)
X_test = test_csv.to_numpy()

y_train = converter(y_train)
y_test = converter(y_test)


p = Perceptron(learning_rate=0.01, n_iters=1)
p.fit(X_train, y_train)
predictions = p.predict(X_test)


print("Perceptron classification accuracy", accuracy(y_test, predictions))



# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# plt.scatter(X_train[:,0], X_train[:,1], marker='o', c=y_train)
#-------------------------------------------------------------------------------------------------------------------------------------
                                                # Code
#-------------------------------------------------------------------------------------------------------------------------------------

# X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05,random_state=2)
# X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# p = Perceptron(learning_rate=0.01, n_iters=1000)
# print(type(X_train))
# p.fit(X_train, y_train)
# predictions = p.predict(X_test)

# print("Perceptron classification accuracy", accuracy(y_test, predictions))

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# plt.scatter(X_train[:,0], X_train[:,1], marker='o', c=y_train)  

# x0_1 = np.amin(X_train[:,0])
# x0_2 = np.amax(X_train[:,0])

# x1_1 = (-p.weights[0] * x0_1 - p.bias)/p.weights[1]
# x1_2 = (-p.weights[0] * x0_2 - p.bias)/p.weights[1]


# ax.plot([x0_1, x0_2],[x1_1, x1_2],'k')


# ymin = np.amin(X_train[:,1])
# ymax = np.amax(X_train[:,1])

# ax.set_ylim([ymin-3, ymax+3])

# plt.show()