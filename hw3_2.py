import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
import sys
import mpl_toolkits.mplot3d 
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import cross_validation,svm, datasets, ensemble, linear_model



# Define function to load data
def load_data(filename):
	with open(filename, 'r') as f:
		data = [row for row in csv.reader(f.read().splitlines())]

	data.remove(['A','B','label'])
	return data


# Load the data and store them in corresponding lists
data = load_data("chessboard.csv")
A = []
B = []
L = []
for row in data:
    A.append(float(row[0]))
    B.append(float(row[1]))
    L.append(int(row[2]))
# Record the number of item in the data
data_len = len(data)

# Plot the data, red for label 0, blue for label 1
for i in range(data_len):
    if L[i] == 0:
        plt.plot(A[i],B[i],"ro")
    else:
        plt.plot(A[i],B[i],"bo")

plt.show()

# Perform stratified sampling, split data into "train" and "set"
sss = StratifiedShuffleSplit(L, n_iter=1, test_size=0.4, random_state=1)
train = []
test = []

for train_index, test_index in sss:
    for index in train_index:
        train.append(data[index])
    for index in test_index:
        test.append(data[index])

# Constructing test data, these data will be used later when scoring
test_A = []
test_B = []
test_L = []
for row in test:
    test_A.append(float(row[0]))
    test_B.append(float(row[1]))
    test_L.append(int(row[2]))

X_test = np.transpose(np.array([test_A,test_B]))
y_test = np.array(test_L)


# Perform k-fold, distribute data into different set
n_fold = 3
kf = cross_validation.KFold(len(train), n_folds=n_fold)

kf_A = [[] for x in range(n_fold)]
kf_B = [[] for x in range(n_fold)]
kf_L = [[] for x in range(n_fold)]
kf_A_valid = [[] for x in range(n_fold)]
kf_B_valid = [[] for x in range(n_fold)]
kf_L_valid = [[] for x in range(n_fold)]

fold = 0
for train_index, valid_index in kf:
    for index in train_index:
        kf_A[fold].append(float(train[index][0]))
        kf_B[fold].append(float(train[index][1]))
        kf_L[fold].append(int(train[index][2]))
    for index in valid_index:
        kf_A_valid[fold].append(float(train[index][0]))
        kf_B_valid[fold].append(float(train[index][1]))
        kf_L_valid[fold].append(int(train[index][2]))
    fold += 1

# This is a function for plotin the decision boundary
# It will plot test and train data also
def plot_boundary(result,X,y):

    # Step size in the mesh
    h = 0.02

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    Z = result.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot training and test points
    for i in range(data_len):
        if L[i] == 0:
            plt.plot(A[i],B[i],"ro")
        else:
            plt.plot(A[i],B[i],"bo")
    plt.xlabel('Feature A')
    plt.ylabel('Feature B')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()

# These code perform different algorithms
# Return the fitting object
def lin_svc(X,y):
    return svm.LinearSVC(C=C).fit(X, y)
def poly_svc(X,y):
    return svm.SVC(kernel='poly', degree=degree, C=C).fit(X, y)
def rbf_svc(X,y):
    return svm.SVC(kernel='rbf', gamma=10, C=C).fit(X, y)
def random_forest_classifier(X,y):
    RFC = ensemble.RandomForestClassifier(n_estimators=50).fit(X,y)
    return RFC.fit(X,y)
def logistic_regression(X,y):
    LR = linear_model.LogisticRegression()
    return LR.fit(X,y)
def perceptron(X,y):
    P = linear_model.Perceptron()
    return P.fit(X,y)
    

# Perform k-fold cross validation
def cross_validation(method):

    best_score = 0.0
    best_result= None

    for i in range(n_fold): 

        X = np.transpose(np.array([kf_A[i],kf_B[i]]))
        y = np.array(kf_L[i])
        X_valid = np.transpose([kf_A_valid[i], kf_B_valid[i]])
        y_valid = np.array(kf_L_valid[i])


        result = method(X,y)        
        current_score = result.score(X_valid, y_valid)

        # Print the score for this fold
        print "Validation score: " , current_score

        # Store the best result and best score
        # We will use this fold and test data to obtain the test score
        if current_score > best_score:
            best_result = result
            best_score = current_score

    print "Best Score:" , best_score
    
    # Plot the boundary with best result
    plot_boundary(best_result,X,y)
    print "Test score: " , best_result.score(X_test, y_test) 


C = 2.0  # SVM regularization parameter
degree = 6 # Polynomial degree
cross_validation(rbf_svc)
cross_validation(lin_svc)
cross_validation(poly_svc)
cross_validation(logistic_regression)
cross_validation(random_forest_classifier)
cross_validation(perceptron)









