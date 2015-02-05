import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
import sys
import mpl_toolkits.mplot3d

# Define function to load data
def load_data(filename):

	# Read in the data
	with open(filename, 'r') as f:
		data = [row for row in csv.reader(f.read().splitlines())]
	return data

# Load the validation data
# Store in "ages_valid" and "height_valid" lists
data = load_data("girls_2_20_validation.csv")
data_len_valid = len(data)
ages_valid = []
heights_valid = []
for row in data:
	ages_valid.append(float(row[0]))
	heights_valid.append(float(row[1]))

# Load the trainig data
# Store in "ages" and "heights" lists
data = load_data("girls_2_20_train.csv")
data_len = len(data)
ages = []
heights = []
for row in data:
	ages.append(float(row[0]))
	heights.append(float(row[1]))

# Plot the training data
plt.plot(ages,heights,"o")
plt.xlabel("age")
plt.ylabel("heights")

# Set up number of degree, since d = 0 - 5, degree = 6
degree = 6
# This is a list store list of "ages" with 0-6 degrees
# The structure looks like the following
# 1 x1 x1^2...
# 1	x2 x2^2...
# 1	x3 x3^2...
# 1	x4 x4^2...
large_X = []
large_X.append([1]*data_len)

for d in range(degree - 1):
	large_X.append([a*b for a,b in zip(large_X[d], ages)])
print large_X

# Use a list MSE to store mean square error
MSE_train = [[]]*degree
MSE_valid = [[]]*degree

# Calculate normal equation solution for 6 different degrees

for d in range(degree):
	XT = np.array(large_X[:d+1])
	X = np.transpose(XT)
	yT = np.array(heights)
	y = np.transpose(yT)

	beta = np.dot(np.dot(np.linalg.inv(np.dot(XT,X)),XT),y)
	print "degree = " , d , " , beta by normal equation: " , beta

	# Now start to plot the fitting polynomial
	# Use a step size of 2
	x = np.arange(0,21,0.2)

	# For every degree, add polydomial term from 0 to d
	# y is the predicted curve for this degree
	y = 0
	for i in range(d + 1):
		y += (x ** i) * beta[i]

	# Now calculate mean square error for training set
	error = 0
	for i in range(data_len):
		predict = 0
		for j in range(d + 1):
			predict += (ages[i] ** j) * beta[j]
		error += (heights[i] - predict) ** 2
	error /= data_len
	MSE_train[d] = error

	# Now calculate mean square error for validation set
	# Just replace all data with validation data
	error = 0
	for i in range(data_len_valid):
		predict = 0
		for j in range(d + 1):
			predict += (ages_valid[i] ** j) * beta[j]
		error += (heights_valid[i] - predict) ** 2
	error /= data_len_valid
	MSE_valid[d] = error

	# Plot predicted curve and the data
	plt.plot(ages,heights,"o") # This line plot the data
	plt.xlabel("age")
	plt.ylabel("heights")
	plt.xlim(0,20)
	plt.ylim(0,200)	
	plt.plot(x,y) # This line plot the prediction curve
	plt.show()

# Print train MSE and validation MSE
print MSE_train
print MSE_valid

# Plot training error and validation error
plt.plot(range(6),MSE_train)
plt.plot(range(6),MSE_valid)
plt.xlim(0,5)
plt.show()








# 




