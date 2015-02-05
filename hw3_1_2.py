# THe result for normal equation and gradient descent is different
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
import sys
import mpl_toolkits.mplot3d

# Define function to load data
def load_data(filename):

	# Read in the fata to lists "ages" and "heights"
	result = csv.reader(open(filename), delimiter="\t")
	ages = []
	weights = []
	heights = []

	for row in result:
		[age, weight, height] = [float(row[0].split(',')[0]), float(row[0].split(',')[1]), float(row[0].split(',')[2])]
		ages.append(age)
		weights.append(weight)
		heights.append(height)

	return [ages, weights, heights]
	
# Load data and scaling
[ages, weights, heights] = load_data("girls_age_weight_height_2_8.csv")
ages = [(x - np.mean(ages))/np.std(ages) for x in ages]
weights = [(x - np.mean(weights))/np.std(weights) for x in weights]
heights = [(x - np.mean(heights))/np.std(heights) for x in heights]

# Recore the number of items in the data, and use 50 iteration
data_len = len(ages)
iteration = 50

# Perform gradient descent and plot

# Since we have top perform gradient descent several times with different alpha
# We construct functions to do it
def cost_func(beta):
	
	cost = 0
	for i in range(data_len):
		cost += (heights[i] - beta[2]*weights[i] - beta[1]*ages[i] - beta[0]) ** 2
	cost /= 2*data_len
	return cost

# Function for gradient descendant
# It will also plot cost versus iteration graph, so that we can see the convergence rate
def grad_descent(alpha):

	global beta
	beta = [0,0,0]
	cost = []
	cost.append(cost_func(beta))

	for it in range(iteration):

		slope = [0,0,0]

		for i in range(data_len):
			slope[0] += (beta[0] + beta[1]*ages[i] + beta[2]*weights[i] - heights[i]) * 1
			slope[1] += (beta[0] + beta[1]*ages[i] + beta[2]*weights[i] - heights[i]) * ages[i]
			slope[2] += (beta[0] + beta[1]*ages[i] + beta[2]*weights[i] - heights[i]) * weights[i]

		slope = [x/data_len for x in slope]

		beta[0] -= alpha * slope[0]
		beta[1] -= alpha * slope[1]
		beta[2] -= alpha * slope[2]
		cost.append(cost_func(beta))

	plt.plot(cost)
	# Return the minimum cost to compare which alpha is best
	return np.amin(cost)


# Perform gradient descent for different alpha and find that has a minimum cost
alpha_set = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
min_cost = sys.maxint
best_alpha = None
beta = [0,0,0]

# Perform gradient descendant for different alpha, and record the best result
# Note, at the same time, the "grad_descent" function will also plot the cost curve for each alpha
for alpha in alpha_set:
	if grad_descent(alpha) < min_cost:
		best_alpha = alpha
print "The best alpha (with least cost) is: " , best_alpha
plt.show()

# Use best_alpha to predict the height for a 5-year old girl weighting 20 kilos 
# First reload the data and scale
# Note here we have to store the mean and std deviation to scale the data back
[ages, weights, heights] = load_data("girls_age_weight_height_2_8.csv")

# Store the std and mean
mu = [np.mean(ages), np.mean(weights), np.mean(heights)]
std = [np.std(ages), np.std(weights), np.std(heights)]

# Scale the data
# age, weight is a number, and ages, weights, heights are the entire data set
age = (5 - mu[0])/std[0]
weight = (20 - mu[1])/std[1]

ages = [(x - mu[0])/std[0] for x in ages]
weights = [(x - mu[1])/std[1] for x in weights]
heights = [(x - mu[2])/std[2] for x in heights]

# Perform gradient descendant for best alpha, and predict
grad_descent(best_alpha)
height = beta[0] + beta[1]*age + beta[2]*weight

# Scale back the data
height = height*std[2] + mu[2]
print "Beta by gradient descent: " , beta
print "The predicted height for a 5-year old girl weighting 20 kilos is: " , height 


# Now use perform linear regression by normal equation
# This time we do not need to scale the data
age = 5
weight = 20
[ages, weights, heights] = load_data("girls_age_weight_height_2_8.csv")

uni = np.ones(data_len)
XT = np.array([uni, ages, weights])
X = np.transpose(XT)
yT = np.array(heights)
y = np.transpose(yT)

beta = np.dot(np.dot(np.linalg.inv(np.dot(XT,X)),XT),y)
print "Beta by normal equation: " , beta
height = beta[0] + beta[1]*age + beta[2]*weight
print "The predicted height for a 5-year old girl weighting 20 kilos is: " , height 







