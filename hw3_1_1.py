import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
import sys
import mpl_toolkits.mplot3d 

# Define function to load data
def load_data(filename):
	
	# Read in the data to list "ages" and "heights" 
	result = csv.reader(open(filename), delimiter="\t")	
	ages = []
	heights = []

	for row in result:
		[age, weight]  = [float(row[0].split(',')[0]), float(row[0].split(',')[1])]
		ages.append(age)
		heights.append(weight)

	return [ages, heights]


# Load training data & plot
[ages, heights] = load_data("girls_train.csv")
plt.plot(ages,heights,"o")
plt.xlabel("age")
plt.ylabel("weight")
plt.show()

# Start gradient descendent
# Setup initial value for parameter
data_len = len(ages)
beta = [0,0]
alpha = 0.05
iteration = 1500

# Perform gradient descent with 1500 iteration
for it in range(iteration):

	slope = [0,0]

	for i in range(data_len):
		slope[0] += (beta[0] + beta[1]*ages[i] - heights[i]) * 1 
		slope[1] += (beta[0] + beta[1]*ages[i] - heights[i]) * ages[i]

	slope[0] /= data_len
	slope[1] /= data_len

	beta[0] -= alpha * slope[0]
	beta[1] -= alpha * slope[1]


# Plot the function with beta[0], beta[1]
plt.plot(ages,heights,"o")
plt.xlabel("age")
plt.ylabel("weight")
x = np.array(range(2,10))
y = eval("x*beta[1] + beta[0]")
plt.plot(x,y)
plt.show()
print "Beta: " , beta

# Calculate mean square error
MSE = 0
for i in range(data_len):
	MSE += (heights[i] - beta[1]*ages[i] - beta[0]) ** 2
MSE /= data_len
print "Mean Square Error for training data: " , MSE


# Define cost function to plot
def cost_func(beta):
	cost = 0
	for i in range(data_len):
		cost += (heights[i] - beta[1]*ages[i] - beta[0]) ** 2
	cost /= 2*data_len
	return cost

# Plot the cost function 
# Use x,y to represent beta 0 and beta 1 avoiding replace beta
[x,y] = np.mgrid[-2:2:500j, -2:2:500j]
cost = cost_func([x,y])
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x,y,cost)
plt.show()

# Predict height for 4.5-year-old girl
answer = beta[0] + beta[1]*4.5
print "Predicted height for 4.5-year-old girl: " , answer

# Load test data and calculate mean square error (MSE)
[ages, heights] = load_data("girls_test.csv")
data_len = len(ages)
MSE = 0
for i in range(data_len):
	MSE += (heights[i] - beta[1]*ages[i] - beta[0]) ** 2
MSE /= data_len
print "Mean Square Error for testing data: " , MSE










