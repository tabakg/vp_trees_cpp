import vp_tree
import time
import numpy as np
from numpy.random import normal
import random

dim = 1
data_size = 1000

mu = 0.0
sigma = 1.0
epsilon = 3.

NN = 50

print("\n Test for euclidean metric #2. : \n")

print("Data are " + str(data_size) + " points of dimension " + str(dim) + " .")
print("Gaussian points with mu = " + str(mu) + ", sigma = "+str(sigma)+"; cutoff epsilon = "+str(epsilon)+" \n")

data = np.reshape(np.array([random.normalvariate(mu,sigma) for i in range(data_size*dim)]),(data_size,dim) ).tolist()

t0 = time.time()
tree = vp_tree.tree_container(data)
t1 = time.time()
s = tree.print_tree() ## print tree contents.
t2 = time.time()
close_points = tree.find_within_epsilon([0.]*dim,epsilon)
t3 = time.time()

ave_num_nbrs = 0
for point in data:
    ave_num_nbrs += len( tree.find_within_epsilon(point,epsilon) )
print("total neighbors found: " +  str(ave_num_nbrs) )
ave_num_nbrs /= data_size
print ("average number of neighbors is " + str(ave_num_nbrs) )

t4 = time.time()


### nearest_neighbrs to zero
# point = [0.]*dim
# nearest_neighbrs = tree.find_N_neighbors(point,NN)

### nearest_neighbrs to each point.
# for point in data:
#     nearest_neighbrs = tree.find_N_neighbors(point,NN)

# print nearest_neighbrs[:100]

t5 = time.time()

print("making tree", t1-t0)
print("making tree print statement",t2-t1)
print("finding near epsilon around zero", t3-t2)
print("finding near epsilon for all points", t4-t3)
print("finding " + str(NN) + " nearest neighbors", t5-t4)
