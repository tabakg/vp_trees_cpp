import vp_tree
import time
import numpy as np
from numpy import linalg as la
from numpy.random import normal
import random

dim = 2
data_size = 10

epsilon = 0.1

print "\n Test for FS metric: \n"

print("Data are " + str(data_size) + " points of dimension " + str(dim) + " .")
print("Points are normalized and have isotropic distribution; cutoff epsilon = " + str(epsilon) + ".\n")

data_not_normalized = np.reshape(np.array([random.normalvariate(0,1) for i in range(data_size*dim)]),(data_size,dim) ).tolist()

data = [ (point / la.norm(point)).tolist() for point in data_not_normalized ]

t0 = time.time()
tree = vp_tree.tree_container(data,"FS_metric")
t1 = time.time()
s = tree.print_tree() ## print tree contents.
t2 = time.time()
default_vec = [0.]*dim
default_vec[0] = 1
close_points = tree.find_within_epsilon(default_vec,epsilon,"FS_metric");
t3 = time.time()

ave_num_nbrs = 0
for point in data:
    ave_num_nbrs += len( tree.find_within_epsilon(point,epsilon,"FS_metric") )
print("total neighbors found: " +  str(ave_num_nbrs) )
ave_num_nbrs /= data_size
print ("average number of neighbors is " + str(ave_num_nbrs) )

t4 = time.time()

print("making tree", t1-t0)
print("making tree print statement",t2-t1)
print("finding near epsilon around zero", t3-t2)
print("finding near epsilon for all points", t4-t3)
