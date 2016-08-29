import vp_tree
import time

print("\n Test for euclidean metric #1. : \n")

mat = [[1.,3.],[2.,4.],[3.,5.],[4.,9.],[5.,7.]];
epsilon = 5.0
vp = [4.,0.]
# print vp_tree.list_double_vec_list_test(mat); ## a test to see converting works.

print("Data are: " + str(mat) + " .")
print("Epsilon cutoff is " + str(epsilon) + " selected vantage point is " + str(vp) + ".")

t0 = time.time()
tree = vp_tree.tree_container(mat)
t1 = time.time()
s = tree.print_tree() ## print tree contents.
t2 = time.time()
close_points = tree.find_within_epsilon(vp,epsilon);
t3 = time.time()

print("making tree", t1-t0)
print("making tree print statement",t2-t1)
print("finding near epsilon", t3-t2)
