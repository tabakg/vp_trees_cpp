import vp_tree
import time

mat = [[1.,3.],[2.,4.],[3.,5.],[4.,9.],[5.,7.]];
# print vp_tree.list_double_vec_list_test(mat); ## a test

t0 = time.time()
tree = vp_tree.tree_container(mat)
t1 = time.time()
s = tree.print_tree() ## print tree contents.
t2 = time.time()
close_points = tree.find_within_epsilon([4.,0.],5.);
t3 = time.time()

print "making tree", t1-t0, "\n"
print "making tree print statement",t2-t1, "\n"
print "finding near epsilon", t3-t2, "\n"
