import vp_tree

mat = [[1.,3.],[2.,4.],[3.,5.],[4.,9.],[5.,7.]];
# print vp_tree.list_double_vec_list_test(mat); ## a test
tree = vp_tree.tree_container(mat)
print tree.print_tree() ## print tree contents.
print tree.find_within_epsilon([4.,0.],5.); 
