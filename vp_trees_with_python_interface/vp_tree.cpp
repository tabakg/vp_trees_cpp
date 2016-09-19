#include <boost/python.hpp> /* python interface */
#include <iostream>
#include <random>
#include <vector>
#include <math.h>       /* sqrt, acos */
#include <queue> /* used for nearest neighbors finding*/
#include <unordered_map> /* dictionary from ID to node pointer */
#include <unordered_set> /* tagging nodes */

typedef boost::python::list pylist;
typedef std::vector<double> vector;
typedef std::vector<vector> double_vec;

template<typename T>
pylist double_vec_to_pylist(T vec){
  pylist list;
  for (unsigned i = 0; i < vec.size(); i++) {
    pylist sublist;
    for (unsigned j = 0; j < vec.at(i).size(); j++){
      sublist.append(vec.at(i).at(j));
    }
    list.append(sublist);
  }
  return list;
}
double_vec pylist_to_double_vec(pylist list){
  unsigned size1 = len(list);
  unsigned size2 = len(list[0]);
  double_vec vec (size1, vector(size2) );
  for (unsigned i = 0; i < size1; ++ i) {
    for (unsigned j = 0; j < size2; ++ j){
      vec.at(i).at(j) = boost::python::extract<double>(list[i][j]);
    }
  }
  return vec;
}
vector pypoint_to_point(pylist list){
  unsigned size = len(list);
  vector vec (size);
  for (unsigned i = 0; i < size; ++ i) {
      vec[i] = boost::python::extract<double>(list[i]);
  }
  return vec;
}

pylist list_double_vec_list_test(pylist list){
  return double_vec_to_pylist(pylist_to_double_vec(list));
}

///////////////////

template <typename R>
std::string vec_to_string(std::vector<R> const& vec){
  std::string s = "[";
  for(int i = 0; i < vec.size(); i++){
    s += std::to_string(vec[i]);
    s += ", ";
  }
  s += "]";
  return s;
}
double l2_norm(vector const& u) {
    double accum = 0.;
    for (double x : u) {
        accum += x * x;
    }
    return sqrt(accum);
}
vector vector_difference(vector const& u, vector const& v){
  if (u.size() != v.size()){
    throw std::invalid_argument("vectors have different sizes.");
  }
  vector w(u.size());
  for(int i = 0; i < w.size(); i++){
    w[i] = u[i] - v[i];
  }
  return w;
}
double inner_prod(vector const& u, vector const& v){
  if (u.size() != v.size()){
    throw std::invalid_argument("vectors have different sizes.");
  }
  double accum = 0.;
  for (unsigned i = 0; i < u.size(); i++){
    accum += u[i] * v[i];
  }
  return accum;
}
double FS_metric(vector const& u, vector const& v){
  /*
  Return the Fubini-Study metric between two vectors u, v.

  Args:
    u,v: Two vectors of even dimension. The first n/2 components represent the real part,
    the next n/2 components represent the imaginary part.

  Returns:
    Fubini-Study metric.
  */
  if (u.size() != v.size()){
    throw std::invalid_argument("vectors have different sizes.");
  }
  unsigned size = u.size();
  if (size % 2 != 0){
    throw std::invalid_argument("dimension must be even!");
  }
  unsigned half_size = size / 2;
  vector u_r (u.begin(), u.begin() + half_size);
  vector u_i (u.begin() + half_size, u.end());
  vector v_r (v.begin(), v.begin() + half_size);
  vector v_i (v.begin() + half_size, v.end());

  double inner = ( pow(inner_prod(u_r,v_r) + inner_prod(u_i,v_i),2)
                     + pow(inner_prod(u_r,v_i) - inner_prod(u_i,v_r),2) );

  if (inner >= 1.){ // this might happen due to numerical error. We don't want to pass this to acos.
    return 0.;
  }
  return acos(sqrt(inner));//acos(inner);
}
double distance(vector const& u, vector const& v, std::string metric){
  if(metric == "FS_metric"){
    return FS_metric(u,v);
  }
  else if(metric == "euclidean"){
    return l2_norm(vector_difference(u,v));
  }
  else{
    throw std::invalid_argument("Not a known metric: " + metric);
  }
}
double FS_metric_py(pylist const& u, pylist const& v){
  return FS_metric(pypoint_to_point(u),pypoint_to_point(v));
}
template <typename T>
class node {
  private:
    T point;
    double distance;
    node<T>* left;
    node<T>* right;
    int ID;
  public:
    node(T point, int ID, node<T> *left = NULL, double distance = -1., node<T>* right = NULL){
      this->point = point;
      this->left = left;
      this->right = right;
      this->distance = distance;
      this->ID = ID;
    }
    int get_ID(){
      return this->ID;
    }
    T get_point(){
      return this->point;
    }
    double get_distance(){
      return this->distance;
    }
    node<T>* get_left_child(){
      return this->left;
    }
    node<T>* get_right_child(){
      return this->right;
    }
    std::string print_tree(){
      std::string s = "{point: " + vec_to_string(this->point);
      s += ", ID: " + std::to_string(this->ID);
      s += ", distance: " + std::to_string(this->distance);
      if (this->left){
        s += ", left child: ";
        s += this->left->print_tree();
      }
      if  (this->right){
        s += ", right child: ";
        s += this->right->print_tree();
      }
      s += "}, ";
      return s;
    }
};
double_vec make_random_data(int num_data_points, int dim){
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0,1.0);
  double_vec data(num_data_points, vector(dim) );
  for(int i = 0; i < num_data_points; i++){
    for(int j = 0; j < dim; j++){
      data[i][j] = distribution(generator);
    }
  }
  return data;
}
double_vec make_normalized_random_data(int num_data_points, int dim){
  double_vec data = make_random_data(num_data_points, dim);
  double norm;
  for (unsigned i = 0; i < num_data_points; i ++){
    norm = l2_norm(data[i]);
    for(unsigned j = 0; j < dim; j++){
      data[i][j] /= norm;
    }
  }
  return data;
}
void print_data(double_vec const& data){
  int num_data_points = data.size();
  int dim = data[0].size();
  std::cout << "Data: [";
  for(int i = 0; i < num_data_points; i++){
    std::cout << "[";
    for(int j = 0; j < dim; j++){
      std::cout << data[i][j] << ", ";
    }
    std::cout << "], ";
  }
  std::cout << "]." << std::endl;

  std::cout << "Norms: [";
  for(int i = 0; i < num_data_points; i++){
    std::cout << l2_norm(data[i]) << ", ";
  }
  std::cout << "]." << std::endl;
}
struct data_point{
  vector* point;
  int ID;
};
node<vector>* vp_tree_helper(std::vector<data_point> data, std::string metric){
  if (data.size() == 0){
    return NULL;
  }
  else if (data.size() == 1){
    data_point vantage_point = data.back();
    return new node<vector>(*(vantage_point.point),vantage_point.ID);
  }
  else if (data.size() == 2){
    data_point vantage_point=data.back();
    data.pop_back();

    std::vector<data_point> singleton (data.begin(), data.end() );

    node<vector>* left = vp_tree_helper(singleton,metric);
    return new node<vector>(*(vantage_point.point),vantage_point.ID,left,distance(*(vantage_point.point), left->get_point(), metric ));
  }
  else{
    data_point vantage_point=data.back();
    data.pop_back();

    auto cmp = [vantage_point,metric](data_point a, data_point b)
      {return distance(*(a.point),*(vantage_point.point),metric) < distance(*(b.point),*(vantage_point.point),metric);};
    sort(data.begin(),data.end(), cmp);

    int half_way = int( data.size() / 2 );

    std::vector<data_point> close_points (data.begin(), data.begin() + half_way );
    std::vector<data_point> far_points (data.begin() + half_way, data.end() );

    node<vector>* left = vp_tree_helper(close_points,metric);
    node<vector>* right = vp_tree_helper(far_points,metric);

    double dist = 0.5 * (distance(*(vantage_point.point), *(close_points.back().point), metric )
                       + distance(*(vantage_point.point), *(far_points.front().point), metric ));
    return new node<vector>(*(vantage_point.point),vantage_point.ID, left,dist,right);
  }
};
node<vector>* vp_tree(double_vec data, std::string metric){
  std::vector<data_point> data_points;
  for (unsigned i = 0; i < data.size(); i++){
    data_point p;
    p.point = &data[i];
    p.ID = i;
    data_points.push_back(p);
  }
  return vp_tree_helper(data_points,metric);
}
void find_within_epsilon_helper(node<vector>* vp_tree,
  vector const& point, double epsilon, std::vector<node<vector>*> & found_points,
  std::string metric){
  if (vp_tree == NULL){
    return;
  }
  else{
    double distance_root_to_point = distance(vp_tree->get_point(), point, metric);
    if (distance_root_to_point <= epsilon){
      found_points.push_back(vp_tree);
    }
    double cutoff_distance = vp_tree->get_distance();

    node<vector>* left_child = vp_tree->get_left_child();
    if (cutoff_distance > - 0.5 && left_child != NULL){
      if (distance_root_to_point - cutoff_distance <= epsilon){
        find_within_epsilon_helper(left_child,point,epsilon,found_points,metric);
      }
      node<vector>* right_child = vp_tree->get_right_child();
      if (right_child != NULL && (cutoff_distance - distance_root_to_point <= epsilon)){
        find_within_epsilon_helper(right_child,point,epsilon,found_points,metric);
      }
    }
  }
}
std::vector<node<vector>*> find_within_epsilon(node<vector>* vp_tree,
  vector const point, double epsilon, std::string metric){
    std::vector<node<vector>*> found_points;
    find_within_epsilon_helper(vp_tree,point,epsilon,found_points,metric);
    return found_points;
}
void make_dict_helper(node<vector>* tree, std::unordered_map<int,node<vector>*>& dict){
  if (tree == NULL){
    return;
  }
  dict.emplace(tree->get_ID(),tree);
  node<vector>* left_child = tree->get_left_child();
  if (left_child != NULL){
    make_dict_helper(left_child, dict);
    node<vector>* right_child = tree->get_right_child();
    if (right_child != NULL){
      make_dict_helper(right_child, dict);
    }
  }
  return;
}
std::unordered_map<int,node<vector>*> make_dict(node<vector>* tree){
  std::unordered_map<int,node<vector>*> dict;
  make_dict_helper(tree, dict);
  return dict;
}
std::vector<node<vector>*> find_N_neighbors(node<vector>* vp_tree,
    vector point, int num, std::string metric,
    double max_dist = std::numeric_limits<double>::infinity()  // upper bound on distance to nearest num neighbors.
    ){
  /*
    Return the num nearest neighbors of point in vp_tree. The first point is the farthest.
    The structured returned is a heap as a function of distance from point.
  */
  if (vp_tree == NULL){
    throw std::invalid_argument("Input pointer to tree is NULL!");
  }
  if (num < 1){
    throw std::invalid_argument("Must look for at least one neighbor. Please set num > 0.");
  }

  std::vector<node<vector>*> neighbors;

  auto cmp = [point,metric](node<vector>* a, node<vector>* b)
      {return distance(a->get_point(),point,metric) < distance(b->get_point(),point,metric);};

  double current_dist;
  std::queue<node<vector>*> node_Q; // nodes to visit, organized as a queue.
  node_Q.push(vp_tree);

  while (!node_Q.empty()){
    node<vector>* current_node = node_Q.front();
    node_Q.pop();
    current_dist = distance(current_node->get_point(),point,metric);

    if (current_dist < max_dist){
      neighbors.push_back(current_node);
      std::push_heap(neighbors.begin(),neighbors.end(),cmp);
      if (neighbors.size() > num){
        std::pop_heap(neighbors.begin(),neighbors.end(),cmp);
        neighbors.pop_back();
      } // keep only the top num neighbors
      if (neighbors.size() == num){
        max_dist = distance(neighbors.front()->get_point(),point,metric);
      } // update max_dist once we have num neighbors.
    }
    double cutoff_distance = current_node->get_distance();

    // // different heuristic mirroring the find_within_epsilon

    node<vector>* left_child = current_node->get_left_child();
    if (cutoff_distance > -0.5 && left_child != NULL){
      if(current_dist - cutoff_distance <= max_dist){
        node_Q.push(left_child);
      }
      node<vector>* right_child = current_node->get_right_child();
      if(right_child != NULL && (cutoff_distance - current_dist <= max_dist)){
        node_Q.push(right_child);
      }
    }

    // // original heuristic used for VP trees, was not substantially slower or faster.

    // node<vector>* left_child = current_node->get_left_child();
    // node<vector>* right_child = current_node->get_right_child();
    // if (current_dist <= cutoff_distance){
    //   if (current_dist - max_dist < cutoff_distance && left_child != NULL){
    //     node_Q.push(left_child);
    //   }
    //   if (current_dist + max_dist >= cutoff_distance && right_child != NULL){
    //     node_Q.push(right_child);
    //   }
    // }
    // else{
    //   if (current_dist + max_dist >= cutoff_distance && right_child != NULL){
    //     node_Q.push(right_child);
    //   }
    //   if (current_dist - max_dist <= cutoff_distance && left_child != NULL){
    //     node_Q.push(left_child);
    //   }
    // }

  }
  std::sort(neighbors.rbegin(),neighbors.rend(),cmp);
  return neighbors;
}
std::vector<std::vector<node<vector>*>> find_all_N_neighbors(
    node<vector>* vp_tree,
    double_vec & data, int num, std::string metric,
    std::unordered_map<int,node<vector>*> dict,
    double numerical_error = 1e-13){
  if (dict.size() == 0){
    throw std::invalid_argument("dict has size zero.");
  }
  // // let's make a set of numbers from 0 to dict.size() to represent ID of untagged nodes.
  std::unordered_set<int> untagged_nodes;
  for (unsigned i = 0; i < dict.size(); i++){
    untagged_nodes.insert(i);
  }

  std::vector<std::vector<node<vector>*>> nearest_neighbrs_vector(dict.size(),std::vector<node<vector>*>(num));
  std::vector<node<vector>*> current_neighborhood;
  node<vector>* current_node;
  node<vector>* past_node;
  double max_dist;

  while (untagged_nodes.size() > 0){ // while there are still unvisited nodes
    bool different_neighborhood = true; // no more nodes in current neighborhood; pick another untagged node.
    // for (auto it = current_neighborhood.rend() - num; it != current_neighborhood.rend(); ++ it){ // iterate through the current neighborhood from nearest to farthest
    for (auto it = current_neighborhood.rbegin(); it != current_neighborhood.rend(); ++ it){ // iterate through the current neighborhood from nearest to farthest
      auto next = untagged_nodes.find((*it)->get_ID());
      if (next != untagged_nodes.end() ){ // if node is untagged
        past_node = current_node;
        current_node = dict[* next]; // update current node.
        untagged_nodes.erase(next); // tag node
        max_dist = distance(past_node->get_point(), current_neighborhood.front()->get_point(), metric)
                 + distance(current_node->get_point(), past_node->get_point(), metric)
                 + numerical_error;
        different_neighborhood = false;
        break; // break out of the loop since we found the nearest untagged neighbor.
      }
    }
    if (different_neighborhood){
      auto next = untagged_nodes.begin(); // choose a node arbitrarily. This could be modified in the future to use a different heuristic.
      current_node = dict[*next]; // tag node.
      untagged_nodes.erase(next);
      max_dist = std::numeric_limits<double>::infinity();
    }
    current_neighborhood = find_N_neighbors(vp_tree, current_node->get_point(), num, metric, max_dist);
    // current_neighborhood = find_within_epsilon(vp_tree, current_node->get_point(), max_dist, metric);
    nearest_neighbrs_vector[current_node->get_ID()] = current_neighborhood;
  }
  return nearest_neighbrs_vector;
}

pylist nodes_to_pylist(std::vector<node<vector>*> vec){
  pylist list;
  for (unsigned i = 0; i < vec.size(); i++) {
    pylist sublist;
    vector point = vec[i]->get_point();
    for (unsigned j = 0; j < point.size(); j++){
      sublist.append(point[j]);
    }
    list.append(sublist);
  }
  return list;
}
pylist double_vec_nodes_to_pylist(std::vector<std::vector<node<vector>*>> vec){
  pylist list;
  for (unsigned i = 0; i < vec.size(); i++){
    pylist sublist;
    std::vector<node<vector>*> neighborhood = vec[i];
    for (unsigned j = 0; j < neighborhood.size(); j++){
      pylist subsublist;
      node<vector>* node = neighborhood[j];
      vector point = node->get_point();
      for (unsigned k = 0; k < point.size(); k++){
        subsublist.append(point[k]);
      }
      sublist.append(subsublist);
    }
    list.append(sublist);
  }
  return list;
}

class tree_container{
  private:
    node<vector>* tree;
    std::unordered_map<int,node<vector>*> dict;
    bool dict_generated = false;
    double_vec data;

  public:
    tree_container(){
      this->tree = NULL;
    }
    tree_container(double_vec data, std::string metric){
      this->tree = ::vp_tree(data,metric);
      this->data = data;
    }
    tree_container(pylist data, std::string metric){
      double_vec data_in_vecs = pylist_to_double_vec(data);
      this->tree = ::vp_tree(data_in_vecs, metric);
      this->data = data_in_vecs;
    }
    tree_container(pylist data){
      double_vec data_in_vecs = pylist_to_double_vec(data);
      this->tree = ::vp_tree(data_in_vecs, "euclidean");
      this->data = data_in_vecs;
    }
    std::vector<node<vector>*> find_within_epsilon(
      vector const point, double epsilon, std::string metric){
        return ::find_within_epsilon(this->tree, point, epsilon, metric);
    }
    pylist find_within_epsilon_py(
      pylist pypoint, double epsilon){
        vector point = pypoint_to_point(pypoint);
        return nodes_to_pylist(
          ::find_within_epsilon(this->tree, point, epsilon, "euclidean"));
    }
    pylist find_within_epsilon_py(
      pylist pypoint, double epsilon, std::string metric){
        vector point = pypoint_to_point(pypoint);
        return nodes_to_pylist(
          ::find_within_epsilon(this->tree, point, epsilon, metric));
    }
    std::string print_tree(){
      return this->tree->print_tree();
    }
    void make_dict(){
      this->dict = ::make_dict(tree);
    }
    pylist find_N_neighbors_py(pylist pypoint, int num, std::string metric){
      vector point = pypoint_to_point(pypoint);
       std::vector<node<vector>*> nodes= ::find_N_neighbors(this->tree, point, num, metric);
       pylist output = nodes_to_pylist(nodes);
      return output;
    }
    pylist find_N_neighbors_py(pylist pypoint, int num){
      vector point = pypoint_to_point(pypoint);
      std::vector<node<vector>*> nodes= ::find_N_neighbors(this->tree, point, num,"euclidean");
      pylist output = nodes_to_pylist(nodes);
     return output;
    }
    pylist find_all_N_neighbors_py(int num, std::string metric){
      if(this->tree == NULL){
        throw std::invalid_argument("The tree is null! Please make a tree.");
      }
      if (this->dict_generated == false){
        this->make_dict();
        this->dict_generated = true;
      }
      if (this->data.size() < 1){
        throw std::invalid_argument("This really shouldn't happen... somehow the data has been constructed as has size zero.");
      }
      std::vector<std::vector<node<vector>*>> neighbors_vector = find_all_N_neighbors(
        this->tree, this->data, num, metric, this->dict);
      return double_vec_nodes_to_pylist(neighbors_vector);
    }
    pylist find_all_N_neighbors_py(int num){
      return find_all_N_neighbors_py(num, "euclidean");
    }
};

BOOST_PYTHON_MODULE(vp_tree) {
    using namespace boost::python;

    def("list_double_vec_list_test",list_double_vec_list_test);
    // def("hello", ex_hello);
    def("FS_metric", FS_metric_py);

    class_<tree_container>("tree_container", init<>()  )
      .def(init<pylist>())
      .def(init<pylist,std::string>())
      // .def("find_within_epsilon",&tree_container::find_within_epsilon_py)
      .def("find_within_epsilon", (pylist (tree_container::*) (pylist, double)) &tree_container::find_within_epsilon_py)
      .def("find_within_epsilon", (pylist (tree_container::*) (pylist, double, std::string)) &tree_container::find_within_epsilon_py)
      .def("print_tree",&tree_container::print_tree)
      .def("find_N_neighbors",(pylist (tree_container::*) (pylist, int)) &tree_container::find_N_neighbors_py )
      .def("find_N_neighbors",(pylist (tree_container::*) (pylist, int, std::string)) &tree_container::find_N_neighbors_py )
      .def("find_all_N_neighbors", (pylist (tree_container::*) (int, std::string)) &tree_container::find_all_N_neighbors_py )
      .def("find_all_N_neighbors", (pylist (tree_container::*) (int) ) &tree_container::find_all_N_neighbors_py )
    ;
}
