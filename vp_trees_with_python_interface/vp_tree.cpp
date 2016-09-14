#include <boost/python.hpp>
#include <iostream>
#include <random>
#include <vector>
#include <math.h>       /* sqrt, acos */
#include <queue> /* nearest neighbors */
#include <unordered_map> /* making dictionaries */
#include <queue>

// #include "conversions.h"

typedef boost::python::list pylist;
typedef std::vector<double> vector;
typedef std::vector<vector> double_vec;

// pylist double_vec_to_pylist(double_vec vec){return conversions::double_vec_to_pylist(vec);}
// std::string hello(){return ex_hello();}

pylist double_vec_to_pylist(double_vec vec){
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
    // std::cout << inner << std::endl;
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
    throw std::invalid_argument("Not a known metric.");
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
  public:
    node(T point){
      this->point = point;
      this->left = NULL;
      this->right = NULL;
      this->distance = -1;
    }
    node(T point, node<T>* left,double distance){
      this->point = point;
      this->left = left;
      this->right = NULL;
      this->distance = distance;
    }
    node(T point, node<T> *left, node<T>* right,double distance){
      this->point = point;
      this->left = left;
      this->right = right;
      this->distance = distance;
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
node<vector>* vp_tree(double_vec data, std::string metric){
  if (data.size() == 0){
    return NULL;
  }
  else if (data.size() == 1){
    return new node<vector>(data.back());
  }
  else if (data.size() == 2){
    vector vantage_point=data.back();
    data.pop_back();

    double_vec singleton (data.begin(), data.end() );
    node<vector>* left = vp_tree(singleton,metric);
    return new node<vector>(vantage_point,left,distance(vantage_point, left->get_point(), metric ));
  }
  else{
    vector vantage_point=data.back();
    data.pop_back();

    auto cmp = [vantage_point,metric](vector a, vector b)
      {return distance(a,vantage_point,metric) < distance(b,vantage_point,metric);};
    sort(data.begin(),data.end(), cmp);

    int half_way = int( data.size() / 2 );

    double_vec close_points (data.begin(), data.begin() + half_way );
    double_vec far_points (data.begin() + half_way, data.end() );

    node<vector>* left = vp_tree(close_points,metric);
    node<vector>* right = vp_tree(far_points,metric);

    return new node<vector>(vantage_point,left,right,
      0.5 * (distance(vantage_point, close_points.back(), metric ) + distance(vantage_point, far_points.front(), metric )) );
  }
};
void find_within_epsilon_helper(node<vector>* vp_tree,
  vector const& point, double epsilon, double_vec& found_points,
  std::string metric){
  if (vp_tree == NULL){
    return;
  }
  else{
    double distance_root_to_point = distance(vp_tree->get_point(), point, metric);
    if (distance_root_to_point <= epsilon){
      found_points.push_back(vp_tree->get_point() );
    }
    double cutoff_distance = vp_tree->get_distance();

    node<vector>* left_child = vp_tree->get_left_child();
    if (left_child != NULL){
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
double_vec find_within_epsilon(node<vector>* vp_tree,
  vector const point, double epsilon, std::string metric){
    double_vec found_points = double_vec();
    find_within_epsilon_helper(vp_tree,point,epsilon,found_points,metric);
    return found_points;
}
std::string hash_vector(vector vec){
  std::string s = "";
  for(unsigned i = 0; i < vec.size(); i++){
    s += std::to_string(vec[i]);
  }
  return s;
}
void make_dict_helper(node<vector>* tree, std::unordered_map<std::string,node<vector>*>& dict){
  if (tree == NULL){
    return;
  }
  dict.emplace(hash_vector(tree->get_point()),tree);
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
std::unordered_map<std::string,node<vector>*> make_dict(node<vector>* tree){
  std::unordered_map<std::string,node<vector>*> dict;
  make_dict_helper(tree, dict);
  return dict;
}

double_vec find_N_neighbors(node<vector>* vp_tree, vector point, int num, std::string metric,
  double max_dist = std::numeric_limits<double>::infinity() // upper bound on distance to nearest num neighbors.
  ){
  if (vp_tree == NULL){
    throw std::invalid_argument("Input tree is NULL!");
  }
  if (num < 1){
    throw std::invalid_argument("Must look for at least one neighbor. Please set num > 0.");
  }

  auto cmp = [point,metric](vector a, vector b) {return distance(a,point,metric) < distance(b,point,metric);};
  std::priority_queue<vector, double_vec, decltype(cmp) > neighbors(cmp);

  double current_dist;
  std::queue<node<vector>*> node_Q; // nodes to visit, organized as a queue.
  node_Q.push(vp_tree);

  while (!node_Q.empty()){
    node<vector>* current_node = node_Q.front();
    node_Q.pop();
    current_dist = distance(current_node->get_point(),point,metric);
    if (current_dist < max_dist){
      neighbors.push(current_node->get_point());
      if (neighbors.size() > num){
        neighbors.pop();
      } // keep only the top num neighbors
      if (neighbors.size() == num){
        max_dist = distance(neighbors.top(),point,metric);
      } // update max_dist once we have num neighbors.
    }
    double cutoff_distance = vp_tree->get_distance();

    // // original heuristic used for VP trees.

    node<vector>* left_child = current_node->get_left_child();
    node<vector>* right_child = current_node->get_right_child();
    if (current_dist <= cutoff_distance){
      if (current_dist - max_dist < cutoff_distance && left_child != NULL){
        node_Q.push(left_child);
      }
      if (current_dist + max_dist >= cutoff_distance && right_child != NULL){
        node_Q.push(right_child);
      }
    }
    else{
      if (current_dist + max_dist >= cutoff_distance && right_child != NULL){
        node_Q.push(right_child);
      }
      if (current_dist - max_dist <= cutoff_distance && left_child != NULL){
        node_Q.push(left_child);
      }
    }

    // // different heuristic for the order, was not substantially slower or faster.

    // node<vector>* left_child = current_node->get_left_child();
    // if (left_child != NULL){
    //   if(current_dist - cutoff_distance <= max_dist){
    //     node_Q.push(left_child);
    //   }
    //   node<vector>* right_child = current_node->get_right_child();
    //   if(right_child != NULL && (cutoff_distance - current_dist <= max_dist)){
    //     node_Q.push(right_child);
    //   }
    // }
  }
  double_vec output_vec (neighbors.size());
  // unsigned i = neighbors.size() - 1;
  // while(!neighbors.empty()){
  //     output_vec[i] = neighbors.top();
  //     i--;
  //     neighbors.pop();
  // }
  return output_vec;
}
// double_vec find_all_N_neighbors(node<vector>* vp_tree, double_vec & data, int num, std::string metric){
//   std::unordered_map<std::string,bool> tagged;
//
// }


class tree_container{
  private:
    node<vector>* tree;
    std::unordered_map<std::string,node<vector>*> dict;

  public:
    tree_container(){
      this->tree = NULL;
    }
    tree_container(double_vec data, std::string metric){
      this->tree = ::vp_tree(data,metric);
    }
    tree_container(pylist data, std::string metric){
      double_vec data_in_vecs = pylist_to_double_vec(data);
      this->tree = ::vp_tree(data_in_vecs, metric);
    }
    tree_container(pylist data){
      double_vec data_in_vecs = pylist_to_double_vec(data);
      this->tree = ::vp_tree(data_in_vecs, "euclidean");
    }
    double_vec find_within_epsilon(
      vector const point, double epsilon, std::string metric){
        return ::find_within_epsilon(this->tree, point, epsilon, metric);
    }
    pylist find_within_epsilon_py(
      pylist pypoint, double epsilon){
        vector point = pypoint_to_point(pypoint);
        return double_vec_to_pylist(
          ::find_within_epsilon(this->tree, point, epsilon, "euclidean"));
    }
    pylist find_within_epsilon_py(
      pylist pypoint, double epsilon, std::string metric){
        vector point = pypoint_to_point(pypoint);
        return double_vec_to_pylist(
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
      double_vec output = ::find_N_neighbors(this->tree, point, num, metric);
      return double_vec_to_pylist(output);
    }
    pylist find_N_neighbors_py(pylist pypoint, int num){
      vector point = pypoint_to_point(pypoint);
      double_vec output = ::find_N_neighbors(this->tree, point, num, "euclidean");
      return double_vec_to_pylist(output);
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
    ;
}
