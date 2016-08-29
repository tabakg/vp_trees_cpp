#include <boost/python.hpp>
#include <iostream>
#include <random>
#include <vector>
#include <math.h>       /* sqrt, acos */

#include "conversions.h"

typedef boost::python::list pylist;
typedef std::vector<std::vector<double>> double_vec;

// pylist double_vec_to_pylist(double_vec vec){return conversions::double_vec_to_pylist(vec);}
// std::string hello(){return conversions::hello();}

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
  double_vec vec (size1, std::vector<double>(size2) );
  for (unsigned i = 0; i < size1; ++ i) {
    for (unsigned j = 0; j < size2; ++ j){
      vec.at(i).at(j) = boost::python::extract<double>(list[i][j]);
    }
  }
  return vec;
}
std::vector<double> pypoint_to_point(pylist list){
  unsigned size = len(list);
  std::vector<double> vec (size);
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
double l2_norm(std::vector<double> const& u) {
    double accum = 0.;
    for (double x : u) {
        accum += x * x;
    }
    return sqrt(accum);
}
std::vector<double> vector_difference(std::vector<double> const& u, std::vector<double> const& v){
  if (u.size() != v.size()){
    throw std::invalid_argument("vectors have different sizes.");
  }
  std::vector<double> w(u.size());
  for(int i = 0; i < w.size(); i++){
    w[i] = u[i] - v[i];
  }
  return w;
}
double inner_prod(std::vector<double> const& u, std::vector<double> const& v){
  if (u.size() != v.size()){
    throw std::invalid_argument("vectors have different sizes.");
  }
  double accum = 0.;
  for (unsigned i = 0; i < u.size(); i++){
    accum += u[i] * v[i];
  }
  return accum;
}
double FS_metric(std::vector<double> const& u, std::vector<double> const& v){
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
  std::vector<double> u_r (u.begin(), u.begin() + half_size);
  std::vector<double> u_i (u.begin() + half_size, u.end());
  std::vector<double> v_r (v.begin(), v.begin() + half_size);
  std::vector<double> v_i (v.begin() + half_size, v.end());

  double inner = ( pow(inner_prod(u_r,v_r) + inner_prod(u_i,v_i),2)
                     + pow(inner_prod(u_r,v_i) - inner_prod(u_i,v_r),2) );

  if (inner >= 1.){ // this might happen due to numerical error. We don't want to pass this to acos.
    return 0.;
  }
  return acos(inner);//acos(inner);
}
double distance(std::vector<double> const& u, std::vector<double> const& v, std::string metric){
  if(metric == "FS_metric"){
    return FS_metric(u,v);
  }
  if(metric == "euclidean"){
    return l2_norm(vector_difference(u,v));
  }
  else{
    throw std::invalid_argument("Not a known metric.");
  }
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
  double_vec data(num_data_points, std::vector<double>(dim) );
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
node<std::vector<double>>* vp_tree(double_vec data, std::string metric){
  if (data.size() == 0){
    return NULL;
  }
  else if (data.size() == 1){
    return new node<std::vector<double>>(data.back());
  }
  else if (data.size() == 2){
    std::vector<double> vantage_point=data.back();
    data.pop_back();

    double_vec singleton (data.begin(), data.end() );
    node<std::vector<double>>* left = vp_tree(singleton,metric);
    return new node<std::vector<double>>(vantage_point,left,distance(vantage_point, left->get_point(), metric ));
  }
  else{
    std::vector<double> vantage_point=data.back();
    data.pop_back();

    sort(data.begin(),data.end(),
      [vantage_point,metric](std::vector<double> a, std::vector<double> b)
      { return distance(a,vantage_point,metric) > distance(b,vantage_point,metric);} );

    int half_way = int( (data.size() + 1 ) / 2 );

    double_vec close_points (data.begin() + half_way, data.end() );
    double_vec far_points (data.begin(), data.begin() + half_way );

    node<std::vector<double>>* left = vp_tree(close_points,metric);
    node<std::vector<double>>* right = vp_tree(far_points,metric);

    return new node<std::vector<double>>(vantage_point,left,right,distance(vantage_point, left->get_point(), metric ) );
  }
};
void find_within_epsilon_helper(node<std::vector<double>>* vp_tree,
  std::vector<double> const& point, double epsilon, double_vec& found_points,
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
    if (cutoff_distance < -0.5){
      return;
    }
    if (distance_root_to_point - cutoff_distance <= epsilon){
      node<std::vector<double>>* left_child = vp_tree->get_left_child();
      if (left_child != NULL){
        find_within_epsilon_helper(left_child,point,epsilon,found_points,metric);
      }
    }
    if (cutoff_distance - distance_root_to_point <= epsilon){
      node<std::vector<double>>* right_child = vp_tree->get_right_child();
      if (right_child != NULL){
        find_within_epsilon_helper(right_child,point,epsilon,found_points,metric);
      }
    }
  }
}
double_vec find_within_epsilon(node<std::vector<double>>* vp_tree,
  std::vector<double> const point, double epsilon, std::string metric){
    double_vec found_points = double_vec();
    find_within_epsilon_helper(vp_tree,point,epsilon,found_points,metric);
    return found_points;
}

class tree_container{
  private:
    node<std::vector<double>>* tree;

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
      std::vector<double> const point, double epsilon, std::string metric){
        return ::find_within_epsilon(this->tree, point, epsilon, metric);
    }
    pylist find_within_epsilon_py(
      pylist pypoint, double epsilon){
        std::vector<double> point = pypoint_to_point(pypoint);
        return double_vec_to_pylist(
          ::find_within_epsilon(this->tree, point, epsilon, "euclidean"));
    }
    pylist find_within_epsilon_py(
      pylist pypoint, double epsilon, std::string metric){
        std::vector<double> point = pypoint_to_point(pypoint);
        return double_vec_to_pylist(
          ::find_within_epsilon(this->tree, point, epsilon, metric));
    }
    std::string print_tree(){
      return this->tree->print_tree();
    }
};

BOOST_PYTHON_MODULE(vp_tree) {
    using namespace boost::python;

    def("list_double_vec_list_test",list_double_vec_list_test);
    // def("hello", hello);

    class_<tree_container>("tree_container", init<>()  )
      .def(init<pylist>())
      .def(init<pylist,std::string>())
      // .def("find_within_epsilon",&tree_container::find_within_epsilon_py)
      .def("find_within_epsilon", (pylist (tree_container::*) (pylist, double)) &tree_container::find_within_epsilon_py)
      .def("find_within_epsilon", (pylist (tree_container::*) (pylist, double, std::string)) &tree_container::find_within_epsilon_py)
      .def("print_tree",&tree_container::print_tree)
    ;
}
