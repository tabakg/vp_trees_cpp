#include <boost/python.hpp>
#include <iostream>
#include <random>
#include <vector>

#include <math.h>       /* sqrt, acos */

typedef boost::python::list pylist;
typedef std::vector<std::vector<double>> double_vec;

using namespace std;

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
vector<double> pypoint_to_point(pylist list){
  unsigned size = len(list);
  vector<double> vec (size);
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
string vec_to_string(vector<R> const& vec){
  string s = "[";
  for(int i = 0; i < vec.size(); i++){
    s += to_string(vec[i]);
    s += ", ";
  }
  s += "]";
  return s;
}
double l2_norm(vector<double> const& u) {
    double accum = 0.;
    for (double x : u) {
        accum += x * x;
    }
    return sqrt(accum);
}
vector<double> vector_difference(vector<double> const& u, vector<double> const& v){
  if (u.size() != v.size()){
    throw invalid_argument("vectors have different sizes.");
  }
  vector<double> w(u.size());
  for(int i = 0; i < w.size(); i++){
    w[i] = u[i] - v[i];
  }
  return w;
}
double inner_prod(vector<double> const& u, vector<double> const& v){
  if (u.size() != v.size()){
    throw invalid_argument("vectors have different sizes.");
  }
  double accum = 0.;
  for (unsigned i = 0; i < u.size(); i++){
    accum += u[i] * v[i];
  }
  return accum;
}
double FS_metric(vector<double> const& u, vector<double> const& v){
  /*
  Return the Fubini-Study metric between two vectors u, v.

  Args:
    u,v: Two vectors of even dimension. The first n/2 components represent the real part,
    the next n/2 components represent the imaginary part.

  Returns:
    Fubini-Study metric.
  */
  if (u.size() != v.size()){
    throw invalid_argument("vectors have different sizes.");
  }
  unsigned size = u.size();
  if (size % 2 != 0){
    throw invalid_argument("dimension must be even!");
  }
  unsigned half_size = size / 2;
  vector<double> u_r (u.begin(), u.begin() + half_size);
  vector<double> u_i (u.begin() + half_size, u.end());
  vector<double> v_r (v.begin(), v.begin() + half_size);
  vector<double> v_i (v.begin() + half_size, v.end());

  double inner = ( pow(inner_prod(u_r,v_r) + inner_prod(u_i,v_i),2)
                     + pow(inner_prod(u_r,v_i) - inner_prod(u_i,v_r),2) );

  // std::cout << inner  << std::endl;

  if (inner >= 1.){ // this might happen due to numerical error. We don't want to pass this to acos.
    return 0.;
  }
  return acos(inner);//acos(inner);
}
double distance(vector<double> const& u, vector<double> const& v, std::string metric){
  if(metric == "FS_metric"){
    return FS_metric(u,v);
  }
  if(metric == "euclidean"){
    return l2_norm(vector_difference(u,v));
  }
  else{
    throw invalid_argument("Not a known metric.");
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
    string print_tree(){
      string s = "{point: " + vec_to_string(this->point);
      s += ", distance: " + to_string(this->distance);
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
vector<vector<double>> make_random_data(int num_data_points, int dim){
  default_random_engine generator;
  normal_distribution<double> distribution(0.0,1.0);
  vector<vector<double>> data(num_data_points, vector<double>(dim) );
  for(int i = 0; i < num_data_points; i++){
    for(int j = 0; j < dim; j++){
      data[i][j] = distribution(generator);
    }
  }
  return data;
}
vector<vector<double>> make_normalized_random_data(int num_data_points, int dim){
  vector<vector<double>> data = make_random_data(num_data_points, dim);
  double norm;
  for (unsigned i = 0; i < num_data_points; i ++){
    norm = l2_norm(data[i]);
    for(unsigned j = 0; j < dim; j++){
      data[i][j] /= norm;
    }
  }
  return data;
}
void print_data(vector<vector<double>> const& data){
  int num_data_points = data.size();
  int dim = data[0].size();
  cout << "Data: [";
  for(int i = 0; i < num_data_points; i++){
    cout << "[";
    for(int j = 0; j < dim; j++){
      cout << data[i][j] << ", ";
    }
    cout << "], ";
  }
  cout << "]." << endl;

  cout << "Norms: [";
  for(int i = 0; i < num_data_points; i++){
    cout << l2_norm(data[i]) << ", ";
  }
  cout << "]." << endl;
}
node<vector<double>>* vp_tree(vector<vector<double>> data, std::string metric){
  if (data.size() == 0){
    return NULL;
  }
  else if (data.size() == 1){
    return new node<vector<double>>(data.back());
  }
  else if (data.size() == 2){
    vector<double> vantage_point=data.back();
    data.pop_back();

    vector<vector<double>> singleton (data.begin(), data.end() );
    node<vector<double>>* left = vp_tree(singleton,metric);
    return new node<vector<double>>(vantage_point,left,distance(vantage_point, left->get_point(), metric ));
  }
  else{
    vector<double> vantage_point=data.back();
    data.pop_back();

    sort(data.begin(),data.end(),
      [vantage_point,metric](vector<double> a, vector<double> b)
      { return distance(a,vantage_point,metric) > distance(b,vantage_point,metric);} );

    int half_way = int( (data.size() + 1 ) / 2 );

    vector<vector<double>> close_points (data.begin() + half_way, data.end() );
    vector<vector<double>> far_points (data.begin(), data.begin() + half_way );

    node<vector<double>>* left = vp_tree(close_points,metric);
    node<vector<double>>* right = vp_tree(far_points,metric);

    return new node<vector<double>>(vantage_point,left,right,distance(vantage_point, left->get_point(), metric ) );
  }
};
void find_within_epsilon_helper(node<vector<double>>* vp_tree,
  vector<double> const& point, double epsilon, vector<vector<double>>& found_points,
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
      node<vector<double>>* left_child = vp_tree->get_left_child();
      if (left_child != NULL){
        find_within_epsilon_helper(left_child,point,epsilon,found_points,metric);
      }
    }
    if (cutoff_distance - distance_root_to_point <= epsilon){
      node<vector<double>>* right_child = vp_tree->get_right_child();
      if (right_child != NULL){
        find_within_epsilon_helper(right_child,point,epsilon,found_points,metric);
      }
    }
  }
}
vector<vector<double>> find_within_epsilon(node<vector<double>>* vp_tree,
  vector<double> const point, double epsilon, std::string metric){
    vector<vector<double>> found_points = vector<vector<double>> ();
    find_within_epsilon_helper(vp_tree,point,epsilon,found_points,metric);
    return found_points;
}

class tree_container{
  private:
    node<vector<double>>* tree;

  public:
    tree_container(){
      this->tree = NULL;
    }
    tree_container(vector<vector<double>> data, std::string metric){
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
    vector<vector<double>> find_within_epsilon(
      vector<double> const point, double epsilon, std::string metric){
        return ::find_within_epsilon(this->tree, point, epsilon, metric);
    }
    pylist find_within_epsilon_py(
      pylist pypoint, double epsilon){
        vector<double> point = pypoint_to_point(pypoint);
        return double_vec_to_pylist(
          ::find_within_epsilon(this->tree, point, epsilon, "euclidean"));
    }
    pylist find_within_epsilon_py(
      pylist pypoint, double epsilon, std::string metric){
        vector<double> point = pypoint_to_point(pypoint);
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

    class_<tree_container>("tree_container", init<>()  )
      .def(init<pylist>())
      .def(init<pylist,std::string>())
      // .def("find_within_epsilon",&tree_container::find_within_epsilon_py)
      .def("find_within_epsilon", (pylist (tree_container::*) (pylist, double)) &tree_container::find_within_epsilon_py)
      .def("find_within_epsilon", (pylist (tree_container::*) (pylist, double, std::string)) &tree_container::find_within_epsilon_py)
      .def("print_tree",&tree_container::print_tree)
    ;
}
