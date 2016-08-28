#include <boost/python.hpp>
#include <iostream>
#include <random>
#include <vector>

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
double euclidean_metric(vector<double> const& u, vector<double> const& v){
  return l2_norm(vector_difference(u,v));
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
node<vector<double>>* vp_tree(vector<vector<double>> data){
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
    node<vector<double>>* left = vp_tree(singleton);
    return new node<vector<double>>(vantage_point,left,euclidean_metric(vantage_point, left->get_point() ));
  }
  else{
    vector<double> vantage_point=data.back();
    data.pop_back();


    sort(data.begin(),data.end(),
      [vantage_point](vector<double> a, vector<double> b)
      { return euclidean_metric(a,vantage_point) > euclidean_metric(b,vantage_point);} );

    int half_way = int( (data.size() + 1 ) / 2 );

    vector<vector<double>> close_points (data.begin() + half_way, data.end() );
    vector<vector<double>> far_points (data.begin(), data.begin() + half_way );

    node<vector<double>>* left = vp_tree(close_points);
    node<vector<double>>* right = vp_tree(far_points);

    return new node<vector<double>>(vantage_point,left,right,euclidean_metric(vantage_point, left->get_point() ) );
  }
};
void find_within_epsilon_helper(node<vector<double>>* vp_tree,
  vector<double> const& point, double epsilon, vector<vector<double>>& found_points){
  if (vp_tree == NULL){
    return;
  }
  else{
    double distance_root_to_point = euclidean_metric(vp_tree->get_point(), point);
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
        find_within_epsilon_helper(left_child,point,epsilon,found_points);
      }
    }
    if (cutoff_distance - distance_root_to_point <= epsilon){
      node<vector<double>>* right_child = vp_tree->get_right_child();
      if (right_child != NULL){
        find_within_epsilon_helper(right_child,point,epsilon,found_points);
      }
    }
  }
}
vector<vector<double>> find_within_epsilon(node<vector<double>>* vp_tree,
  vector<double> const point, double epsilon){
    vector<vector<double>> found_points = vector<vector<double>> ();
    find_within_epsilon_helper(vp_tree,point,epsilon,found_points);
    return found_points;
}

class tree_container{
  private:
    node<vector<double>>* tree;

  public:
    tree_container(){
      this->tree = NULL;
    }
    tree_container(vector<vector<double>> data){
      this->tree = ::vp_tree(data);
    }
    tree_container(pylist data){
      double_vec data_in_vecs = pylist_to_double_vec(data);
      this->tree = ::vp_tree(data_in_vecs);
    }
    vector<vector<double>> find_within_epsilon(
      vector<double> const point, double epsilon){
        return ::find_within_epsilon(this->tree, point, epsilon);
    }
    pylist find_within_epsilon_py(
      pylist pypoint, double epsilon){
        vector<double> point = pypoint_to_point(pypoint);
        return double_vec_to_pylist(
          ::find_within_epsilon(this->tree, point, epsilon));
    }
    std::string print_tree(){
      return this->tree->print_tree();
    }
};

BOOST_PYTHON_MODULE(vp_tree) {
    using namespace boost::python;

    def("list_double_vec_list_test",list_double_vec_list_test);

    class_<tree_container>("tree_container", init<>()  )
      .def(init<double_vec>()) // this constructor woudln't work using python list of lists
      .def(init<pylist>()) // this costructor will work with python list of lists
      // .def("find_within_epsilon",&tree_container::find_within_epsilon) // not compatible with python because of output types
      .def("find_within_epsilon",&tree_container::find_within_epsilon_py)
      .def("print_tree",&tree_container::print_tree)
    ;
}
