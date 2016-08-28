#include <iostream>
#include <random>
#include <vector>
#include <chrono>
using namespace std;

template <typename Output, typename... Arguments>
Output time_it(Output (*func) (Arguments... ), Arguments... args ){
  auto start = chrono::steady_clock::now();
  Output val = func(args...);
  auto duration = chrono::steady_clock::now() - start;
  cout << duration.count() * 1e-9 << " seconds.\n";
  return val;
}

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

// NOTE: as written here, data may be modified (i.e. re-ordered).
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

int all_points_within_epsilon_timer(vector<vector<double>> data,node<vector<double>>* tree, double epsilon){
  for (int i = 0; i < data.size(); i++){
    find_within_epsilon(tree, data[i], epsilon);
  }
  return 0;
}

int main(){

  // // make zero vector
  vector<double> zero_vec = vector<double>(1);
  // cout << vec_to_string(zero_vec);

  const int num_data_points = 1000;
  const int dim = 1;
  vector<vector<double>> data = make_random_data(num_data_points,dim);

  // // print out a vector.
  // cout << vec_to_string(data[0]);

  // sample trees
  // node<vector<double>> *n = new node<vector<double>>(zero_vec);
  // cout << n->print_tree() << endl;
  //
  // node<vector<double>> *m = new node<vector<double>>(zero_vec,n,5.0);
  // cout << m->print_tree() << endl;

  // // sample data printed out
  // print_data(data);

  // // make a node in the tree and print the tree.
  node<vector<double>>* tree = vp_tree(data);


  // // sort the data by distance from zero_vec using euclidean_metric.
  // sort(data.begin(),data.end(),
  //   [zero_vec](vector<double> a, vector<double> b)
  //   { return euclidean_metric(a,zero_vec) > euclidean_metric(b,zero_vec);} );
  // print_data(data);

  // // let's get only the points within 1.0 of zero_vec.
  vector<vector<double>> close_points = find_within_epsilon(tree, zero_vec, 1.0);
  // print_data(close_points);

  // // let's time the results.
  time_it(vp_tree,data);
  time_it(find_within_epsilon,tree, zero_vec, 1.0);
  time_it(all_points_within_epsilon_timer,data,tree,1.0);

}
