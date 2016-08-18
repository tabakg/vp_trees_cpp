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
    node* left;
    node* right;
  public:
    node(T point){
      this->point = point;
      this->left = NULL;
      this->right = NULL;
    }
    node(T point, node<T> left){
      this->point = point;
      this->left = left;
    }
    node(T point, node<T> left, node<T> right){
      this->point = point;
      this->left = left;
      this->right = right;
    }
    void print_tree(){
      cout << "{point: " << this->point;
      if (this->left){
        cout << ", left child: ";
        this->left->print_tree();
      }
      if  (this->right){
        cout << ", right child: ";
        this->right->print_tree();
      }
      cout << "}.";
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
void print_data(vector<vector<double>> data){
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

int main(){

  const int num_data_points = 4;
  const int dim = 2;
  vector<vector<double>> data = make_random_data(num_data_points,dim);
  print_data(data);

}
