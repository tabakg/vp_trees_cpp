#include <iostream>
#include <random>
#include <vector>
#include "vector_stuff.h"

using namespace std;

template <typename T>
void print_vector_contents(vector<T> const& data){
    int size = data.size();
    for(int i = 0; i < size; i++){
      cout << data[i] << endl;
    }
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

int main(){

  default_random_engine generator;
  normal_distribution<double> distribution(0.0,1.0);

  const int data_points = 100;
  vector<double> data(data_points);
  for(int i = 0; i < data_points; i++){
    data[i] = distribution(generator);
  }

  cout << "size of vector: " <<  data.size() << endl;
  cout << "norm of vector: " << l2_norm(data) << endl;
  cout << "norm of difference between data and itself: " << euclidean_metric(data,data) << endl;

  print_vector_contents(data);
}
