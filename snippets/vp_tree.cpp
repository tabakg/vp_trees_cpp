#include <iostream>
#include <random>
#include <vector>
#include "vector_stuff.h"

using namespace std;

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

  // print_vector_contents(data);
}
