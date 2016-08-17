#ifndef VECTOR_STUFF_H
#define VECTOR_STUFF_H

#include <iostream>
#include <random>
#include <vector>
using namespace std;

template <typename T>
void print_vector_contents(vector<T> const& data);

string example();

double l2_norm(vector<double> const& u);
vector<double> vector_difference(vector<double> const& u, vector<double> const& v);
double euclidean_metric(vector<double> const& u, vector<double> const& v);

#endif
