#ifndef SAMPLE_TREE_H
#define SAMPLE_TREE_H

#include <iostream>
#include <random>
#include "vector_stuff.h"

using namespace std;

template <typename T>
class node {
  private:
    T point;
    node* left;
    node* right;
  public:
    node(T point);
    node(T point, node<T> left);
    node(T point, node<T> left, node<T> right);
    void print_tree();
};

#endif
