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


int main(){
  node<int> * myNode = new node<int>(5);
  myNode->print_tree();
  cout << endl;
}
