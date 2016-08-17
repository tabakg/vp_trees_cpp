#include <iostream>
#include <chrono>
#include "time_functor.h"

using namespace std;

template <typename Output, typename... Arguments>
Output time_it(Output (*func) (Arguments... ), Arguments... args ){
  auto start = chrono::steady_clock::now();
  Output val = func(args...);
  auto duration = chrono::steady_clock::now() - start;
  cout << duration.count() * 1e-9 << " seconds.\n";
  return val;
}

int add_one(int x){
  return x + 1;
}

int add_together(int x, int y){
  return x + y;
}

int add_one_incrementally(int x, int y){
  for(int i = 0; i < y; i++){
    x += 1;
  }
  return x;
}

int main(){
  cout<< "Adding one to 3: " <<time_it(add_one,3) << endl;
  cout<< "Adding together 5 and 6: "<<time_it(add_together,5,6) << endl;
  cout<< "Adding 3 and 1e9 incrementally: "<<time_it(add_one_incrementally,3,int(1e9)) << endl;
}
