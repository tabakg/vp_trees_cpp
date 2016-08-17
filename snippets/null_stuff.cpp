#include <iostream>

using namespace std;

int main(){
  int * ptr = NULL;
  cout << ptr << endl;
  cout << NULL << endl;
  if (ptr){
    cout << "ptr is not NULL" << endl;
  }
  else{
    cout << "ptr is NULL" << endl;
  }
}
