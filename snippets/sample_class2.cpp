#include <iostream>
using namespace std;

class temp
{
    private:
        int data;
    public:
       void set_data(int d){
          data  =d;
         }
       int get_data(){
          return data;
         }
};

int main(){
      temp obj;
      obj.set_data(5);
      cout << "number is " << obj.get_data() << endl;
 }
