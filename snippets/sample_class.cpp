/* Program to illustrate working of Objects and Class in C++ Programming */
#include <iostream>
using namespace std;
class temp
{
    private:
        int data1;
        float data2;
    public:
       void int_data(int d){
          data1=d;
          cout<<"Number: "<<data1 << endl;
         }
       float float_data(){
           cout<<"\nEnter data: ";
           cin>>data2;
           return data2;
         }
};
 int main(){
      temp obj1, obj2;
      obj1.int_data(12);
      cout<<"You entered "<<obj2.float_data();
      return 0;
 }
