#include <iostream>
#include "sample.h"

using namespace std;

void addOne(int &y)
{
  y += 1;
}

int main (void)
{       int a;
        cout << "Give me a number: ";
        cin >> a;
        cout << "You gave me: " << a << "!" << endl;
        addOne(a);
        cout << "One more than is " <<  a << endl;
        return 0;
}
