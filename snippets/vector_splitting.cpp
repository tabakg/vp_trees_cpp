#include <vector>
#include <iostream>

using namespace std;

int main(){
  const int number = 10;
  vector<int> lines = vector<int>(number);
  for (int i = 0; i < number; i++){
    lines[i] = i;
  }

  vector<int> lines1(lines.begin(),lines.begin() + 3);
  vector<int> lines2(lines.begin() + 3,lines.end() );

  for (int i = 0; i < lines1.size(); i++){
    cout << lines1[i] << endl;
  }
  for (int i = 0; i < lines2.size(); i++){
    cout << lines2[i] << endl;
  }

  lines1[2] = 999;

  for (int i = 0; i < lines1.size(); i++){
    cout << lines1[i] << endl;
  }
  for (int i = 0; i < lines2.size(); i++){
    cout << lines2[i] << endl;
  }

  for (int i = 0; i < lines.size(); i++){
    cout << lines[i] << endl;
  }
}
