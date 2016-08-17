#ifndef TIME_FUNCTOR_H
#define TIME_FUNCTOR_H

#include <iostream>
#include <chrono>
using namespace std;

int add_one(int x);
int add_together(int x, int y);
int add_one_incrementally(int x, int y);

template <typename Output, typename... Arguments>
Output time_it(Output (*func) (Arguments... ), Arguments... args );

#endif
