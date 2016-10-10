#ifndef CONVERSIONS_H
#define CONVERSIONS_H

#include <boost/python.hpp>
#include <vector>
#include <iostream>

#include "conversions.h"

typedef boost::python::list pylist;
typedef std::vector<double> vector;
typedef std::vector<vector> double_vec;

std::string ex_hello();

namespace conversions{
  // pylist double_vec_to_pylist(double_vec vec);
  // double_vec pylist_to_double_vec(pylist list);
  // std::vector<double> pypoint_to_point(pylist list);
  // pylist list_double_vec_list_test(pylist list);
}


#endif
