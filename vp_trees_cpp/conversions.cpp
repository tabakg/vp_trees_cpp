#include <boost/python.hpp>
#include <vector>
#include "conversions.h"
#include <iostream>


typedef boost::python::list pylist;
typedef std::vector<double> vector;
typedef std::vector<vector> double_vec;

std::string ex_hello(){
  return "hello";
}

namespace conversions{
  // pylist double_vec_to_pylist(double_vec vec){
  //   pylist list;
  //   for (unsigned i = 0; i < vec.size(); i++) {
  //     pylist sublist;
  //     for (unsigned j = 0; j < vec.at(i).size(); j++){
  //       sublist.append(vec.at(i).at(j));
  //     }
  //     list.append(sublist);
  //   }
  //   return list;
  // }
  // double_vec pylist_to_double_vec(pylist list){
  //   unsigned size1 = len(list);
  //   unsigned size2 = len(list[0]);
  //   double_vec vec (size1, std::vector<double>(size2) );
  //   for (unsigned i = 0; i < size1; ++ i) {
  //     for (unsigned j = 0; j < size2; ++ j){
  //       vec.at(i).at(j) = boost::python::extract<double>(list[i][j]);
  //     }
  //   }
  //   return vec;
  // }
  // std::vector<double> pypoint_to_point(pylist list){
  //   unsigned size = len(list);
  //   std::vector<double> vec (size);
  //   for (unsigned i = 0; i < size; ++ i) {
  //       vec[i] = boost::python::extract<double>(list[i]);
  //   }
  //   return vec;
  // }
  //
  // pylist list_double_vec_list_test(pylist list){
  //   return double_vec_to_pylist(pylist_to_double_vec(list));
  // }
  //
}
