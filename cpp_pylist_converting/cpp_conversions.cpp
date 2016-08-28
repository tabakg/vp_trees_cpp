#include <boost/python.hpp>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace ublas = boost::numeric::ublas;
typedef ublas::matrix<double> matrix;
typedef ublas::vector<double> vector;
typedef boost::python::list pylist;
typedef std::vector<std::vector<double>> double_vec;

pylist double_vec_to_pylist(double_vec vec){
  pylist list;
  for (unsigned i = 0; i < vec.size(); i++) {
    pylist sublist;
    for (unsigned j = 0; j < vec.at(i).size(); j++){
      sublist.append(vec.at(i).at(j));
    }
    list.append(sublist);
  }
  return list;
}

double_vec pylist_to_double_vec(pylist list){
  unsigned size1 = len(list);
  unsigned size2 = len(list[0]);
  double_vec vec (size1, std::vector<double>(size2) );
  for (unsigned i = 0; i < size1; ++ i) {
    for (unsigned j = 0; j < size2; ++ j){
      vec.at(i).at(j) = boost::python::extract<double>(list[i][j]);
    }
  }
  return vec;
}

pylist list_double_vec_list_test(pylist list){
  return double_vec_to_pylist(pylist_to_double_vec(list));
}

/////////////////

pylist matrix_to_python_list(matrix m) {
  pylist list;
	for (unsigned i = 0; i < m.size1 (); ++ i) {
    pylist sublist;
    for (unsigned j = 0; j < m.size2 (); ++ j){
      sublist.append(m (i,j) );
    }
		list.append(sublist);
	}
	return list;
}

// assume we have a list of lists of doubles.
// Also, each list has the same size.
matrix python_list_to_matrix(pylist list) {
  unsigned size1 = len(list);
  unsigned size2 = len(list[0]);
  matrix m (size1,size2);
	for (unsigned i = 0; i < size1; ++ i) {
    for (unsigned j = 0; j < size2; ++ j){
      m(i, j) = boost::python::extract<double>(list[i][j]);
    }
	}
	return m;
}

pylist list_matrix_list_test(pylist list){
  return matrix_to_python_list(python_list_to_matrix(list));
}

BOOST_PYTHON_MODULE(cpp_conversions) {
    using namespace boost::python;

    def("list_matrix_list_test",list_matrix_list_test);
    def("list_double_vec_list_test",list_double_vec_list_test);
}
