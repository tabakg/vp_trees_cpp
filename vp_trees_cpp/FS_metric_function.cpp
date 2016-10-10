#include <vector>
#include <math.h>       /* sqrt, acos */
#include <iostream>

typedef std::vector<double> vector;

double inner_prod(vector const& u, vector const& v){
  if (u.size() != v.size()){
    throw std::invalid_argument("vectors have different sizes.");
  }
  double accum = 0.;
  for (unsigned i = 0; i < u.size(); i++){
    accum += u[i] * v[i];
  }
  return accum;
}
double FS_metric(vector const& u, vector const& v){
  /*
  Return the Fubini-Study metric between two vectors u, v.

  Args:
    u,v: Two vectors of even dimension. The first n/2 components represent the real part,
    the next n/2 components represent the imaginary part. I assume here both
    u and v have been normalized.

  Returns:
    Fubini-Study metric between u and v.
  */
  if (u.size() != v.size()){
    throw std::invalid_argument("vectors have different sizes.");
  }
  unsigned size = u.size();
  if (size % 2 != 0){
    throw std::invalid_argument("dimension must be even!");
  }
  unsigned half_size = size / 2;
  vector u_r (u.begin(), u.begin() + half_size);
  vector u_i (u.begin() + half_size, u.end());
  vector v_r (v.begin(), v.begin() + half_size);
  vector v_i (v.begin() + half_size, v.end());

  double inner = ( pow(inner_prod(u_r,v_r) + inner_prod(u_i,v_i),2)
                     + pow(inner_prod(u_r,v_i) - inner_prod(u_i,v_r),2) );

  if (inner >= 1.){ // this might happen due to numerical error. We don't want to pass this to acos.
    return 0.;
  }
  return acos(sqrt(inner));
}
void normalize_vec(vector & v){
  double v_norm = sqrt(inner_prod(v,v));
  for(unsigned i = 0; i < v.size(); i++){
    v[i] /= v_norm;
  }
}

int main(){
  vector v = {1,2,3,4};
  normalize_vec(v);
  vector u = {2,3,4,5};
  normalize_vec(u);
  std::cout << FS_metric(u,v) << std::endl;
}
