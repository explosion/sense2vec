#ifndef USE_BLAS
#include "third_party/Eigen/Core"
#endif
#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus
#ifdef USE_BLAS
#include <cblas.h>

int _use_blas()
{
    return 1;
}
#else // USE_BLAS

Eigen::VectorXf arr_to_vecf(const int N, const float *arr) {
    // need non-const for vector
    Eigen::Map<Eigen::VectorXf> v((float*)arr, N);
    return v;
}


float cblas_snrm2(const int N, const float *X, const int incX)
{
    Eigen::VectorXf v = arr_to_vecf(N, X);
    return v.norm();
}

float cblas_sdot(const int N, const float *X, const int incX,
                 const float *Y, const int incY)
{
    Eigen::VectorXf a = arr_to_vecf(N, X);
    Eigen::VectorXf b = arr_to_vecf(N, Y);
    return a.dot(b);
}

int _use_blas()
{
    return 0;
}

#endif // USE_BLAS
#ifdef __cplusplus
}
#endif // __cplusplus

