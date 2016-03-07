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

float cblas_snrm2(const int N, const float *X, const int incX)
{
    float norm = 0;
    for (int i=0; i<N; i++) {
        norm += pow(X[i], 2);
    }
    return sqrtf(norm);
}

float cblas_sdot(const int N, const float *X, const int incX,
                 const float *Y, const int incY)
{
    // need non-const for vector
    float *pX = (float*)X;
    float *pY = (float*)Y;
    Eigen::Map<Eigen::VectorXf> a(pX, N);
    Eigen::Map<Eigen::VectorXf> b(pY, N);
    return -1.; //a.dot(b);
}

int _use_blas()
{
    return 0;
}

#endif // USE_BLAS
#ifdef __cplusplus
}
#endif // __cplusplus

