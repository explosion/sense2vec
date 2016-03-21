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
#if defined(_MSC_VER)
/* Microsoft C/C++-compatible compiler */
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* GCC-compatible compiler, targeting x86/x86-64 */
#include <x86intrin.h>
#endif

float cblas_snrm2(const int N, const float *m1, const int incX)
{
    if (N % 4 != 0) {
        fprintf(stderr, "cblas_snrm2() expects N to be a multiple of 4.\n");
        exit(EXIT_FAILURE);
    }

    float norm = 0;
    __m128 X;
    __m128 Z = _mm_setzero_ps();

    for (int i=0; i<N; i+=4) {
        X = _mm_load_ps(&m1[i]);
        X = _mm_mul_ps(X, X);
        Z = _mm_add_ps(X, Z);
    }

    norm += Z[0] + Z[1] + Z[2] + Z[3];
    return sqrtf(norm);
}

float cblas_sdot(const int N, const float *m1, const int incX,
                 const float *m2, const int incY)
{
    if (N % 4 != 0) {
        fprintf(stderr, "cblas_sdot() expects N to be a multiple of 4.\n");
        exit(EXIT_FAILURE);
    }

    float dot = 0;
    __m128 X, Y;
    __m128 Z = _mm_setzero_ps();

    for (int i=0; i<N; i+=4) {
        X = _mm_load_ps(&m1[i]);
        Y = _mm_load_ps(&m2[i]);
        X = _mm_mul_ps(X, Y);
        Z = _mm_add_ps(X, Z);
    }

    dot += Z[0] + Z[1] + Z[2] + Z[3];
    return dot;
}

int _use_blas()
{
    return 0;
}
#endif // USE_BLAS
#ifdef __cplusplus
}
#endif // __cplusplus
