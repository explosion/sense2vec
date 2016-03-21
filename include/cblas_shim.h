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
#include <xmmintrin.h>

#if defined(_MSC_VER)
#define ALIGNAS(byte_alignment) __declspec(align(byte_alignment))
#elif defined(__GNUC__)
#define ALIGNAS(byte_alignment) __attribute__((aligned(byte_alignment)))
#endif

float cblas_snrm2(const int N, const float *m1, const int incX)
{
    if (N % 4 != 0) {
        fprintf(stderr, "cblas_snrm2() expects N to be a multiple of 4.\n");
        exit(EXIT_FAILURE);
    }

    float norm = 0;
    ALIGNAS(16) float z[4];
    __m128 X;
    __m128 Z = _mm_setzero_ps();

    for (int i=0; i<N; i+=4) {
        X = _mm_load_ps(&m1[i]);
        X = _mm_mul_ps(X, X);
        Z = _mm_add_ps(X, Z);
    }

    _mm_store_ps(z, Z);
    norm += z[0] + z[1] + z[2] + z[3];
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
    ALIGNAS(16) float z[4];
    __m128 X, Y;
    __m128 Z = _mm_setzero_ps();

    for (int i=0; i<N; i+=4) {
        X = _mm_load_ps(&m1[i]);
        Y = _mm_load_ps(&m2[i]);
        X = _mm_mul_ps(X, Y);
        Z = _mm_add_ps(X, Z);
    }

    _mm_store_ps(z, Z);
    dot += z[0] + z[1] + z[2] + z[3];
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
