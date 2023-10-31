// # coding = utf-8
// # @Time    : 2023-04-25  15:22:35
// # @Author  : zhaosheng@nuaa.edu.cn
// # @Describe: .

#ifndef _COSINE_SIMILARITY_
#define _COSINE_SIMILARITY_

#include <cmath>
#include <float.h>
#include <immintrin.h>

template <typename T>
T Cosine_similarity(const T* __restrict__ const vectorA, // first vector's address
                    const T* __restrict__ const vectorB, // second vector's address
                    const int len) // length of the vector
{
    T mult_add = 0.0f;
    T a_norm2  = 0.0f;
    T b_norm2  = 0.0f;
    // Calculate the cosine similarity of two vectors
    for(int i = 0; i < len; i++) {
        mult_add += vectorA[i] * vectorB[i];
        a_norm2  += vectorA[i] * vectorA[i];
        b_norm2  += vectorB[i] * vectorB[i];
    }
    T a_norm = sqrt(a_norm2);
    T b_norm = sqrt(b_norm2);
    T similarity = mult_add / (a_norm * b_norm);
    return similarity;
}

// Step 7, SIMD
float inline reduceM128(const __m128 r)
{
    // 128位操作只需要16字节对齐
    __attribute__((aligned(16))) float f[4] = {0.0f};
//    assert(reinterpret_cast<int>(f)%16 == 0);
    _mm_store_ps(f, r); // f必须16字节对齐
//    _mm_storeu_ps(f, r); // f可以不必对齐

    return (f[0]+f[1]) + (f[2]+f[3]);
}

float inline reduceM256(const __m256 r)
{
    const __m128 hi  = _mm256_extractf128_ps(r, 1);
    const __m128 lo  = _mm256_extractf128_ps(r, 0);
    const __m128 sum = _mm_add_ps(hi, lo);

    return reduceM128(sum);
}

// Step 8, Fast InvSqrt from QUAKE-III
float InvSqrt(float x){
    const float xhalf = 0.5f*x;
    int i = *(int*)&x;
//    i = 0x5f3759df - (i>>1);
    i = 0x5f375a86 - (i>>1);
    x = *(float*)&i;
    x = x*(1.5f - xhalf*x*x);
    return x;
}

float
Cosine_similarity_avx(const float* const vectorA, const float* const vectorB, const int len)
{
    // 一个AVX指令每次可以计算8个32位浮点(float)，
    // 暂不考虑有不能整除部分的情况
    assert(len%8 == 0);

    const int step = len / 8;
    const __m256* one = (__m256*) vectorA;
    const __m256* two = (__m256*) vectorB;

    __m256 mult_add_m256 = _mm256_setzero_ps();
    __m256 a_norm_m256   = _mm256_setzero_ps();
    // Step 9，考虑b_norm2预计算
    __m256 b_norm_m256   = _mm256_setzero_ps();
    for(int i = 0; i < step; i++) {
        mult_add_m256 = _mm256_fmadd_ps(one[i], two[i], mult_add_m256); // a * b + c
        a_norm_m256   = _mm256_fmadd_ps(one[i], one[i], a_norm_m256);
        b_norm_m256   = _mm256_fmadd_ps(two[i], two[i], b_norm_m256);
    }

    const float mult_add = reduceM256(mult_add_m256);
    const float a_norm2  = reduceM256(a_norm_m256);
    const float b_norm2  = reduceM256(b_norm_m256);

    const float similarity = mult_add / (sqrt(a_norm2 * b_norm2) + FLT_MIN);
    //const float similarity = mult_add * InvSqrt(a_norm2 * b_norm2);

    return similarity;
}

#endif // !_COSINE_SIMILARITY_

