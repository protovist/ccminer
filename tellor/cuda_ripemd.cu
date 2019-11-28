/*
 * sha256 + ripemd CUDA implementation for SM 5.0 GPUS (and lower)
 * Tanguy Pruvot and Provos Alexis - Jul / Sep 2016
 * Sponsored by TELLOR.IO team
 */

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include <cuda_helper.h>
#include <cuda_vector_uint2x4.h>

#include <miner.h>

__global__ void tellor_difficulty_gpu(const uint64_t thread_count,
                                      uint32_t *d_hash,
                                      uint64_t *d_nonce);

__constant__ static uint32_t c_difficulty[8];

#if defined(__CUDA_ARCH__)
#include "cgbn/cgbn.h"

#define TPI 4

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, 256> env256_t;
typedef env256_t::cgbn_t cgbn_t;

__global__ void tellor_difficulty_gpu(const uint64_t thread_count,
                                      uint32_t *d_hash,
                                      uint64_t *d_nonce) {
  context_t bn_context;
  env256_t bn256_env(bn_context);
  cgbn_t hash, difficulty, remainder;

  const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
  if (thread >= thread_count) {
    return;
  }
  const uint32_t instance = thread / TPI;
  const uint32_t* input = &d_hash[instance<<3];
#if 0
  if (thread == 0) {
    for (int i = 0; i < 8; i++) {
    printf("%08x\n", c_difficulty[i]);
  }
  }
#endif
  
  cgbn_load(bn256_env, hash, (cgbn_mem_t<256>*)input);
  cgbn_load(bn256_env, difficulty, (cgbn_mem_t<256>*)&c_difficulty);
  cgbn_rem(bn256_env, remainder, hash, difficulty);

#if 0
  printf("%d %d: %08x mod %08x == %08x\n", instance, thread,
         cgbn_get_ui32(bn256_env, hash),
         cgbn_get_ui32(bn256_env, difficulty),
         cgbn_get_ui32(bn256_env, remainder));
#endif

if (cgbn_equals_ui32(bn256_env, remainder, 0)) {
    d_nonce[0] = instance;
}
  //  output[0] = (uint8_t)cgbn_equals_ui32(bn256_env, remainder, 0);
  //  cgbn_store(bn256_env, (cgbn_mem_t<256>*)&d_remainder[instance<<3], remainder);
}

#endif

#define TPI 4

__constant__  const uint32_t c_H256[8] = {
	0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
	0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U
};
__constant__ static uint32_t _ALIGN(16) c_K[64] = {
	0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
	0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
	0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
	0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
	0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
	0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
	0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
	0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
};

#ifdef __INTELLISENSE__
#define atomicExch(p,y) y
#define __byte_perm(x,y,c) x
#endif

// ------------------------------------------------------------------------------------------------

static const uint32_t cpu_H256[8] = {
	0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
	0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U
};

static const uint32_t cpu_K[64] = {
	0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
	0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
	0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
	0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
	0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
	0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
	0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
	0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
};

#define xor3b(a,b,c) (a ^ b ^ c)

__device__ __forceinline__ uint32_t bsg2_0(const uint32_t x)
{
	return xor3b(ROTR32(x,2),ROTR32(x,13),ROTR32(x,22));
}

__device__ __forceinline__ uint32_t bsg2_1(const uint32_t x)
{
	return xor3b(ROTR32(x,6),ROTR32(x,11),ROTR32(x,25));
}

__device__ __forceinline__ uint32_t ssg2_0(const uint32_t x)
{
	return xor3b(ROTR32(x,7),ROTR32(x,18),(x>>3));
}

__device__ __forceinline__ uint32_t ssg2_1(const uint32_t x)
{
	return xor3b(ROTR32(x,17),ROTR32(x,19),(x>>10));
}

__device__ __forceinline__ uint2 vectorizeswap(uint64_t v)
{
	uint2 result;
	asm("mov.b64 {%0,%1},%2; // vectorizeswap \n\t"
		: "=r"(result.y), "=r"(result.x) : "l"(v));
	return result;
}

#define Maj(x, y, z)    ((x & (y | z)) | (y & z))
#define Ch(a, b, c)     (((b^c) & a) ^ c)

__device__ __forceinline__
static void sha2_step(const uint32_t a,const uint32_t b,const uint32_t c, uint32_t &d,const uint32_t e,const uint32_t f,const uint32_t g, uint32_t &h,const uint32_t in, const uint32_t Kshared)
{
	const uint32_t t1 = h + bsg2_1(e) + Ch(e, f, g) + Kshared + in;
	h = t1 + bsg2_0(a) + Maj(a, b, c);
	d+= t1;

}

__device__ __forceinline__
static void sha256_round_body(uint32_t *const __restrict__ in, uint32_t * __restrict__ state,const uint32_t *const __restrict__ Kshared)
{
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	uint32_t f = state[5];
	uint32_t g = state[6];
	uint32_t h = state[7];

        sha2_step(a,b,c,d,e,f,g,h,in[0], Kshared[0]);
	sha2_step(h,a,b,c,d,e,f,g,in[1], Kshared[1]);
	sha2_step(g,h,a,b,c,d,e,f,in[2], Kshared[2]);
	sha2_step(f,g,h,a,b,c,d,e,in[3], Kshared[3]);
	sha2_step(e,f,g,h,a,b,c,d,in[4], Kshared[4]);
	sha2_step(d,e,f,g,h,a,b,c,in[5], Kshared[5]);
	sha2_step(c,d,e,f,g,h,a,b,in[6], Kshared[6]);
	sha2_step(b,c,d,e,f,g,h,a,in[7], Kshared[7]);
	sha2_step(a,b,c,d,e,f,g,h,in[8], Kshared[8]);
	sha2_step(h,a,b,c,d,e,f,g,in[9], Kshared[9]);
	sha2_step(g,h,a,b,c,d,e,f,in[10],Kshared[10]);
	sha2_step(f,g,h,a,b,c,d,e,in[11],Kshared[11]);
	sha2_step(e,f,g,h,a,b,c,d,in[12],Kshared[12]);
	sha2_step(d,e,f,g,h,a,b,c,in[13],Kshared[13]);
	sha2_step(c,d,e,f,g,h,a,b,in[14],Kshared[14]);
	sha2_step(b,c,d,e,f,g,h,a,in[15],Kshared[15]);

	#pragma unroll 3
	for (uint32_t i=0; i<3; i++)
	{
		#pragma unroll 16
		for (uint32_t j = 0; j < 16; j++){
			in[j] = in[j] + in[(j + 9) & 15] + ssg2_0(in[(j + 1) & 15]) + ssg2_1(in[(j + 14) & 15]);
		}
		sha2_step(a, b, c, d, e, f, g, h, in[0], Kshared[16 + 16 * i]);
		sha2_step(h, a, b, c, d, e, f, g, in[1], Kshared[17 + 16 * i]);
		sha2_step(g, h, a, b, c, d, e, f, in[2], Kshared[18 + 16 * i]);
		sha2_step(f, g, h, a, b, c, d, e, in[3], Kshared[19 + 16 * i]);
		sha2_step(e, f, g, h, a, b, c, d, in[4], Kshared[20 + 16 * i]);
		sha2_step(d, e, f, g, h, a, b, c, in[5], Kshared[21 + 16 * i]);
		sha2_step(c, d, e, f, g, h, a, b, in[6], Kshared[22 + 16 * i]);
		sha2_step(b, c, d, e, f, g, h, a, in[7], Kshared[23 + 16 * i]);
		sha2_step(a, b, c, d, e, f, g, h, in[8], Kshared[24 + 16 * i]);
		sha2_step(h, a, b, c, d, e, f, g, in[9], Kshared[25 + 16 * i]);
		sha2_step(g, h, a, b, c, d, e, f, in[10], Kshared[26 + 16 * i]);
		sha2_step(f, g, h, a, b, c, d, e, in[11], Kshared[27 + 16 * i]);
		sha2_step(e, f, g, h, a, b, c, d, in[12], Kshared[28 + 16 * i]);
		sha2_step(d, e, f, g, h, a, b, c, in[13], Kshared[29 + 16 * i]);
		sha2_step(c, d, e, f, g, h, a, b, in[14], Kshared[30 + 16 * i]);
		sha2_step(b, c, d, e, f, g, h, a, in[15], Kshared[31 + 16 * i]);
	}

	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
}

 __device__ __forceinline__
uint64_t cuda_swab64ll(const uint32_t x, const uint32_t y) {
	uint64_t r;
	asm("prmt.b32 %1, %1, 0, 0x0123; // swab64ll\n\t"
	    "prmt.b32 %2, %2, 0, 0x0123;\n\t"
	    "mov.b64 %0, {%1,%2};\n\t"
	  : "=l"(r): "r"(x), "r"(y) );
	return r;
}

__host__
void tellor_sha256_init(int thr_id)
{
	cudaMemcpyToSymbol(c_K, cpu_K, sizeof(cpu_K), 0, cudaMemcpyHostToDevice);
}

__host__ void tellor_set_difficulty(const uint32_t *difficulty) {
  cudaMemcpyToSymbol(c_difficulty, difficulty, sizeof(uint32_t) * 8, 0,
                     cudaMemcpyHostToDevice);
}

// ------------------------------------------------------------------------------------------

static __constant__ const uint32_t c_IV[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };

static __constant__ const uint32_t c_K1[5] = { 0, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E };

static __constant__ const uint32_t c_K2[5] = { 0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0 };

__device__ __forceinline__
static uint32_t ROTATE(const uint32_t x,const uint32_t r){
	if(r==8)
		return __byte_perm(x, 0, 0x2103);
	else
		return ROTL32(x,r);
}

/*
 * Round functions for RIPEMD-160.
 */
#define F1(x, y, z)   (x ^ y ^ z)
#define F2(x, y, z)   ((x & (y ^ z)) ^ z)
#define F3(x, y, z)   ((x | ~y) ^ z)
#define F4(x, y, z)   (y ^ ((x ^ y) & z))
#define F5(x, y, z)   (x ^ (y | ~z))

/*
 * Round constants for RIPEMD-160.
 */
#define RR(a, b, c, d, e, f, s, r, k) { \
	a = e + ROTATE((a + r + k + f(b, c, d)), s); \
	c = ROTL32(c, 10); \
}

#define ROUND1(a, b, c, d, e, f, s, r, k) \
	RR(a[0], b[0], c[0], d[0], e[0], f, s, r, c_K1[k])

#define ROUND2(a, b, c, d, e, f, s, r, k) \
	RR(a[1], b[1], c[1], d[1], e[1], f, s, r, c_K2[k])

#define RIPEMD160_ROUND_BODY(in, h) { \
	uint32_t A[2], B[2], C[2], D[2], E[2]; \
	uint32_t tmp; \
\
	A[0] = A[1] = h[0]; \
	B[0] = B[1] = h[1]; \
	C[0] = C[1] = h[2]; \
	D[0] = D[1] = h[3]; \
	E[0] = E[1] = h[4]; \
\
	ROUND1(A, B, C, D, E, F1, 11, in[ 0],  0); \
	ROUND1(E, A, B, C, D, F1, 14, in[ 1],  0); \
	ROUND1(D, E, A, B, C, F1, 15, in[ 2],  0); \
	ROUND1(C, D, E, A, B, F1, 12, in[ 3],  0); \
	ROUND1(B, C, D, E, A, F1,  5, in[ 4],  0); \
	ROUND1(A, B, C, D, E, F1,  8, in[ 5],  0); \
	ROUND1(E, A, B, C, D, F1,  7, in[ 6],  0); \
	ROUND1(D, E, A, B, C, F1,  9, in[ 7],  0); \
	ROUND1(C, D, E, A, B, F1, 11, in[ 8],  0); \
	ROUND1(B, C, D, E, A, F1, 13, in[ 9],  0); \
	ROUND1(A, B, C, D, E, F1, 14, in[10],  0); \
	ROUND1(E, A, B, C, D, F1, 15, in[11],  0); \
	ROUND1(D, E, A, B, C, F1,  6, in[12],  0); \
	ROUND1(C, D, E, A, B, F1,  7, in[13],  0); \
	ROUND1(B, C, D, E, A, F1,  9, in[14],  0); \
	ROUND1(A, B, C, D, E, F1,  8, in[15],  0); \
\
	ROUND1(E, A, B, C, D, F2,  7, in[ 7],  1); \
	ROUND1(D, E, A, B, C, F2,  6, in[ 4],  1); \
	ROUND1(C, D, E, A, B, F2,  8, in[13],  1); \
	ROUND1(B, C, D, E, A, F2, 13, in[ 1],  1); \
	ROUND1(A, B, C, D, E, F2, 11, in[10],  1); \
	ROUND1(E, A, B, C, D, F2,  9, in[ 6],  1); \
	ROUND1(D, E, A, B, C, F2,  7, in[15],  1); \
	ROUND1(C, D, E, A, B, F2, 15, in[ 3],  1); \
	ROUND1(B, C, D, E, A, F2,  7, in[12],  1); \
	ROUND1(A, B, C, D, E, F2, 12, in[ 0],  1); \
	ROUND1(E, A, B, C, D, F2, 15, in[ 9],  1); \
	ROUND1(D, E, A, B, C, F2,  9, in[ 5],  1); \
	ROUND1(C, D, E, A, B, F2, 11, in[ 2],  1); \
	ROUND1(B, C, D, E, A, F2,  7, in[14],  1); \
	ROUND1(A, B, C, D, E, F2, 13, in[11],  1); \
	ROUND1(E, A, B, C, D, F2, 12, in[ 8],  1); \
\
	ROUND1(D, E, A, B, C, F3, 11, in[ 3],  2); \
	ROUND1(C, D, E, A, B, F3, 13, in[10],  2); \
	ROUND1(B, C, D, E, A, F3,  6, in[14],  2); \
	ROUND1(A, B, C, D, E, F3,  7, in[ 4],  2); \
	ROUND1(E, A, B, C, D, F3, 14, in[ 9],  2); \
	ROUND1(D, E, A, B, C, F3,  9, in[15],  2); \
	ROUND1(C, D, E, A, B, F3, 13, in[ 8],  2); \
	ROUND1(B, C, D, E, A, F3, 15, in[ 1],  2); \
	ROUND1(A, B, C, D, E, F3, 14, in[ 2],  2); \
	ROUND1(E, A, B, C, D, F3,  8, in[ 7],  2); \
	ROUND1(D, E, A, B, C, F3, 13, in[ 0],  2); \
	ROUND1(C, D, E, A, B, F3,  6, in[ 6],  2); \
	ROUND1(B, C, D, E, A, F3,  5, in[13],  2); \
	ROUND1(A, B, C, D, E, F3, 12, in[11],  2); \
	ROUND1(E, A, B, C, D, F3,  7, in[ 5],  2); \
	ROUND1(D, E, A, B, C, F3,  5, in[12],  2); \
\
	ROUND1(C, D, E, A, B, F4, 11, in[ 1],  3); \
	ROUND1(B, C, D, E, A, F4, 12, in[ 9],  3); \
	ROUND1(A, B, C, D, E, F4, 14, in[11],  3); \
	ROUND1(E, A, B, C, D, F4, 15, in[10],  3); \
	ROUND1(D, E, A, B, C, F4, 14, in[ 0],  3); \
	ROUND1(C, D, E, A, B, F4, 15, in[ 8],  3); \
	ROUND1(B, C, D, E, A, F4,  9, in[12],  3); \
	ROUND1(A, B, C, D, E, F4,  8, in[ 4],  3); \
	ROUND1(E, A, B, C, D, F4,  9, in[13],  3); \
	ROUND1(D, E, A, B, C, F4, 14, in[ 3],  3); \
	ROUND1(C, D, E, A, B, F4,  5, in[ 7],  3); \
	ROUND1(B, C, D, E, A, F4,  6, in[15],  3); \
	ROUND1(A, B, C, D, E, F4,  8, in[14],  3); \
	ROUND1(E, A, B, C, D, F4,  6, in[ 5],  3); \
	ROUND1(D, E, A, B, C, F4,  5, in[ 6],  3); \
	ROUND1(C, D, E, A, B, F4, 12, in[ 2],  3); \
\
	ROUND1(B, C, D, E, A, F5,  9, in[ 4],  4); \
	ROUND1(A, B, C, D, E, F5, 15, in[ 0],  4); \
	ROUND1(E, A, B, C, D, F5,  5, in[ 5],  4); \
	ROUND1(D, E, A, B, C, F5, 11, in[ 9],  4); \
	ROUND1(C, D, E, A, B, F5,  6, in[ 7],  4); \
	ROUND1(B, C, D, E, A, F5,  8, in[12],  4); \
	ROUND1(A, B, C, D, E, F5, 13, in[ 2],  4); \
	ROUND1(E, A, B, C, D, F5, 12, in[10],  4); \
	ROUND1(D, E, A, B, C, F5,  5, in[14],  4); \
	ROUND1(C, D, E, A, B, F5, 12, in[ 1],  4); \
	ROUND1(B, C, D, E, A, F5, 13, in[ 3],  4); \
	ROUND1(A, B, C, D, E, F5, 14, in[ 8],  4); \
	ROUND1(E, A, B, C, D, F5, 11, in[11],  4); \
	ROUND1(D, E, A, B, C, F5,  8, in[ 6],  4); \
	ROUND1(C, D, E, A, B, F5,  5, in[15],  4); \
	ROUND1(B, C, D, E, A, F5,  6, in[13],  4); \
\
	ROUND2(A, B, C, D, E, F5,  8, in[ 5],  0); \
	ROUND2(E, A, B, C, D, F5,  9, in[14],  0); \
	ROUND2(D, E, A, B, C, F5,  9, in[ 7],  0); \
	ROUND2(C, D, E, A, B, F5, 11, in[ 0],  0); \
	ROUND2(B, C, D, E, A, F5, 13, in[ 9],  0); \
	ROUND2(A, B, C, D, E, F5, 15, in[ 2],  0); \
	ROUND2(E, A, B, C, D, F5, 15, in[11],  0); \
	ROUND2(D, E, A, B, C, F5,  5, in[ 4],  0); \
	ROUND2(C, D, E, A, B, F5,  7, in[13],  0); \
	ROUND2(B, C, D, E, A, F5,  7, in[ 6],  0); \
	ROUND2(A, B, C, D, E, F5,  8, in[15],  0); \
	ROUND2(E, A, B, C, D, F5, 11, in[ 8],  0); \
	ROUND2(D, E, A, B, C, F5, 14, in[ 1],  0); \
	ROUND2(C, D, E, A, B, F5, 14, in[10],  0); \
	ROUND2(B, C, D, E, A, F5, 12, in[ 3],  0); \
	ROUND2(A, B, C, D, E, F5,  6, in[12],  0); \
\
	ROUND2(E, A, B, C, D, F4,  9, in[ 6],  1); \
	ROUND2(D, E, A, B, C, F4, 13, in[11],  1); \
	ROUND2(C, D, E, A, B, F4, 15, in[ 3],  1); \
	ROUND2(B, C, D, E, A, F4,  7, in[ 7],  1); \
	ROUND2(A, B, C, D, E, F4, 12, in[ 0],  1); \
	ROUND2(E, A, B, C, D, F4,  8, in[13],  1); \
	ROUND2(D, E, A, B, C, F4,  9, in[ 5],  1); \
	ROUND2(C, D, E, A, B, F4, 11, in[10],  1); \
	ROUND2(B, C, D, E, A, F4,  7, in[14],  1); \
	ROUND2(A, B, C, D, E, F4,  7, in[15],  1); \
	ROUND2(E, A, B, C, D, F4, 12, in[ 8],  1); \
	ROUND2(D, E, A, B, C, F4,  7, in[12],  1); \
	ROUND2(C, D, E, A, B, F4,  6, in[ 4],  1); \
	ROUND2(B, C, D, E, A, F4, 15, in[ 9],  1); \
	ROUND2(A, B, C, D, E, F4, 13, in[ 1],  1); \
	ROUND2(E, A, B, C, D, F4, 11, in[ 2],  1); \
\
	ROUND2(D, E, A, B, C, F3,  9, in[15],  2); \
	ROUND2(C, D, E, A, B, F3,  7, in[ 5],  2); \
	ROUND2(B, C, D, E, A, F3, 15, in[ 1],  2); \
	ROUND2(A, B, C, D, E, F3, 11, in[ 3],  2); \
	ROUND2(E, A, B, C, D, F3,  8, in[ 7],  2); \
	ROUND2(D, E, A, B, C, F3,  6, in[14],  2); \
	ROUND2(C, D, E, A, B, F3,  6, in[ 6],  2); \
	ROUND2(B, C, D, E, A, F3, 14, in[ 9],  2); \
	ROUND2(A, B, C, D, E, F3, 12, in[11],  2); \
	ROUND2(E, A, B, C, D, F3, 13, in[ 8],  2); \
	ROUND2(D, E, A, B, C, F3,  5, in[12],  2); \
	ROUND2(C, D, E, A, B, F3, 14, in[ 2],  2); \
	ROUND2(B, C, D, E, A, F3, 13, in[10],  2); \
	ROUND2(A, B, C, D, E, F3, 13, in[ 0],  2); \
	ROUND2(E, A, B, C, D, F3,  7, in[ 4],  2); \
	ROUND2(D, E, A, B, C, F3,  5, in[13],  2); \
\
	ROUND2(C, D, E, A, B, F2, 15, in[ 8],  3); \
	ROUND2(B, C, D, E, A, F2,  5, in[ 6],  3); \
	ROUND2(A, B, C, D, E, F2,  8, in[ 4],  3); \
	ROUND2(E, A, B, C, D, F2, 11, in[ 1],  3); \
	ROUND2(D, E, A, B, C, F2, 14, in[ 3],  3); \
	ROUND2(C, D, E, A, B, F2, 14, in[11],  3); \
	ROUND2(B, C, D, E, A, F2,  6, in[15],  3); \
	ROUND2(A, B, C, D, E, F2, 14, in[ 0],  3); \
	ROUND2(E, A, B, C, D, F2,  6, in[ 5],  3); \
	ROUND2(D, E, A, B, C, F2,  9, in[12],  3); \
	ROUND2(C, D, E, A, B, F2, 12, in[ 2],  3); \
	ROUND2(B, C, D, E, A, F2,  9, in[13],  3); \
	ROUND2(A, B, C, D, E, F2, 12, in[ 9],  3); \
	ROUND2(E, A, B, C, D, F2,  5, in[ 7],  3); \
	ROUND2(D, E, A, B, C, F2, 15, in[10],  3); \
	ROUND2(C, D, E, A, B, F2,  8, in[14],  3); \
\
	ROUND2(B, C, D, E, A, F1,  8, in[12],  4); \
	ROUND2(A, B, C, D, E, F1,  5, in[15],  4); \
	ROUND2(E, A, B, C, D, F1, 12, in[10],  4); \
	ROUND2(D, E, A, B, C, F1,  9, in[ 4],  4); \
	ROUND2(C, D, E, A, B, F1, 12, in[ 1],  4); \
	ROUND2(B, C, D, E, A, F1,  5, in[ 5],  4); \
	ROUND2(A, B, C, D, E, F1, 14, in[ 8],  4); \
	ROUND2(E, A, B, C, D, F1,  6, in[ 7],  4); \
	ROUND2(D, E, A, B, C, F1,  8, in[ 6],  4); \
	ROUND2(C, D, E, A, B, F1, 13, in[ 2],  4); \
	ROUND2(B, C, D, E, A, F1,  6, in[13],  4); \
	ROUND2(A, B, C, D, E, F1,  5, in[14],  4); \
	ROUND2(E, A, B, C, D, F1, 15, in[ 0],  4); \
	ROUND2(D, E, A, B, C, F1, 13, in[ 3],  4); \
	ROUND2(C, D, E, A, B, F1, 11, in[ 9],  4); \
	ROUND2(B, C, D, E, A, F1, 11, in[11],  4); \
\
	tmp  = h[1] + C[0] + D[1]; \
	h[1] = h[2] + D[0] + E[1]; \
	h[2] = h[3] + E[0] + A[1]; \
	h[3] = h[4] + A[0] + B[1]; \
	h[4] = h[0] + B[0] + C[1]; \
	h[0] = tmp; \
}

__global__
#if CUDA_VERSION > 6050
__launch_bounds__(1024,2) /* to force 32 regs */
#endif
void tellor_ripemd(const uint64_t threads, uint32_t *Hash256){
	uint32_t dat[16];
	uint32_t h[5];
	uint32_t buf[8]; // align for vectorize

        int thread = blockIdx.x * blockDim.x + threadIdx.x;
	if(thread<threads){

		uint32_t* input = (uint32_t*) (&Hash256[thread<<3]);

		*(uint2x4*)&dat[0] = __ldg4((uint2x4*)&input[0]);

		dat[8] = 0x80;

		#pragma unroll 7
		for (int i=9;i<16;i++) dat[i] = 0;

		dat[14] = 0x100; // size in bits

		#pragma unroll 5
		for (int i=0; i<5; i++)
			h[i] = c_IV[i];

                RIPEMD160_ROUND_BODY(dat, h);

                //		#pragma unroll 5
                //		for (int i=0; i<5; i++)
                //			buf[i] = h[i];

		// second 32 bytes block hash

                //		*(uint2x4*)&dat[0] = __ldg4((uint2x4*)&input[8]);

                //		dat[8] = 0x80;

                //		#pragma unroll 7
                //		for (int i=9;i<16;i++) dat[i] = 0;

                //		dat[14] = 0x100; // size in bits

                //		#pragma unroll 5
                //		for (int i=0; i<5; i++)
                //			h[i] = c_IV[i];

                //		RIPEMD160_ROUND_BODY(dat, h);

		#pragma unroll 5
		for (int i=0;i<5;i++) dat[i] = h[i];
                //		#pragma unroll 5
                //		for (int i=0;i<5;i++) dat[i+5] = cuda_swab32(h[i]);

		*(uint2x4*)&input[ 0] = *(uint2x4*)&dat[ 0];
                //		*(uint2*)&input[ 8] = *(uint2*)&dat[ 8];
#if 0
                printf("GPU RIPEMD HASH:\n");
                for (int i = 0; i < 5; i++) {
                  printf("%08x ", Hash256[i]);
                }
                printf("\n");
#endif
	}
}

__global__
#if CUDA_VERSION > 6050
__launch_bounds__(768,2) /* to force 32 regs */
#endif
    void tellor_sha256_gpu_hash_final(const uint64_t threads, uint32_t *Hash256)
{
	const uint64_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	uint32_t dat[16] = {0};
	uint32_t buf[8]; // align for vectorize
	if (thread < threads)
	{

          uint32_t* input = (uint32_t*) (&Hash256[thread<<3]);
          //*(uint2x4*)&dat[0] = __ldg4((uint2x4*)&input[0]);
          //          *(uint2*)&dat[8] = __ldg((uint2*)&input[8]);
#pragma unroll 5
          for (int i = 0; i < 5; i++) {
            dat[i] = cuda_swab32(input[i]);
          }
          dat[5] = 0x80000000;
#pragma unroll 10
          for (int i=6; i<15; i++) dat[i] = 0;
          dat[15] = 0xa0;

          *(uint2x4*)&buf[0] = *(uint2x4*)&c_H256[0];

          sha256_round_body(dat, buf, c_K); // s_K uses too many regs
          //          *(uint2x4*)&Hash256[thread<<3] = *(uint2x4*)&buf[0];

          //          printf("GPU SHA256 HASH:\n");
                #pragma unroll 8
                for (int i = 0; i < 8; i++) {
                  input[i] = (buf[7-i]);
                  //                  printf("%08x ", input[i]);
                }
                //                printf("\n");
	}
}

__host__
    void tellor_sha256_hash_final(int thr_id, uint64_t threads, uint32_t *d_inputHash)
{
	int threadsperblock;

	threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

        tellor_sha256_gpu_hash_final <<<grid, block>>> (threads, (uint32_t*) d_inputHash);
        //tellor_sha256_gpu_hash_final <<<4,1>>> (threads, (uint32_t*) d_inputHash);
}

__host__
    void tellor_ripemd_hash(int thr_id, uint64_t threads, uint32_t *d_inputHash)
{
	int threadsperblock;

	threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

        tellor_ripemd <<<grid, block>>> (threads,(uint32_t*) d_inputHash);
        //tellor_ripemd <<<4,1>>> (threads,(uint32_t*) d_inputHash);
}

__host__ void tellor_difficulty(int gpu_id, uint64_t thread_count, uint32_t* d_hash,
                                uint64_t *d_nonce) {
  int threadsperblock = 512;
  int ipb = threadsperblock/TPI;
  dim3 grid((thread_count + ipb - 1) / threadsperblock);
  dim3 block(threadsperblock);
  
  tellor_difficulty_gpu <<<grid, block>>> (thread_count, d_hash, d_nonce);
  //tellor_difficulty_gpu<<<4,4>>>(thread_count, d_hash, d_nonce);
}
