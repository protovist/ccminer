#include "miner.h"

extern "C" {
#include <stdint.h>
#include <memory.h>
}

#include "cuda_helper.h"

static const uint64_t host_sha3_round_constants[24] = {
	0x0000000000000001ull, 0x0000000000008082ull,
	0x800000000000808aull, 0x8000000080008000ull,
	0x000000000000808bull, 0x0000000080000001ull,
	0x8000000080008081ull, 0x8000000000008009ull,
	0x000000000000008aull, 0x0000000000000088ull,
	0x0000000080008009ull, 0x000000008000000aull,
	0x000000008000808bull, 0x800000000000008bull,
	0x8000000000008089ull, 0x8000000000008003ull,
	0x8000000000008002ull, 0x8000000000000080ull,
	0x000000000000800aull, 0x800000008000000aull,
	0x8000000080008081ull, 0x8000000000008080ull,
	0x0000000080000001ull, 0x8000000080008008ull
};

static uint64_t *d_KNonce[MAX_GPUS];

__constant__ uint32_t pTarget[2];
__constant__ uint64_t sha3_round_constants[24];
__constant__ uint64_t c_PaddedMessage80[25]; // padded message (80 bytes + padding?)

__device__ __forceinline__
static void sha3_blockv30(uint64_t *s, const uint64_t *sha3_round_constants)
{
	size_t i;
	uint64_t t[5], u[5], v, w;

	/* absorb input */

	for (i = 0; i < 24; i++) {
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[0] = t[4] ^ ROTL64(t[1], 1);
		u[1] = t[0] ^ ROTL64(t[2], 1);
		u[2] = t[1] ^ ROTL64(t[3], 1);
		u[3] = t[2] ^ ROTL64(t[4], 1);
		u[4] = t[3] ^ ROTL64(t[0], 1);

		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
		s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
		s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
		s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
		s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
		s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[ 1];
		s[ 1] = ROTL64(s[ 6], 44);
		s[ 6] = ROTL64(s[ 9], 20);
		s[ 9] = ROTL64(s[22], 61);
		s[22] = ROTL64(s[14], 39);
		s[14] = ROTL64(s[20], 18);
		s[20] = ROTL64(s[ 2], 62);
		s[ 2] = ROTL64(s[12], 43);
		s[12] = ROTL64(s[13], 25);
		s[13] = ROTL64(s[19],  8);
		s[19] = ROTL64(s[23], 56);
		s[23] = ROTL64(s[15], 41);
		s[15] = ROTL64(s[ 4], 27);
		s[ 4] = ROTL64(s[24], 14);
		s[24] = ROTL64(s[21],  2);
		s[21] = ROTL64(s[ 8], 55);
		s[ 8] = ROTL64(s[16], 45);
		s[16] = ROTL64(s[ 5], 36);
		s[ 5] = ROTL64(s[ 3], 28);
		s[ 3] = ROTL64(s[18], 21);
		s[18] = ROTL64(s[17], 15);
		s[17] = ROTL64(s[11], 10);
		s[11] = ROTL64(s[ 7],  6);
		s[ 7] = ROTL64(s[10],  3);
		s[10] = ROTL64(    v,  1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v = s[ 0]; w = s[ 1]; s[ 0] ^= (~w) & s[ 2]; s[ 1] ^= (~s[ 2]) & s[ 3]; s[ 2] ^= (~s[ 3]) & s[ 4]; s[ 3] ^= (~s[ 4]) & v; s[ 4] ^= (~v) & w;
		v = s[ 5]; w = s[ 6]; s[ 5] ^= (~w) & s[ 7]; s[ 6] ^= (~s[ 7]) & s[ 8]; s[ 7] ^= (~s[ 8]) & s[ 9]; s[ 8] ^= (~s[ 9]) & v; s[ 9] ^= (~v) & w;
		v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
		v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
		v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

		/* iota: a[0,0] ^= round constant */
		s[0] ^= sha3_round_constants[i];
	}
}

__device__
char* format_nonce(uint64_t nonce, char* buffer) { 
  const char digits[] =
      "0001020304050607080910111213141516171819"
      "2021222324252627282930313233343536373839"
      "4041424344454647484950515253545556575859"
      "6061626364656667686970717273747576777879"
      "8081828384858687888990919293949596979899";
  char* position = buffer + 19;
  while (nonce >= 100) {
    unsigned index = static_cast<unsigned>((nonce % 100) * 2);
    nonce /= 100;
    *--position = digits[index + 1];
    *--position = digits[index];
  }
  if (nonce < 10) {
    *--position = static_cast<char>('0' + nonce);
    return position;
  }
  unsigned index = static_cast<unsigned>(nonce * 2);
  *--position = digits[index + 1];
  *--position = digits[index];
  return position;
} 

__global__
void cruz_gpu_hash(uint32_t threads, uint64_t startNounce, uint64_t *resNounce)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
          uint64_t nounce = startNounce + thread + 1000000000000000;
		uint64_t sha3_gpu_state[25];
#pragma unroll 25
		for (int i=0; i<25; i++) {
                  sha3_gpu_state[i] = c_PaddedMessage80[i];
		}
                //                uint64_t n = 6372506688733637;
                //                printf("format nonce: %s\n", format_nonce(n));
                
                //                char* nonce = "6372506688733637";
                char buffer[20];
                char* nonce = format_nonce(nounce, buffer);
                //                if (nounce % (65536 * 4) == 0) {
                //                                  printf("%s\n", nonce);
                //                }
#pragma unroll 16
                for (int i = 0; i < 16; i++) {
                  ((uint8_t*)sha3_gpu_state)[73 + i] ^= nonce[i];
                }
                //                memcpy(&((uint8_t*)sha3_gpu_state)[73], nonce, 16);
		sha3_blockv30(sha3_gpu_state, sha3_round_constants);
                //                for (int i = 0; i < 8; i++) {
                //                  printf("%08x ", pTarget[i]);
                //                }
                //                  printf("%016lx %016lx (%" PRIu64 ")\n",
                //                         (sha3_gpu_state)[0],
                //                (sha3_gpu_state)[1],
                //                         nounce);
		if (cuda_swab32(((uint32_t*)sha3_gpu_state)[0]) <= pTarget[0] &&
                    cuda_swab32(((uint32_t*)sha3_gpu_state)[1]) <= pTarget[1]) {
                  //                  printf("nonce: %s\n", nonce);
                  //                  printf("%08x %08x (%08x %08x) (%" PRIu64 ")\n",
                  //                         ((uint32_t*)sha3_gpu_state)[0],
                  //                         ((uint32_t*)sha3_gpu_state)[1],
                  //                         pTarget[0], pTarget[1],
                  //                         nounce);
                  resNounce[0] = nounce;
                }
	}
}

__host__
void cruz_cpu_hash(int thr_id, uint32_t threads, uint64_t startNounce, uint64_t *resNonces, int order)
{
	cudaMemset(d_KNonce[thr_id], 0xff, sizeof(uint64_t));
	const uint32_t threadsperblock = 128;

        dim3 grid((threads + threadsperblock-1)/threadsperblock);
        dim3 block(threadsperblock);
        //dim3 grid = 1;
        //dim3 block = 1;

	size_t shared_size = 0;

        cruz_gpu_hash<<<grid, block, shared_size>>>(threads, startNounce, d_KNonce[thr_id]);
        CUDA_SAFE_CALL(cudaMemcpy(resNonces, d_KNonce[thr_id], sizeof(uint64_t), cudaMemcpyDeviceToHost));
        // 	cudaThreadSynchronize();
}

#if 0
__global__ __launch_bounds__(256,3)
void sha3256_sm3_gpu_hash_32(uint32_t threads, uint32_t startNounce, uint64_t *outputHash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
#if __CUDA_ARCH__ >= 350 /* tpr: to double check if faster on SM5+ */
		uint2 sha3_gpu_state[25];
		#pragma unroll 25
		for (int i = 0; i<25; i++) {
			if (i<4) sha3_gpu_state[i] = vectorize(outputHash[i*threads+thread]);
			else     sha3_gpu_state[i] = make_uint2(0, 0);
		}
		sha3_gpu_state[4]  = make_uint2(6, 0);
		sha3_gpu_state[16] = make_uint2(0, 0x80000000);
		sha3_blockv35(sha3_gpu_state, sha3_round_constants);

		#pragma unroll 4
		for (int i=0; i<4; i++)
			outputHash[i*threads+thread] = devectorize(sha3_gpu_state[i]);
#else
		uint64_t sha3_gpu_state[25];
		#pragma unroll 25
		for (int i = 0; i<25; i++) {
			if (i<4)
				sha3_gpu_state[i] = outputHash[i*threads+thread];
			else
				sha3_gpu_state[i] = 0;
		}
		sha3_gpu_state[4]  = 0x0000000000000006;
		sha3_gpu_state[16] = 0x8000000000000000;

		sha3_blockv30(sha3_gpu_state, sha3_round_constants);
		#pragma unroll 4
		for (int i = 0; i<4; i++)
			outputHash[i*threads + thread] = sha3_gpu_state[i];
#endif
	}
}

__host__
void cruz_sm3_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, int order)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	sha3256_sm3_gpu_hash_32 <<<grid, block>>> (threads, startNounce, d_outputHash);
	MyStreamSynchronize(NULL, order, thr_id);
}
#endif

__host__
void sha3_keccakf_cu(uint64_t st[25]) {
  // constants
  const uint64_t keccakf_rndc[24] = {
      0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
      0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
      0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
      0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
      0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
      0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
      0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
      0x8000000000008080, 0x0000000080000001, 0x8000000080008008};
  const int keccakf_rotc[24] = {1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
                                27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44};
  const int keccakf_piln[24] = {10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
                                15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1};

  // variables
  int i, j, r;
  uint64_t t, bc[5];

#if 0
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
  uint8_t *v;

  // endianess conversion. this is redundant on little-endian targets
  for (i = 0; i < 25; i++) {
    v = (uint8_t *)&st[i];
    st[i] = ((uint64_t)v[0]) | (((uint64_t)v[1]) << 8) |
            (((uint64_t)v[2]) << 16) | (((uint64_t)v[3]) << 24) |
            (((uint64_t)v[4]) << 32) | (((uint64_t)v[5]) << 40) |
            (((uint64_t)v[6]) << 48) | (((uint64_t)v[7]) << 56);
  }
#endif
#endif

  // actual iteration
  for (r = 0; r < 24; r++) {

    // Theta
    for (i = 0; i < 5; i++)
      bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

    for (i = 0; i < 5; i++) {
      t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
      for (j = 0; j < 25; j += 5)
        st[j + i] ^= t;
    }

    // Rho Pi
    t = st[1];
    for (i = 0; i < 24; i++) {
      j = keccakf_piln[i];
      bc[0] = st[j];
      st[j] = ROTL64(t, keccakf_rotc[i]);
      t = bc[0];
    }

    //  Chi
    for (j = 0; j < 25; j += 5) {
      for (i = 0; i < 5; i++)
        bc[i] = st[j + i];
      for (i = 0; i < 5; i++)
        st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
    }

    //  Iota
    st[0] ^= keccakf_rndc[r];
  }

#if 0
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
  // endianess conversion. this is redundant on little-endian targets
  for (i = 0; i < 25; i++) {
    v = (uint8_t *)&st[i];
    t = st[i];
    v[0] = t & 0xFF;
    v[1] = (t >> 8) & 0xFF;
    v[2] = (t >> 16) & 0xFF;
    v[3] = (t >> 24) & 0xFF;
    v[4] = (t >> 32) & 0xFF;
    v[5] = (t >> 40) & 0xFF;
    v[6] = (t >> 48) & 0xFF;
    v[7] = (t >> 56) & 0xFF;
  }
#endif
#endif
}

__host__
static void sha3_block(uint64_t *s, const uint64_t *sha3_round_constants)
{
	size_t i;
	uint64_t t[5], u[5], v, w;

	/* absorb input */

	for (i = 0; i < 24; i++) {
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[0] = t[4] ^ ROTL64(t[1], 1);
		u[1] = t[0] ^ ROTL64(t[2], 1);
		u[2] = t[1] ^ ROTL64(t[3], 1);
		u[3] = t[2] ^ ROTL64(t[4], 1);
		u[4] = t[3] ^ ROTL64(t[0], 1);

		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
		s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
		s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
		s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
		s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
		s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[ 1];
		s[ 1] = ROTL64(s[ 6], 44);
		s[ 6] = ROTL64(s[ 9], 20);
		s[ 9] = ROTL64(s[22], 61);
		s[22] = ROTL64(s[14], 39);
		s[14] = ROTL64(s[20], 18);
		s[20] = ROTL64(s[ 2], 62);
		s[ 2] = ROTL64(s[12], 43);
		s[12] = ROTL64(s[13], 25);
		s[13] = ROTL64(s[19],  8);
		s[19] = ROTL64(s[23], 56);
		s[23] = ROTL64(s[15], 41);
		s[15] = ROTL64(s[ 4], 27);
		s[ 4] = ROTL64(s[24], 14);
		s[24] = ROTL64(s[21],  2);
		s[21] = ROTL64(s[ 8], 55);
		s[ 8] = ROTL64(s[16], 45);
		s[16] = ROTL64(s[ 5], 36);
		s[ 5] = ROTL64(s[ 3], 28);
		s[ 3] = ROTL64(s[18], 21);
		s[18] = ROTL64(s[17], 15);
		s[17] = ROTL64(s[11], 10);
		s[11] = ROTL64(s[ 7],  6);
		s[ 7] = ROTL64(s[10],  3);
		s[10] = ROTL64(    v,  1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v = s[ 0]; w = s[ 1]; s[ 0] ^= (~w) & s[ 2]; s[ 1] ^= (~s[ 2]) & s[ 3]; s[ 2] ^= (~s[ 3]) & s[ 4]; s[ 3] ^= (~s[ 4]) & v; s[ 4] ^= (~v) & w;
		v = s[ 5]; w = s[ 6]; s[ 5] ^= (~w) & s[ 7]; s[ 6] ^= (~s[ 7]) & s[ 8]; s[ 7] ^= (~s[ 8]) & s[ 9]; s[ 8] ^= (~s[ 9]) & v; s[ 9] ^= (~v) & w;
		v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
		v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
		v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

		/* iota: a[0,0] ^= round constant */
		s[0] ^= sha3_round_constants[i];
	}
}

__host__
void cruz_setBlock_345(uint64_t* data, const void *pTargetIn, size_t block_size) {
  uint64_t state[25] = {0};
  
#if 0
  uint64_t state[25] = {
    0x59830c7b2e360638, 0x25c8054aa1fe772b, 0xa63a7afb2adab057, 0x8bc6407a70705868,
    0x2ab5a20f0f258c5a, 0x50f42c70cc6b62ab, 0xee7a078b1654c33a, 0x0995782d81eef31e,
    0xf9f8765b14dbfbf1, 0x4c0152ba68c7c487, 0xa2d4c455c5cc1b4c, 0x74427549fe2e0263,
    0x74ff6836e434a5d7, 0x11c2fb9ac37cebcf, 0x0c088eff2caad46b, 0x1d4c9465b20a030e,
    0xf2d74572132d0014, 0x3a93d6c7d3a14db2, 0x9eaa277d7c920f16, 0x683a03db87cfa81c,
    0x05ce9391364f79f0, 0xac50db98756c54f5, 0x0539ab0cd4f1c619, 0xcd346a07f393ffba,
    0xefacb5358cb691f6};
  #endif

  for (int i = 0; i < 17; i++) {
    state[i] = data[i];
  }
  //  printf("**** UPDATE:\n");
  //  for (int i = 0; i < 25; i++) {
  //    printf("%016lx ", state[i]);
  //  }
  //  printf("\n");
  //  sha3_block(state, sha3_round_constants);
  sha3_keccakf_cu(state);
  //  printf("**** STATE:\n");
  //  for (int i = 0; i < 25; i++) {
  //    printf("%016lx ", state[i]);
  //  }
  //  printf("\n");

  for (int i = 0; i < 17; i++) {
    state[i] ^= data[17 + i];
  }
  //  printf("**** UPDATE:\n");
  //  for (int i = 0; i < 25; i++) {
  //    printf("%016lx ", state[i]);
  //  }
  //  printf("\n");
  sha3_keccakf_cu(state);
  //  sha3_block(state, sha3_round_constants);

  //  printf("**** STATE:\n");
  //  for (int i = 0; i < 25; i++) {
  //    printf("%016lx ", state[i]);
  //  }
  //  printf("\n");

  for (int i = 0; i < 73; i++) {
    ((uint8_t*)state)[i] ^= ((uint8_t*)data)[272 + i];
  }
  for (int i = 89; i < block_size; i++) {
    ((uint8_t*)state)[i] ^= ((uint8_t*)data)[272 + i];
  }
  ((uint8_t*)state)[block_size] ^= 0x06;
  ((uint8_t*)state)[135] ^= 0x80;
  //  printf("**** UPDATE:\n");
  //  for (int i = 0; i < 25; i++) {
  //    printf("%016lx ", state[i]);
  //  }
  //  printf("\n");
  //  sha3_block(state, sha3_round_constants);
  // sha3_keccakf_cu(state);

  //  printf("**** FINAL:\n");
  //  for (int i = 0; i < 25; i++) {
  //    printf("%016lx ", state[i]);
  //  }
  //  printf("\n");

  
  //  unsigned char PaddedMessage[80];
  //	memcpy(PaddedMessage, pdata, 80);
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(pTarget, pTargetIn, 2*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_PaddedMessage80, state, 25*sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
}

__host__
void cruz_sm3_init(int thr_id, uint32_t threads)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(sha3_round_constants, host_sha3_round_constants,
				sizeof(host_sha3_round_constants), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc(&d_KNonce[thr_id], sizeof(uint64_t)));
}

__host__
void cruz_sm3_free(int thr_id)
{
	cudaFree(d_KNonce[thr_id]);
}
