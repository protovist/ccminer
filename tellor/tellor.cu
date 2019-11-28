#include <string>

#include <gmp.h>

extern "C"
{
#include <sph/sph_ripemd.h>
#include <sph/sph_sha2.h>
#include "sph/sph_keccak.h"

#include "miner.h"
}

#include "cuda_helper.h"
#include "gpu_support.h"

static uint32_t* d_hash[MAX_GPUS];
static uint32_t* h_hash[MAX_GPUS];

static uint8_t* d_remainder[MAX_GPUS];
static uint8_t* h_remainder[MAX_GPUS];

extern void tellor_setBlock_56(uint32_t* data);
extern void tellor_keccak256_init(int thr_id);
extern void tellor_sha256_init(int thr_id);
extern void tellor_keccak256_hash(int thr_id, uint64_t threads, uint64_t startNonce, uint32_t* d_hash);
extern void tellor_ripemd_hash(int thr_id, uint64_t threads, uint32_t* d_hash);
extern void tellor_sha256_hash_final(int thr_id, uint64_t threads, uint32_t* d_hash);
extern void tellor_difficulty(int thr_id, uint64_t threads, uint32_t* d_hash, uint8_t *d_remainder);
extern void tellor_set_difficulty(const uint32_t* difficulty);

extern "C" void free_tellor(int thr_id);

void tellor_format_nonce(uint64_t nonce, char* nonce_text) { 
  const char digits[] =
      "0001020304050607080910111213141516171819"
      "2021222324252627282930313233343536373839"
      "4041424344454647484950515253545556575859"
      "6061626364656667686970717273747576777879"
      "8081828384858687888990919293949596979899";
  char* position = nonce_text + 19;
  while (nonce >= 100) {
    unsigned index = static_cast<unsigned>((nonce % 100) * 2);
    nonce /= 100;
    *--position = digits[index + 1];
    *--position = digits[index];
  }
  if (nonce < 10) {
    *--position = static_cast<char>('0' + nonce);
    return;
    //    return position;
  }
  unsigned index = static_cast<unsigned>(nonce * 2);
  *--position = digits[index + 1];
  *--position = digits[index];
  //  return position;
} 

// CPU Hash
extern "C" void tellor_hash(void *output, const void *input, size_t count)
{
	uint32_t _ALIGN(64) hash_keccak[8];
	uint32_t _ALIGN(64) hash_ripemd[8];
	uint32_t _ALIGN(64) hash_sha256[8];
	sph_keccak256_context ctx_keccak;
	sph_ripemd160_context ctx_ripemd;
	sph_sha256_context ctx_sha256;
#if 0
        printf("DATA: %d\n", count);
        for (int i = 0; i < count; i++) {
          printf("%02x", ((uint8_t*)input)[i]);
        }
        printf("\n");
#endif
	sph_keccak256_init(&ctx_keccak);
	sph_keccak256(&ctx_keccak, input, count);
	sph_keccak256_close(&ctx_keccak, (void*)hash_keccak);
#if 0
        printf("### CPU KECCAK HASH:\n");
        for (int i = 0; i < 8; i++) {
          printf("%08x ", hash_keccak[i]);
        }
        printf("\n");
#endif
        sph_ripemd160_init(&ctx_ripemd);
	sph_ripemd160(&ctx_ripemd, hash_keccak, 32);
	sph_ripemd160_close(&ctx_ripemd, (void*)hash_ripemd);
#if 0
        printf("### CPU RIPEMD HASH:\n");
        for (int i = 0; i < 8; i++) {
          printf("%08x ", hash_ripemd[i]);
        }
        printf("\n");
#endif
	sph_sha256_init(&ctx_sha256);
	sph_sha256(&ctx_sha256, hash_ripemd, 20);
	sph_sha256_close(&ctx_sha256, hash_sha256);

	memcpy(output, hash_sha256, 32);
#if 0
        uint32_t hash[8];
        printf("### CPU SHA256 HASH:\n");
        for (int i = 0; i < 8; i++) {
          be32enc(&hash[i], hash_sha256[i]);
          printf("%08x ", hash[i]);
        }
        printf("\n");
#endif
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_tellor(int thr_id, struct work* work, uint64_t max_nonce, uint64_t *hashes_done)
{
	uint32_t _ALIGN(64) work_data[71];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
        //work->current_nonce = 5048335987331992217;
        //work->current_nonce = 5048335987331992200;
        //work->current_nonce = 5048335987331990000;
        //work->current_nonce = 5040000000000000000;
	const uint64_t first_nonce = work->current_nonce;
        const int dev_id = device_map[thr_id];
	uint64_t throughput;
	uint32_t intensity = 23;
	throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x000f;

        if (!init[thr_id])
	{
                CUDA_CHECK(cudaSetDevice(dev_id));
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
                  CUDA_CHECK(cudaDeviceReset());
                  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
                  //                  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize,
                  //                                                16 * sizeof(uint32_t) * throughput));
                  CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
                  CUDA_LOG_ERROR();
		}
                tellor_keccak256_init(thr_id);
                tellor_sha256_init(thr_id);

                CUDA_CHECK(
                    cudaMalloc(&d_hash[thr_id], 8 * sizeof(uint32_t) * (size_t)throughput));
                CUDA_LOG_ERROR();
                
                //                CUDA_CALL_OR_RET_X(
                //                    cudaMalloc(&d_difficulty[thr_id],
                //                    sizeof(uint32_t) * 8), 0);
                CUDA_CHECK(cudaMalloc(&d_remainder[thr_id], sizeof(uint8_t) * (size_t)throughput));
                CUDA_LOG_ERROR();
                //                CUDA_CALL_OR_RET_X(
                //                    cudaMemset(d_hash[thr_id], 0,
                //                               sizeof(uint32_t) * 8 * throughput),
                //                                      0);
                //                CUDA_CALL_OR_RET_X(
                //                    cudaMemset(d_difficulty[thr_id], 0, sizeof(uint32_t) * 8),
                //                    0);

                //                h_hash[thr_id] = (uint32_t*)malloc(sizeof(uint32_t) * 8 * throughput);
                h_remainder[thr_id] = (uint8_t*)malloc(sizeof(uint8_t) * (size_t)throughput);

		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		init[thr_id] = true;
	}

        /*
        challenge = "bc4a80ff6b8ab0a7ef20ba8cb57d678977235423b771a7ee2ffc29e32f1917f0"
        address = "11fa934f6754076aeb7cf0a72a1c2d2518aa4c77"
        difficulty = "10708507298"
        nonce = "5048335987331992217"

        In [23]: puzzle = challenge + address + nonce.encode()

In [30]: puzzle.hex()
Out[30]: 'bc4a80ff6b8ab0a7ef20ba8cb57d678977235423b771a7ee2ffc29e32f1917f011fa934f6754076aeb7cf0a72a1c2d2518aa4c7735303438333335393837333331393932323137'
          bc4a80ff6b8ab0a7ef20ba8cb57d678977235423b771a7ee2ffc29e32f1917f011fa934f6754076aeb7cf0a72a1c2d2518aa4c7735303438333335393837333331393932323137

In [24]: keccak(puzzle).hex()
Out[24]: 'ae08c30b03416952636d2a3b00909ecbb21e69d7480ef2e2a586c6403302cc59'

In [25]: ripemd160(keccak(puzzle)).hex()
Out[25]: '49294227e26ec6609fd0972fb2315f845c3a1d25'

In [26]: sha256(ripemd160(keccak(puzzle))).hex()
Out[26]: 'cfdeb6c43fa367b83447b3ecd9eb99095c0e677944526e97ba3757c8f4e6a7ae'

In [27]: uint(sha256(ripemd160(keccak(puzzle))))
Out[27]: 94022261113803501715297430037142381858971503356692188005047773840353184360366

In [28]: uint(sha256(ripemd160(keccak(puzzle)))) % difficulty
Out[28]: 0

        */

#if 0
        uint8_t challenge[32] = {
            0xbc, 0x4a, 0x80, 0xff, 0x6b, 0x8a, 0xb0, 0xa7, 0xef, 0x20, 0xba,
            0x8c, 0xb5, 0x7d, 0x67, 0x89, 0x77, 0x23, 0x54, 0x23, 0xb7, 0x71,
            0xa7, 0xee, 0x2f, 0xfc, 0x29, 0xe3, 0x2f, 0x19, 0x17, 0xf0,
        };
        uint8_t address[20] = {
            0x11, 0xfa, 0x93, 0x4f, 0x67, 0x54, 0x07, 0x6a, 0xeb, 0x7c,
            0xf0, 0xa7, 0x2a, 0x1c, 0x2d, 0x25, 0x18, 0xaa, 0x4c, 0x77,
        };
        
        memcpy(&((uint8_t*)work->data)[0], challenge, 32);
        memcpy(&((uint8_t*)work->data)[32], address, 20);

        #endif

        char nonce_text[38] = {0};
        tellor_format_nonce(work->current_nonce, nonce_text);
        memcpy(&((uint8_t*)work->data)[52], nonce_text, 19);
        
        tellor_setBlock_56(work->data);
        
        //        uint32_t hash[8];
        //        tellor_hash(&hash, work->data, 71);

        //char* difficulty_text = "10708507298";
        mpz_t difficulty_mpz;
        mpz_init(difficulty_mpz);
        //mpz_set_str(difficulty_mpz, difficulty_text, 10);
        mpz_set_str(difficulty_mpz, work->difficulty, 10);

        uint32_t difficulty_words[8];
        mpz_export(&difficulty_words, NULL, -1, 32, -1, 0, difficulty_mpz);

#if 0        
        printf("MPZ DIFF: ");
        for (int i = 0; i < 8; i++) {
          printf("%08x ", difficulty_words[i]);
        }
        printf("\n");
#endif        
        mpz_clear(difficulty_mpz);

        tellor_set_difficulty(difficulty_words);
        //cudaMemset(d_remainder[thr_id], 0xFF, 8 * sizeof(uint8_t) * throughput);

        //        exit(1);

        //        work->work_time = time(NULL);
        //        memcpy(&((uint8_t*)work->data)[170], std::to_string(work->work_time).c_str(), 10);


        work->valid_nonces = 0;
	do {
          //          cudaPointerAttributes attr;
          //          CUDA_CHECK(cudaPointerGetAttributes(&attr, d_remainder[thr_id]));
          //          printf("memory type: %d\n", attr.memoryType);
          //          printf("type: %d\n", attr.type);
          //          printf("device: %d\n", attr.device);
          //          printf("device ptr: %p\n", attr.devicePointer);
          //          printf("host ptr: %p\n", attr.hostPointer);
          //          printf("managed: %d\n", attr.isManaged);
          //                printf("START: %lu, FIRST: %lu, DONE: %lu\n",
          //                work->current_nonce,
          // first_nonce, throughput);

          //		*hashes_done = pdata[19] - first_nonce + throughput;
          //        time_t start = time(NULL);
          //cudaEvent_t start_event, stop_event;
                //                cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
                //                cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);
                //                cudaEventRecord(start_event, 0);

          tellor_keccak256_hash(thr_id, throughput, work->current_nonce, d_hash[thr_id]);
          //          CUDA_CHECK(cudaThreadSynchronize());
          //          CUDA_LOG_ERROR();
          tellor_ripemd_hash(thr_id, throughput, d_hash[thr_id]);
          //          CUDA_CHECK(cudaThreadSynchronize());
          //          CUDA_LOG_ERROR();
          tellor_sha256_hash_final(thr_id, throughput, d_hash[thr_id]);
          //          CUDA_CHECK(cudaThreadSynchronize());
          CUDA_LOG_ERROR();

#if 0
          cudaMemcpy(h_hash[thr_id], d_hash[thr_id], sizeof(uint32_t) * 8 * throughput,
                     cudaMemcpyDeviceToHost);

          uint32_t* h = (uint32_t*)&h_hash[thr_id];
          for (int i = 0; i < 32; i++) {
            printf("%08x ", h[i]);
            if ((i+1) % 8 == 0) printf("\n");
          }
          //          exit(1);
#endif          
          tellor_difficulty(thr_id, throughput, d_hash[thr_id], d_remainder[thr_id]);
          //                printf("diff d_hash: %08x\n", d_hash[thr_id]);
          //                printf("d_remainder: %08x\n", d_remainder[thr_id]);
                //          CUDA_CHECK(cudaThreadSynchronize());
          CUDA_LOG_ERROR();

                //                exit(1);
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(
              cudaMemcpy(h_remainder[thr_id], d_remainder[thr_id], sizeof(uint8_t) * throughput,
                         cudaMemcpyDeviceToHost));
          CUDA_LOG_ERROR();

          work->nonces[0] = UINT64_MAX;
          uint8_t* output = (uint8_t*)h_remainder[thr_id];
          for (int i = 0; i < throughput; i++) {
          //          for (int i = 0; i < 8; i++) {
            //printf("%d: %d\n", i, output[i]);
          if (output[i] == 1) {
              work->nonces[0] = work->current_nonce + i;
              break;
            }
          }
          //          exit(1);
          *hashes_done = work->current_nonce - first_nonce + throughput;
          
          //                cudaEventRecord(stop_event, 0);
                //                cudaEventSynchronize(stop_event);
                //                float time_elapsed;
                //                cudaEventElapsedTime(&time_elapsed, start_event, stop_event);
                //                printf("HASHES: %lu %.2f MH/s\n", *hashes_done,  (float)*hashes_done / time_elapsed / 1000);
                //                printf("*** TIME: %.2f\n", time_elapsed);
                //        time_t stop = time(NULL);
                //        printf("*** TIME: %lu, %lu, %lu\n", stop, start, stop - start);
                //        printf("*** HASHES: %lu %0.2f\n", *hashes_done, *hashes_done/(stop-start));
                //                exit(1);
                //                printf("WORKNONCE: %" PRIu64 "\n", work->nonces[0]);
		if (work->nonces[0] > 0 && work->nonces[0] != UINT64_MAX && bench_algo < 0)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
                        std::string nonce = std::to_string(work->nonces[0]);
                        memcpy(&((uint8_t*)work->data)[52], nonce.c_str(), 19);                   
                        tellor_hash(vhash, work->data, 71);
                        //                        exit(1);
                        
                        //			if (vhash[0] <= ptarget[7] && fulltest(vhash, ptarget)) {
                        //			if (vhash[0] <= ptarget[7]) {
                        if (true) {
                        work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
                                work->current_nonce = work->nonces[0] + 1;
                                printf("FOUND NONCE: %" PRIu64 " %" PRIu64
                                       "\n", work->nonces[0], work->current_nonce);

                                //                                free_tellor(thr_id);
				return work->valid_nonces;
			}
			else if (vhash[0] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				work->current_nonce = work->nonces[0] + 1;
				//tellor_setOutput(thr_id);
				continue;
			} else {
                          printf("oops\n");
                          exit(1);
                        }
		}

		if ((uint64_t) throughput + work->current_nonce >= max_nonce) {
                  //printf("BREAK: %016lx + %016lx > %016lx\n", throughput, work->current_nonce, max_nonce);
                  work->current_nonce = max_nonce;
                  break;
		}

                work->current_nonce += throughput;
		//work->current_nonce += 8;
                //if (work->current_nonce > 5048335987331992217) exit(1);
                //                printf("NEW NONCE: %" PRIu64 "\n", work->current_nonce);

	} while (!work_restart[thr_id].restart);

	*hashes_done = work->current_nonce - first_nonce;
        //        free_tellor(thr_id);

	return 0;
}

// cleanup
extern "C" void free_tellor(int thr_id)
{
#if 0
  if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_remainder[thr_id]);
        //	cudaFree(d_difficulty[thr_id]);

        free(h_remainder[thr_id]);

        //	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
#endif        
}

