/*
 * Keccak 256
 *
 */

#include <string>

extern "C"
{
#include "sph/sph_sha3.h"

#include "miner.h"
}

#include "cuda_helper.h"

// compat
extern void cruz_sm3_init(int thr_id, uint32_t threads);
extern void cruz_sm3_free(int thr_id);
extern void cruz_setBlock_345(uint64_t* data, const void* ptarget, size_t block_size);
extern void cruz_cpu_hash(int thr_id, uint32_t threads, uint64_t startNonce, uint64_t* resNonces, int order);

// CPU Hash
extern "C" void cruz_hash(void *state, const void *input, size_t count)
{
	uint32_t _ALIGN(64) hash[8];
	sph_sha3_context ctx_sha3;

        printf("SIZE: %d\n", count);
        for (int i = 0; i < count; i++) {
          printf("%c", ((char*)input)[i]);
        }
        printf("\n");

	sph_sha3256_init(&ctx_sha3);
	sph_sha3256(&ctx_sha3, input, count);
	sph_sha3256_close(&ctx_sha3, (void*)hash);
	memcpy(state, hash, 32);

        printf("### CPU HASH:\n");
        for (int i = 0; i < 8; i++) {
          printf("%08x ", hash[i]);
        }
        printf("\n");
}

extern "C" void cruz_midstate(const void *input, void *output)
{
  sph_sha3_context ctx_sha3;
  uint32_t hash[8];
  sph_sha3256_init(&ctx_sha3);
  sph_sha3256(&ctx_sha3, input, 345);
  //  sph_sha3256_close(&ctx_sha3, (void*)hash);

  memcpy(output, ctx_sha3.u.wide, 200);
  printf("\nMIDSTATE:\n");
  for (int i = 0; i < 25; i++) {
    printf("%016lx ", ((uint64_t*)output)[i]);
  }
  printf("\n");

  sph_sha3_context ctx_sha3_mid;
  sph_sha3256_init(&ctx_sha3_mid);
  memcpy(ctx_sha3_mid.u.wide, &ctx_sha3.u.wide, 200);
  
  sph_sha3256(&ctx_sha3_mid, ((uint8_t*)input)+345, 41);
  sph_sha3256_close(&ctx_sha3_mid, (void*)hash);
        printf("### MID HASH:\n");
        for (int i = 0; i < 8; i++) {
          printf("%08x ", hash[i]);
        }
        printf("\n");

}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_cruz(int thr_id, struct work* work, uint64_t max_nonce, uint64_t *hashes_done)
{
	uint32_t _ALIGN(64) work_data[128];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = work->nonces[1];
	const int dev_id = device_map[thr_id];
	uint64_t throughput;
	uint32_t intensity = 23;
	throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x000f;

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		cuda_get_arch(thr_id);
                cruz_sm3_init(thr_id, throughput);

		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		init[thr_id] = true;
	}

        //        for (int k = 0; k < 128; k++) {
        //          be32enc(&work_data[k], work->data[k]);
        //        }

        uint64_t state[25];
        //                be32enc(&endiandata[19], work->nonces[0]);
        //        cruz_midstate(work->data, &state);
        //        exit(1);

        uint32_t hash[8];
        //        cruz_hash(&state, work->data, 396);
        //        cruz_hash(&state, work->data, 396);

	// const uint2 highTarget = make_uint2(ptarget[6], ptarget[7]);
        //cruz_setBlock_345((uint64_t*)work_data);
        uint8_t block_size = work->work_size - 272;
        cruz_setBlock_345((uint64_t*)work->data, ptarget, block_size);
        work->valid_nonces = 0;
        
	do {
		int order = 0;

		*hashes_done = work->nonces[1] - first_nonce + throughput;
                //		*hashes_done = pdata[19] - first_nonce + throughput;
                cruz_cpu_hash(thr_id, throughput, work->nonces[1], work->nonces, order++);
                //                exit(1);
                //                printf("WORKNONCE: %" PRIu64 "\n", work->nonces[0]);
		if (work->nonces[0] > 0 && work->nonces[0] != UINT64_MAX && bench_algo < 0)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
                        std::string nonce = std::to_string(work->nonces[0]);
                        memcpy(&((uint8_t*)work->data)[345], nonce.c_str(), 16);                   
                        cruz_hash(vhash, work->data, work->work_size);

                        //			if (vhash[0] <= ptarget[7] && fulltest(vhash, ptarget)) {
			if (vhash[0] <= ptarget[7]) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
                                work->nonces[1] = work->nonces[0] + 1;
				return work->valid_nonces;
			}
			else if (vhash[0] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				work->nonces[1] = work->nonces[0] + 1;
				//cruz_setOutput(thr_id);
				continue;
			} else {
                          printf("oops\n");
                          exit(1);
                        }
		}

		if ((uint64_t) throughput + work->nonces[1] >= max_nonce) {
                  //                  printf("BREAK\n");
                  work->nonces[1] = max_nonce;
                  break;
		}

		work->nonces[1] += throughput;
                //                printf("NEW NONCE: %" PRIu64 "\n", work->nonces[1]);

	} while (!work_restart[thr_id].restart);

	*hashes_done = work->nonces[1] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_cruz(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();
        cruz_sm3_free(thr_id);
	cudaDeviceSynchronize();
	init[thr_id] = false;
}
