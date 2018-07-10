/**
 * Zen Protocol SHA3.
 */

#include <miner.h>
#include <cuda_helper.h>
#include <openssl/evp.h>

static uint32_t *d_resultNonce[MAX_GPUS];

extern "C" void zenprotocol_hash(void *output, const uint32_t *input)
{

#if 0
  printf("CPU INPUT\n");
  for (int i = 0; i < 25; ++i) {
    printf("%08x ", input[i]);
  }
  printf("\n");
#endif

  uint32_t hash[8];

  EVP_MD_CTX *ctx = EVP_MD_CTX_create();
  EVP_DigestInit(ctx, EVP_sha3_256());
  EVP_DigestUpdate(ctx, (uint8_t*)input, 100);
  EVP_DigestFinal(ctx, (uint8_t*)hash, NULL);

#if 0  
  printf("############ SHA3 CPU HASH:\n");
  for (int i = 0; i < 8; ++i) {
    printf("%08x ", ((uint32_t*)hash)[i]);
  }
  printf("\n");
#endif
}


static bool init[MAX_GPUS] = { 0 };

extern void zenprotocol_init(int thr_id);
extern void zenprotocol_free(int thr_id);
extern void zenprotocol_setBlock_100(uint32_t *pdata);
extern void zenprotocol_setTarget(const uint32_t *ptarget);
extern void zenprotocol_cpu_hash(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *resultNonces);

extern "C" int scanhash_zenprotocol(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
        uint32_t *nonceptr = &work->data[24];
        const uint32_t first_nonce = *nonceptr;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 26);
	if (init[thr_id]) throughput = min(throughput, (max_nonce - *nonceptr));

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x03;
        
	if (!init[thr_id])
	{
          CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		zenprotocol_init(thr_id);
                CUDA_SAFE_CALL(cudaMalloc(&d_resultNonce[thr_id], 2 * sizeof(uint32_t)));

		init[thr_id] = true;
	}

	zenprotocol_setBlock_100(work->data);
        zenprotocol_setTarget(ptarget);
        work->valid_nonces = 0;

        CUDA_SAFE_CALL(cudaMemset(d_resultNonce[thr_id], 0xFF, 2 * sizeof(uint32_t)));
	do {
		uint32_t start_nonce = *nonceptr;
                *hashes_done = *nonceptr - first_nonce + throughput;
                zenprotocol_cpu_hash(thr_id, throughput, start_nonce, d_resultNonce[thr_id]);
                CUDA_SAFE_CALL(cudaMemcpy(&work->nonces[0], d_resultNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash[8];
		        work->nonces[0] += start_nonce;
			work->data[24] = swab32(work->nonces[0]);
			zenprotocol_hash(vhash, work->data);

                        cudaMemset(d_resultNonce[thr_id], 0xFF, 2 * sizeof(uint32_t));
			if (true) {
			  work->valid_nonces = 1;
			  work_set_target_ratio(work, vhash);
			  return 1;
			} else {
			  gpu_increment_reject(thr_id);
			  if (!opt_quiet)
			    gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
                          start_nonce += throughput;
			}
		}

		if ((uint64_t) throughput + *nonceptr >= max_nonce) {
			*nonceptr = max_nonce;
			break;
		}
		*nonceptr += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = *nonceptr - first_nonce;

	return 0;
}

extern "C" void free_zenprotocol(int thr_id)
{
	if (!init[thr_id])
		return;

        if (d_resultNonce[thr_id]) {
          cudaFree(d_resultNonce[thr_id]);
        }
        d_resultNonce[thr_id] = NULL;
	init[thr_id] = false;
}
