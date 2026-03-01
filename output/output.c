#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>

#ifndef PRAM_MIN
#define PRAM_MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef PRAM_MAX
#define PRAM_MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef PRAM_SWAP
#define PRAM_SWAP(type, a, b) do { type _tmp = (a); (a) = (b); (b) = _tmp; } while(0)
#endif

static void* pram_malloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr && size > 0) {
        fprintf(stderr, "pram_malloc: out of memory (requested %zu bytes)\n", size);
        exit(1);
    }
    return ptr;
}

static void* pram_calloc(size_t count, size_t size) {
    void* ptr = calloc(count, size);
    if (!ptr && count > 0 && size > 0) {
        fprintf(stderr, "pram_calloc: out of memory (requested %zu * %zu bytes)\n", count, size);
        exit(1);
    }
    return ptr;
}

static void pram_free(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

/* PRAM program: bitonic_sort | model: EREW */
/* Batcher's bitonic sort. O(log^2 n) time, n/2 processors, EREW. */
/* Work bound: O(n log^2 n) */
/* Time bound: O(log^2 n) */

/* Generated C99 code for PRAM program: bitonic_sort */
/* Memory model: EREW */
/* Batcher's bitonic sort. O(log^2 n) time, n/2 processors, EREW. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = (n / 2LL);

    int64_t* A = (int64_t*)pram_calloc(n, sizeof(int64_t));

    /* Bitonic sort network: O(log n) stages × O(log n) steps */
    int64_t log_n = ((int64_t)(log2(((double)(n)))));
    for (int64_t k = 1LL; k < (log_n + 1LL); k += 1LL) {
        for (int64_t j_fwd = 0LL; j_fwd < k; j_fwd += 1LL) {
            int64_t j_actual = (k - j_fwd);
            int64_t d = (1LL << (j_actual - 1LL));
            int64_t block_size = (1LL << k);
            for (int64_t pid = 0; pid < (n / 2LL); pid++) {
                int64_t block_id = (pid / d);
                int64_t pos_in_block = (pid % d);
                int64_t i = ((block_id * (2LL * d)) + pos_in_block);
                int64_t j_idx = (i + d);
                if ((j_idx < n)) {
                    int64_t dir_bit = ((i / block_size) & 1LL);
                    int64_t ai = A[i];
                    int64_t aj = A[j_idx];
                    if ((dir_bit == 0LL)) {
                        if ((ai > aj)) {
                            A[i] = aj;
                            A[j_idx] = ai;
                        }
                    } else {
                        if ((ai < aj)) {
                            A[i] = aj;
                            A[j_idx] = ai;
                        }
                    }
                }
            }
            /* ---- barrier (end of phase 0) ---- */
        }
    }

    pram_free(A);
    return 0;
}
