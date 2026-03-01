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

/* PRAM program: odd_even_merge_sort | model: EREW */
/* Odd-even merge sort (Batcher). O(log^2 n) time, n/2 processors, EREW. */
/* Work bound: O(n log^2 n) */
/* Time bound: O(log^2 n) */

/* Generated C99 code for PRAM program: odd_even_merge_sort */
/* Memory model: EREW */
/* Odd-even merge sort (Batcher). O(log^2 n) time, n/2 processors, EREW. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = (n / 2LL);

    int64_t* A = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* T = (int64_t*)pram_calloc(n, sizeof(int64_t));

    /* Odd-even merge sort: O(log n) stages, each with O(log n) merge steps */
    int64_t log_n = ((int64_t)(log2(((double)(n)))));
    for (int64_t stage = 0LL; stage < log_n; stage += 1LL) {
        int64_t merge_len = (2LL << stage);
        int64_t half = (1LL << stage);
        for (int64_t step = 0LL; step < (stage + 1LL); step += 1LL) {
            int64_t step_dist = (1LL << (stage - step));
            for (int64_t pid = 0; pid < (n / 2LL); pid++) {
                int64_t group = (pid / half);
                int64_t pos = (pid % half);
                int64_t base = (group * merge_len);
                if ((step == 0LL)) {
                    int64_t i = (base + (2LL * pos));
                    int64_t j_idx = (i + 1LL);
                } else {
                    int64_t within_group = (pid % step_dist);
                    int64_t pair_base = (base + ((pos / step_dist) * (2LL * step_dist)));
                    int64_t i = (pair_base + within_group);
                    int64_t j_idx = (i + step_dist);
                }
                if ((j_idx < PRAM_MIN((base + merge_len), n))) {
                    int64_t ai = A[i];
                    int64_t aj = A[j_idx];
                    if ((ai > aj)) {
                        A[i] = aj;
                        A[j_idx] = ai;
                    }
                }
            }
            /* ---- barrier (end of phase 0) ---- */
        }
    }

    pram_free(A);
    pram_free(T);
    return 0;
}
