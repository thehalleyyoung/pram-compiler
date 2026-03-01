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

/* PRAM program: parallel_batch_search | model: CREW */
/* Parallel batch search for multiple keys. CREW, O(log n) time, n*k processors. */
/* Work bound: O(k * log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: parallel_batch_search */
/* Memory model: CREW */
/* Parallel batch search for multiple keys. CREW, O(log n) time, n*k processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t k = 0;
    int64_t _num_procs = k;

    int64_t* A = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* keys = (int64_t*)pram_calloc(k, sizeof(int64_t));
    int64_t* results = (int64_t*)pram_calloc(k, sizeof(int64_t));
    int64_t* lo_arr = (int64_t*)pram_calloc(k, sizeof(int64_t));
    int64_t* hi_arr = (int64_t*)pram_calloc(k, sizeof(int64_t));
    int64_t* done_arr = (int64_t*)pram_calloc(k, sizeof(int64_t));

    /* Phase 1: initialise search state */
    for (int64_t pid = 0; pid < k; pid++) {
        lo_arr[pid] = 0LL;
        hi_arr[pid] = (n - 1LL);
        done_arr[pid] = 0LL;
        results[pid] = (-1LL);
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Phase 2: O(log n) rounds of parallel binary search */
    int64_t log_n = (((int64_t)(log2(((double)(n))))) + 2LL);
    for (int64_t round = 0LL; round < log_n; round += 1LL) {
        for (int64_t pid = 0; pid < k; pid++) {
            if ((done_arr[pid] == 0LL)) {
                int64_t lo = lo_arr[pid];
                int64_t hi = hi_arr[pid];
                if ((lo > hi)) {
                    done_arr[pid] = 1LL;
                } else {
                    int64_t mid = (lo + ((hi - lo) / 2LL));
                    int64_t mid_val = A[mid];
                    int64_t my_key = keys[pid];
                    if ((mid_val == my_key)) {
                        results[pid] = mid;
                        done_arr[pid] = 1LL;
                    } else {
                        if ((mid_val < my_key)) {
                            lo_arr[pid] = (mid + 1LL);
                        } else {
                            hi_arr[pid] = (mid - 1LL);
                        }
                    }
                }
            }
        }
        /* ---- barrier (end of phase 1) ---- */
    }

    pram_free(A);
    pram_free(keys);
    pram_free(results);
    pram_free(lo_arr);
    pram_free(hi_arr);
    pram_free(done_arr);
    return 0;
}
