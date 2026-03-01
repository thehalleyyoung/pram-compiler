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

/* PRAM program: parallel_interpolation_search | model: CREW */
/* Parallel interpolation search. CREW, O(log log n) expected time. */
/* Work bound: O(sqrt(n)) */
/* Time bound: O(log log n) */

/* Generated C99 code for PRAM program: parallel_interpolation_search */
/* Memory model: CREW */
/* Parallel interpolation search. CREW, O(log log n) expected time. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t key = 0;
    int64_t _num_procs = (((int64_t)(sqrt(((double)(n))))) + 1LL);

    int64_t* A = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* probe_result = (int64_t*)pram_calloc((((int64_t)(sqrt(((double)(n))))) + 1LL), sizeof(int64_t));
    int64_t* lo = (int64_t*)pram_calloc(1LL, sizeof(int64_t));
    int64_t* hi = (int64_t*)pram_calloc(1LL, sizeof(int64_t));
    int64_t* found_idx = (int64_t*)pram_calloc(1LL, sizeof(int64_t));
    int64_t* done = (int64_t*)pram_calloc(1LL, sizeof(int64_t));

    /* Initialise search bounds */
    lo[0LL] = 0LL;
    hi[0LL] = (n - 1LL);
    found_idx[0LL] = (-1LL);
    done[0LL] = 0LL;
    /* ---- barrier (end of phase 0) ---- */
    int64_t max_rounds = (((int64_t)(log2(log2(((double)((n + 1LL))))))) + 4LL);
    int64_t p_count = ((int64_t)(sqrt(((double)(n)))));
    /* Interpolation search loop: O(log log n) rounds */
    for (int64_t round = 0LL; round < max_rounds; round += 1LL) {
        for (int64_t pid = 0; pid < p_count; pid++) {
            if ((done[0LL] == 0LL)) {
                int64_t cur_lo = lo[0LL];
                int64_t cur_hi = hi[0LL];
                if ((cur_lo <= cur_hi)) {
                    int64_t a_lo = A[cur_lo];
                    int64_t a_hi = A[cur_hi];
                    int64_t range = (cur_hi - cur_lo);
                    int64_t est = ((a_hi == a_lo) ? cur_lo : (cur_lo + (((key - a_lo) * range) / (a_hi - a_lo))));
                    int64_t offset = (pid - (p_count / 2LL));
                    int64_t probe = (est + offset);
                    int64_t probe_c = ((probe < cur_lo) ? cur_lo : ((probe > cur_hi) ? cur_hi : probe));
                    int64_t val = A[probe_c];
                    if ((val == key)) {
                        probe_result[pid] = 0LL;
                        found_idx[0LL] = probe_c;
                        done[0LL] = 1LL;
                    } else {
                        if ((val < key)) {
                            probe_result[pid] = (-1LL);
                        } else {
                            probe_result[pid] = 1LL;
                        }
                    }
                }
            }
        }
        /* ---- barrier (end of phase 1) ---- */
        if ((done[0LL] == 0LL)) {
            for (int64_t pid = 0; pid < 1LL; pid++) {
                int64_t cur_lo = lo[0LL];
                int64_t cur_hi = hi[0LL];
                int64_t new_lo = cur_lo;
                int64_t new_hi = cur_hi;
                int64_t range = (cur_hi - cur_lo);
                int64_t a_lo = A[cur_lo];
                int64_t a_hi = A[cur_hi];
                int64_t est = ((a_hi == a_lo) ? cur_lo : (cur_lo + (((key - a_lo) * range) / (a_hi - a_lo))));
                for (int64_t i = 0LL; i < p_count; i += 1LL) {
                    int64_t pr = probe_result[i];
                    int64_t offset = (i - (p_count / 2LL));
                    int64_t probe_pos = (est + offset);
                    int64_t probe_clamped = ((probe_pos < cur_lo) ? cur_lo : ((probe_pos > cur_hi) ? cur_hi : probe_pos));
                    if ((pr == (-1LL))) {
                        new_lo = (probe_clamped + 1LL);
                    }
                    if ((pr == 1LL)) {
                        new_hi = (probe_clamped - 1LL);
                    }
                }
                lo[0LL] = new_lo;
                hi[0LL] = new_hi;
            }
        }
        /* ---- barrier (end of phase 2) ---- */
        if (((lo[0LL] > hi[0LL]) && (done[0LL] == 0LL))) {
            done[0LL] = 1LL;
        }
        /* ---- barrier (end of phase 3) ---- */
    }

    pram_free(A);
    pram_free(probe_result);
    pram_free(lo);
    pram_free(hi);
    pram_free(found_idx);
    pram_free(done);
    return 0;
}
