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

/* PRAM program: parallel_prefix_multiplication | model: CREW */
/* Parallel prefix multiplication. CREW, O(log n) time, n processors. */
/* Work bound: O(n log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: parallel_prefix_multiplication */
/* Memory model: CREW */
/* Parallel prefix multiplication. CREW, O(log n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    int64_t* A = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* B = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* temp = (int64_t*)pram_calloc(n, sizeof(int64_t));

    /* Phase 1: copy A to B */
    for (int64_t pid = 0; pid < n; pid++) {
        B[pid] = A[pid];
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Phase 2: up-sweep reduce with multiplication */
    int64_t log_n = ((int64_t)(log2(((double)(n)))));
    for (int64_t d = 0LL; d < log_n; d += 1LL) {
        int64_t stride = (1LL << (d + 1LL));
        int64_t half = (1LL << d);
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t idx = pid;
            if (((idx >= stride) && (((idx + 1LL) % stride) == 0LL))) {
                int64_t src = (idx - half);
                int64_t bsrc = B[src];
                int64_t bidx = B[idx];
                B[idx] = (bsrc * bidx);
            }
        }
        /* ---- barrier (end of phase 1) ---- */
    }
    /* Phase 3: down-sweep propagation */
    for (int64_t d_fwd = 0LL; d_fwd < (log_n - 1LL); d_fwd += 1LL) {
        int64_t d_actual = ((log_n - 2LL) - d_fwd);
        int64_t stride_d = (1LL << (d_actual + 1LL));
        int64_t half_d = (1LL << d_actual);
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t idx = pid;
            int64_t base_idx = (idx - half_d);
            if ((((idx >= stride_d) && (((idx + 1LL) % stride_d) != 0LL)) && (((idx + 1LL) % half_d) == 0LL))) {
                int64_t b_base = B[base_idx];
                int64_t b_target = B[idx];
                B[idx] = (b_base * b_target);
            }
        }
        /* ---- barrier (end of phase 2) ---- */
    }

    pram_free(A);
    pram_free(B);
    pram_free(temp);
    return 0;
}
