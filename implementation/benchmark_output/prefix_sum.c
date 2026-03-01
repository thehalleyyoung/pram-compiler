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

/* PRAM program: prefix_sum | model: EREW */
/* Parallel prefix sum (Blelloch scan). EREW, O(log n) time, n/2 processors. */
/* Work bound: O(n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: prefix_sum */
/* Memory model: EREW */
/* Parallel prefix sum (Blelloch scan). EREW, O(log n) time, n/2 processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = (n / 2LL);

    int64_t* A = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* B = (int64_t*)pram_calloc(n, sizeof(int64_t));

    /* Copy input to working array */
    for (int64_t pid = 0; pid < n; pid++) {
        B[pid] = A[pid];
    }
    /* ---- barrier (end of phase 0) ---- */
    int64_t log_n = ((int64_t)(log2(((double)(n)))));
    /* Up-sweep (reduce) phase */
    for (int64_t d = 0LL; d < log_n; d += 1LL) {
        int64_t stride = (2LL << d);
        int64_t half_stride = (1LL << d);
        int64_t num_active = (n / stride);
        for (int64_t pid = 0; pid < num_active; pid++) {
            int64_t idx = (((pid + 1LL) * stride) - 1LL);
            int64_t left_idx = (idx - half_stride);
            if ((idx < n)) {
                int64_t left_val = B[left_idx];
                int64_t right_val = B[idx];
                B[idx] = (left_val + right_val);
            }
        }
        /* ---- barrier (end of phase 1) ---- */
    }
    /* Down-sweep phase */
    for (int64_t d_fwd = 0LL; d_fwd < (log_n - 1LL); d_fwd += 1LL) {
        int64_t d_actual = ((log_n - 2LL) - d_fwd);
        int64_t stride = (2LL << d_actual);
        int64_t half_stride = (1LL << d_actual);
        int64_t num_active = (n / stride);
        for (int64_t pid = 0; pid < num_active; pid++) {
            int64_t base = (((pid + 1LL) * stride) - 1LL);
            int64_t target = (base + half_stride);
            if ((target < n)) {
                int64_t parent_val = B[base];
                int64_t cur_val = B[target];
                B[target] = (parent_val + cur_val);
            }
        }
        /* ---- barrier (end of phase 2) ---- */
    }

    pram_free(A);
    pram_free(B);
    return 0;
}
