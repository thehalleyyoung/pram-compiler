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

/* PRAM program: parallel_multiplication | model: CREW */
/* Parallel integer multiplication. CREW, O(log n) time, n^2 processors. */
/* Work bound: O(n^2) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: parallel_multiplication */
/* Memory model: CREW */
/* Parallel integer multiplication. CREW, O(log n) time, n^2 processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = (n * n);

    int64_t* a_bits = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* b_bits = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* partial = (int64_t*)pram_calloc((n * n), sizeof(int64_t));
    int64_t* result = (int64_t*)pram_calloc((2LL * n), sizeof(int64_t));

    /* Step 1: compute partial products a[i] & b[j] */
    for (int64_t pid = 0; pid < (n * n); pid++) {
        int64_t i = (pid / n);
        int64_t j = (pid % n);
        int64_t ai = a_bits[i];
        int64_t bj = b_bits[j];
        partial[pid] = (ai & bj);
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Step 2: reduce partial products along diagonals for each result bit */
    int64_t log_n = ((int64_t)(log2(((double)(n)))));
    for (int64_t pid = 0; pid < (2LL * n); pid++) {
        result[pid] = 0LL;
    }
    /* ---- barrier (end of phase 1) ---- */
    for (int64_t d = 0LL; d < log_n; d += 1LL) {
        int64_t stride = (1LL << d);
        for (int64_t pid = 0; pid < (n * n); pid++) {
            int64_t i = (pid / n);
            int64_t j = (pid % n);
            int64_t diag = (i + j);
            int64_t partner_j = (j + stride);
            int64_t partner_i = (i - stride);
            if (((partner_i >= 0LL) && (partner_j < n))) {
                int64_t partner_idx = ((partner_i * n) + partner_j);
                int64_t pv = partial[partner_idx];
                partial[pid] = (partial[pid] + pv);
            }
        }
        /* ---- barrier (end of phase 2) ---- */
    }
    /* Write diagonal sums to result (with carry propagation) */
    for (int64_t pid = 0; pid < (2LL * n); pid++) {
        int64_t row = ((pid < n) ? pid : (n - 1LL));
        int64_t col = (pid - row);
        int64_t flat_idx = ((row * n) + col);
        if ((col < n)) {
            result[pid] = partial[flat_idx];
        }
    }
    /* ---- barrier (end of phase 3) ---- */
    /* Carry propagation */
    for (int64_t bit = 0LL; bit < ((2LL * n) - 1LL); bit += 1LL) {
        int64_t val = result[bit];
        int64_t carry_out = (val / 2LL);
        int64_t bit_val = (val % 2LL);
        result[bit] = bit_val;
        if ((carry_out > 0LL)) {
            result[(bit + 1LL)] = (result[(bit + 1LL)] + carry_out);
        }
    }

    pram_free(a_bits);
    pram_free(b_bits);
    pram_free(partial);
    pram_free(result);
    return 0;
}
