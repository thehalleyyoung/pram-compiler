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

/* PRAM program: parallel_addition | model: CREW */
/* Carry-lookahead parallel addition. CREW, O(log n) time, n processors. */
/* Work bound: O(n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: parallel_addition */
/* Memory model: CREW */
/* Carry-lookahead parallel addition. CREW, O(log n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    int64_t* a_bits = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* b_bits = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* g = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* p = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* carry = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* sum = (int64_t*)pram_calloc((n + 1LL), sizeof(int64_t));

    /* Phase 1: generate / propagate */
    for (int64_t pid = 0; pid < n; pid++) {
        int64_t ai = a_bits[pid];
        int64_t bi = b_bits[pid];
        g[pid] = (ai & bi);
        p[pid] = (ai | bi);
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Phase 2: parallel prefix on (g,p) pairs to compute all carries */
    int64_t log_n = ((int64_t)(log2(((double)(n)))));
    for (int64_t d = 0LL; d < log_n; d += 1LL) {
        int64_t stride = (1LL << d);
        for (int64_t pid = 0; pid < n; pid++) {
            if ((pid >= stride)) {
                int64_t j = (pid - stride);
                int64_t g_j = g[j];
                int64_t p_j = p[j];
                int64_t g_i = g[pid];
                int64_t p_i = p[pid];
                g[pid] = (g_i | (p_i & g_j));
                p[pid] = (p_i & p_j);
            }
        }
        /* ---- barrier (end of phase 1) ---- */
    }
    for (int64_t pid = 0; pid < n; pid++) {
        carry[pid] = g[pid];
    }
    /* ---- barrier (end of phase 2) ---- */
    /* Phase 3: sum[i] = a[i] ^ b[i] ^ carry[i-1] */
    for (int64_t pid = 0; pid < n; pid++) {
        int64_t ai = a_bits[pid];
        int64_t bi = b_bits[pid];
        int64_t xor_ab = (ai ^ bi);
        if ((pid > 0LL)) {
            int64_t c_prev = carry[(pid - 1LL)];
            sum[pid] = (xor_ab ^ c_prev);
        } else {
            sum[pid] = xor_ab;
        }
    }
    /* ---- barrier (end of phase 3) ---- */
    /* Carry out bit */
    sum[n] = carry[(n - 1LL)];

    pram_free(a_bits);
    pram_free(b_bits);
    pram_free(g);
    pram_free(p);
    pram_free(carry);
    pram_free(sum);
    return 0;
}
