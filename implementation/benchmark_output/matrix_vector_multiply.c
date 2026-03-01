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

/* PRAM program: matrix_vector_multiply | model: CREW */
/* Parallel matrix-vector multiply. CREW, n^2 processors, O(log n) time. */
/* Work bound: O(n^2) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: matrix_vector_multiply */
/* Memory model: CREW */
/* Parallel matrix-vector multiply. CREW, n^2 processors, O(log n) time. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = (n * n);

    int64_t* A = (int64_t*)pram_calloc((n * n), sizeof(int64_t));
    int64_t* x = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* y = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* T = (int64_t*)pram_calloc((n * n), sizeof(int64_t));

    /* Step 1: each processor (i,j) computes A[i,j] * x[j] */
    for (int64_t pid = 0; pid < (n * n); pid++) {
        int64_t i = (pid / n);
        int64_t j = (pid % n);
        int64_t a_val = A[pid];
        int64_t x_val = x[j];
        T[pid] = (a_val * x_val);
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Step 2: parallel reduction along j dimension */
    int64_t log_n = ((int64_t)(log2(((double)(n)))));
    for (int64_t d = 0LL; d < log_n; d += 1LL) {
        int64_t stride = (1LL << d);
        for (int64_t pid = 0; pid < (n * n); pid++) {
            int64_t i = (pid / n);
            int64_t j = (pid % n);
            if ((((j % (2LL * stride)) == 0LL) && ((j + stride) < n))) {
                int64_t partner = (pid + stride);
                T[pid] = (T[pid] + T[partner]);
            }
        }
        /* ---- barrier (end of phase 1) ---- */
    }
    /* Step 3: write y[i] = T[i*n + 0] */
    for (int64_t pid = 0; pid < n; pid++) {
        y[pid] = T[(pid * n)];
    }

    pram_free(A);
    pram_free(x);
    pram_free(y);
    pram_free(T);
    return 0;
}
