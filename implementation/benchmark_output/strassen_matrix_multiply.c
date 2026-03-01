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

/* PRAM program: strassen_matrix_multiply | model: CREW */
/* Strassen's parallel matrix multiply. CREW, O(n^log2(7)) work, O(log^2 n) time. */
/* Work bound: O(n^log2(7)) */
/* Time bound: O(log^2 n) */

/* Generated C99 code for PRAM program: strassen_matrix_multiply */
/* Memory model: CREW */
/* Strassen's parallel matrix multiply. CREW, O(n^log2(7)) work, O(log^2 n) time. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = (n * n);

    int64_t* A = (int64_t*)pram_calloc((n * n), sizeof(int64_t));
    int64_t* B = (int64_t*)pram_calloc((n * n), sizeof(int64_t));
    int64_t* C = (int64_t*)pram_calloc((n * n), sizeof(int64_t));
    int64_t* M1 = (int64_t*)pram_calloc((n * n), sizeof(int64_t));
    int64_t* M2 = (int64_t*)pram_calloc((n * n), sizeof(int64_t));
    int64_t* M3 = (int64_t*)pram_calloc((n * n), sizeof(int64_t));
    int64_t* M4 = (int64_t*)pram_calloc((n * n), sizeof(int64_t));
    int64_t* temp1 = (int64_t*)pram_calloc((n * n), sizeof(int64_t));
    int64_t* temp2 = (int64_t*)pram_calloc((n * n), sizeof(int64_t));

    /* Phase 1: compute sub-matrix sums for Strassen decomposition */
    for (int64_t pid = 0; pid < (n * n); pid++) {
        int64_t i = (pid / n);
        int64_t j = (pid % n);
        int64_t half = (n / 2LL);
        int64_t a11 = A[((i * n) + j)];
        int64_t a22 = A[(((i + half) * n) + (j + half))];
        int64_t b11 = B[((i * n) + j)];
        int64_t b22 = B[(((i + half) * n) + (j + half))];
        if (((i < half) && (j < half))) {
            temp1[pid] = (a11 + a22);
            temp2[pid] = (b11 + b22);
        }
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Phase 2: compute Strassen intermediate products via parallel multiply */
    int64_t log_n = ((int64_t)(log2(((double)(n)))));
    for (int64_t pid = 0; pid < (n * n); pid++) {
        int64_t i = (pid / n);
        int64_t j = (pid % n);
        int64_t acc = 0LL;
        M1[pid] = acc;
    }
    /* ---- barrier (end of phase 1) ---- */
    for (int64_t k = 0LL; k < n; k += 1LL) {
        for (int64_t pid = 0; pid < (n * n); pid++) {
            int64_t i = (pid / n);
            int64_t j = (pid % n);
            int64_t t1_val = temp1[((i * n) + k)];
            int64_t t2_val = temp2[((k * n) + j)];
            int64_t prev = M1[pid];
            M1[pid] = (prev + (t1_val * t2_val));
        }
        /* ---- barrier (end of phase 2) ---- */
    }
    for (int64_t pid = 0; pid < (n * n); pid++) {
        int64_t i = (pid / n);
        int64_t j = (pid % n);
        int64_t half = (n / 2LL);
        int64_t a21 = A[(((i + half) * n) + j)];
        int64_t a22 = A[(((i + half) * n) + (j + half))];
        M2[pid] = (a21 + a22);
        int64_t b12 = B[((i * n) + (j + half))];
        int64_t b22 = B[(((i + half) * n) + (j + half))];
        M3[pid] = (b12 - b22);
        int64_t b21 = B[(((i + half) * n) + j)];
        int64_t b11 = B[((i * n) + j)];
        M4[pid] = (b21 - b11);
    }
    /* ---- barrier (end of phase 3) ---- */
    /* Phase 3: combine Strassen products into C quadrants */
    for (int64_t pid = 0; pid < (n * n); pid++) {
        int64_t i = (pid / n);
        int64_t j = (pid % n);
        int64_t half = (n / 2LL);
        int64_t m1 = M1[pid];
        int64_t m2 = M2[pid];
        int64_t m3 = M3[pid];
        int64_t m4 = M4[pid];
        if (((i < half) && (j < half))) {
            C[pid] = (m1 + m4);
        }
        if (((i < half) && (j >= half))) {
            C[pid] = m3;
        }
        if (((i >= half) && (j < half))) {
            C[pid] = (m2 + m4);
        }
        if (((i >= half) && (j >= half))) {
            C[pid] = ((m1 - m2) + m3);
        }
    }
    /* ---- barrier (end of phase 4) ---- */

    pram_free(A);
    pram_free(B);
    pram_free(C);
    pram_free(M1);
    pram_free(M2);
    pram_free(M3);
    pram_free(M4);
    pram_free(temp1);
    pram_free(temp2);
    return 0;
}
