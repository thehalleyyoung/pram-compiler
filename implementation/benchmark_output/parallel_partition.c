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

/* PRAM program: parallel_partition | model: EREW */
/* Parallel array partitioning. EREW, O(log n) time, n processors. */
/* Work bound: O(n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: parallel_partition */
/* Memory model: EREW */
/* Parallel array partitioning. EREW, O(log n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t pivot_val = 0;
    int64_t _num_procs = n;

    int64_t* A = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* B = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* lt_flag = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* ge_flag = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* lt_dest = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* ge_dest = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* partition_point = (int64_t*)pram_calloc(1LL, sizeof(int64_t));

    /* Phase 1: classify elements */
    for (int64_t pid = 0; pid < n; pid++) {
        if ((A[pid] < pivot_val)) {
            lt_flag[pid] = 1LL;
            ge_flag[pid] = 0LL;
        } else {
            lt_flag[pid] = 0LL;
            ge_flag[pid] = 1LL;
        }
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Phase 2: prefix sums for destinations */
    /* Prefix sum (+) lt_flag -> lt_dest */
    lt_dest[0] = lt_flag[0];
    for (int64_t _ps_i = 1; _ps_i < n; _ps_i++) {
        lt_dest[_ps_i] = lt_dest[_ps_i - 1] + lt_flag[_ps_i];
    }
    /* Prefix sum (+) ge_flag -> ge_dest */
    ge_dest[0] = ge_flag[0];
    for (int64_t _ps_i = 1; _ps_i < n; _ps_i++) {
        ge_dest[_ps_i] = ge_dest[_ps_i - 1] + ge_flag[_ps_i];
    }
    /* ---- barrier (end of phase 1) ---- */
    /* Phase 3: store partition point */
    partition_point[0LL] = lt_dest[(n - 1LL)];
    /* ---- barrier (end of phase 2) ---- */
    /* Phase 4: scatter elements */
    for (int64_t pid = 0; pid < n; pid++) {
        if ((lt_flag[pid] == 1LL)) {
            B[(lt_dest[pid] - 1LL)] = A[pid];
        } else {
            B[(partition_point[0LL] + (ge_dest[pid] - 1LL))] = A[pid];
        }
    }
    /* ---- barrier (end of phase 3) ---- */
    /* Phase 5: copy B to A */
    for (int64_t pid = 0; pid < n; pid++) {
        A[pid] = B[pid];
    }
    /* ---- barrier (end of phase 4) ---- */

    pram_free(A);
    pram_free(B);
    pram_free(lt_flag);
    pram_free(ge_flag);
    pram_free(lt_dest);
    pram_free(ge_dest);
    pram_free(partition_point);
    return 0;
}
