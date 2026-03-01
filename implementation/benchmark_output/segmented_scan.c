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

/* PRAM program: segmented_scan | model: EREW */
/* Parallel segmented prefix sum. EREW, O(log n) time, n processors. */
/* Work bound: O(n log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: segmented_scan */
/* Memory model: EREW */
/* Parallel segmented prefix sum. EREW, O(log n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    int64_t* A = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* B = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* seg_head = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* temp = (int64_t*)pram_calloc(n, sizeof(int64_t));

    /* Phase 1: initialise B and segment flags */
    for (int64_t pid = 0; pid < n; pid++) {
        B[pid] = A[pid];
        temp[pid] = seg_head[pid];
    }
    /* ---- barrier (end of phase 0) ---- */
    int64_t log_n = (((int64_t)(log2(((double)(n))))) + 1LL);
    /* Phase 2: segmented prefix sum via pointer jumping */
    for (int64_t d = 0LL; d < log_n; d += 1LL) {
        int64_t stride = (1LL << d);
        for (int64_t pid = 0; pid < n; pid++) {
            if ((pid >= stride)) {
                int64_t src = (pid - stride);
                int64_t flag_src = temp[src];
                if ((flag_src == 0LL)) {
                    B[pid] = (B[pid] + B[src]);
                }
                temp[pid] = (temp[pid] | flag_src);
            }
        }
        /* ---- barrier (end of phase 1) ---- */
    }

    pram_free(A);
    pram_free(B);
    pram_free(seg_head);
    pram_free(temp);
    return 0;
}
