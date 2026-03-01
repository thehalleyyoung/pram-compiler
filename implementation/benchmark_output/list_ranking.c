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

/* PRAM program: list_ranking | model: EREW */
/* Pointer-jumping list ranking. EREW, O(log n) time, n processors. */
/* Work bound: O(n log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: list_ranking */
/* Memory model: EREW */
/* Pointer-jumping list ranking. EREW, O(log n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    int64_t* next = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* rank = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* succ = (int64_t*)pram_calloc(n, sizeof(int64_t));

    /* Initialise rank and working successor array */
    for (int64_t pid = 0; pid < n; pid++) {
        int64_t nxt = next[pid];
        succ[pid] = nxt;
        if ((nxt >= 0LL)) {
            rank[pid] = 1LL;
        } else {
            rank[pid] = 0LL;
        }
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Pointer jumping: O(log n) rounds */
    int64_t log_n = (((int64_t)(log2(((double)(n))))) + 1LL);
    for (int64_t step = 0LL; step < log_n; step += 1LL) {
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t s = succ[pid];
            if ((s >= 0LL)) {
                int64_t ss = succ[s];
                int64_t rank_s = rank[s];
                rank[pid] = (rank[pid] + rank_s);
                succ[pid] = ss;
            }
        }
        /* ---- barrier (end of phase 1) ---- */
    }

    pram_free(next);
    pram_free(rank);
    pram_free(succ);
    return 0;
}
