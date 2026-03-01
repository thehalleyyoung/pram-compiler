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

/* PRAM program: tree_contraction | model: EREW */
/* Rake/compress tree contraction. EREW, O(log n) time, n processors. */
/* Work bound: O(n log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: tree_contraction */
/* Memory model: EREW */
/* Rake/compress tree contraction. EREW, O(log n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    int64_t* parent = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* left_child = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* right_child = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* value = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* active = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* degree = (int64_t*)pram_calloc(n, sizeof(int64_t));

    /* Initialise: all nodes active */
    for (int64_t pid = 0; pid < n; pid++) {
        active[pid] = 1LL;
        int64_t deg = 0LL;
        if ((left_child[pid] >= 0LL)) {
            deg = (deg + 1LL);
        }
        if ((right_child[pid] >= 0LL)) {
            deg = (deg + 1LL);
        }
        degree[pid] = deg;
    }
    /* ---- barrier (end of phase 0) ---- */
    /* O(log n) rounds of rake + compress */
    int64_t log_n = (((int64_t)(log2(((double)(n))))) + 1LL);
    for (int64_t round = 0LL; round < log_n; round += 1LL) {
        /* Rake: remove leaf nodes */
        for (int64_t pid = 0; pid < n; pid++) {
            if (((active[pid] == 1LL) && ((degree[pid] == 0LL) && (parent[pid] >= 0LL)))) {
                int64_t par = parent[pid];
                value[par] = (value[par] + value[pid]);
                degree[par] = (degree[par] - 1LL);
                active[pid] = 0LL;
            }
        }
        /* ---- barrier (end of phase 1) ---- */
        /* Compress: remove degree-1 chain nodes */
        for (int64_t pid = 0; pid < n; pid++) {
            if (((active[pid] == 1LL) && ((degree[pid] == 1LL) && (parent[pid] >= 0LL)))) {
                int64_t par = parent[pid];
                int64_t lc = left_child[pid];
                int64_t rc = right_child[pid];
                int64_t child = (((lc >= 0LL) && (active[lc] == 1LL)) ? lc : rc);
                if ((child >= 0LL)) {
                    parent[child] = par;
                }
                value[par] = (value[par] + value[pid]);
                active[pid] = 0LL;
            }
        }
        /* ---- barrier (end of phase 2) ---- */
    }

    pram_free(parent);
    pram_free(left_child);
    pram_free(right_child);
    pram_free(value);
    pram_free(active);
    pram_free(degree);
    return 0;
}
