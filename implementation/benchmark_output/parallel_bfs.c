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

/* PRAM program: parallel_bfs | model: CREW */
/* Level-synchronous parallel BFS. CREW, O(D) time, n+m processors. */
/* Work bound: O(n + m) */
/* Time bound: O(D) */

/* Generated C99 code for PRAM program: parallel_bfs */
/* Memory model: CREW */
/* Level-synchronous parallel BFS. CREW, O(D) time, n+m processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t m = 0;
    int64_t source = 0;
    int64_t _num_procs = (n + m);

    int64_t* row_ptr = (int64_t*)pram_calloc((n + 1LL), sizeof(int64_t));
    int64_t* col_idx = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* dist = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* frontier = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* next_frontier = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* frontier_size = (int64_t*)pram_calloc(1LL, sizeof(int64_t));

    /* Initialise dist to -1, source to 0 */
    for (int64_t pid = 0; pid < n; pid++) {
        dist[pid] = (-1LL);
        frontier[pid] = 0LL;
        next_frontier[pid] = 0LL;
    }
    /* ---- barrier (end of phase 0) ---- */
    dist[source] = 0LL;
    frontier[source] = 1LL;
    frontier_size[0LL] = 1LL;
    /* ---- barrier (end of phase 1) ---- */
    /* BFS levels: iterate while frontier is non-empty */
    int64_t level = 0LL;
    while ((frontier_size[0LL] > 0LL)) {
        level = (level + 1LL);
        /* Explore neighbors of frontier vertices */
        for (int64_t pid = 0; pid < n; pid++) {
            if ((frontier[pid] == 1LL)) {
                int64_t row_start = row_ptr[pid];
                int64_t row_end = row_ptr[(pid + 1LL)];
                for (int64_t e = row_start; e < row_end; e += 1LL) {
                    int64_t neighbor = col_idx[e];
                    if ((dist[neighbor] == (-1LL))) {
                        dist[neighbor] = level;
                        next_frontier[neighbor] = 1LL;
                    }
                }
            }
        }
        /* ---- barrier (end of phase 2) ---- */
        /* Swap frontier and next_frontier */
        frontier_size[0LL] = 0LL;
        for (int64_t pid = 0; pid < n; pid++) {
            frontier[pid] = next_frontier[pid];
            next_frontier[pid] = 0LL;
        }
        /* ---- barrier (end of phase 3) ---- */
        /* Prefix sum (+) frontier -> frontier_size */
        frontier_size[0] = frontier[0];
        for (int64_t _ps_i = 1; _ps_i < n; _ps_i++) {
            frontier_size[_ps_i] = frontier_size[_ps_i - 1] + frontier[_ps_i];
        }
        /* ---- barrier (end of phase 4) ---- */
    }

    pram_free(row_ptr);
    pram_free(col_idx);
    pram_free(dist);
    pram_free(frontier);
    pram_free(next_frontier);
    pram_free(frontier_size);
    return 0;
}
