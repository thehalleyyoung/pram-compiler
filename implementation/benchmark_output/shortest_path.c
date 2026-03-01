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

/* PRAM program: shortest_path | model: CREW */
/* Parallel Bellman-Ford shortest paths. CREW, O(n log n) time, m processors. */
/* Work bound: O(n * m) */
/* Time bound: O(n log n) */

/* Generated C99 code for PRAM program: shortest_path */
/* Memory model: CREW */
/* Parallel Bellman-Ford shortest paths. CREW, O(n log n) time, m processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t m = 0;
    int64_t source = 0;
    int64_t _num_procs = m;

    int64_t* edge_src = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* edge_dst = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* edge_w = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* dist = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* new_dist = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* changed = (int64_t*)pram_calloc(1LL, sizeof(int64_t));

    /* Initialise dist to MAX, dist[source]=0 */
    for (int64_t pid = 0; pid < n; pid++) {
        dist[pid] = 9223372036854775807LL;
        new_dist[pid] = 9223372036854775807LL;
    }
    /* ---- barrier (end of phase 0) ---- */
    dist[source] = 0LL;
    new_dist[source] = 0LL;
    /* ---- barrier (end of phase 1) ---- */
    /* Bellman-Ford: n-1 relaxation rounds */
    for (int64_t iter = 0LL; iter < (n - 1LL); iter += 1LL) {
        /* Phase 1: reset changed flag */
        changed[0LL] = 0LL;
        /* ---- barrier (end of phase 2) ---- */
        /* Phase 2: relax edges */
        for (int64_t pid = 0; pid < m; pid++) {
            int64_t u = edge_src[pid];
            int64_t v_node = edge_dst[pid];
            int64_t w = edge_w[pid];
            int64_t du = dist[u];
            if ((du < 9223372036854775807LL)) {
                int64_t new_d = (du + w);
                int64_t cur_dv = dist[v_node];
                if ((new_d < cur_dv)) {
                    new_dist[v_node] = new_d;
                    changed[0LL] = 1LL;
                }
            }
        }
        /* ---- barrier (end of phase 3) ---- */
        /* Phase 3: merge new_dist into dist */
        for (int64_t pid = 0; pid < n; pid++) {
            dist[pid] = PRAM_MIN(dist[pid], new_dist[pid]);
        }
        /* ---- barrier (end of phase 4) ---- */
        /* Phase 4: early exit if no changes */
        if ((changed[0LL] == 0LL)) {
            /* Converged – break */
        }
    }

    pram_free(edge_src);
    pram_free(edge_dst);
    pram_free(edge_w);
    pram_free(dist);
    pram_free(new_dist);
    pram_free(changed);
    return 0;
}
