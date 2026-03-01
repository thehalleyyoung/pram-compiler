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

/* PRAM program: biconnected_components | model: CREW */
/* Parallel biconnected components via ear decomposition. CREW, O(log^2 n) time, n+m processors. */
/* Work bound: O((n+m) log^2 n) */
/* Time bound: O(log^2 n) */

/* Generated C99 code for PRAM program: biconnected_components */
/* Memory model: CREW */
/* Parallel biconnected components via ear decomposition. CREW, O(log^2 n) time, n+m processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t m = 0;
    int64_t _num_procs = (n + m);

    int64_t* row_ptr = (int64_t*)pram_calloc((n + 1LL), sizeof(int64_t));
    int64_t* col_idx = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* parent = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* depth = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* low = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* bicomp_id = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* is_artic = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* stack_arr = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* comp_count = (int64_t*)pram_calloc(1LL, sizeof(int64_t));

    /* Phase 1: compute depth via parent pointers */
    for (int64_t pid = 0; pid < n; pid++) {
        int64_t d = 0LL;
        int64_t cur = pid;
        while ((parent[cur] >= 0LL)) {
            d = (d + 1LL);
            cur = parent[cur];
        }
        depth[pid] = d;
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Phase 2: initialise low = depth */
    for (int64_t pid = 0; pid < n; pid++) {
        low[pid] = depth[pid];
    }
    /* ---- barrier (end of phase 1) ---- */
    /* Phase 3: update low values via neighbours and pointer-jumping */
    int64_t log_n = (((int64_t)(log2(((double)(n))))) + 2LL);
    for (int64_t round = 0LL; round < log_n; round += 1LL) {
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t start = row_ptr[pid];
            int64_t end = row_ptr[(pid + 1LL)];
            for (int64_t e = start; e < end; e += 1LL) {
                int64_t nbr = col_idx[e];
                int64_t d_nbr = depth[nbr];
                if ((d_nbr < low[pid])) {
                    low[pid] = d_nbr;
                }
            }
            int64_t p = parent[pid];
            if ((p >= 0LL)) {
                int64_t par_low = low[p];
                if ((par_low < low[pid])) {
                    low[pid] = par_low;
                }
            }
        }
        /* ---- barrier (end of phase 2) ---- */
    }
    /* Phase 4: identify articulation points */
    for (int64_t pid = 0; pid < n; pid++) {
        is_artic[pid] = 0LL;
        int64_t start = row_ptr[pid];
        int64_t end = row_ptr[(pid + 1LL)];
        for (int64_t e = start; e < end; e += 1LL) {
            int64_t nbr = col_idx[e];
            if ((parent[nbr] == pid)) {
                if ((low[nbr] >= depth[pid])) {
                    is_artic[pid] = 1LL;
                }
            }
        }
    }
    /* ---- barrier (end of phase 3) ---- */
    /* Phase 5: assign biconnected component ids */
    for (int64_t pid = 0; pid < m; pid++) {
        bicomp_id[pid] = pid;
    }
    /* ---- barrier (end of phase 4) ---- */
    for (int64_t step = 0LL; step < log_n; step += 1LL) {
        for (int64_t pid = 0; pid < m; pid++) {
            int64_t src = col_idx[pid];
            int64_t my_id = bicomp_id[pid];
            int64_t src_id = bicomp_id[src];
            if (((src_id < my_id) && (is_artic[src] == 0LL))) {
                bicomp_id[pid] = src_id;
            }
        }
        /* ---- barrier (end of phase 5) ---- */
    }

    pram_free(row_ptr);
    pram_free(col_idx);
    pram_free(parent);
    pram_free(depth);
    pram_free(low);
    pram_free(bicomp_id);
    pram_free(is_artic);
    pram_free(stack_arr);
    pram_free(comp_count);
    return 0;
}
