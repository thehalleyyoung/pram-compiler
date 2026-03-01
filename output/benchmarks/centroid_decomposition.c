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

/* PRAM program: centroid_decomposition | model: EREW */
/* Parallel centroid decomposition. EREW, O(log^2 n) time, n processors. */
/* Work bound: O(n log^2 n) */
/* Time bound: O(log^2 n) */

/* Generated C99 code for PRAM program: centroid_decomposition */
/* Memory model: EREW */
/* Parallel centroid decomposition. EREW, O(log^2 n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    int64_t* parent = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* left_child = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* right_child = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* subtree_size = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* centroid_parent = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* centroid_depth = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* active = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* temp = (int64_t*)pram_calloc(n, sizeof(int64_t));

    /* Phase 1: initialise */
    for (int64_t pid = 0; pid < n; pid++) {
        active[pid] = 1LL;
        centroid_parent[pid] = (-1LL);
        centroid_depth[pid] = 0LL;
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Phase 2: iterative centroid finding */
    int64_t log_n = (((int64_t)(log2(((double)(n))))) + 1LL);
    for (int64_t round = 0LL; round < log_n; round += 1LL) {
        /* Phase 2a: compute subtree sizes */
        for (int64_t pid = 0; pid < n; pid++) {
            subtree_size[pid] = active[pid];
        }
        /* ---- barrier (end of phase 1) ---- */
        for (int64_t step = 0LL; step < log_n; step += 1LL) {
            for (int64_t pid = 0; pid < n; pid++) {
                if ((active[pid] == 1LL)) {
                    int64_t lc = left_child[pid];
                    int64_t rc = right_child[pid];
                    int64_t sz = subtree_size[pid];
                    if (((lc >= 0LL) && (active[lc] == 1LL))) {
                        sz = (sz + subtree_size[lc]);
                    }
                    if (((rc >= 0LL) && (active[rc] == 1LL))) {
                        sz = (sz + subtree_size[rc]);
                    }
                    subtree_size[pid] = sz;
                }
            }
            /* ---- barrier (end of phase 2) ---- */
        }
        /* Phase 2b: identify centroids */
        for (int64_t pid = 0; pid < n; pid++) {
            if ((active[pid] == 1LL)) {
                int64_t my_sz = subtree_size[pid];
                int64_t lc = left_child[pid];
                int64_t rc = right_child[pid];
                int64_t lc_sz = (((lc >= 0LL) && (active[lc] == 1LL)) ? subtree_size[lc] : 0LL);
                int64_t rc_sz = (((rc >= 0LL) && (active[rc] == 1LL)) ? subtree_size[rc] : 0LL);
                if (((lc_sz <= (my_sz / 2LL)) && (rc_sz <= (my_sz / 2LL)))) {
                    temp[pid] = 1LL;
                    centroid_depth[pid] = round;
                }
            }
        }
        /* ---- barrier (end of phase 3) ---- */
        /* Phase 2c: deactivate centroids */
        for (int64_t pid = 0; pid < n; pid++) {
            if (((active[pid] == 1LL) && (temp[pid] == 1LL))) {
                active[pid] = 0LL;
                temp[pid] = 0LL;
                int64_t lc = left_child[pid];
                int64_t rc = right_child[pid];
                if (((lc >= 0LL) && (active[lc] == 1LL))) {
                    centroid_parent[lc] = pid;
                }
                if (((rc >= 0LL) && (active[rc] == 1LL))) {
                    centroid_parent[rc] = pid;
                }
            }
        }
        /* ---- barrier (end of phase 4) ---- */
    }

    pram_free(parent);
    pram_free(left_child);
    pram_free(right_child);
    pram_free(subtree_size);
    pram_free(centroid_parent);
    pram_free(centroid_depth);
    pram_free(active);
    pram_free(temp);
    return 0;
}
