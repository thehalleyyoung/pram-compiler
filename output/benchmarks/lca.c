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

/* PRAM program: lca | model: CREW */
/* LCA via Euler tour + sparse table RMQ. CREW, O(log n) preprocessing, n processors. */
/* Work bound: O(n log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: lca */
/* Memory model: CREW */
/* LCA via Euler tour + sparse table RMQ. CREW, O(log n) preprocessing, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    int64_t* parent = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* first_child = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* next_sibling = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* euler_node = (int64_t*)pram_calloc(((2LL * n) - 1LL), sizeof(int64_t));
    int64_t* euler_depth = (int64_t*)pram_calloc(((2LL * n) - 1LL), sizeof(int64_t));
    int64_t* first_occ = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* succ = (int64_t*)pram_calloc((2LL * n), sizeof(int64_t));
    int64_t* rank_arr = (int64_t*)pram_calloc((2LL * n), sizeof(int64_t));
    int64_t* sparse = (int64_t*)pram_calloc(((((int64_t)(log2(((double)(n))))) + 2LL) * ((2LL * n) - 1LL)), sizeof(int64_t));

    /* Step 1: build Euler tour via successor + list ranking */
    for (int64_t pid = 0; pid < n; pid++) {
        int64_t fc = first_child[pid];
        int64_t ns = next_sibling[pid];
        int64_t par = parent[pid];
        if ((fc >= 0LL)) {
            succ[(2LL * pid)] = (2LL * fc);
        } else {
            succ[(2LL * pid)] = ((2LL * pid) + 1LL);
        }
        if ((ns >= 0LL)) {
            succ[((2LL * pid) + 1LL)] = (2LL * ns);
        } else {
            if ((par >= 0LL)) {
                succ[((2LL * pid) + 1LL)] = ((2LL * par) + 1LL);
            } else {
                succ[((2LL * pid) + 1LL)] = ((2LL * pid) + 1LL);
            }
        }
    }
    /* ---- barrier (end of phase 0) ---- */
    for (int64_t pid = 0; pid < (2LL * n); pid++) {
        if ((succ[pid] != pid)) {
            rank_arr[pid] = 1LL;
        } else {
            rank_arr[pid] = 0LL;
        }
    }
    /* ---- barrier (end of phase 1) ---- */
    int64_t log_n = (((int64_t)(log2(((double)(n))))) + 2LL);
    for (int64_t step = 0LL; step < log_n; step += 1LL) {
        for (int64_t pid = 0; pid < (2LL * n); pid++) {
            int64_t s = succ[pid];
            if ((s != pid)) {
                rank_arr[pid] = (rank_arr[pid] + rank_arr[s]);
                succ[pid] = succ[s];
            }
        }
        /* ---- barrier (end of phase 2) ---- */
    }
    /* Fill Euler tour arrays from ranks */
    for (int64_t pid = 0; pid < (2LL * n); pid++) {
        int64_t pos = rank_arr[pid];
        int64_t node_id = (pid / 2LL);
        euler_node[pos] = node_id;
    }
    /* ---- barrier (end of phase 3) ---- */
    for (int64_t pid = 0; pid < n; pid++) {
        int64_t d = 0LL;
        int64_t cur = pid;
        while ((parent[cur] >= 0LL)) {
            d = (d + 1LL);
            cur = parent[cur];
        }
        int64_t first_pos = rank_arr[(2LL * pid)];
        euler_depth[first_pos] = d;
        first_occ[pid] = first_pos;
    }
    /* ---- barrier (end of phase 4) ---- */
    /* Step 2: sparse table for RMQ on euler_depth */
    for (int64_t pid = 0; pid < ((2LL * n) - 1LL); pid++) {
        sparse[pid] = pid;
    }
    /* ---- barrier (end of phase 5) ---- */
    for (int64_t k = 1LL; k < log_n; k += 1LL) {
        int64_t half_span = (1LL << (k - 1LL));
        int64_t base = (k * ((2LL * n) - 1LL));
        int64_t prev_base = ((k - 1LL) * ((2LL * n) - 1LL));
        for (int64_t pid = 0; pid < ((2LL * n) - 1LL); pid++) {
            int64_t right_start = (pid + half_span);
            if ((right_start < ((2LL * n) - 1LL))) {
                int64_t left_idx = sparse[(prev_base + pid)];
                int64_t right_idx = sparse[(prev_base + right_start)];
                int64_t left_depth = euler_depth[left_idx];
                int64_t right_depth = euler_depth[right_idx];
                if ((left_depth <= right_depth)) {
                    sparse[(base + pid)] = left_idx;
                } else {
                    sparse[(base + pid)] = right_idx;
                }
            }
        }
        /* ---- barrier (end of phase 6) ---- */
    }

    pram_free(parent);
    pram_free(first_child);
    pram_free(next_sibling);
    pram_free(euler_node);
    pram_free(euler_depth);
    pram_free(first_occ);
    pram_free(succ);
    pram_free(rank_arr);
    pram_free(sparse);
    return 0;
}
