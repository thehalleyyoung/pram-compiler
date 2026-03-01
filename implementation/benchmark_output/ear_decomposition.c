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

/* PRAM program: ear_decomposition | model: CREW */
/* Reif's ear decomposition. CREW, O(log n) time, m processors. */
/* Work bound: O(m log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: ear_decomposition */
/* Memory model: CREW */
/* Reif's ear decomposition. CREW, O(log n) time, m processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t m = 0;
    int64_t _num_procs = m;

    int64_t* parent = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* depth = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* nt_src = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* nt_dst = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* nt_count = (int64_t*)pram_calloc(1LL, sizeof(int64_t));
    int64_t* euler_succ = (int64_t*)pram_calloc((2LL * n), sizeof(int64_t));
    int64_t* euler_rank = (int64_t*)pram_calloc((2LL * n), sizeof(int64_t));
    int64_t* ear_id = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* lca_arr = (int64_t*)pram_calloc(m, sizeof(int64_t));

    /* Step 1: compute depths via parent pointers */
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
    /* Step 2: Euler tour via pointer jumping */
    for (int64_t pid = 0; pid < (2LL * n); pid++) {
        euler_rank[pid] = 1LL;
    }
    /* ---- barrier (end of phase 1) ---- */
    int64_t log_n = (((int64_t)(log2(((double)(n))))) + 2LL);
    for (int64_t step = 0LL; step < log_n; step += 1LL) {
        for (int64_t pid = 0; pid < (2LL * n); pid++) {
            int64_t s = euler_succ[pid];
            if ((s != pid)) {
                euler_rank[pid] = (euler_rank[pid] + euler_rank[s]);
                euler_succ[pid] = euler_succ[s];
            }
        }
        /* ---- barrier (end of phase 2) ---- */
    }
    /* Step 3: compute LCA for each non-tree edge, assign ear ids */
    for (int64_t pid = 0; pid < nt_count[0LL]; pid++) {
        int64_t u = nt_src[pid];
        int64_t v_node = nt_dst[pid];
        int64_t du = depth[u];
        int64_t dv = depth[v_node];
        while ((du > dv)) {
            u = parent[u];
            du = (du - 1LL);
        }
        while ((dv > du)) {
            v_node = parent[v_node];
            dv = (dv - 1LL);
        }
        while ((u != v_node)) {
            u = parent[u];
            v_node = parent[v_node];
        }
        lca_arr[pid] = u;
        ear_id[u] = pid;
    }
    /* ---- barrier (end of phase 3) ---- */
    /* Step 4: propagate ear ids down from LCA */
    for (int64_t step = 0LL; step < log_n; step += 1LL) {
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t my_ear = ear_id[pid];
            int64_t par_ear = ear_id[parent[pid]];
            if (((my_ear == (-1LL)) && (par_ear != (-1LL)))) {
                ear_id[pid] = par_ear;
            }
        }
        /* ---- barrier (end of phase 4) ---- */
    }

    pram_free(parent);
    pram_free(depth);
    pram_free(nt_src);
    pram_free(nt_dst);
    pram_free(nt_count);
    pram_free(euler_succ);
    pram_free(euler_rank);
    pram_free(ear_id);
    pram_free(lca_arr);
    return 0;
}
