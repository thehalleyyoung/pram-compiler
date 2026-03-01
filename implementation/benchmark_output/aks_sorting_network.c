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

/* PRAM program: aks_sorting_network | model: EREW */
/* AKS optimal sorting network. EREW, O(log n) depth, n log n comparators. */
/* Work bound: O(n log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: aks_sorting_network */
/* Memory model: EREW */
/* AKS optimal sorting network. EREW, O(log n) depth, n log n comparators. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    int64_t* A = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* B = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* perm = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* rank = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* temp = (int64_t*)pram_calloc(n, sizeof(int64_t));

    /* Phase 1: initialize permutation and rank */
    for (int64_t pid = 0; pid < n; pid++) {
        perm[pid] = pid;
        rank[pid] = pid;
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Phase 2: O(log n) rounds of expander-based sorting */
    int64_t log_n = ((int64_t)(log2(((double)(n)))));
    for (int64_t stage = 0LL; stage < log_n; stage += 1LL) {
        /* Phase 2a: expander-graph compare-swap on pairs */
        for (int64_t pid = 0; pid < (n / 2LL); pid++) {
            int64_t left = (2LL * pid);
            int64_t right = ((2LL * pid) + 1LL);
            if ((right < n)) {
                int64_t l_idx = perm[left];
                int64_t r_idx = perm[right];
                int64_t l_val = A[l_idx];
                int64_t r_val = A[r_idx];
                if ((l_val > r_val)) {
                    perm[left] = r_idx;
                    perm[right] = l_idx;
                }
            }
        }
        /* ---- barrier (end of phase 1) ---- */
        /* Phase 2b: halving network substeps */
        for (int64_t substep = 0LL; substep < (stage + 1LL); substep += 1LL) {
            int64_t dist = (1LL << substep);
            for (int64_t pid = 0; pid < (n / 2LL); pid++) {
                int64_t group = (pid / dist);
                int64_t pos = (pid % dist);
                int64_t i = ((group * (2LL * dist)) + pos);
                int64_t j_idx = (i + dist);
                if ((j_idx < n)) {
                    int64_t i_perm = perm[i];
                    int64_t j_perm = perm[j_idx];
                    int64_t i_val = A[i_perm];
                    int64_t j_val = A[j_perm];
                    if ((i_val > j_val)) {
                        perm[i] = j_perm;
                        perm[j_idx] = i_perm;
                    }
                }
            }
            /* ---- barrier (end of phase 2) ---- */
        }
    }
    /* Phase 3: apply permutation to produce sorted output */
    for (int64_t pid = 0; pid < n; pid++) {
        B[pid] = A[perm[pid]];
    }
    /* ---- barrier (end of phase 3) ---- */
    /* Phase 4: copy sorted result back to A */
    for (int64_t pid = 0; pid < n; pid++) {
        A[pid] = B[pid];
    }

    pram_free(A);
    pram_free(B);
    pram_free(perm);
    pram_free(rank);
    pram_free(temp);
    return 0;
}
