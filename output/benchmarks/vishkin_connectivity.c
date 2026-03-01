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

/* PRAM program: vishkin_connectivity | model: CREW */
/* Vishkin's deterministic connectivity. CREW, O(log n) time, m+n processors. */
/* Work bound: O((m+n) log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: vishkin_connectivity */
/* Memory model: CREW */
/* Vishkin's deterministic connectivity. CREW, O(log n) time, m+n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t m = 0;
    int64_t _num_procs = (m + n);

    int64_t* row_ptr = (int64_t*)pram_calloc((n + 1LL), sizeof(int64_t));
    int64_t* col_idx = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* comp = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* min_neighbor_comp = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* changed = (int64_t*)pram_calloc(1LL, sizeof(int64_t));

    /* Initialise: comp[i] = i */
    for (int64_t pid = 0; pid < n; pid++) {
        comp[pid] = pid;
    }
    /* ---- barrier (end of phase 0) ---- */
    int64_t max_rounds = (((int64_t)(log2(((double)(n))))) + 1LL);
    for (int64_t round = 0LL; round < max_rounds; round += 1LL) {
        changed[0LL] = 0LL;
        /* ---- barrier (end of phase 1) ---- */
        /* Find minimum neighbor component for each vertex */
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t my_comp = comp[pid];
            int64_t best = my_comp;
            int64_t start = row_ptr[pid];
            int64_t end = row_ptr[(pid + 1LL)];
            for (int64_t e = start; e < end; e += 1LL) {
                int64_t nbr = col_idx[e];
                int64_t nbr_comp = comp[nbr];
                best = PRAM_MIN(best, nbr_comp);
            }
            min_neighbor_comp[pid] = best;
        }
        /* ---- barrier (end of phase 2) ---- */
        /* Deterministic hooking: each vertex hooks to min neighbor comp */
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t mc = min_neighbor_comp[pid];
            int64_t cc = comp[pid];
            if ((mc < cc)) {
                comp[pid] = mc;
                changed[0LL] = 1LL;
            }
        }
        /* ---- barrier (end of phase 3) ---- */
        /* Pointer jumping: comp[i] <- comp[comp[i]] */
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t c = comp[pid];
            int64_t cc = comp[c];
            if ((c != cc)) {
                comp[pid] = cc;
                changed[0LL] = 1LL;
            }
        }
        /* ---- barrier (end of phase 4) ---- */
    }

    pram_free(row_ptr);
    pram_free(col_idx);
    pram_free(comp);
    pram_free(min_neighbor_comp);
    pram_free(changed);
    return 0;
}
