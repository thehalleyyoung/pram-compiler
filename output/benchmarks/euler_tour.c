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

/* PRAM program: euler_tour | model: CREW */
/* Euler tour construction. CREW, O(log n) time, n processors. */
/* Work bound: O(n log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: euler_tour */
/* Memory model: CREW */
/* Euler tour construction. CREW, O(log n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    int64_t* parent = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* first_child = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* next_sibling = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* succ = (int64_t*)pram_calloc((2LL * n), sizeof(int64_t));
    int64_t* tour_pos = (int64_t*)pram_calloc((2LL * n), sizeof(int64_t));
    int64_t* rank_arr = (int64_t*)pram_calloc((2LL * n), sizeof(int64_t));

    /* Build Euler-tour successor pointers for each tree edge */
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
    /* List-ranking via pointer jumping on successor list: O(log n) rounds */
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
                int64_t ss = succ[s];
                int64_t r_s = rank_arr[s];
                rank_arr[pid] = (rank_arr[pid] + r_s);
                succ[pid] = ss;
            }
        }
        /* ---- barrier (end of phase 2) ---- */
    }
    /* tour_pos = rank_arr (Euler tour order) */
    for (int64_t pid = 0; pid < (2LL * n); pid++) {
        tour_pos[pid] = rank_arr[pid];
    }

    pram_free(parent);
    pram_free(first_child);
    pram_free(next_sibling);
    pram_free(succ);
    pram_free(tour_pos);
    pram_free(rank_arr);
    return 0;
}
