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

/* PRAM program: list_split | model: EREW */
/* Parallel list splitting. EREW, O(log n) time, n processors. */
/* Work bound: O(n log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: list_split */
/* Memory model: EREW */
/* Parallel list splitting. EREW, O(log n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    int64_t* next = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* color = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* next_even = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* next_odd = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* rank_even = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* rank_odd = (int64_t*)pram_calloc(n, sizeof(int64_t));

    /* Phase 1: build same-color sublist pointers */
    for (int64_t pid = 0; pid < n; pid++) {
        int64_t nxt = next[pid];
        int64_t my_color = color[pid];
        if ((nxt >= 0LL)) {
            int64_t nxt_color = color[nxt];
            if ((nxt_color == my_color)) {
                if ((my_color == 0LL)) {
                    next_even[pid] = nxt;
                } else {
                    next_odd[pid] = nxt;
                }
            } else {
                int64_t nxt2 = next[nxt];
                if ((my_color == 0LL)) {
                    next_even[pid] = nxt2;
                } else {
                    next_odd[pid] = nxt2;
                }
            }
            if ((my_color == 0LL)) {
                rank_even[pid] = 1LL;
            } else {
                rank_odd[pid] = 1LL;
            }
        } else {
            if ((my_color == 0LL)) {
                next_even[pid] = (-1LL);
                rank_even[pid] = 0LL;
            } else {
                next_odd[pid] = (-1LL);
                rank_odd[pid] = 0LL;
            }
        }
    }
    /* ---- barrier (end of phase 0) ---- */
    int64_t log_n = (((int64_t)(log2(((double)(n))))) + 1LL);
    /* Phase 2: pointer jumping to rank even sublist */
    for (int64_t step = 0LL; step < log_n; step += 1LL) {
        for (int64_t pid = 0; pid < n; pid++) {
            if ((color[pid] == 0LL)) {
                int64_t s = next_even[pid];
                if ((s >= 0LL)) {
                    rank_even[pid] = (rank_even[pid] + rank_even[s]);
                    next_even[pid] = next_even[s];
                }
            }
        }
        /* ---- barrier (end of phase 1) ---- */
    }
    /* Phase 3: pointer jumping to rank odd sublist */
    for (int64_t step = 0LL; step < log_n; step += 1LL) {
        for (int64_t pid = 0; pid < n; pid++) {
            if ((color[pid] == 1LL)) {
                int64_t s = next_odd[pid];
                if ((s >= 0LL)) {
                    rank_odd[pid] = (rank_odd[pid] + rank_odd[s]);
                    next_odd[pid] = next_odd[s];
                }
            }
        }
        /* ---- barrier (end of phase 2) ---- */
    }

    pram_free(next);
    pram_free(color);
    pram_free(next_even);
    pram_free(next_odd);
    pram_free(rank_even);
    pram_free(rank_odd);
    return 0;
}
