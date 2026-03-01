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

/* PRAM program: symmetry_breaking | model: EREW */
/* Deterministic symmetry breaking (Cole-Vishkin). EREW, O(log* n) time, n processors. */
/* Work bound: O(n log* n) */
/* Time bound: O(log* n) */

/* Generated C99 code for PRAM program: symmetry_breaking */
/* Memory model: EREW */
/* Deterministic symmetry breaking (Cole-Vishkin). EREW, O(log* n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    int64_t* next = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* color = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* new_color = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* num_rounds = (int64_t*)pram_calloc(1LL, sizeof(int64_t));

    /* Phase 1: unique initial coloring */
    for (int64_t pid = 0; pid < n; pid++) {
        color[pid] = pid;
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Phase 2: compute number of Cole-Vishkin rounds */
    int64_t nr = ((3LL * ((int64_t)(log2((log2(((double)(n))) + 1.0))))) + 6LL);
    num_rounds[0LL] = nr;
    /* ---- barrier (end of phase 1) ---- */
    /* Phase 3: Cole-Vishkin color reduction */
    for (int64_t round = 0LL; round < nr; round += 1LL) {
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t c1 = color[pid];
            int64_t nxt = next[pid];
            if ((nxt >= 0LL)) {
                int64_t c2 = color[nxt];
            } else {
                int64_t c2 = c1;
            }
            int64_t diff = (c1 ^ c2);
            int64_t bit_pos = 0LL;
            int64_t tmp_diff = diff;
            if ((tmp_diff != 0LL)) {
                while (((tmp_diff & 1LL) == 0LL)) {
                    tmp_diff = (tmp_diff >> 1LL);
                    bit_pos = (bit_pos + 1LL);
                }
            }
            int64_t bit_val = ((c1 >> bit_pos) & 1LL);
            new_color[pid] = ((2LL * bit_pos) + bit_val);
        }
        /* ---- barrier (end of phase 2) ---- */
        for (int64_t pid = 0; pid < n; pid++) {
            color[pid] = new_color[pid];
        }
        /* ---- barrier (end of phase 3) ---- */
    }

    pram_free(next);
    pram_free(color);
    pram_free(new_color);
    pram_free(num_rounds);
    return 0;
}
