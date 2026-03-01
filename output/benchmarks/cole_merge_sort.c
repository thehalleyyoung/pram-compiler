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

/* PRAM program: cole_merge_sort | model: CREW */
/* Cole's O(log n) pipelined merge sort. Each of O(log n) phases advances every merge-tree level by one pipelined merge step. */
/* Work bound: O(n log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: cole_merge_sort */
/* Memory model: CREW */
/* Cole's O(log n) pipelined merge sort. Each of O(log n) phases advances every merge-tree level by one pipelined merge step. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    int64_t* A = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* B = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* rank = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* level_sorted = (int64_t*)pram_calloc(n, sizeof(int64_t));

    /* Phase 0: initialise ranks – each element is its own sorted run */
    for (int64_t pid = 0; pid < n; pid++) {
        rank[pid] = pid;
        level_sorted[pid] = A[pid];
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Pipelined merge: O(log n) super-steps; each level advances one merge step */
    int64_t num_phases = ((int64_t)(log2(((double)(n)))));
    for (int64_t phase = 0LL; phase < num_phases; phase += 1LL) {
        /* Compute current run length = 2^phase */
        int64_t run_len = (1LL << phase);
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t run_id = (pid / run_len);
            int64_t pos_in_run = (pid % run_len);
            int64_t partner_start_val = (((run_id % 2LL) == 0LL) ? ((run_id * run_len) + run_len) : ((run_id - 1LL) * run_len));
            int64_t lo = 0LL;
            int64_t hi = (run_len - 1LL);
            int64_t my_val = level_sorted[pid];
            while ((lo <= hi)) {
                int64_t mid = ((lo + hi) / 2LL);
                int64_t partner_idx = (partner_start_val + mid);
                if ((partner_idx < n)) {
                    int64_t partner_val = level_sorted[partner_idx];
                    if ((partner_val <= my_val)) {
                        lo = (mid + 1LL);
                    } else {
                        hi = (mid - 1LL);
                    }
                } else {
                    hi = (mid - 1LL);
                }
            }
            rank[pid] = (pos_in_run + lo);
        }
        /* ---- barrier (end of phase 1) ---- */
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t dest = rank[pid];
            int64_t run_id = (pid / run_len);
            int64_t merged_start = ((run_id / 2LL) * (2LL * run_len));
            int64_t final_pos = (merged_start + dest);
            if ((final_pos < n)) {
                B[final_pos] = level_sorted[pid];
            }
        }
        /* ---- barrier (end of phase 2) ---- */
        for (int64_t pid = 0; pid < n; pid++) {
            level_sorted[pid] = B[pid];
        }
        /* ---- barrier (end of phase 3) ---- */
    }
    /* Copy sorted result to A */
    for (int64_t pid = 0; pid < n; pid++) {
        A[pid] = level_sorted[pid];
    }

    pram_free(A);
    pram_free(B);
    pram_free(rank);
    pram_free(level_sorted);
    return 0;
}
