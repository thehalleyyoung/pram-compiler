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

/* PRAM program: parallel_selection | model: CREW */
/* Parallel kth-element selection. CREW, O(log n) expected time, n processors. */
/* Work bound: O(n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: parallel_selection */
/* Memory model: CREW */
/* Parallel kth-element selection. CREW, O(log n) expected time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t k = 0;
    int64_t _num_procs = n;

    int64_t* A = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* B = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* lt_flag = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* eq_flag = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* gt_flag = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* lt_dest = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* gt_dest = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* pivot = (int64_t*)pram_calloc(1LL, sizeof(int64_t));
    int64_t* result = (int64_t*)pram_calloc(1LL, sizeof(int64_t));
    int64_t* cur_n = (int64_t*)pram_calloc(1LL, sizeof(int64_t));
    int64_t* cur_k = (int64_t*)pram_calloc(1LL, sizeof(int64_t));
    int64_t* done = (int64_t*)pram_calloc(1LL, sizeof(int64_t));

    /* Initialise: cur_n = n, cur_k = k */
    cur_n[0LL] = n;
    cur_k[0LL] = k;
    done[0LL] = 0LL;
    /* ---- barrier (end of phase 0) ---- */
    /* Selection loop: O(log n) expected rounds */
    int64_t max_rounds = ((4LL * ((int64_t)(log2(((double)(n)))))) + 2LL);
    for (int64_t round = 0LL; round < max_rounds; round += 1LL) {
        if ((done[0LL] == 1LL)) {
            /* Already found result */
        }
        if (((cur_n[0LL] <= 1LL) && (done[0LL] == 0LL))) {
            result[0LL] = A[0LL];
            done[0LL] = 1LL;
        }
        /* ---- barrier (end of phase 1) ---- */
        /* Pick pivot from median of three */
        for (int64_t pid = 0; pid < 1LL; pid++) {
            if ((done[0LL] == 0LL)) {
                int64_t cn = cur_n[0LL];
                int64_t first = A[0LL];
                int64_t mid_val = A[(cn / 2LL)];
                int64_t last = A[(cn - 1LL)];
                int64_t med = (((first >= mid_val) && (first <= last)) ? first : (((mid_val >= first) && (mid_val <= last)) ? mid_val : last));
                pivot[0LL] = med;
            }
        }
        /* ---- barrier (end of phase 2) ---- */
        /* Partition elements around pivot */
        for (int64_t pid = 0; pid < n; pid++) {
            if (((pid < cur_n[0LL]) && (done[0LL] == 0LL))) {
                int64_t val = A[pid];
                int64_t piv = pivot[0LL];
                if ((val < piv)) {
                    lt_flag[pid] = 1LL;
                    eq_flag[pid] = 0LL;
                    gt_flag[pid] = 0LL;
                } else {
                    if ((val == piv)) {
                        lt_flag[pid] = 0LL;
                        eq_flag[pid] = 1LL;
                        gt_flag[pid] = 0LL;
                    } else {
                        lt_flag[pid] = 0LL;
                        eq_flag[pid] = 0LL;
                        gt_flag[pid] = 1LL;
                    }
                }
            }
        }
        /* ---- barrier (end of phase 3) ---- */
        /* Prefix sum (+) lt_flag -> lt_dest */
        lt_dest[0] = lt_flag[0];
        for (int64_t _ps_i = 1; _ps_i < n; _ps_i++) {
            lt_dest[_ps_i] = lt_dest[_ps_i - 1] + lt_flag[_ps_i];
        }
        /* Prefix sum (+) gt_flag -> gt_dest */
        gt_dest[0] = gt_flag[0];
        for (int64_t _ps_i = 1; _ps_i < n; _ps_i++) {
            gt_dest[_ps_i] = gt_dest[_ps_i - 1] + gt_flag[_ps_i];
        }
        /* ---- barrier (end of phase 4) ---- */
        /* Determine which partition contains k */
        for (int64_t pid = 0; pid < 1LL; pid++) {
            if ((done[0LL] == 0LL)) {
                int64_t cn = cur_n[0LL];
                int64_t ck = cur_k[0LL];
                int64_t lt_count = lt_dest[(cn - 1LL)];
                int64_t eq_count = ((cn - lt_count) - gt_dest[(cn - 1LL)]);
                if ((ck < lt_count)) {
                    cur_n[0LL] = lt_count;
                } else {
                    if ((ck < (lt_count + eq_count))) {
                        result[0LL] = pivot[0LL];
                        done[0LL] = 1LL;
                    } else {
                        cur_k[0LL] = (ck - (lt_count + eq_count));
                        cur_n[0LL] = gt_dest[(cn - 1LL)];
                    }
                }
            }
        }
        /* ---- barrier (end of phase 5) ---- */
        /* Compact chosen partition into B, then copy back to A */
        for (int64_t pid = 0; pid < n; pid++) {
            if (((pid < cur_n[0LL]) && (done[0LL] == 0LL))) {
                if ((lt_flag[pid] == 1LL)) {
                    int64_t d = (lt_dest[pid] - 1LL);
                    B[d] = A[pid];
                }
                if ((gt_flag[pid] == 1LL)) {
                    int64_t d = (gt_dest[pid] - 1LL);
                    B[d] = A[pid];
                }
            }
        }
        /* ---- barrier (end of phase 6) ---- */
        for (int64_t pid = 0; pid < n; pid++) {
            if ((pid < cur_n[0LL])) {
                A[pid] = B[pid];
            }
        }
        /* ---- barrier (end of phase 7) ---- */
    }

    pram_free(A);
    pram_free(B);
    pram_free(lt_flag);
    pram_free(eq_flag);
    pram_free(gt_flag);
    pram_free(lt_dest);
    pram_free(gt_dest);
    pram_free(pivot);
    pram_free(result);
    pram_free(cur_n);
    pram_free(cur_k);
    pram_free(done);
    return 0;
}
