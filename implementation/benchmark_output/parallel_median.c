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

/* PRAM program: parallel_median | model: CREW */
/* Parallel weighted median finding. CREW, O(log^2 n) time, n processors. */
/* Work bound: O(n log^2 n) */
/* Time bound: O(log^2 n) */

/* Generated C99 code for PRAM program: parallel_median */
/* Memory model: CREW */
/* Parallel weighted median finding. CREW, O(log^2 n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    int64_t* A = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* weights = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* B = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* lt_flag = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* lt_dest = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* gt_flag = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* gt_dest = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* weight_sum = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* pivot = (int64_t*)pram_calloc(1LL, sizeof(int64_t));
    int64_t* result = (int64_t*)pram_calloc(1LL, sizeof(int64_t));

    /* Phase 1: copy A to B */
    for (int64_t pid = 0; pid < n; pid++) {
        B[pid] = A[pid];
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Phase 2: weighted median via pivot-based selection */
    int64_t max_rounds = ((4LL * ((int64_t)(log2(((double)(n)))))) + 2LL);
    for (int64_t round = 0LL; round < max_rounds; round += 1LL) {
        /* Pick median-of-three pivot */
        for (int64_t pid = 0; pid < 1LL; pid++) {
            int64_t first = B[0LL];
            int64_t mid_val = B[(n / 2LL)];
            int64_t last = B[(n - 1LL)];
            int64_t med = (((first >= mid_val) && (first <= last)) ? first : (((mid_val >= first) && (mid_val <= last)) ? mid_val : last));
            pivot[0LL] = med;
        }
        /* ---- barrier (end of phase 1) ---- */
        /* Partition elements around pivot */
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t val = B[pid];
            int64_t piv = pivot[0LL];
            if ((val < piv)) {
                lt_flag[pid] = 1LL;
                gt_flag[pid] = 0LL;
            } else {
                if ((val > piv)) {
                    lt_flag[pid] = 0LL;
                    gt_flag[pid] = 1LL;
                } else {
                    lt_flag[pid] = 0LL;
                    gt_flag[pid] = 0LL;
                }
            }
        }
        /* ---- barrier (end of phase 2) ---- */
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
        /* ---- barrier (end of phase 3) ---- */
        /* Prefix sum (+) weights -> weight_sum */
        weight_sum[0] = weights[0];
        for (int64_t _ps_i = 1; _ps_i < n; _ps_i++) {
            weight_sum[_ps_i] = weight_sum[_ps_i - 1] + weights[_ps_i];
        }
        /* ---- barrier (end of phase 4) ---- */
        /* Determine weighted median partition */
        for (int64_t pid = 0; pid < 1LL; pid++) {
            int64_t total_w = weight_sum[(n - 1LL)];
            int64_t half_w = (total_w / 2LL);
            int64_t lt_w = 0LL;
            for (int64_t i = 0LL; i < n; i += 1LL) {
                if ((lt_flag[i] == 1LL)) {
                    lt_w = (lt_w + weights[i]);
                }
            }
            if ((lt_w >= half_w)) {
                result[0LL] = pivot[0LL];
            }
        }
        /* ---- barrier (end of phase 5) ---- */
    }

    pram_free(A);
    pram_free(weights);
    pram_free(B);
    pram_free(lt_flag);
    pram_free(lt_dest);
    pram_free(gt_flag);
    pram_free(gt_dest);
    pram_free(weight_sum);
    pram_free(pivot);
    pram_free(result);
    return 0;
}
