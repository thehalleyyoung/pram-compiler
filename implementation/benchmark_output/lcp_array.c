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

/* PRAM program: lcp_array | model: CREW */
/* Parallel LCP array construction. CREW, O(log^2 n) time, n processors. */
/* Work bound: O(n log^2 n) */
/* Time bound: O(log^2 n) */

/* Generated C99 code for PRAM program: lcp_array */
/* Memory model: CREW */
/* Parallel LCP array construction. CREW, O(log^2 n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    int64_t* text = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* sa = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* rank_arr = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* lcp = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* temp = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* inv_sa = (int64_t*)pram_calloc(n, sizeof(int64_t));

    /* Phase 1: inv_sa[sa[i]] = i */
    for (int64_t pid = 0; pid < n; pid++) {
        inv_sa[sa[pid]] = pid;
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Phase 2: lcp[0] = 0 */
    lcp[0LL] = 0LL;
    /* ---- barrier (end of phase 1) ---- */
    /* Phase 3: Kasai-like parallel LCP computation */
    for (int64_t pid = 0; pid < n; pid++) {
        int64_t rank_i = inv_sa[pid];
        if ((rank_i > 0LL)) {
            int64_t j = sa[(rank_i - 1LL)];
            int64_t match_len = 0LL;
            for (int64_t c = 0LL; c < n; c += 1LL) {
                int64_t pos_a = (pid + match_len);
                int64_t pos_b = (j + match_len);
                if (((pos_a < n) && ((pos_b < n) && (text[pos_a] == text[pos_b])))) {
                    match_len = (match_len + 1LL);
                }
            }
            lcp[rank_i] = match_len;
        }
    }
    /* ---- barrier (end of phase 2) ---- */
    /* Phase 4: propagate minimum LCP information */
    int64_t log_n = (((int64_t)(log2(((double)(n))))) + 1LL);
    for (int64_t d = 0LL; d < log_n; d += 1LL) {
        int64_t stride = (1LL << d);
        for (int64_t pid = 0; pid < n; pid++) {
            if (((pid >= stride) && (pid < n))) {
                int64_t a_val = lcp[pid];
                int64_t b_val = lcp[(pid - stride)];
                temp[pid] = ((a_val < b_val) ? a_val : b_val);
            }
        }
        /* ---- barrier (end of phase 3) ---- */
    }

    pram_free(text);
    pram_free(sa);
    pram_free(rank_arr);
    pram_free(lcp);
    pram_free(temp);
    pram_free(inv_sa);
    return 0;
}
