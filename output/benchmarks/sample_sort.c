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

/* PRAM program: sample_sort | model: CREW */
/* Parallel sample sort. n/log n processors, O(log n) time, CREW. */
/* Work bound: O(n log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: sample_sort */
/* Memory model: CREW */
/* Parallel sample sort. n/log n processors, O(log n) time, CREW. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = (n / log2(((double)(n))));

    int64_t* A = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* B = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* samples = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* bucket_id = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* bucket_count = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* bucket_offset = (int64_t*)pram_calloc(n, sizeof(int64_t));

    /* Step 1: local sort of blocks of size log n */
    for (int64_t pid = 0; pid < (n / log2(((double)(n)))); pid++) {
        int64_t start = (pid * ((int64_t)(log2(((double)(n))))));
        int64_t end = PRAM_MIN((start + ((int64_t)(log2(((double)(n)))))), n);
        for (int64_t i_outer = (start + 1LL); i_outer < end; i_outer += 1LL) {
            int64_t key = A[i_outer];
            int64_t j_inner = (i_outer - 1LL);
            while (((j_inner >= start) && (A[j_inner] > key))) {
                A[(j_inner + 1LL)] = A[j_inner];
                j_inner = (j_inner - 1LL);
            }
            A[(j_inner + 1LL)] = key;
        }
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Step 2: pick p-1 splitters from sorted blocks */
    for (int64_t pid = 0; pid < ((n / log2(((double)(n)))) - 1LL); pid++) {
        int64_t sample_idx = (((pid + 1LL) * ((int64_t)(log2(((double)(n)))))) - 1LL);
        if ((sample_idx < n)) {
            samples[pid] = A[sample_idx];
        }
    }
    /* ---- barrier (end of phase 1) ---- */
    /* Step 3: sort splitters sequentially (p-1 elements) */
    for (int64_t pid = 0; pid < 1LL; pid++) {
        int64_t num_samples = ((n / log2(((double)(n)))) - 1LL);
        for (int64_t si = 1LL; si < num_samples; si += 1LL) {
            int64_t skey = samples[si];
            int64_t sj = (si - 1LL);
            while (((sj >= 0LL) && (samples[sj] > skey))) {
                samples[(sj + 1LL)] = samples[sj];
                sj = (sj - 1LL);
            }
            samples[(sj + 1LL)] = skey;
        }
    }
    /* ---- barrier (end of phase 2) ---- */
    /* Step 4: assign bucket ids via binary search on splitters */
    for (int64_t pid = 0; pid < n; pid++) {
        int64_t val = A[pid];
        int64_t lo = 0LL;
        int64_t hi = ((n / log2(((double)(n)))) - 2LL);
        int64_t bucket = ((n / log2(((double)(n)))) - 1LL);
        while ((lo <= hi)) {
            int64_t mid = ((lo + hi) / 2LL);
            if ((val <= samples[mid])) {
                bucket = mid;
                hi = (mid - 1LL);
            } else {
                lo = (mid + 1LL);
            }
        }
        bucket_id[pid] = bucket;
    }
    /* ---- barrier (end of phase 3) ---- */
    /* Step 5: prefix-sum bucket counts and scatter into B */
    /* Prefix sum (+) bucket_count -> bucket_offset */
    bucket_offset[0] = bucket_count[0];
    for (int64_t _ps_i = 1; _ps_i < (n / log2(((double)(n)))); _ps_i++) {
        bucket_offset[_ps_i] = bucket_offset[_ps_i - 1] + bucket_count[_ps_i];
    }
    /* ---- barrier (end of phase 4) ---- */
    for (int64_t pid = 0; pid < n; pid++) {
        int64_t b = bucket_id[pid];
        int64_t off = bucket_offset[b];
        B[off] = A[pid];
    }
    /* ---- barrier (end of phase 5) ---- */
    /* Step 6: copy back */
    for (int64_t pid = 0; pid < n; pid++) {
        A[pid] = B[pid];
    }

    pram_free(A);
    pram_free(B);
    pram_free(samples);
    pram_free(bucket_id);
    pram_free(bucket_count);
    pram_free(bucket_offset);
    return 0;
}
