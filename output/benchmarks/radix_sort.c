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

/* PRAM program: radix_sort | model: EREW */
/* Parallel radix sort via prefix sums. EREW, O(b log n) time, n processors. */
/* Work bound: O(b * n) */
/* Time bound: O(b log n) */

/* Generated C99 code for PRAM program: radix_sort */
/* Memory model: EREW */
/* Parallel radix sort via prefix sums. EREW, O(b log n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t b = 0;
    int64_t _num_procs = n;

    int64_t* A = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* B = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* bit_flags = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* zero_flags = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* one_flags = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* zero_dest = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* one_dest = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* zero_count = (int64_t*)pram_calloc(1LL, sizeof(int64_t));

    for (int64_t d = 0LL; d < b; d += 1LL) {
        /* Phase 1: extract bit d from each element */
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t bit_val = ((A[pid] >> d) & 1LL);
            bit_flags[pid] = bit_val;
            zero_flags[pid] = (1LL - bit_val);
            one_flags[pid] = bit_val;
        }
        /* ---- barrier (end of phase 0) ---- */
        /* Phase 2: prefix sum on zero and one flags */
        /* Prefix sum (+) zero_flags -> zero_dest */
        zero_dest[0] = zero_flags[0];
        for (int64_t _ps_i = 1; _ps_i < n; _ps_i++) {
            zero_dest[_ps_i] = zero_dest[_ps_i - 1] + zero_flags[_ps_i];
        }
        /* Prefix sum (+) one_flags -> one_dest */
        one_dest[0] = one_flags[0];
        for (int64_t _ps_i = 1; _ps_i < n; _ps_i++) {
            one_dest[_ps_i] = one_dest[_ps_i - 1] + one_flags[_ps_i];
        }
        /* ---- barrier (end of phase 1) ---- */
        /* Phase 3: get total zero count */
        zero_count[0LL] = zero_dest[(n - 1LL)];
        /* ---- barrier (end of phase 2) ---- */
        /* Phase 4: scatter to sorted positions */
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t bit_val = bit_flags[pid];
            if ((bit_val == 0LL)) {
                int64_t dest = (zero_dest[pid] - 1LL);
                B[dest] = A[pid];
            } else {
                int64_t dest = (zero_count[0LL] + (one_dest[pid] - 1LL));
                B[dest] = A[pid];
            }
        }
        /* ---- barrier (end of phase 3) ---- */
        /* Phase 5: copy B back to A */
        for (int64_t pid = 0; pid < n; pid++) {
            A[pid] = B[pid];
        }
        /* ---- barrier (end of phase 4) ---- */
    }

    pram_free(A);
    pram_free(B);
    pram_free(bit_flags);
    pram_free(zero_flags);
    pram_free(one_flags);
    pram_free(zero_dest);
    pram_free(one_dest);
    pram_free(zero_count);
    return 0;
}
