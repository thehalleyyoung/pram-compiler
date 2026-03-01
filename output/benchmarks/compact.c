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

/* PRAM program: compact | model: CREW */
/* Parallel compaction. CREW, O(log n) time, n processors. */
/* Work bound: O(n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: compact */
/* Memory model: CREW */
/* Parallel compaction. CREW, O(log n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    int64_t* A = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* flags = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* dest = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* B = (int64_t*)pram_calloc(n, sizeof(int64_t));

    /* Step 1: prefix sum on flags to get destination indices */
    /* Prefix sum (+) flags -> dest */
    dest[0] = flags[0];
    for (int64_t _ps_i = 1; _ps_i < n; _ps_i++) {
        dest[_ps_i] = dest[_ps_i - 1] + flags[_ps_i];
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Step 2: scatter flagged elements to their destinations */
    for (int64_t pid = 0; pid < n; pid++) {
        if ((flags[pid] == 1LL)) {
            int64_t d = (dest[pid] - 1LL);
            B[d] = A[pid];
        }
    }

    pram_free(A);
    pram_free(flags);
    pram_free(dest);
    pram_free(B);
    return 0;
}
