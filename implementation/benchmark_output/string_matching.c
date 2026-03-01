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

/* PRAM program: string_matching | model: CREW */
/* Parallel brute-force string matching. CREW, O(log m) time, n*m processors. */
/* Work bound: O(n * m) */
/* Time bound: O(log m) */

/* Generated C99 code for PRAM program: string_matching */
/* Memory model: CREW */
/* Parallel brute-force string matching. CREW, O(log m) time, n*m processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t m = 0;
    int64_t _num_procs = (n * m);

    int64_t* text = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* pattern = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* match_flag = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* match_grid = (int64_t*)pram_calloc((n * m), sizeof(int64_t));

    /* Step 1: each processor (i,j) compares text[i+j] == pattern[j] */
    for (int64_t pid = 0; pid < (n * m); pid++) {
        int64_t i = (pid / m);
        int64_t j = (pid % m);
        int64_t text_pos = (i + j);
        if ((text_pos < n)) {
            if ((text[text_pos] == pattern[j])) {
                match_grid[pid] = 1LL;
            } else {
                match_grid[pid] = 0LL;
            }
        } else {
            match_grid[pid] = 0LL;
        }
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Step 2: parallel AND-reduction along j dimension, O(log m) steps */
    int64_t log_m = ((int64_t)(log2(((double)(m)))));
    for (int64_t d = 0LL; d < log_m; d += 1LL) {
        int64_t stride = (1LL << d);
        for (int64_t pid = 0; pid < (n * m); pid++) {
            int64_t i = (pid / m);
            int64_t j = (pid % m);
            if ((((j % (2LL * stride)) == 0LL) && ((j + stride) < m))) {
                int64_t partner = (pid + stride);
                int64_t a_val = match_grid[pid];
                int64_t b_val = match_grid[partner];
                match_grid[pid] = (a_val & b_val);
            }
        }
        /* ---- barrier (end of phase 1) ---- */
    }
    /* Step 3: write match results */
    for (int64_t pid = 0; pid < ((n - m) + 1LL); pid++) {
        match_flag[pid] = match_grid[(pid * m)];
    }

    pram_free(text);
    pram_free(pattern);
    pram_free(match_flag);
    pram_free(match_grid);
    return 0;
}
