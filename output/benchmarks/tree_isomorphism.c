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

/* PRAM program: tree_isomorphism | model: CREW */
/* Parallel tree isomorphism testing. CREW, O(log^2 n) time, n processors. */
/* Work bound: O(n log^2 n) */
/* Time bound: O(log^2 n) */

/* Generated C99 code for PRAM program: tree_isomorphism */
/* Memory model: CREW */
/* Parallel tree isomorphism testing. CREW, O(log^2 n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    int64_t* parent1 = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* parent2 = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* left_child1 = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* left_child2 = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* right_child1 = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* right_child2 = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* label1 = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* label2 = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* new_label1 = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* new_label2 = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* is_iso = (int64_t*)pram_calloc(1LL, sizeof(int64_t));

    /* Phase 1: initialise labels from degree */
    for (int64_t pid = 0; pid < n; pid++) {
        int64_t deg1 = 0LL;
        if ((left_child1[pid] >= 0LL)) {
            deg1 = (deg1 + 1LL);
        }
        if ((right_child1[pid] >= 0LL)) {
            deg1 = (deg1 + 1LL);
        }
        label1[pid] = deg1;
        int64_t deg2 = 0LL;
        if ((left_child2[pid] >= 0LL)) {
            deg2 = (deg2 + 1LL);
        }
        if ((right_child2[pid] >= 0LL)) {
            deg2 = (deg2 + 1LL);
        }
        label2[pid] = deg2;
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Phase 2: label refinement via hashing */
    int64_t log_n = (((int64_t)(log2(((double)(n))))) + 1LL);
    for (int64_t round = 0LL; round < log_n; round += 1LL) {
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t lc1 = left_child1[pid];
            int64_t rc1 = right_child1[pid];
            int64_t lv1 = label1[pid];
            int64_t lc_lab1 = ((lc1 >= 0LL) ? label1[lc1] : 0LL);
            int64_t rc_lab1 = ((rc1 >= 0LL) ? label1[rc1] : 0LL);
            new_label1[pid] = (((lv1 * 31LL) + lc_lab1) + rc_lab1);
            int64_t lc2 = left_child2[pid];
            int64_t rc2 = right_child2[pid];
            int64_t lv2 = label2[pid];
            int64_t lc_lab2 = ((lc2 >= 0LL) ? label2[lc2] : 0LL);
            int64_t rc_lab2 = ((rc2 >= 0LL) ? label2[rc2] : 0LL);
            new_label2[pid] = (((lv2 * 31LL) + lc_lab2) + rc_lab2);
        }
        /* ---- barrier (end of phase 1) ---- */
        for (int64_t pid = 0; pid < n; pid++) {
            label1[pid] = new_label1[pid];
            label2[pid] = new_label2[pid];
        }
        /* ---- barrier (end of phase 2) ---- */
    }
    /* Phase 3: compare root labels */
    for (int64_t pid = 0; pid < 1LL; pid++) {
        if ((label1[0LL] == label2[0LL])) {
            is_iso[0LL] = 1LL;
        } else {
            is_iso[0LL] = 0LL;
        }
    }
    /* ---- barrier (end of phase 3) ---- */
    /* Phase 4: verify sorted label sequences match */
    for (int64_t pid = 0; pid < n; pid++) {
        int64_t rank1 = 0LL;
        int64_t rank2 = 0LL;
        new_label1[pid] = label1[pid];
        new_label2[pid] = label2[pid];
    }
    /* ---- barrier (end of phase 4) ---- */
    for (int64_t pid = 0; pid < n; pid++) {
        if ((new_label1[pid] != new_label2[pid])) {
            is_iso[0LL] = 0LL;
        }
    }
    /* ---- barrier (end of phase 5) ---- */

    pram_free(parent1);
    pram_free(parent2);
    pram_free(left_child1);
    pram_free(left_child2);
    pram_free(right_child1);
    pram_free(right_child2);
    pram_free(label1);
    pram_free(label2);
    pram_free(new_label1);
    pram_free(new_label2);
    pram_free(is_iso);
    return 0;
}
