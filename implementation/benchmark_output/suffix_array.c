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

/* PRAM program: suffix_array | model: CREW */
/* Parallel suffix array (prefix-doubling). CREW, O(log^2 n) time, n processors. */
/* Work bound: O(n log^2 n) */
/* Time bound: O(log^2 n) */

/* Generated C99 code for PRAM program: suffix_array */
/* Memory model: CREW */
/* Parallel suffix array (prefix-doubling). CREW, O(log^2 n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    int64_t* text = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* sa = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* rank_arr = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* new_rank = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* key1 = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* key2 = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* temp = (int64_t*)pram_calloc(n, sizeof(int64_t));

    /* Initialise: rank[i] = text[i], sa[i] = i */
    for (int64_t pid = 0; pid < n; pid++) {
        rank_arr[pid] = text[pid];
        sa[pid] = pid;
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Prefix-doubling: O(log n) rounds */
    int64_t log_n = (((int64_t)(log2(((double)(n))))) + 1LL);
    for (int64_t round = 0LL; round < log_n; round += 1LL) {
        int64_t offset = (1LL << round);
        /* Build (key1, key2) pairs for sorting */
        for (int64_t pid = 0; pid < n; pid++) {
            key1[pid] = rank_arr[pid];
            if (((pid + offset) < n)) {
                key2[pid] = rank_arr[(pid + offset)];
            } else {
                key2[pid] = (-1LL);
            }
        }
        /* ---- barrier (end of phase 1) ---- */
        /* Rank suffixes by (key1, key2) pairs */
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t my_k1 = key1[pid];
            int64_t my_k2 = key2[pid];
            int64_t my_new_rank = 0LL;
            for (int64_t j = 0LL; j < n; j += 1LL) {
                int64_t other_k1 = key1[j];
                int64_t other_k2 = key2[j];
                if (((other_k1 < my_k1) || ((other_k1 == my_k1) && (other_k2 < my_k2)))) {
                    my_new_rank = (my_new_rank + 1LL);
                }
                if (((other_k1 == my_k1) && ((other_k2 == my_k2) && (j < pid)))) {
                    my_new_rank = (my_new_rank + 1LL);
                }
            }
            new_rank[pid] = my_new_rank;
            sa[my_new_rank] = pid;
        }
        /* ---- barrier (end of phase 2) ---- */
        for (int64_t pid = 0; pid < n; pid++) {
            rank_arr[pid] = new_rank[pid];
        }
        /* ---- barrier (end of phase 3) ---- */
    }

    pram_free(text);
    pram_free(sa);
    pram_free(rank_arr);
    pram_free(new_rank);
    pram_free(key1);
    pram_free(key2);
    pram_free(temp);
    return 0;
}
