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

/* PRAM program: maximal_independent_set | model: CRCW-Arbitrary */
/* Luby's parallel MIS algorithm. CRCW-Arbitrary, O(log n) expected time, n+m processors. */
/* Work bound: O((n+m) log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: maximal_independent_set */
/* Memory model: CRCW-Arbitrary */
/* Luby's parallel MIS algorithm. CRCW-Arbitrary, O(log n) expected time, n+m processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t m = 0;
    int64_t _num_procs = (n + m);

    int64_t* row_ptr = (int64_t*)pram_calloc((n + 1LL), sizeof(int64_t));
    int64_t* col_idx = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* in_mis = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* removed = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* priority = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* is_local_max = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* changed = (int64_t*)pram_calloc(1LL, sizeof(int64_t));

    /* Initialise in_mis=0, removed=0 */
    for (int64_t pid = 0; pid < n; pid++) {
        /* Tiled access for spatial locality (tile_var=__tile_pid) */
        for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
            {
                int64_t __buf_pid_1 = 0LL;
                int64_t __buf_idx_pid_1 = pid;
                if ((__buf_idx_pid_1 >= 0LL)) {
                    in_mis[__buf_idx_pid_1] = __buf_pid_1;
                }
            }
            {
                int64_t __buf_pid_0 = 0LL;
                int64_t __buf_idx_pid_0 = pid;
                if ((__buf_idx_pid_0 >= 0LL)) {
                    removed[__buf_idx_pid_0] = __buf_pid_0;
                }
            }
        }
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Luby's MIS: O(log n) rounds */
    int64_t max_rounds = (((int64_t)(log2(((double)(n))))) + 2LL);
    for (int64_t round = 0LL; round < max_rounds; round += 1LL) {
        /* Phase 1: assign random priorities via hash */
        for (int64_t pid = 0; pid < n; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                if ((removed[pid] == 0LL)) {
                    {
                        int64_t __buf_pid_0 = hash((pid + (round * n)));
                        int64_t __buf_idx_pid_0 = pid;
                        if ((__buf_idx_pid_0 >= 0LL)) {
                            priority[__buf_idx_pid_0] = __buf_pid_0;
                        }
                    }
                }
            }
        }
        /* ---- barrier (end of phase 1) ---- */
        /* Phase 2: check if vertex has max priority among non-removed neighbors */
        for (int64_t pid = 0; pid < n; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                {
                    int64_t __buf_pid_0 = 0LL;
                    int64_t __buf_idx_pid_0 = pid;
                    if ((__buf_idx_pid_0 >= 0LL)) {
                        is_local_max[__buf_idx_pid_0] = __buf_pid_0;
                    }
                }
                if ((removed[pid] == 0LL)) {
                    int64_t my_pri = priority[pid];
                    int64_t is_max = 1LL;
                    int64_t row_start = row_ptr[pid];
                    int64_t row_end = row_ptr[(pid + 1LL)];
                    for (int64_t e = row_start; e < row_end; e += 1LL) {
                        int64_t neighbor = col_idx[e];
                        if ((removed[neighbor] == 0LL)) {
                            int64_t npri = priority[neighbor];
                            if ((npri >= my_pri)) {
                                if (((npri > my_pri) || (neighbor > pid))) {
                                    is_max = 0LL;
                                }
                            }
                        }
                    }
                    {
                        int64_t __buf_pid_0 = is_max;
                        int64_t __buf_idx_pid_0 = pid;
                        if ((__buf_idx_pid_0 >= 0LL)) {
                            is_local_max[__buf_idx_pid_0] = __buf_pid_0;
                        }
                    }
                }
            }
        }
        /* ---- barrier (end of phase 2) ---- */
        /* Phase 3: add local maxima to MIS */
        for (int64_t pid = 0; pid < n; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                if ((is_local_max[pid] == 1LL)) {
                    {
                        int64_t __buf_pid_0 = 1LL;
                        int64_t __buf_idx_pid_0 = pid;
                        if ((__buf_idx_pid_0 >= 0LL)) {
                            in_mis[__buf_idx_pid_0] = __buf_pid_0;
                        }
                    }
                }
            }
        }
        /* ---- barrier (end of phase 3) ---- */
        /* Phase 4: remove MIS vertices and neighbors */
        for (int64_t pid = 0; pid < n; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                if ((removed[pid] == 0LL)) {
                    if ((in_mis[pid] == 1LL)) {
                        {
                            int64_t __buf_pid_0 = 1LL;
                            int64_t __buf_idx_pid_0 = pid;
                            if ((__buf_idx_pid_0 >= 0LL)) {
                                removed[__buf_idx_pid_0] = __buf_pid_0;
                            }
                        }
                    }
                    int64_t row_start = row_ptr[pid];
                    int64_t row_end = row_ptr[(pid + 1LL)];
                    for (int64_t e = row_start; e < row_end; e += 1LL) {
                        int64_t neighbor = col_idx[e];
                        if ((in_mis[neighbor] == 1LL)) {
                            {
                                int64_t __buf_pid_1 = 1LL;
                                int64_t __buf_idx_pid_1 = pid;
                                if ((__buf_idx_pid_1 >= 0LL)) {
                                    removed[__buf_idx_pid_1] = __buf_pid_1;
                                }
                            }
                        }
                    }
                }
            }
        }
        /* ---- barrier (end of phase 4) ---- */
    }

    pram_free(row_ptr);
    pram_free(col_idx);
    pram_free(in_mis);
    pram_free(removed);
    pram_free(priority);
    pram_free(is_local_max);
    pram_free(changed);
    return 0;
}
