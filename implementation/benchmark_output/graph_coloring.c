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

/* PRAM program: graph_coloring | model: CRCW-Common */
/* Parallel graph coloring (greedy). CRCW-Common, O(Delta log n) time, n+m processors. */
/* Work bound: O((n+m) Delta) */
/* Time bound: O(Delta log n) */

/* Generated C99 code for PRAM program: graph_coloring */
/* Memory model: CRCW-Common */
/* Parallel graph coloring (greedy). CRCW-Common, O(Delta log n) time, n+m processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t m = 0;
    int64_t _num_procs = (n + m);

    int64_t* row_ptr = (int64_t*)pram_calloc((n + 1LL), sizeof(int64_t));
    int64_t* col_idx = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* color = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* colored = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* available = (int64_t*)pram_calloc((n * n), sizeof(int64_t));
    int64_t* remaining = (int64_t*)pram_calloc(1LL, sizeof(int64_t));

    /* Initialise color=-1, colored=0, remaining=n */
    for (int64_t pid = 0; pid < n; pid++) {
        /* Tiled access for spatial locality (tile_var=__tile_pid) */
        for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
            {
                int64_t __buf_pid_1 = (-1LL);
                int64_t __buf_idx_pid_1 = pid;
                if ((__buf_idx_pid_1 >= 0LL)) {
                    color[__buf_idx_pid_1] = __buf_pid_1;
                }
            }
            {
                int64_t __buf_pid_0 = 0LL;
                int64_t __buf_idx_pid_0 = pid;
                if ((__buf_idx_pid_0 >= 0LL)) {
                    colored[__buf_idx_pid_0] = __buf_pid_0;
                }
            }
        }
    }
    remaining[0LL] = n;
    /* ---- barrier (end of phase 0) ---- */
    /* Coloring rounds: scan neighbors, pick min available color */
    for (int64_t round = 0LL; round < n; round += 1LL) {
        /* Phase 1: mark unavailable colors */
        for (int64_t pid = 0; pid < n; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                if ((colored[pid] == 0LL)) {
                    for (int64_t c = 0LL; c < n; c += 1LL) {
                        {
                            int64_t __buf_pid_0 = 1LL;
                            int64_t __buf_idx_pid_0 = ((pid * n) + c);
                            if ((__buf_idx_pid_0 >= 0LL)) {
                                available[__buf_idx_pid_0] = __buf_pid_0;
                            }
                        }
                    }
                    int64_t row_start = row_ptr[pid];
                    int64_t row_end = row_ptr[(pid + 1LL)];
                    for (int64_t e = row_start; e < row_end; e += 1LL) {
                        int64_t neighbor = col_idx[e];
                        int64_t nc = color[neighbor];
                        if ((nc >= 0LL)) {
                            {
                                int64_t __buf_pid_1 = 0LL;
                                int64_t __buf_idx_pid_1 = ((pid * n) + nc);
                                if ((__buf_idx_pid_1 >= 0LL)) {
                                    available[__buf_idx_pid_1] = __buf_pid_1;
                                }
                            }
                        }
                    }
                }
            }
        }
        /* ---- barrier (end of phase 1) ---- */
        /* Phase 2: assign minimum available color */
        for (int64_t pid = 0; pid < n; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                if ((colored[pid] == 0LL)) {
                    int64_t chosen = (-1LL);
                    for (int64_t c = 0LL; c < n; c += 1LL) {
                        if (((chosen == (-1LL)) && (available[((pid * n) + c)] == 1LL))) {
                            chosen = c;
                        }
                    }
                    if ((chosen >= 0LL)) {
                        {
                            int64_t __buf_pid_1 = chosen;
                            int64_t __buf_idx_pid_1 = pid;
                            if ((__buf_idx_pid_1 >= 0LL)) {
                                color[__buf_idx_pid_1] = __buf_pid_1;
                            }
                        }
                        {
                            int64_t __buf_pid_0 = 1LL;
                            int64_t __buf_idx_pid_0 = pid;
                            if ((__buf_idx_pid_0 >= 0LL)) {
                                colored[__buf_idx_pid_0] = __buf_pid_0;
                            }
                        }
                    }
                }
            }
        }
        /* ---- barrier (end of phase 2) ---- */
        /* Phase 3: update remaining count */
        for (int64_t pid = 0; pid < n; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                if (((colored[pid] == 1LL) && (color[pid] >= 0LL))) {
                }
            }
        }
        /* ---- barrier (end of phase 3) ---- */
    }

    pram_free(row_ptr);
    pram_free(col_idx);
    pram_free(color);
    pram_free(colored);
    pram_free(available);
    pram_free(remaining);
    return 0;
}
