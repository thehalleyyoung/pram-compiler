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

/* PRAM program: strongly_connected | model: CREW */
/* Parallel strongly connected components (forward-backward). CREW, O(log^2 n) time, n+m processors. [small_input_crossover=10000] */
/* Work bound: O((n+m) log^2 n) */
/* Time bound: O(log^2 n) */

/* Generated C99 code for PRAM program: strongly_connected */
/* Memory model: CREW */
/* Parallel strongly connected components (forward-backward). CREW, O(log^2 n) time, n+m processors. [small_input_crossover=10000] */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t m = 0;
    int64_t _num_procs = (n + m);

    int64_t* row_ptr = (int64_t*)pram_calloc((n + 1LL), sizeof(int64_t));
    int64_t* col_idx = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* rev_row_ptr = (int64_t*)pram_calloc((n + 1LL), sizeof(int64_t));
    int64_t* rev_col_idx = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* scc_id = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* fw_reach = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* bw_reach = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* pivot = (int64_t*)pram_calloc(1LL, sizeof(int64_t));
    int64_t* active = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* changed = (int64_t*)pram_calloc(1LL, sizeof(int64_t));

    /* Phase 1: initialise scc_id and active flags */
    for (int64_t pid = 0; pid < n; pid++) {
        /* Tiled access for spatial locality (tile_var=__tile_pid) */
        for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
            scc_id[pid] = (-1LL);
            active[pid] = 1LL;
        }
    }
    /* ---- barrier (end of phase 0) ---- */
    int64_t log_n = (((int64_t)(log2(((double)(n))))) + 2LL);
    /* Phase 2-6: forward-backward reachability loop */
    for (int64_t outer = 0LL; outer < log_n; outer += 1LL) {
        /* Pick pivot: first active vertex */
        pivot[0LL] = (-1LL);
        /* ---- barrier (end of phase 1) ---- */
        for (int64_t pid = 0; pid < n; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                if (((active[pid] == 1LL) && (pivot[0LL] == (-1LL)))) {
                    pivot[0LL] = pid;
                }
            }
        }
        /* ---- barrier (end of phase 2) ---- */
        /* Phase 3: forward BFS from pivot */
        for (int64_t pid = 0; pid < n; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                fw_reach[pid] = 0LL;
            }
        }
        /* ---- barrier (end of phase 3) ---- */
        for (int64_t pid = 0; pid < 1LL; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < 1LL; __tile_pid += 64LL) {
                int64_t pv = pivot[0LL];
                if ((pv >= 0LL)) {
                    fw_reach[pv] = 1LL;
                }
            }
        }
        /* ---- barrier (end of phase 4) ---- */
        for (int64_t bfs_step = 0LL; bfs_step < log_n; bfs_step += 1LL) {
            changed[0LL] = 0LL;
            /* ---- barrier (end of phase 5) ---- */
            for (int64_t pid = 0; pid < n; pid++) {
                /* Tiled access for spatial locality (tile_var=__tile_pid) */
                for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                    if (((fw_reach[pid] == 1LL) && (active[pid] == 1LL))) {
                        int64_t start = row_ptr[pid];
                        int64_t end = row_ptr[(pid + 1LL)];
                        for (int64_t e = start; e < end; e += 1LL) {
                            int64_t nbr = col_idx[e];
                            if (((fw_reach[nbr] == 0LL) && (active[nbr] == 1LL))) {
                                fw_reach[nbr] = 1LL;
                                changed[0LL] = 1LL;
                            }
                        }
                    }
                }
            }
            /* ---- barrier (end of phase 6) ---- */
        }
        /* Phase 4: backward BFS from pivot */
        for (int64_t pid = 0; pid < n; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                bw_reach[pid] = 0LL;
            }
        }
        /* ---- barrier (end of phase 7) ---- */
        for (int64_t pid = 0; pid < 1LL; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < 1LL; __tile_pid += 64LL) {
                int64_t pv = pivot[0LL];
                if ((pv >= 0LL)) {
                    bw_reach[pv] = 1LL;
                }
            }
        }
        /* ---- barrier (end of phase 8) ---- */
        for (int64_t bfs_step = 0LL; bfs_step < log_n; bfs_step += 1LL) {
            changed[0LL] = 0LL;
            /* ---- barrier (end of phase 9) ---- */
            for (int64_t pid = 0; pid < n; pid++) {
                /* Tiled access for spatial locality (tile_var=__tile_pid) */
                for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                    if (((bw_reach[pid] == 1LL) && (active[pid] == 1LL))) {
                        int64_t start = rev_row_ptr[pid];
                        int64_t end = rev_row_ptr[(pid + 1LL)];
                        for (int64_t e = start; e < end; e += 1LL) {
                            int64_t nbr = rev_col_idx[e];
                            if (((bw_reach[nbr] == 0LL) && (active[nbr] == 1LL))) {
                                bw_reach[nbr] = 1LL;
                                changed[0LL] = 1LL;
                            }
                        }
                    }
                }
            }
            /* ---- barrier (end of phase 10) ---- */
        }
        /* Phase 5: assign SCC and deactivate */
        for (int64_t pid = 0; pid < n; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                if (((fw_reach[pid] == 1LL) && (bw_reach[pid] == 1LL))) {
                    scc_id[pid] = pivot[0LL];
                    active[pid] = 0LL;
                }
            }
        }
        /* ---- barrier (end of phase 11) ---- */
    }

    pram_free(row_ptr);
    pram_free(col_idx);
    pram_free(rev_row_ptr);
    pram_free(rev_col_idx);
    pram_free(scc_id);
    pram_free(fw_reach);
    pram_free(bw_reach);
    pram_free(pivot);
    pram_free(active);
    pram_free(changed);
    return 0;
}
