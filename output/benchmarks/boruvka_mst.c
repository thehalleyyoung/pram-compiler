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

/* PRAM program: boruvka_mst | model: CRCW-Priority */
/* Borůvka's MST. CRCW-Priority, O(log n) phases, m processors per phase. */
/* Work bound: O(m log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: boruvka_mst */
/* Memory model: CRCW-Priority */
/* Borůvka's MST. CRCW-Priority, O(log n) phases, m processors per phase. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t m = 0;
    int64_t _num_procs = m;

    int64_t* edge_src = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* _wpid_edge_src = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* _stg_edge_src = (int64_t*)pram_calloc(m, sizeof(int64_t));
    for (int64_t _i = 0; _i < m; _i++) { _wpid_edge_src[_i] = INT64_MAX; }
    int64_t* edge_dst = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* _wpid_edge_dst = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* _stg_edge_dst = (int64_t*)pram_calloc(m, sizeof(int64_t));
    for (int64_t _i = 0; _i < m; _i++) { _wpid_edge_dst[_i] = INT64_MAX; }
    int64_t* edge_w = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* _wpid_edge_w = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* _stg_edge_w = (int64_t*)pram_calloc(m, sizeof(int64_t));
    for (int64_t _i = 0; _i < m; _i++) { _wpid_edge_w[_i] = INT64_MAX; }
    int64_t* comp = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* _wpid_comp = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* _stg_comp = (int64_t*)pram_calloc(n, sizeof(int64_t));
    for (int64_t _i = 0; _i < n; _i++) { _wpid_comp[_i] = INT64_MAX; }
    int64_t* min_edge = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* _wpid_min_edge = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* _stg_min_edge = (int64_t*)pram_calloc(n, sizeof(int64_t));
    for (int64_t _i = 0; _i < n; _i++) { _wpid_min_edge[_i] = INT64_MAX; }
    int64_t* min_w = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* _wpid_min_w = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* _stg_min_w = (int64_t*)pram_calloc(n, sizeof(int64_t));
    for (int64_t _i = 0; _i < n; _i++) { _wpid_min_w[_i] = INT64_MAX; }
    int64_t* mst_flag = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* _wpid_mst_flag = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* _stg_mst_flag = (int64_t*)pram_calloc(m, sizeof(int64_t));
    for (int64_t _i = 0; _i < m; _i++) { _wpid_mst_flag[_i] = INT64_MAX; }
    int64_t* active = (int64_t*)pram_calloc(1LL, sizeof(int64_t));
    int64_t* _wpid_active = (int64_t*)pram_calloc(1LL, sizeof(int64_t));
    int64_t* _stg_active = (int64_t*)pram_calloc(1LL, sizeof(int64_t));
    for (int64_t _i = 0; _i < 1LL; _i++) { _wpid_active[_i] = INT64_MAX; }

    /* Initialise: comp[i] = i */
    for (int64_t pid = 0; pid < n; pid++) {
        /* Tiled access for spatial locality (tile_var=__tile_pid) */
        for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
            {
                int64_t __buf_pid_0 = pid;
                int64_t __buf_idx_pid_0 = pid;
                if ((__buf_idx_pid_0 >= 0LL)) {
                    if (pid < _wpid_comp[__buf_idx_pid_0]) {
                        _wpid_comp[__buf_idx_pid_0] = pid;
                        _stg_comp[__buf_idx_pid_0] = __buf_pid_0;
                    }
                }
            }
        }
    }
    /* ---- barrier (end of phase 0) ---- */
    int64_t max_phases = (((int64_t)(log2(((double)(n))))) + 1LL);
    for (int64_t phase = 0LL; phase < max_phases; phase += 1LL) {
        /* Reset min_w to MAX */
        for (int64_t pid = 0; pid < n; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                {
                    int64_t __buf_pid_1 = 9223372036854775807LL;
                    int64_t __buf_idx_pid_1 = pid;
                    if ((__buf_idx_pid_1 >= 0LL)) {
                        if (pid < _wpid_min_w[__buf_idx_pid_1]) {
                            _wpid_min_w[__buf_idx_pid_1] = pid;
                            _stg_min_w[__buf_idx_pid_1] = __buf_pid_1;
                        }
                    }
                }
                {
                    int64_t __buf_pid_0 = (-1LL);
                    int64_t __buf_idx_pid_0 = pid;
                    if ((__buf_idx_pid_0 >= 0LL)) {
                        if (pid < _wpid_min_edge[__buf_idx_pid_0]) {
                            _wpid_min_edge[__buf_idx_pid_0] = pid;
                            _stg_min_edge[__buf_idx_pid_0] = __buf_pid_0;
                        }
                    }
                }
            }
        }
        /* ---- barrier (end of phase 1) ---- */
        /* Find min-weight edge per component (priority write) */
        for (int64_t pid = 0; pid < m; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < m; __tile_pid += 64LL) {
                int64_t u = edge_src[pid];
                int64_t v_node = edge_dst[pid];
                int64_t cu = comp[u];
                int64_t cv = comp[v_node];
                if ((cu != cv)) {
                    int64_t w = edge_w[pid];
                    {
                        int64_t __buf_pid_3 = w;
                        int64_t __buf_idx_pid_3 = cu;
                        if ((__buf_idx_pid_3 >= 0LL)) {
                            if (pid < _wpid_min_w[__buf_idx_pid_3]) {
                                _wpid_min_w[__buf_idx_pid_3] = pid;
                                _stg_min_w[__buf_idx_pid_3] = __buf_pid_3;
                            }
                        }
                    }
                    {
                        int64_t __buf_pid_2 = pid;
                        int64_t __buf_idx_pid_2 = cu;
                        if ((__buf_idx_pid_2 >= 0LL)) {
                            if (pid < _wpid_min_edge[__buf_idx_pid_2]) {
                                _wpid_min_edge[__buf_idx_pid_2] = pid;
                                _stg_min_edge[__buf_idx_pid_2] = __buf_pid_2;
                            }
                        }
                    }
                    {
                        int64_t __buf_pid_1 = w;
                        int64_t __buf_idx_pid_1 = cv;
                        if ((__buf_idx_pid_1 >= 0LL)) {
                            if (pid < _wpid_min_w[__buf_idx_pid_1]) {
                                _wpid_min_w[__buf_idx_pid_1] = pid;
                                _stg_min_w[__buf_idx_pid_1] = __buf_pid_1;
                            }
                        }
                    }
                    {
                        int64_t __buf_pid_0 = pid;
                        int64_t __buf_idx_pid_0 = cv;
                        if ((__buf_idx_pid_0 >= 0LL)) {
                            if (pid < _wpid_min_edge[__buf_idx_pid_0]) {
                                _wpid_min_edge[__buf_idx_pid_0] = pid;
                                _stg_min_edge[__buf_idx_pid_0] = __buf_pid_0;
                            }
                        }
                    }
                }
            }
        }
        /* ---- barrier (end of phase 2) ---- */
        /* Mark MST edges and hook components */
        for (int64_t pid = 0; pid < n; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                int64_t me = min_edge[pid];
                if ((me >= 0LL)) {
                    {
                        int64_t __buf_pid_0 = 1LL;
                        int64_t __buf_idx_pid_0 = me;
                        if ((__buf_idx_pid_0 >= 0LL)) {
                            if (pid < _wpid_mst_flag[__buf_idx_pid_0]) {
                                _wpid_mst_flag[__buf_idx_pid_0] = pid;
                                _stg_mst_flag[__buf_idx_pid_0] = __buf_pid_0;
                            }
                        }
                    }
                    int64_t u = edge_src[me];
                    int64_t v_node = edge_dst[me];
                    int64_t cu = comp[u];
                    int64_t cv = comp[v_node];
                    if ((cu < cv)) {
                        {
                            int64_t __buf_pid_1 = cu;
                            int64_t __buf_idx_pid_1 = cv;
                            if ((__buf_idx_pid_1 >= 0LL)) {
                                if (pid < _wpid_comp[__buf_idx_pid_1]) {
                                    _wpid_comp[__buf_idx_pid_1] = pid;
                                    _stg_comp[__buf_idx_pid_1] = __buf_pid_1;
                                }
                            }
                        }
                    } else {
                        {
                            int64_t __buf_pid_2 = cv;
                            int64_t __buf_idx_pid_2 = cu;
                            if ((__buf_idx_pid_2 >= 0LL)) {
                                if (pid < _wpid_comp[__buf_idx_pid_2]) {
                                    _wpid_comp[__buf_idx_pid_2] = pid;
                                    _stg_comp[__buf_idx_pid_2] = __buf_pid_2;
                                }
                            }
                        }
                    }
                }
            }
        }
        /* ---- barrier (end of phase 3) ---- */
        /* Pointer-jump to flatten component ids */
        for (int64_t pid = 0; pid < n; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                int64_t c = comp[pid];
                int64_t cc = comp[c];
                while ((c != cc)) {
                    c = cc;
                    cc = comp[c];
                }
                {
                    int64_t __buf_pid_0 = c;
                    int64_t __buf_idx_pid_0 = pid;
                    if ((__buf_idx_pid_0 >= 0LL)) {
                        if (pid < _wpid_comp[__buf_idx_pid_0]) {
                            _wpid_comp[__buf_idx_pid_0] = pid;
                            _stg_comp[__buf_idx_pid_0] = __buf_pid_0;
                        }
                    }
                }
            }
        }
        /* ---- barrier (end of phase 4) ---- */
    }

    pram_free(edge_src);
    pram_free(_wpid_edge_src);
    pram_free(_stg_edge_src);
    pram_free(edge_dst);
    pram_free(_wpid_edge_dst);
    pram_free(_stg_edge_dst);
    pram_free(edge_w);
    pram_free(_wpid_edge_w);
    pram_free(_stg_edge_w);
    pram_free(comp);
    pram_free(_wpid_comp);
    pram_free(_stg_comp);
    pram_free(min_edge);
    pram_free(_wpid_min_edge);
    pram_free(_stg_min_edge);
    pram_free(min_w);
    pram_free(_wpid_min_w);
    pram_free(_stg_min_w);
    pram_free(mst_flag);
    pram_free(_wpid_mst_flag);
    pram_free(_stg_mst_flag);
    pram_free(active);
    pram_free(_wpid_active);
    pram_free(_stg_active);
    return 0;
}
