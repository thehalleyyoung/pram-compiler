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

/* PRAM program: string_sorting | model: CREW */
/* Parallel string sorting (MSD radix). CREW, O(L log n) time, n processors. */
/* Work bound: O(n * L) */
/* Time bound: O(L log n) */

/* Generated C99 code for PRAM program: string_sorting */
/* Memory model: CREW */
/* Parallel string sorting (MSD radix). CREW, O(L log n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t max_len = 0;
    int64_t _num_procs = n;

    int64_t* strings = (int64_t*)pram_calloc((n * max_len), sizeof(int64_t));
    int64_t* order = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* new_order = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* keys = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* bucket_id = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* bucket_offset = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* bucket_count = (int64_t*)pram_calloc(n, sizeof(int64_t));

    /* Phase 1: initialise order */
    for (int64_t pid = 0; pid < n; pid++) {
        /* Tiled access for spatial locality (tile_var=__tile_pid) */
        for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
            order[pid] = pid;
        }
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Phase 2: MSD radix sort over character positions */
    for (int64_t d = 0LL; d < max_len; d += 1LL) {
        /* Phase 2a: extract key at position d */
        for (int64_t pid = 0; pid < n; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                keys[pid] = strings[((order[pid] * max_len) + d)];
            }
        }
        /* ---- barrier (end of phase 1) ---- */
        /* Phase 2b: rank by key via counting */
        for (int64_t pid = 0; pid < n; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                int64_t my_key = keys[pid];
                int64_t cnt = 0LL;
                for (int64_t j = 0LL; j < n; j += 1LL) {
                    if (((keys[j] < my_key) || ((keys[j] == my_key) && (j < pid)))) {
                        cnt = (cnt + 1LL);
                    }
                }
                bucket_offset[pid] = cnt;
            }
        }
        /* ---- barrier (end of phase 2) ---- */
        /* Prefix sum (+) bucket_offset -> bucket_id */
        bucket_id[0] = bucket_offset[0];
        for (int64_t _ps_i = 1; _ps_i < n; _ps_i++) {
            bucket_id[_ps_i] = bucket_id[_ps_i - 1] + bucket_offset[_ps_i];
        }
        /* ---- barrier (end of phase 3) ---- */
        /* Phase 2d: scatter into new_order */
        for (int64_t pid = 0; pid < n; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                new_order[bucket_offset[pid]] = order[pid];
            }
        }
        /* ---- barrier (end of phase 4) ---- */
        /* Phase 2e: copy new_order to order */
        for (int64_t pid = 0; pid < n; pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
                order[pid] = new_order[pid];
            }
        }
        /* ---- barrier (end of phase 5) ---- */
    }

    pram_free(strings);
    pram_free(order);
    pram_free(new_order);
    pram_free(keys);
    pram_free(bucket_id);
    pram_free(bucket_offset);
    pram_free(bucket_count);
    return 0;
}
