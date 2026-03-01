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

/* PRAM program: flashsort | model: CREW */
/* Distribution-based parallel flashsort. CREW, O(log n) time, n processors. [small_input_crossover=10000] */
/* Work bound: O(n log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: flashsort */
/* Memory model: CREW */
/* Distribution-based parallel flashsort. CREW, O(log n) time, n processors. [small_input_crossover=10000] */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    int64_t* A = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* B = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* class = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* class_count = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* class_offset = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* global_min = (int64_t*)pram_calloc(1LL, sizeof(int64_t));
    int64_t* global_max = (int64_t*)pram_calloc(1LL, sizeof(int64_t));
    int64_t* num_classes = (int64_t*)pram_calloc(1LL, sizeof(int64_t));

    /* Phase 1: parallel reduction to find global min and max */
    for (int64_t pid = 0; pid < n; pid++) {
        /* Tiled access for spatial locality (tile_var=__tile_pid) */
        for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
            B[pid] = A[pid];
        }
    }
    /* ---- barrier (end of phase 0) ---- */
    int64_t log_n = ((int64_t)(log2(((double)(n)))));
    for (int64_t r = 0LL; r < log_n; r += 1LL) {
        int64_t stride = (1LL << (r + 1LL));
        for (int64_t pid = 0; pid < (n / 2LL); pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < (n / 2LL); __tile_pid += 64LL) {
                int64_t i = (pid * stride);
                int64_t j_idx = (i + (stride / 2LL));
                if ((j_idx < n)) {
                    int64_t bi = B[i];
                    int64_t bj = B[j_idx];
                    if ((bi > bj)) {
                        B[i] = bj;
                    }
                }
            }
        }
        /* ---- barrier (end of phase 1) ---- */
    }
    global_min[0LL] = B[0LL];
    /* ---- barrier (end of phase 2) ---- */
    for (int64_t pid = 0; pid < n; pid++) {
        /* Tiled access for spatial locality (tile_var=__tile_pid) */
        for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
            B[pid] = A[pid];
        }
    }
    /* ---- barrier (end of phase 3) ---- */
    for (int64_t r = 0LL; r < log_n; r += 1LL) {
        int64_t stride = (1LL << (r + 1LL));
        for (int64_t pid = 0; pid < (n / 2LL); pid++) {
            /* Tiled access for spatial locality (tile_var=__tile_pid) */
            for (int64_t __tile_pid = 0LL; __tile_pid < (n / 2LL); __tile_pid += 64LL) {
                int64_t i = (pid * stride);
                int64_t j_idx = (i + (stride / 2LL));
                if ((j_idx < n)) {
                    int64_t bi = B[i];
                    int64_t bj = B[j_idx];
                    if ((bi < bj)) {
                        B[i] = bj;
                    }
                }
            }
        }
        /* ---- barrier (end of phase 4) ---- */
    }
    global_max[0LL] = B[0LL];
    /* Phase 2: classify each element into a class */
    num_classes[0LL] = (n / ((int64_t)(log2(((double)(n))))));
    /* ---- barrier (end of phase 5) ---- */
    for (int64_t pid = 0; pid < n; pid++) {
        /* Tiled access for spatial locality (tile_var=__tile_pid) */
        for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
            int64_t range = ((global_max[0LL] - global_min[0LL]) + 1LL);
            int64_t c = (((A[pid] - global_min[0LL]) * num_classes[0LL]) / range);
            int64_t c_clamped = PRAM_MIN(c, (num_classes[0LL] - 1LL));
            class[pid] = c_clamped;
        }
    }
    /* ---- barrier (end of phase 6) ---- */
    /* Phase 3: count elements per class */
    for (int64_t pid = 0; pid < n; pid++) {
        /* Tiled access for spatial locality (tile_var=__tile_pid) */
        for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
            class_count[pid] = 0LL;
        }
        /* --- fused from adjacent parallel phase --- */
        /* Tiled access for spatial locality (tile_var=__tile_pid) */
        for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
            int64_t c = class[pid];
            old_count = class_count[c]; class_count[c] += 1LL;
        }
    }
    /* ---- barrier (end of phase 7) ---- */
    /* Phase 4: prefix sum on class counts */
    /* Prefix sum (+) class_count -> class_offset */
    class_offset[0] = class_count[0];
    for (int64_t _ps_i = 1; _ps_i < num_classes[0LL]; _ps_i++) {
        class_offset[_ps_i] = class_offset[_ps_i - 1] + class_count[_ps_i];
    }
    /* Phase 5: scatter elements to sorted positions in B */
    for (int64_t pid = 0; pid < n; pid++) {
        /* Tiled access for spatial locality (tile_var=__tile_pid) */
        for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
            int64_t c = class[pid];
            int64_t off = class_offset[c];
            B[off] = A[pid];
        }
    }
    /* ---- barrier (end of phase 8) ---- */
    /* Phase 6: insertion sort within each bucket */
    for (int64_t pid = 0; pid < num_classes[0LL]; pid++) {
        /* Tiled access for spatial locality (tile_var=__tile_pid) */
        for (int64_t __tile_pid = 0LL; __tile_pid < num_classes[0LL]; __tile_pid += 64LL) {
            int64_t bucket_start = ((pid == 0LL) ? 0LL : class_offset[(pid - 1LL)]);
            int64_t bucket_end = class_offset[pid];
            for (int64_t i_outer = (bucket_start + 1LL); i_outer < bucket_end; i_outer += 1LL) {
                int64_t key = B[i_outer];
                int64_t j_inner = (i_outer - 1LL);
                while (((j_inner >= bucket_start) && (B[j_inner] > key))) {
                    B[(j_inner + 1LL)] = B[j_inner];
                    j_inner = (j_inner - 1LL);
                }
                B[(j_inner + 1LL)] = key;
            }
        }
    }
    /* ---- barrier (end of phase 9) ---- */
    /* Phase 7: copy sorted result back to A */
    for (int64_t pid = 0; pid < n; pid++) {
        /* Tiled access for spatial locality (tile_var=__tile_pid) */
        for (int64_t __tile_pid = 0LL; __tile_pid < n; __tile_pid += 64LL) {
            A[pid] = B[pid];
        }
    }

    pram_free(A);
    pram_free(B);
    pram_free(class);
    pram_free(class_count);
    pram_free(class_offset);
    pram_free(global_min);
    pram_free(global_max);
    pram_free(num_classes);
    return 0;
}
