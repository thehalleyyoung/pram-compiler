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

/* PRAM program: point_location | model: CREW */
/* Parallel point location in planar subdivision. CREW, O(log n) time, n processors. */
/* Work bound: O(n log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: point_location */
/* Memory model: CREW */
/* Parallel point location in planar subdivision. CREW, O(log n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t q = 0;
    int64_t _num_procs = (n * q);

    double* edge_x1 = (double*)pram_calloc(n, sizeof(double));
    double* edge_y1 = (double*)pram_calloc(n, sizeof(double));
    double* edge_x2 = (double*)pram_calloc(n, sizeof(double));
    double* edge_y2 = (double*)pram_calloc(n, sizeof(double));
    int64_t* face_id = (int64_t*)pram_calloc(n, sizeof(int64_t));
    double* query_x = (double*)pram_calloc(q, sizeof(double));
    double* query_y = (double*)pram_calloc(q, sizeof(double));
    int64_t* result_face = (int64_t*)pram_calloc(q, sizeof(int64_t));
    int64_t* slab_id = (int64_t*)pram_calloc(q, sizeof(int64_t));
    int64_t* rank_arr = (int64_t*)pram_calloc(n, sizeof(int64_t));

    /* Phase 1: rank edges by x-midpoint */
    for (int64_t pid = 0; pid < n; pid++) {
        double my_mid = ((edge_x1[pid] + edge_x2[pid]) * 0.5);
        int64_t my_rank = 0LL;
        for (int64_t j = 0LL; j < n; j += 1LL) {
            double other_mid = ((edge_x1[j] + edge_x2[j]) * 0.5);
            if (((other_mid < my_mid) || ((other_mid == my_mid) && (j < pid)))) {
                my_rank = (my_rank + 1LL);
            }
        }
        rank_arr[pid] = my_rank;
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Phase 2: binary search to find slab for each query */
    for (int64_t pid = 0; pid < q; pid++) {
        double qx = query_x[pid];
        int64_t lo = 0LL;
        int64_t hi = (n - 1LL);
        int64_t slab = 0LL;
        for (int64_t _step = 0LL; _step < ((int64_t)(log2(((double)(n))))); _step += 1LL) {
            int64_t mid = ((lo + hi) / 2LL);
            double mid_edge_x = ((edge_x1[mid] + edge_x2[mid]) * 0.5);
            if ((mid_edge_x <= qx)) {
                lo = (mid + 1LL);
                slab = mid;
            } else {
                hi = (mid - 1LL);
            }
        }
        slab_id[pid] = slab;
    }
    /* ---- barrier (end of phase 1) ---- */
    /* Phase 3: find face within slab via y-comparison */
    for (int64_t pid = 0; pid < q; pid++) {
        double qy = query_y[pid];
        int64_t my_slab = slab_id[pid];
        int64_t found_face = 0LL;
        for (int64_t e = 0LL; e < n; e += 1LL) {
            if ((rank_arr[e] == my_slab)) {
                double ey1 = edge_y1[e];
                double ey2 = edge_y2[e];
                double avg_ey = ((ey1 + ey2) * 0.5);
                if ((qy >= avg_ey)) {
                    found_face = face_id[e];
                }
            }
        }
        result_face[pid] = found_face;
    }
    /* ---- barrier (end of phase 2) ---- */
    /* Phase 4: results written to result_face */
    /* ---- barrier (end of phase 3) ---- */

    pram_free(edge_x1);
    pram_free(edge_y1);
    pram_free(edge_x2);
    pram_free(edge_y2);
    pram_free(face_id);
    pram_free(query_x);
    pram_free(query_y);
    pram_free(result_face);
    pram_free(slab_id);
    pram_free(rank_arr);
    return 0;
}
