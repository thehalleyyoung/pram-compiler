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

/* PRAM program: line_segment_intersection | model: CREW */
/* Parallel line segment intersection detection. CREW, O(log n) time, n^2 processors. */
/* Work bound: O(n^2) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: line_segment_intersection */
/* Memory model: CREW */
/* Parallel line segment intersection detection. CREW, O(log n) time, n^2 processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = (n * n);

    double* seg_x1 = (double*)pram_calloc(n, sizeof(double));
    double* seg_y1 = (double*)pram_calloc(n, sizeof(double));
    double* seg_x2 = (double*)pram_calloc(n, sizeof(double));
    double* seg_y2 = (double*)pram_calloc(n, sizeof(double));
    int64_t* intersects = (int64_t*)pram_calloc((n * n), sizeof(int64_t));
    int64_t* intersection_count = (int64_t*)pram_calloc(1LL, sizeof(int64_t));

    /* Phase 1: all-pairs intersection test */
    for (int64_t pid = 0; pid < (n * n); pid++) {
        int64_t i = (pid / n);
        int64_t j = (pid % n);
        if ((i < j)) {
            double ax = (seg_x2[i] - seg_x1[i]);
            double ay = (seg_y2[i] - seg_y1[i]);
            double d1 = ((ax * (seg_y1[j] - seg_y1[i])) - (ay * (seg_x1[j] - seg_x1[i])));
            double d2 = ((ax * (seg_y2[j] - seg_y1[i])) - (ay * (seg_x2[j] - seg_x1[i])));
            double bx = (seg_x2[j] - seg_x1[j]);
            double by = (seg_y2[j] - seg_y1[j]);
            double d3 = ((bx * (seg_y1[i] - seg_y1[j])) - (by * (seg_x1[i] - seg_x1[j])));
            double d4 = ((bx * (seg_y2[i] - seg_y1[j])) - (by * (seg_x2[i] - seg_x1[j])));
            if ((((d1 * d2) < 0.0) && ((d3 * d4) < 0.0))) {
                intersects[((i * n) + j)] = 1LL;
            }
        }
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Phase 2: count intersections via parallel reduction */
    /* Prefix sum (+) intersects -> intersects */
    intersects[0] = intersects[0];
    for (int64_t _ps_i = 1; _ps_i < (n * n); _ps_i++) {
        intersects[_ps_i] = intersects[_ps_i - 1] + intersects[_ps_i];
    }
    /* ---- barrier (end of phase 1) ---- */
    /* Phase 3: write intersection count */
    for (int64_t pid = 0; pid < 1LL; pid++) {
        intersection_count[0LL] = intersects[((n * n) - 1LL)];
    }
    /* ---- barrier (end of phase 2) ---- */

    pram_free(seg_x1);
    pram_free(seg_y1);
    pram_free(seg_x2);
    pram_free(seg_y2);
    pram_free(intersects);
    pram_free(intersection_count);
    return 0;
}
