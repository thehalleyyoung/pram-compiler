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

/* PRAM program: voronoi_diagram | model: CREW */
/* Parallel Voronoi diagram construction. CREW, O(log^2 n) time, n^2 processors. */
/* Work bound: O(n^2 log n) */
/* Time bound: O(log^2 n) */

/* Generated C99 code for PRAM program: voronoi_diagram */
/* Memory model: CREW */
/* Parallel Voronoi diagram construction. CREW, O(log^2 n) time, n^2 processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = (n * n);

    double* px = (double*)pram_calloc(n, sizeof(double));
    double* py = (double*)pram_calloc(n, sizeof(double));
    int64_t* nearest = (int64_t*)pram_calloc((n * n), sizeof(int64_t));
    double* nearest_dist = (double*)pram_calloc((n * n), sizeof(double));
    int64_t* grid_size = (int64_t*)pram_calloc(1LL, sizeof(int64_t));
    int64_t* voronoi_site = (int64_t*)pram_calloc((n * n), sizeof(int64_t));

    /* Phase 1: compute nearest site for each grid point */
    for (int64_t pid = 0; pid < (n * n); pid++) {
        int64_t gi = (pid / n);
        int64_t gj = (pid % n);
        double gx = ((double)(gi));
        double gy = ((double)(gj));
        int64_t best_site = 0LL;
        double best_d = 1000000000000000000.0;
        for (int64_t s = 0LL; s < n; s += 1LL) {
            double dx = (px[s] - gx);
            double dy = (py[s] - gy);
            double d = ((dx * dx) + (dy * dy));
            if ((d < best_d)) {
                best_d = d;
                best_site = s;
            }
        }
        nearest[pid] = best_site;
        nearest_dist[pid] = best_d;
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Phase 2: O(log n) merge refinement rounds */
    int64_t log_n = ((int64_t)(log2(((double)(n)))));
    for (int64_t level = 0LL; level < log_n; level += 1LL) {
        int64_t half = (1LL << level);
        for (int64_t pid = 0; pid < (n * n); pid++) {
            int64_t col = (pid % n);
            if (((col % half) == 0LL)) {
                int64_t gi2 = (pid / n);
                double gx2 = ((double)(gi2));
                double gy2 = ((double)(col));
                int64_t cur_site = nearest[pid];
                double cur_d = nearest_dist[pid];
                for (int64_t s = 0LL; s < n; s += 1LL) {
                    double dx3 = (px[s] - gx2);
                    double dy3 = (py[s] - gy2);
                    double d3 = ((dx3 * dx3) + (dy3 * dy3));
                    if ((d3 < cur_d)) {
                        cur_d = d3;
                        cur_site = s;
                    }
                }
                nearest[pid] = cur_site;
                nearest_dist[pid] = cur_d;
            }
        }
        /* ---- barrier (end of phase 1) ---- */
    }
    /* Phase 3: label Voronoi regions */
    for (int64_t pid = 0; pid < (n * n); pid++) {
        voronoi_site[pid] = nearest[pid];
    }
    /* ---- barrier (end of phase 2) ---- */

    pram_free(px);
    pram_free(py);
    pram_free(nearest);
    pram_free(nearest_dist);
    pram_free(grid_size);
    pram_free(voronoi_site);
    return 0;
}
