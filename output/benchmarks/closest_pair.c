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

/* PRAM program: closest_pair | model: CREW */
/* Parallel closest pair (divide & conquer). CREW, O(log^2 n) time, n processors. */
/* Work bound: O(n log^2 n) */
/* Time bound: O(log^2 n) */

/* Generated C99 code for PRAM program: closest_pair */
/* Memory model: CREW */
/* Parallel closest pair (divide & conquer). CREW, O(log^2 n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    double* px = (double*)pram_calloc(n, sizeof(double));
    double* py = (double*)pram_calloc(n, sizeof(double));
    int64_t* sorted_idx = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* rank = (int64_t*)pram_calloc(n, sizeof(int64_t));
    double* best_dist = (double*)pram_calloc(n, sizeof(double));
    int64_t* best_i = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* best_j = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* in_strip = (int64_t*)pram_calloc(n, sizeof(int64_t));
    double* global_best = (double*)pram_calloc(1LL, sizeof(double));

    /* Step 1: sort points by x */
    for (int64_t pid = 0; pid < n; pid++) {
        int64_t my_rank = 0LL;
        double my_x = px[pid];
        for (int64_t j = 0LL; j < n; j += 1LL) {
            if (((px[j] < my_x) || ((px[j] == my_x) && (j < pid)))) {
                my_rank = (my_rank + 1LL);
            }
        }
        rank[pid] = my_rank;
        sorted_idx[my_rank] = pid;
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Step 2: base case — distance between adjacent sorted points */
    for (int64_t pid = 0; pid < (n - 1LL); pid++) {
        int64_t i_pt = sorted_idx[pid];
        int64_t j_pt = sorted_idx[(pid + 1LL)];
        double dx = (px[j_pt] - px[i_pt]);
        double dy = (py[j_pt] - py[i_pt]);
        double dist = ((dx * dx) + (dy * dy));
        best_dist[pid] = dist;
        best_i[pid] = i_pt;
        best_j[pid] = j_pt;
    }
    /* ---- barrier (end of phase 1) ---- */
    /* Step 3: O(log n) merge levels */
    int64_t log_n = ((int64_t)(log2(((double)(n)))));
    for (int64_t level = 1LL; level < (log_n + 1LL); level += 1LL) {
        int64_t group_size = (1LL << level);
        int64_t half = (1LL << (level - 1LL));
        for (int64_t pid = 0; pid < (n / group_size); pid++) {
            int64_t left_start = (pid * group_size);
            int64_t right_start = (left_start + half);
            double delta_left = best_dist[left_start];
            double delta_right = best_dist[right_start];
            double delta = PRAM_MIN(delta_left, delta_right);
            int64_t mid_idx = sorted_idx[(right_start - 1LL)];
            double mid_x = px[mid_idx];
            for (int64_t si = left_start; si < (left_start + group_size); si += 1LL) {
                if ((si < n)) {
                    int64_t pt = sorted_idx[si];
                    double pt_x = px[pt];
                    double diff_x = (pt_x - mid_x);
                    if (((diff_x * diff_x) <= delta)) {
                        for (int64_t sj = (si + 1LL); sj < PRAM_MIN((si + 8LL), (left_start + group_size)); sj += 1LL) {
                            if ((sj < n)) {
                                int64_t pt2 = sorted_idx[sj];
                                double dx2 = (px[pt2] - px[pt]);
                                double dy2 = (py[pt2] - py[pt]);
                                double d2 = ((dx2 * dx2) + (dy2 * dy2));
                                if ((d2 < delta)) {
                                    delta = d2;
                                    best_dist[left_start] = d2;
                                    best_i[left_start] = pt;
                                    best_j[left_start] = pt2;
                                }
                            }
                        }
                    }
                }
            }
        }
        /* ---- barrier (end of phase 2) ---- */
    }
    /* Final: reduce to find global closest pair */
    for (int64_t d = 0LL; d < log_n; d += 1LL) {
        int64_t stride = (1LL << d);
        for (int64_t pid = 0; pid < (n / (2LL * stride)); pid++) {
            int64_t i_idx = (pid * (2LL * stride));
            int64_t j_idx = (i_idx + stride);
            if ((j_idx < n)) {
                if ((best_dist[j_idx] < best_dist[i_idx])) {
                    best_dist[i_idx] = best_dist[j_idx];
                    best_i[i_idx] = best_i[j_idx];
                    best_j[i_idx] = best_j[j_idx];
                }
            }
        }
        /* ---- barrier (end of phase 3) ---- */
    }
    global_best[0LL] = best_dist[0LL];

    pram_free(px);
    pram_free(py);
    pram_free(sorted_idx);
    pram_free(rank);
    pram_free(best_dist);
    pram_free(best_i);
    pram_free(best_j);
    pram_free(in_strip);
    pram_free(global_best);
    return 0;
}
