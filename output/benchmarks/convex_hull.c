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

/* PRAM program: convex_hull | model: CREW */
/* Parallel convex hull (divide & conquer). CREW, O(log n) time, n processors. */
/* Work bound: O(n log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: convex_hull */
/* Memory model: CREW */
/* Parallel convex hull (divide & conquer). CREW, O(log n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    double* px = (double*)pram_calloc(n, sizeof(double));
    double* py = (double*)pram_calloc(n, sizeof(double));
    int64_t* sorted_idx = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* hull_flag = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* hull_next = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* hull_prev = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* rank = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* hull_size = (int64_t*)pram_calloc(1LL, sizeof(int64_t));

    /* Step 1: sort points by x-coordinate via ranking */
    for (int64_t pid = 0; pid < n; pid++) {
        int64_t my_x_rank = 0LL;
        double my_x = px[pid];
        for (int64_t j = 0LL; j < n; j += 1LL) {
            if (((px[j] < my_x) || ((px[j] == my_x) && (j < pid)))) {
                my_x_rank = (my_x_rank + 1LL);
            }
        }
        rank[pid] = my_x_rank;
        sorted_idx[my_x_rank] = pid;
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Step 2: initialise trivial hulls */
    for (int64_t pid = 0; pid < n; pid++) {
        hull_flag[pid] = 1LL;
        hull_next[pid] = pid;
        hull_prev[pid] = pid;
    }
    /* ---- barrier (end of phase 1) ---- */
    /* Step 3: O(log n) merge levels */
    int64_t log_n = ((int64_t)(log2(((double)(n)))));
    for (int64_t level = 0LL; level < log_n; level += 1LL) {
        int64_t group_size = (2LL << level);
        int64_t half_group = (1LL << level);
        /* Find tangent lines and merge hulls at this level */
        for (int64_t pid = 0; pid < (n / group_size); pid++) {
            int64_t left_start = (pid * group_size);
            int64_t right_start = (left_start + half_group);
            int64_t left_right = (right_start - 1LL);
            int64_t left_idx = sorted_idx[left_right];
            int64_t right_idx = sorted_idx[right_start];
            hull_next[left_idx] = right_idx;
            hull_prev[right_idx] = left_idx;
            int64_t left_left = sorted_idx[left_start];
            int64_t right_right_pos = PRAM_MIN(((right_start + half_group) - 1LL), (n - 1LL));
            int64_t right_right_idx = sorted_idx[right_right_pos];
            hull_next[right_right_idx] = left_left;
            hull_prev[left_left] = right_right_idx;
        }
        /* ---- barrier (end of phase 2) ---- */
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t nxt = hull_next[pid];
            int64_t prv = hull_prev[pid];
            if (((nxt != pid) && (hull_prev[nxt] == pid))) {
            }
        }
        /* ---- barrier (end of phase 3) ---- */
    }
    /* Count hull vertices */
    /* Prefix sum (+) hull_flag -> hull_size */
    hull_size[0] = hull_flag[0];
    for (int64_t _ps_i = 1; _ps_i < n; _ps_i++) {
        hull_size[_ps_i] = hull_size[_ps_i - 1] + hull_flag[_ps_i];
    }

    pram_free(px);
    pram_free(py);
    pram_free(sorted_idx);
    pram_free(hull_flag);
    pram_free(hull_next);
    pram_free(hull_prev);
    pram_free(rank);
    pram_free(hull_size);
    return 0;
}
