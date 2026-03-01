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

/* PRAM program: parallel_dfs | model: CREW */
/* Parallel depth-first search. CREW, O(D log n) time, n+m processors. */
/* Work bound: O((n+m) log n) */
/* Time bound: O(D log n) */

/* Generated C99 code for PRAM program: parallel_dfs */
/* Memory model: CREW */
/* Parallel depth-first search. CREW, O(D log n) time, n+m processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t m = 0;
    int64_t source = 0;
    int64_t _num_procs = (n + m);

    int64_t* row_ptr = (int64_t*)pram_calloc((n + 1LL), sizeof(int64_t));
    int64_t* col_idx = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* visited = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* dfs_order = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* parent_arr = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* stack = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* stack_top = (int64_t*)pram_calloc(1LL, sizeof(int64_t));
    int64_t* counter = (int64_t*)pram_calloc(1LL, sizeof(int64_t));

    /* Initialise visited=0, parent=-1, dfs_order=-1 */
    for (int64_t pid = 0; pid < n; pid++) {
        visited[pid] = 0LL;
        parent_arr[pid] = (-1LL);
        dfs_order[pid] = (-1LL);
    }
    /* ---- barrier (end of phase 0) ---- */
    stack[0LL] = source;
    stack_top[0LL] = 1LL;
    counter[0LL] = 0LL;
    /* ---- barrier (end of phase 1) ---- */
    /* Main DFS loop: pop stack, mark visited, push neighbors */
    for (int64_t iter = 0LL; iter < n; iter += 1LL) {
        /* Pop top vertex from stack */
        for (int64_t pid = 0; pid < 1LL; pid++) {
            int64_t top = stack_top[0LL];
            if ((top > 0LL)) {
                int64_t cur = stack[(top - 1LL)];
                stack_top[0LL] = (top - 1LL);
                if ((visited[cur] == 0LL)) {
                    visited[cur] = 1LL;
                    int64_t ord = counter[0LL];
                    dfs_order[cur] = ord;
                    counter[0LL] = (ord + 1LL);
                }
            }
        }
        /* ---- barrier (end of phase 2) ---- */
        /* Push unvisited neighbors onto stack */
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t ord = dfs_order[pid];
            if (((visited[pid] == 1LL) && (ord == (counter[0LL] - 1LL)))) {
                int64_t row_start = row_ptr[pid];
                int64_t row_end = row_ptr[(pid + 1LL)];
                for (int64_t e = row_start; e < row_end; e += 1LL) {
                    int64_t neighbor = col_idx[e];
                    if ((visited[neighbor] == 0LL)) {
                        int64_t st = stack_top[0LL];
                        stack[st] = neighbor;
                        stack_top[0LL] = (st + 1LL);
                        parent_arr[neighbor] = pid;
                    }
                }
            }
        }
        /* ---- barrier (end of phase 3) ---- */
    }

    pram_free(row_ptr);
    pram_free(col_idx);
    pram_free(visited);
    pram_free(dfs_order);
    pram_free(parent_arr);
    pram_free(stack);
    pram_free(stack_top);
    pram_free(counter);
    return 0;
}
