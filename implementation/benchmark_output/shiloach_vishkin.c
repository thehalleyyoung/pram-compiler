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

/* PRAM program: shiloach_vishkin | model: CRCW-Arbitrary */
/* Shiloach-Vishkin connected components. CRCW-Arbitrary, O(log n) time, m+n processors. */
/* Work bound: O((m+n) log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: shiloach_vishkin */
/* Memory model: CRCW-Arbitrary */
/* Shiloach-Vishkin connected components. CRCW-Arbitrary, O(log n) time, m+n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t m = 0;
    int64_t _num_procs = (m + n);

    int64_t* src = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* dst = (int64_t*)pram_calloc(m, sizeof(int64_t));
    int64_t* D = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* changed = (int64_t*)pram_calloc(1LL, sizeof(int64_t));

    /* Initialise: D[i] = i */
    for (int64_t pid = 0; pid < n; pid++) {
        {
            int64_t __buf_pid_0 = pid;
            int64_t __buf_idx_pid_0 = pid;
            if ((__buf_idx_pid_0 >= 0LL)) {
                D[__buf_idx_pid_0] = __buf_pid_0;
            }
        }
    }
    /* ---- barrier (end of phase 0) ---- */
    /* Main loop: O(log n) rounds of hook + pointer-jump */
    int64_t max_rounds = (((int64_t)(log2(((double)(n))))) + 1LL);
    for (int64_t round = 0LL; round < max_rounds; round += 1LL) {
        changed[0LL] = 0LL;
        /* ---- barrier (end of phase 1) ---- */
        /* Hooking: for each edge (u,v) hook higher root under lower */
        for (int64_t pid = 0; pid < m; pid++) {
            int64_t u = src[pid];
            int64_t v_node = dst[pid];
            int64_t du = D[u];
            int64_t dv = D[v_node];
            if ((du != dv)) {
                if ((du < dv)) {
                    {
                        int64_t __buf_pid_1 = du;
                        int64_t __buf_idx_pid_1 = dv;
                        if ((__buf_idx_pid_1 >= 0LL)) {
                            D[__buf_idx_pid_1] = __buf_pid_1;
                        }
                    }
                    {
                        int64_t __buf_pid_0 = 1LL;
                        int64_t __buf_idx_pid_0 = 0LL;
                        if ((__buf_idx_pid_0 >= 0LL)) {
                            changed[__buf_idx_pid_0] = __buf_pid_0;
                        }
                    }
                } else {
                    {
                        int64_t __buf_pid_3 = dv;
                        int64_t __buf_idx_pid_3 = du;
                        if ((__buf_idx_pid_3 >= 0LL)) {
                            D[__buf_idx_pid_3] = __buf_pid_3;
                        }
                    }
                    {
                        int64_t __buf_pid_2 = 1LL;
                        int64_t __buf_idx_pid_2 = 0LL;
                        if ((__buf_idx_pid_2 >= 0LL)) {
                            changed[__buf_idx_pid_2] = __buf_pid_2;
                        }
                    }
                }
            }
        }
        /* ---- barrier (end of phase 2) ---- */
        /* Pointer jumping: D[i] <- D[D[i]] */
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t parent = D[pid];
            int64_t grandparent = D[parent];
            if ((parent != grandparent)) {
                {
                    int64_t __buf_pid_1 = grandparent;
                    int64_t __buf_idx_pid_1 = pid;
                    if ((__buf_idx_pid_1 >= 0LL)) {
                        D[__buf_idx_pid_1] = __buf_pid_1;
                    }
                }
                {
                    int64_t __buf_pid_0 = 1LL;
                    int64_t __buf_idx_pid_0 = 0LL;
                    if ((__buf_idx_pid_0 >= 0LL)) {
                        changed[__buf_idx_pid_0] = __buf_pid_0;
                    }
                }
            }
        }
        /* ---- barrier (end of phase 3) ---- */
        if ((changed[0LL] == 0LL)) {
            /* Converged – break */
        }
    }

    pram_free(src);
    pram_free(dst);
    pram_free(D);
    pram_free(changed);
    return 0;
}
