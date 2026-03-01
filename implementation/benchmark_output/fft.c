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

/* PRAM program: fft | model: CREW */
/* Parallel FFT butterfly computation. CREW, O(log n) time, n processors. */
/* Work bound: O(n log n) */
/* Time bound: O(log n) */

/* Generated C99 code for PRAM program: fft */
/* Memory model: CREW */
/* Parallel FFT butterfly computation. CREW, O(log n) time, n processors. */

int main(int argc, char* argv[]) {
    int64_t n = 0;
    int64_t _num_procs = n;

    int64_t* real = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* imag = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* real_temp = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* imag_temp = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* twiddle_re = (int64_t*)pram_calloc(n, sizeof(int64_t));
    int64_t* twiddle_im = (int64_t*)pram_calloc(n, sizeof(int64_t));

    int64_t log_n = ((int64_t)(log2(((double)(n)))));
    /* Phase 1: bit-reversal permutation */
    for (int64_t pid = 0; pid < n; pid++) {
        int64_t rev = 0LL;
        int64_t tmp_pid = pid;
        int64_t bit_count = log_n;
        real_temp[pid] = real[pid];
        imag_temp[pid] = imag[pid];
    }
    /* ---- barrier (end of phase 0) ---- */
    for (int64_t pid = 0; pid < n; pid++) {
        int64_t rev = 0LL;
        int64_t val = pid;
        int64_t b = 0LL;
        rev = ((rev << 1LL) | (val & 1LL));
        val = (val >> 1LL);
        real[rev] = real_temp[pid];
        imag[rev] = imag_temp[pid];
    }
    /* ---- barrier (end of phase 1) ---- */
    /* Phase 2: butterfly stages */
    for (int64_t stage = 0LL; stage < log_n; stage += 1LL) {
        int64_t block_size = (1LL << (stage + 1LL));
        int64_t half_block = (1LL << stage);
        for (int64_t pid = 0; pid < n; pid++) {
            int64_t group = (pid / block_size);
            int64_t pos = (pid % block_size);
            if ((pos < half_block)) {
                int64_t even_idx = ((group * block_size) + pos);
                int64_t odd_idx = (even_idx + half_block);
                int64_t tw_idx = (pos * (n / block_size));
                int64_t re_even = real[even_idx];
                int64_t im_even = imag[even_idx];
                int64_t re_odd = real[odd_idx];
                int64_t im_odd = imag[odd_idx];
                int64_t tw_re = twiddle_re[tw_idx];
                int64_t tw_im = twiddle_im[tw_idx];
                int64_t t_re = ((re_odd * tw_re) - (im_odd * tw_im));
                int64_t t_im = ((re_odd * tw_im) + (im_odd * tw_re));
                real_temp[even_idx] = (re_even + t_re);
                imag_temp[even_idx] = (im_even + t_im);
                real_temp[odd_idx] = (re_even - t_re);
                imag_temp[odd_idx] = (im_even - t_im);
            }
        }
        /* ---- barrier (end of phase 2) ---- */
        for (int64_t pid = 0; pid < n; pid++) {
            real[pid] = real_temp[pid];
            imag[pid] = imag_temp[pid];
        }
        /* ---- barrier (end of phase 3) ---- */
    }

    pram_free(real);
    pram_free(imag);
    pram_free(real_temp);
    pram_free(imag_temp);
    pram_free(twiddle_re);
    pram_free(twiddle_im);
    return 0;
}
