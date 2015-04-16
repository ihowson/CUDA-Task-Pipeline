#ifndef LP_SUM_H
#define LP_SUM_H

// always launch this many threads per block
// 64 seems to be the minimum
#define LP_SUM_BLOCK_SIZE 64

__global__ void lp_sum_kernel(const double *g_input, double *g_sum, unsigned n);

#endif /* LP_SUM_H */
