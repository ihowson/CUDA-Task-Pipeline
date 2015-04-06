#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <math_constants.h>

#define NUM_MIXTURE_COMPONENTS 2
#define M NUM_MIXTURE_COMPONENTS
// FIXME: should be set within the kernel func
#define N CHUNK_ENTRIES

// TODO: this should be configurable and/or read from command line params
#define MAX_ITERATIONS 100

// FIXME: 40kchunks is taking too long right now
// #define NUM_CHUNKS 40000
#define NUM_CHUNKS 100
#define CHUNK_ENTRIES 2000

#define CHUNK_BYTES (CHUNK_ENTRIES * sizeof(double))
#define POSTERIOR_CHUNK_BYTES (CHUNK_ENTRIES * sizeof(posterior_t))
#define ALL_CHUNK_BYTES (DATASET_ENTRIES * sizeof(posterior_t))
#define DATASET_ENTRIES (CHUNK_ENTRIES * NUM_CHUNKS)
#define DATASET_BYTES (sizeof(double) * DATASET_ENTRIES)

#define ALL_POSTERIOR_BYTES (sizeof(posterior_t) * DATASET_ENTRIES)

#define CONTROL_BYTES (NUM_CHUNKS * sizeof(control_t))

// this is the MaxOccupancy suggested block size for GTX660
#define BLOCK_SIZE 1024

// posterior probability for each component
typedef struct _posterior_t
{
    double component[NUM_MIXTURE_COMPONENTS];
} posterior_t;

typedef struct _invgauss_params_t
{
    double mu;
    double lambda;
    double alpha; // mixing proportion
} invgauss_params_t;

typedef struct _invgauss_control_t
{
    unsigned iterations;
    double log_likelihood;

    // fitted distribution parameters
    invgauss_params_t params[NUM_MIXTURE_COMPONENTS];
} invgauss_control_t;

typedef invgauss_control_t control_t;

enum execution_policy {
    EP_CPU, // run on CPU
    EP_GPU, // run on GPU on default stream
    EP_STREAM // run on GPU and allocate yourself a new stream
};

extern void stream(double *dataset, int32_t *g_chunk_id);
// extern __device__ double dinvgauss(double x, double mu, double lambda);

#endif /* COMMON_H */
