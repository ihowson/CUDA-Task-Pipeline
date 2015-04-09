// a single thread
// no Thrust dependency; using CUB instead as it (should) have full streams support

#define CUB_STDERR

#include <pthread.h>

#include <cuda_runtime.h>

#include <math.h>

// NOTE: OS X only
#include <libkern/OSAtomic.h>

#include <math_constants.h>
#include <iostream>

#include <cub/cub.cuh>

// TODO: this is only used for checkCudaErrors - try to remove it
#include "helper_cuda.h"

#include "common.h"


__host__ __device__ double dinvgauss(double x, double mu, double lambda)
{
    // TODO would be nice to assert that x > 0 and lambda = 0

    double x_minus_mu = x - mu;
    // using the definition from Wikipedia
    return sqrt(lambda / (2 * CUDART_PI * pow(x, 3.0))) * exp((-lambda * x_minus_mu * x_minus_mu) / (2 * mu * mu * x));
}

// http://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf
__device__ int getGlobalIdx_1D_1D()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void member_prob_kernel(
    const double *g_chunk,
    const invgauss_params_t *g_params,
    double *g_member_prob,
    double *g_x_times_member_prob,
    double *g_lambda_sum_arg,
    double *g_log_sum_prob)
{
    // g_chunk: 1xN input
    // g_params: 1xM input
    // g_member_prob: MxN output
    // TODO document the rest

    /*
        x.prob <- dinvgauss(x_expanded, mean=mu_expanded, shape=lambda_expanded)  # N x 2 matrix
        weighted.prob <- alpha_expanded * x.prob  # per-component weighted sum of probabilities (Nx2)
        sum.prob <- rowSums(weighted.prob)  # second line of the T function from wikipedia (Nx1)
        member.prob <- weighted.prob / sum.prob
        */

    int t = getGlobalIdx_1D_1D();

    if (t >= N)
    {
        // The recommended grid/block sizes often launch more kernels than we
        // have data for. Don't do anything if this is an excess kernel.
        return;
    }

    double weighted_prob[M];
    double sum_prob = 0.0f;
    double x = g_chunk[t];

    #pragma unroll
    for (int m = 0; m < M; m++)
    {
        weighted_prob[m] = g_params[m].alpha * dinvgauss(x, g_params[m].mu, g_params[m].lambda);
        sum_prob += weighted_prob[m];
    }

    // used for convergence check; this is log(rowSums(alpha_expanded * x.prob)))
    g_log_sum_prob[t] = log(sum_prob);

    #pragma unroll
    for (int m = 0; m < M; m++)
    {
        unsigned index = m * N + t;

        double mp = weighted_prob[m] / sum_prob;
        g_member_prob[index] = mp;

        // we don't use this until the end, but it's convenient to calculate here
        // it's the argument to colSums in:
        // mu.new <- colSums(x * member.prob) / member.prob.sum  # should be 1x2 matrix
        g_x_times_member_prob[index] = mp * x;

        // also used right at the end; this is the argument to colSums in:
        // lambda.new <- member.prob.sum / colSums(((x_expanded - mu_expanded) ^ 2 * member.prob) / (mu_expanded ^ 2 * x_expanded))
        // i.e. ((x_expanded - mu_expanded) ^ 2 * member.prob) / (mu_expanded ^ 2 * x_expanded)
        double x_minus_mu = x - g_params[m].mu;
        g_lambda_sum_arg[index] = (x_minus_mu * x_minus_mu * mp) / (g_params[m].mu * g_params[m].mu * x);
    }
}

void dump_dev_array(const char *msg, double *dev_array, int offset = 0)
{
    printf("tid %p %s: ", pthread_self(), msg);
    for (int i = offset; i < offset + 4; i++)
    {
        double val;

        checkCudaErrors(cudaMemcpyAsync(&val, &dev_array[i], sizeof(double), cudaMemcpyDeviceToHost, 0));
        checkCudaErrors(cudaStreamSynchronize(0));
        printf("%lf ", val);
    }

    printf("\n");
}

void dump_dev_value(const char *msg, double *dev_ptr)
{
    double val;

    checkCudaErrors(cudaMemcpyAsync(&val, dev_ptr, sizeof(double), cudaMemcpyDeviceToHost, 0));
    checkCudaErrors(cudaStreamSynchronize(0));
    printf("tid %p %s: %lf \n", pthread_self(), msg, val);
}

/* So that we only have one memcpy back from the device, we combine the results into this struct and copy them in one operation */
typedef struct _igresults
{
    double xmp_sum;
    double lambda_sum;
    double member_prob_sum;
} igresults;

void stream(double *dataset, int32_t *g_chunk_id)
{
    // TODO: grid/block size calculations could be moved into main code, outside thread
    // http://stackoverflow.com/a/25010560/591483
    int blockSize;      // The launch configurator returned block size 
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
    int gridSize;       // The actual grid size needed, based on input size 

    checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (const void *)member_prob_kernel, 0, N));
    // Round up according to array size 
    gridSize = (N + blockSize - 1) / blockSize; 


// gridSize = 16;
// blockSize = 128;
    printf("using blockSize=%d gridSize=%d\n", blockSize, gridSize);

    invgauss_params_t *params_new;
    checkCudaErrors(cudaMallocHost(&params_new, sizeof(invgauss_params_t) * M));
    invgauss_params_t *start_params;
    checkCudaErrors(cudaMallocHost(&start_params, sizeof(invgauss_params_t) * M));
    igresults *host_igresults;
    checkCudaErrors(cudaMallocHost(&host_igresults, sizeof(igresults)));
    double *host_loglik;
    checkCudaErrors(cudaMallocHost(&host_loglik, sizeof(double)));

    cudaStream_t stream;
    // cudaSetDevice(0); // TODO: adjust when we have multiple GPUs; probably assign a new group of threads to each GPU
    checkCudaErrors(cudaStreamCreate(&stream));

    // persistent device memory
    // allocated out here to avoid cudaMalloc in main loop
    double *dev_chunk; // N elements
    checkCudaErrors(cudaMalloc(&dev_chunk, sizeof(double) * N));

    invgauss_params_t *dev_params;
    checkCudaErrors(cudaMalloc(&dev_params, sizeof(invgauss_params_t) * M));

    double *dev_member_prob;
    checkCudaErrors(cudaMalloc(&dev_member_prob, sizeof(double) * M * N));

    double *dev_x_times_member_prob;
    checkCudaErrors(cudaMalloc(&dev_x_times_member_prob, sizeof(double) * M * N));

    igresults *dev_igresults;
    checkCudaErrors(cudaMalloc(&dev_igresults, sizeof(igresults)));

    double *dev_lambda_sum_arg;
    checkCudaErrors(cudaMalloc(&dev_lambda_sum_arg, sizeof(double) * M * N));

    double *dev_log_sum_prob;
    checkCudaErrors(cudaMalloc(&dev_log_sum_prob, sizeof(double) * N));

    double *dev_loglik;
    checkCudaErrors(cudaMalloc(&dev_loglik, sizeof(double)));

    // temp storage for reductions
    // we pretty much only do a sum reduction across N, so this ought to persist for the lifetime of the thread
    // we do the actual malloc once in the main loop and check that this is sufficient each time through
    void *dev_temp = NULL;
    size_t temp_size = 0;

    // thrust::device_vector<double> dev_posterior(N);

    // Lots of computations use 2D matrices.
    // - We don't use BLAS as there are no mat-mults, only elementwise mult/add
    // - We store in standard vectors to simplify expression
    // - We store columns grouped together as this is the common access pattern
    // - We often use for loops as (for the target dataset) N ~= 2000 and this is
    //   already more parallel than we have capacity for. We also have many
    //   datasets to run and the infrastructure to run tasks in parallel, which
    //   covers up many sins.

    // set up initial parameters
    for (unsigned m = 0; m < M; m++)
    {
        start_params[m].mu = 0.99 + 0.02 * m;
        start_params[m].lambda = 1.0;
        start_params[m].alpha = 0.5;
    }

    // FIXME FIXME: we're getting a crash on the last 8; presumably we have an out-of-range memory access

    // Sum reductions require some temporary storage. We always do sums over N
    // elements of type 'double', so we assume that the storage requirement is
    // always the same and do the malloc once here.
    // If we pass in NULL for needed_bytes, we get back the required size
    cub::DeviceReduce::Sum(dev_temp, temp_size, dev_member_prob, &dev_igresults->member_prob_sum, N, stream);
    printf("malloc dev_temp to %zd bytes\n", temp_size);
    checkCudaErrors(cudaMalloc(&dev_temp, temp_size));

    int32_t chunk_id = OSAtomicIncrement32(g_chunk_id);
    // TODO: I'm not convinced that this is working.
    // TODO: we're definitely missing chunk 0, so bodge around it here
    chunk_id--;

    while (chunk_id < NUM_CHUNKS)
    {
        // printf("thread %p chunk %d\n", pthread_self(), chunk_id);

        // init device parameters
        for (unsigned m = 0; m < M; m++)
        {
            params_new[m] = start_params[m];
        }

        // copy chunk to device
        double *chunk_host_ptr = &dataset[chunk_id * CHUNK_ENTRIES];
        checkCudaErrors(cudaMemcpyAsync(dev_chunk, chunk_host_ptr, CHUNK_BYTES, cudaMemcpyHostToDevice, stream));

        bool converged = false;
        bool failed = false;
        double old_loglik = -INFINITY;
        double epsilon = 0.000001;

        // run EM algorithm
        unsigned iteration = 0; // FIXME: nasty. Better to explicitly count.
        while (converged == false && failed == false && iteration < MAX_ITERATIONS)
        {
            iteration++;
            checkCudaErrors(cudaMemcpyAsync(dev_params, params_new, sizeof(invgauss_params_t) * M, cudaMemcpyHostToDevice, stream));
            checkCudaErrors(cudaStreamSynchronize(stream));

            //////// PROCESS CHUNK

            // calculate member.prob
            // x.prob <- dinvgauss(x_expanded, mean=mu_expanded, shape=lambda_expanded)  # N x 2 matrix
            // weighted.prob <- alpha_expanded * x.prob  # per-component weighted sum of probabilities (Nx2)
            // sum.prob <- rowSums(weighted.prob)  # second line of the T function from wikipedia (Nx1)
            // member.prob <- weighted.prob / sum.prob

            // There are a whole bunch of operations on the data and
            // parameters that can be calculated in one pass. We do them all
            // here to minimise kernel launch overhead. This does make the
            // program flow a bit confusing.
            member_prob_kernel<<<gridSize, blockSize, 0, stream>>>(dev_chunk, dev_params, dev_member_prob, dev_x_times_member_prob, dev_lambda_sum_arg, dev_log_sum_prob);
            // The remaining operations are all summations over various
            // outputs from this kernel.

            checkCudaErrors(cudaStreamSynchronize(stream));

            // have we converged?
            // TODO(perf): we could probably save some time by only doing this check once every few iterations - it's slow relative to the rest of the iteration time
            // log.lik <- sum(log(rowSums(alpha_expanded * x.prob)))
            cub::DeviceReduce::Sum(dev_temp, temp_size, dev_log_sum_prob, dev_loglik, N, stream);
            // copy new log-likelihood back
            checkCudaErrors(cudaMemcpyAsync(host_loglik, dev_loglik, sizeof(double), cudaMemcpyDeviceToHost, stream));
            checkCudaErrors(cudaStreamSynchronize(stream)); // wait for copy to complete

            // printf("old ll = %lf, new ll = %lf\n", old_loglik, *host_loglik);

            if (old_loglik > *host_loglik)
            {
                // we're going backwards
                printf("FAILED TO CONVERGE. Giving up.\n");
                failed = true;
                break;
            }

            double diff = *host_loglik - old_loglik;
            if (diff < epsilon) {
                converged = true;
                break;
            }

            // didn't converge, continue optimising

            // member.prob.sum <- colSums(member.prob)
            for (int m = 0; m < M; m++)
            {
                // TODO(perf): the components are independent, so we could sum them in parallel
                cub::DeviceReduce::Sum(dev_temp, temp_size, dev_member_prob + m * N, &dev_igresults->member_prob_sum, N, stream);

                // dump_dev_array("dev_member_prob", dev_member_prob + m * N);
                // dump_dev_array("dev_member_prob end", dev_member_prob + m * N + N - 4);
                // dump_dev_value("member_prob_sum", &dev_igresults->member_prob_sum);

                // mu.new <- colSums(x * member.prob) / member.prob.sum  # should be 1x2 matrix
                // x * member.prob was calculated earlier into dev_x_times_member_prob
                // perform the sum
                // TODO check that the dev_temp size is still suitable
                cub::DeviceReduce::Sum(dev_temp, temp_size, dev_x_times_member_prob + m * N, &dev_igresults->xmp_sum, N, stream);

                // do the colSums for lambda
                cub::DeviceReduce::Sum(dev_temp, temp_size, dev_lambda_sum_arg + m * N, &dev_igresults->lambda_sum, N, stream);

                // determine new parameters
                checkCudaErrors(cudaMemcpyAsync(host_igresults, dev_igresults, sizeof(igresults), cudaMemcpyDeviceToHost, stream));
                checkCudaErrors(cudaStreamSynchronize(stream)); // wait for copy to complete

                // FIXME: sum here is grossly wrong
                // printf("thread %p iter %d comp %d, sum is %lf\n", pthread_self(), iteration, m, host_igresults->member_prob_sum);

                // alpha.new <- member.prob.sum / N  # should be 1x2 matrix
                params_new[m].alpha = host_igresults->member_prob_sum / N;
                params_new[m].mu = host_igresults->xmp_sum / host_igresults->member_prob_sum;
                params_new[m].lambda = host_igresults->member_prob_sum / host_igresults->lambda_sum;
            }

            // check for parameter sanity
            for (int m = 0; m < M; m++)
            {
                invgauss_params_t *p = &params_new[m];
                if (isnan(p->alpha) || isnan(p->mu) || isnan(p->lambda))
                {
                    printf("bogus!\n");
                    printf("m = %d alpha = %lf mu = %lf lambda = %lf\n", m, p->alpha, p->mu, p->lambda);
                    abort();
                }
            }
            /*
            printf("thread %p iter %d\n", pthread_self(), iteration);
            for (int m = 0; m < M; m++)
            {
                invgauss_params_t *p = &params_new[m];
                printf("\tcomp %d alpha=%lf mu=%lf lambda=%lf\n", m, p->alpha, p->mu, p->lambda);
            }
            */
            old_loglik = *host_loglik;
        }

        // Disabling this gains about .5 seconds on a 5 second run
        printf("thread %p fit chunk %d after %d iterations\n", pthread_self(), chunk_id, iteration);
        for (int m = 0; m < M; m++)
        {
            invgauss_params_t *p = &params_new[m];
            printf("\tcomp %d alpha=%lf mu=%lf lambda=%lf\n", m, p->alpha, p->mu, p->lambda);
        }

        // FIXME BODGE
        chunk_id = OSAtomicIncrement32(g_chunk_id);
        chunk_id--;
    }
}
