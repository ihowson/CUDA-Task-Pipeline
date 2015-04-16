#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/time.h>

#include "helper_cuda.h"

#include "common.h"


// SMS=30 dbg=1 make

// This is OS X specific, unfortunately; haven't found a good cross-platform way to do this
// C++ <atomic> doesn't work on Mac for whatever reason
#ifdef DARWIN
#include <libkern/OSAtomic.h>
#else
//#include <atomic>
//#include <stdatomic.h>
//#include <glib.h>
//#include <linux/arch/arm64/include/asm/atomic.h>
#endif

// consider electricfence
// http://lh3lh3.users.sourceforge.net/memdebug.shtml
// http://thingsilearned.com/2007/09/30/electric-fence-on-os-x/

int g_runcpu = 1;
int g_runser = 1;
int g_runpar = 1;

void *thread(void *void_args);

void usage(char **argv)
{
    printf("Usage: %s [options below]\n", argv[0]);
    /*
    printf("\t--sync_method=n for CPU/GPU synchronization\n");
    printf("\t             n=%s\n", sSyncMethod[0]);
    printf("\t             n=%s\n", sSyncMethod[1]);
    printf("\t             n=%s\n", sSyncMethod[2]);
    printf("\t   <Default> n=%s\n", sSyncMethod[4]);
    printf("\t--use_generic_memory (default) use generic page-aligned for system memory\n");
    printf("\t--use_cuda_malloc_host (optional) use cudaMallocHost to allocate system memory\n");
    */
}

/*
#if defined(__APPLE__) || defined(MACOSX)
#define DEFAULT_PINNED_GENERIC_MEMORY false
#else
#define DEFAULT_PINNED_GENERIC_MEMORY true
#endif

*/

// How many chunks to attempt to process at the same time. 32 is the number of
// simultaneous kernel executions supported on the Kepler architecture.
// Adjusting this may or may not provide performance improvement; I'm guessing
// that it's best to keep it near the device configuration.
#define SIMULTANEOUS_KERNELS 32
// #define SIMULTANEOUS_KERNELS 16
// 8 ran in 1.3 seconds
// 16 in 1.14 seconds
// 32 in 1.11 seconds

typedef struct _thread_args_t
{
    unsigned thread_id;
    double *device_chunk; // this is a device pointer

    posterior_t *device_posterior; // this is a device pointer

    control_t *device_control; // this is a device pointer
    double *host_dataset; // full dataset stored on host
    posterior_t *host_posterior; // 1:1 posterior probabilities for each observation
    control_t *host_controls; // controls stored on host
    int *current_chunk_id; // pointer to shared chunk_id counter
} thread_args_t;

__host__ __device__ double old_dinvgauss(double x, double mu, double lambda)
{
    // TODO would be nice to assert that x > 0 and lambda = 0

    double x_minus_mu = x - mu;
    return sqrt(lambda / (2 * CUDART_PI * pow(x, 3.0))) * exp((-lambda * x_minus_mu * x_minus_mu) / (2 * mu * mu * x));
    // http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.invgauss.html
    // invgauss.pdf(x, mu) = 1 / sqrt(2*pi*x**3) * exp(-(x-mu)**2/(2*x*mu**2))
}

/*
or, http://en.wikipedia.org/wiki/Inverse_Gaussian_distribution#Generating_random_variates_from_an_inverse-Gaussian_distribution
// ported from invgauss.R from the statmod package
// this version only generates a single random number at a time, not a matrix
double rinvgauss(double mu, double lambda)
{
    // Random variates from inverse Gaussian distribution
    // Gordon Smyth (with a correction by Trevor Park 14 June 2005)
    // Created 15 Jan 1998.  Last revised 27 May 2014.

    // Check input
    double phi = 1.0 / shape;

    double r;

    // Divide out mu
    phi *= mu;

    // TODO up to here
    // need a replacement for rchisq
    Y <- rchisq(n,df=1)
    X1 <- 1 + phi[i]/2 * (Y - sqrt(4*Y/phi[i]+Y^2))
    firstroot <- as.logical(rbinom(n,size=1L,prob=1/(1+X1)))
    r[i][firstroot] <- X1[firstroot]
    r[i][!firstroot] <- 1/X1[!firstroot]

    mu*r
}
*/

__global__ void inverse_gaussian_em_kernel(double *g_chunk, invgauss_control_t *g_control, posterior_t *g_posterior)
{
    // the global memory cache might catch these, so probably no strong reason to put then in shared memory
    __shared__ invgauss_params_t params[M];
    __shared__ invgauss_params_t params_new[M];
    //__shared__ double mu[M], double lambda[M], double alpha[M];
    //__shared__ double mu_new[M], double lambda_new[M], double alpha_new[M];

    // TODO register use is currently 46 bytes. If you can bring it <= 32 bytes, you double occupancy. Consider moving data to shared memory and restricting scope of register variables to reduce peak usage.

    unsigned t = threadIdx.x;

    // init
    if (t < M)
    {
        params[t] = g_control->params[t];
    }

    __syncthreads();

    // Each thread of this kernel will handle one observation. This is my x.
    double x = g_chunk[t];

    double x_prob[M]; // p(l|x_i, Theta^g)
    double weighted_prob[M]; // per-component weighted sum of probabilities

    // each iteration of the EM algorithm
    unsigned iteration = 0; // FIXME: nasty. Better to explicitly count.
    for (; iteration < MAX_ITERATIONS; iteration++)
    {
        // This could be expressed more parallel, but we have plenty of parallelism already across tasks; this is much easier to understand
        // it is a little awkward that we're continually running for loops over the components; a matrix-like representation would be cleaner. We don't actually need matrix ops, though - we only do element-wise multiply and col/row sums.
        for (unsigned m = 0; m < M; m++)
        {
            // calculate p(l|x_i, Theta^g)
            x_prob[m] = old_dinvgauss(x, params[m].mu, params[m].lambda);
            weighted_prob[m] = params[m].alpha * x_prob[m];
        }

        double sum_prob = 0.0f;
        for (unsigned m = 0; m < M; m++)
        {
            sum_prob += weighted_prob[m];
        }

        __shared__ double member_prob[M][N];
        for (unsigned m = 0; m < M; m++)
        {
            member_prob[m][t] = weighted_prob[m] / sum_prob;
        }

        // column sum member_prob and put results in member_prob_sum
        double member_prob_sum[M];
        __shared__ double temp[N]; // we reuse this array every time we do a column sum
        for (unsigned m = 0; m < M; m++)
        {
            temp[t] = member_prob[m][t]; // memcpy!
            __syncthreads();
            // FIXME: BROKEN
            // sum_reduce(temp, CHUNK_ENTRIES);

            // NOTE: this does the divide on all threads, which is wasteful, but we'd have to syncthreads anyway, so it probably does no harm. Probably better to keep this in registers rather than push to shared memory.
            member_prob_sum[m] = temp[0];

            if (t == 0)
            {
                params_new[m].alpha = member_prob_sum[m] / N;
            }
        }

        // calculate mu.new
        for (unsigned m = 0; m < M; m++)
        {
            temp[t] = member_prob[m][t] * x;
            __syncthreads();
            // FIXME: BROKEN
            // sum_reduce(temp, CHUNK_ENTRIES);

            if (t == 0)
            {
                params_new[m].mu = temp[0] / member_prob_sum[m];
            }
        }

        // calculate lambda.new
        for (unsigned m = 0; m < M; m++)
        {
            double x_minus_mu = x - params[m].mu;
            temp[t] = x_minus_mu * x_minus_mu * member_prob[m][t] / (params[m].mu * params[m].mu * x);
            __syncthreads();
            // FIXME: BROKEN
            // sum_reduce(temp, CHUNK_ENTRIES);

            // TODO: you could probably optimise this to minimise thread divergence or parallelise some of the work, though potential gains are pretty trivial
            if (t == 0)
            {
                params_new[m].lambda = member_prob_sum[m] / temp[0];
            }
        }
        
        // TODO: have we converged?
        // might work to keep an array of parameter attempts - good for logging

        // copy params_new to params
        if (t < M)
        {
            params[t] = params_new[t];
        }

        // TODO remove, debug only
        /*
        if (t == 0)
        {
            for (unsigned m = 0; m < M; m++)
            {
                printf("iteration %d\n", iteration);
                printf("\tcomp %d: alpha=%f, mu=%f, lambda=%f\n", m, params[m].alpha, params[m].mu, params[m].lambda);
            }
        }
        __syncthreads();
        */
    }

    // EM has converged. Copy result back to global memory.
    if (t == 0)
    {
        g_control->iterations = iteration;
        //g_control->loglik = TODO;
    }

    if (t < M)
    {
        g_control->params[t] = params[t];
    }

    /*
    stuff to return:
        iterations 
        loglike
        params
    */

    // TODO: also calculate membership probabilities and copy them back to global memory for collection by host
}

void print_time_elapsed(struct timeval start_time, struct timeval end_time)
{
    double elapsed_time = (1000000.0 * (end_time.tv_sec - start_time.tv_sec) + end_time.tv_usec - start_time.tv_usec) / 1000000.0;
    printf("%f seconds elapsed\n", elapsed_time);
}

int main(int argc, char **argv)
{
    struct timeval start_time, end_time;
    int num_gpus;

    /*
    for (unsigned i = 0; i < argc; i++)
    {
        if (0 == strcmp(argv[i], "--paronly"))
        {
            g_runcpu = 0;
            g_runser = 0;
            g_runpar = 1;
        }

        if (0 == strcmp(argv[i], "--seronly"))
        {
            g_runcpu = 0;
            g_runser = 1;
            g_runpar = 0;
        }
    }
    */

    g_runcpu = 0;
    g_runser = 0;
    g_runpar = 1;

    printf("If this fails, make sure you're running as root (on Linux)\n");
    checkCudaErrors(cudaGetDeviceCount(&num_gpus));

    assert(num_gpus == 1); // don't know what to do with multiple GPUs yet
    printf("There are %d GPUs\n", num_gpus);

    // For now, I'm assuming that the entire dataset resides within host RAM, in one single huge block

    // you might also consider unified memory: http://devblogs.nvidia.com/parallelforall/unified-memory-in-cuda-6/
    // probably lower performance, but easier programming model

    // this is pinned RAM, so bad things will happen if you allocate too much
    // (there's no paging; either the system will die or this process will be
    // OOM killed)
    // pinned RAM is a requirement for CUDA stream concurrency
    double *dataset;
    printf("dataset is %lu bytes\n", DATASET_BYTES);
    checkCudaErrors(cudaMallocHost(&dataset, DATASET_BYTES));

    // for each input observation, we generate a posterior probability
    posterior_t *posterior;
    checkCudaErrors(cudaMallocHost(&posterior, ALL_POSTERIOR_BYTES));

    printf("Loading test data...");
    // for now, we just load one chunk and repeat it across the entire dataset
    for (unsigned i = 0; i < NUM_CHUNKS; i++)
    {
        printf(".");
        FILE *f = fopen("test.data", "r");

        if (f == NULL)
        {
            printf("Couldn't open file (errno = %d)\n", errno);
            return 1;
        }

        for (unsigned data_index = 0; data_index < N; data_index++)
        {
            if (1 != fscanf(f, "%lf", &dataset[i * N + data_index]))
            {
                printf("ERROR: ran out of data to read at i=%d index=%d\n", i, data_index);
            }
        }

        fclose(f);
    }
    printf("\n");

#if 0
    printf("Making up some data...\n");

    // TODO you could fill it with inv gaussian data
    // fill it in with Uniform(0, 1)
    for (uint64_t i = 0; i < DATASET_ENTRIES; i++)
    {
        dataset[i] = (double)rand() / (double)INT_MAX;
    }
#endif
    int dump_entries = 8;
    printf("The first %d entries look like ", dump_entries);
    for (int i = 0; i < dump_entries; i++)
    {
        printf("%lf ", dataset[i]);
    }
    printf("\n");

    // http://stackoverflow.com/a/25010560/591483

    if (g_runcpu)
    {
        printf("CPU calculation is disabled; it's not implemented yet\n");

        printf("Clearing posterior probabilities...\n");
        memset(posterior, 0, ALL_POSTERIOR_BYTES);

    }

    if (g_runser)
    {
        /*
        if (ep == EP_STREAM)
        {
            // TODO: check return value
            cudaStreamCreate(&stream);
            policy = thrust::cuda::par.on(stream);
        }
        else if (ep == EP_GPU)
        {
            policy = thrust::cuda::par;
        }
        else
        {
            policy = thrust::cpp::par;
        }
        */

        // FIXME: this will definitely crash
        stream(dataset, NULL);
    }

    if (g_runpar)
    {
        // int32_t g_chunk_id = 0;

        printf("Processing %d chunks simultaneously\n", SIMULTANEOUS_KERNELS);
        // fire off 16-32 threads; that'll make the logic simpler for you

/*
        double *device_chunks[SIMULTANEOUS_KERNELS];
        control_t *device_controls[SIMULTANEOUS_KERNELS];
        posterior_t *device_posteriors[SIMULTANEOUS_KERNELS];
        control_t host_controls[NUM_CHUNKS];
        
        for (unsigned i = 0; i < SIMULTANEOUS_KERNELS; i++)
        {
            checkCudaErrors(cudaMalloc((void **)&(device_chunks[i]), CHUNK_BYTES));
            checkCudaErrors(cudaMalloc((void **)&(device_controls[i]), CONTROL_BYTES));
            checkCudaErrors(cudaMalloc((void **)&(device_posteriors[i]), POSTERIOR_CHUNK_BYTES));
        }
        */

        pthread_t threads[SIMULTANEOUS_KERNELS];
        thread_args_t args[SIMULTANEOUS_KERNELS];

        int32_t current_chunk_id = 0;

        gettimeofday(&start_time, 0);

        // launch threads
        for (unsigned i = 0; i < SIMULTANEOUS_KERNELS; i++)
        {
            // set up arguments
            args[i].thread_id = i;
            // args[i].device_chunk = device_chunks[i];
            // args[i].device_posterior = device_posteriors[i];
            // args[i].device_control = device_controls[i];
            args[i].current_chunk_id = &current_chunk_id;
            args[i].host_dataset = dataset;
            // args[i].host_controls = host_controls;
            // args[i].host_posterior = posterior;
            int rc = pthread_create(&threads[i], NULL, thread, (void *)&args[i]);
            assert(rc == 0);
        }

        // join threads
        for (unsigned i = 0; i < SIMULTANEOUS_KERNELS; i++)
        {
            pthread_join(threads[i], NULL);
        }

        gettimeofday(&end_time, 0);
        printf("Processed %i chunks\n", NUM_CHUNKS);

        printf("cuda parallel: ");
        print_time_elapsed(start_time, end_time);
        
#if 0
        for (unsigned i = 0; i < 4; i++)
        {
            // printf("cuda parallel: chunk %d sum is %f\n", i, host_controls[i].sum);
            // FIXME: hardcoded component numbers
            printf("cuda parallel: posterior %d is %f %f\n", i, posterior[i].component[0], posterior[i].component[1]);
        }
#endif

        //cudaFree(device_chunk);
        //cudaFree(device_control);
    }
#if 0
old version
        int32_t current_chunk_id = 0;

        printf("Processing %d chunks simultaneously\n", SIMULTANEOUS_KERNELS);
        // fire off 16-32 threads; that'll make the logic simpler for you

        double *device_chunks[SIMULTANEOUS_KERNELS];
        control_t *device_controls[SIMULTANEOUS_KERNELS];
        posterior_t *device_posteriors[SIMULTANEOUS_KERNELS];
        control_t host_controls[NUM_CHUNKS];
        
        for (unsigned i = 0; i < SIMULTANEOUS_KERNELS; i++)
        {
            checkCudaErrors(cudaMalloc((void **)&(device_chunks[i]), CHUNK_BYTES));
            checkCudaErrors(cudaMalloc((void **)&(device_controls[i]), CONTROL_BYTES));
            checkCudaErrors(cudaMalloc((void **)&(device_posteriors[i]), POSTERIOR_CHUNK_BYTES));
        }

        pthread_t threads[SIMULTANEOUS_KERNELS];
        thread_args_t args[SIMULTANEOUS_KERNELS];

        gettimeofday(&start_time, 0);

        // launch threads
        for (unsigned i = 0; i < SIMULTANEOUS_KERNELS; i++)
        {
            // set up arguments
            args[i].thread_id = i;
            args[i].device_chunk = device_chunks[i];
            args[i].device_posterior = device_posteriors[i];
            args[i].device_control = device_controls[i];
            args[i].current_chunk_id = &current_chunk_id;
            args[i].host_dataset = dataset;
            args[i].host_controls = host_controls;
            args[i].host_posterior = posterior;

            int rc = pthread_create(&threads[i], NULL, thread, (void *)&args[i]);
            assert(rc == 0);
        }

        // join threads
        for (unsigned i = 0; i < SIMULTANEOUS_KERNELS; i++)
        {
            pthread_join(threads[i], NULL);
        }

        gettimeofday(&end_time, 0);
        printf("Processed %i chunks\n", NUM_CHUNKS);

        printf("cuda parallel: ");
        print_time_elapsed(start_time, end_time);
        
        for (unsigned i = 0; i < 4; i++)
        {
            // printf("cuda parallel: chunk %d sum is %f\n", i, host_controls[i].sum);
            // FIXME: hardcoded component numbers
            printf("cuda parallel: posterior %d is %f %f\n", i, posterior[i].component[0], posterior[i].component[1]);
        }

        //cudaFree(device_chunk);
        //cudaFree(device_control);
    }
#endif
}

void *thread(void *void_args)
{
    thread_args_t *args = (thread_args_t *)void_args;
    // cudaStream_t stream;

    // cudaSetDevice(0); // TODO: adjust when we have multiple GPUs; probably assign a new group of threads to each GPU

    // checkCudaErrors(cudaStreamCreate(&stream));

    // int32_t chunk_id = OSAtomicIncrement32(args->current_chunk_id);
    // TODO: I'm not convinced that this is working. 
    // TODO: we're definitely missing chunk 0; so bodge around it here
    // chunk_id--;

    // while (chunk_id < NUM_CHUNKS)
    {
        // double *host_chunk_ptr = &args->host_dataset[chunk_id * CHUNK_ENTRIES];
        // posterior_t *host_posterior_ptr = &args->host_posterior[chunk_id * CHUNK_ENTRIES];

        // printf("thread %d chunk %d host_ptr %p\n", args->thread_id, chunk_id, host_chunk_ptr);

        // cudaMemcpyAsync(args->device_chunk, host_chunk_ptr, CHUNK_BYTES, cudaMemcpyHostToDevice, stream);
        // cudaStreamSynchronize(stream);

        // do work on gpu
        stream(args->host_dataset, args->current_chunk_id);

        // These small copies are relatively inefficient. Computation time dominates, however, so it hardly matters.
        // cudaMemcpyAsync(&(args->host_controls[chunk_id]), args->device_control, sizeof(control_t), cudaMemcpyDeviceToHost, stream);
        // TODO: check that you can fire off two copies like this without synchronizing
        // cudaMemcpyAsync(host_posterior_ptr, args->device_posterior, POSTERIOR_CHUNK_BYTES, cudaMemcpyDeviceToHost, stream);
        // cudaStreamSynchronize(stream);

        // TODO check return value

        // printf("chunk %d complete\n", chunk_id);

        // FIXME BODGE
        // chunk_id = OSAtomicIncrement32(args->current_chunk_id);
        // chunk_id--;
    }

    printf("thread %d complete\n", args->thread_id);
    return 0;
}

#if 0
void *old_thread(void *void_args)
{
    thread_args_t *args = (thread_args_t *)void_args;
    cudaStream_t stream;

    cudaSetDevice(0); // TODO: adjust when we have multiple GPUs; probably assign a new group of threads to each GPU

    checkCudaErrors(cudaStreamCreate(&stream));

    int32_t chunk_id = OSAtomicIncrement32(args->current_chunk_id);
    // TODO: I'm not convinced that this is working. 
    // TODO: we're definitely missing chunk 0; so bodge around it here
    chunk_id--;

    while (chunk_id < NUM_CHUNKS)
    {
        double *host_chunk_ptr = &args->host_dataset[chunk_id * CHUNK_ENTRIES];
        posterior_t *host_posterior_ptr = &args->host_posterior[chunk_id * CHUNK_ENTRIES];

        // printf("thread %d chunk %d host_ptr %p\n", args->thread_id, chunk_id, host_chunk_ptr);

        cudaMemcpyAsync(args->device_chunk, host_chunk_ptr, CHUNK_BYTES, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        // do work on gpu
        inverse_gaussian_em_kernel<<<args->gridSize, args->blockSize, 0, stream>>>(args->device_chunk, args->device_control, args->device_posterior);
        cudaStreamSynchronize(stream);

        // These small copies are relatively inefficient. Computation time dominates, however, so it hardly matters.
        cudaMemcpyAsync(&(args->host_controls[chunk_id]), args->device_control, sizeof(control_t), cudaMemcpyDeviceToHost, stream);
        // TODO: check that you can fire off two copies like this without synchronizing
        cudaMemcpyAsync(host_posterior_ptr, args->device_posterior, POSTERIOR_CHUNK_BYTES, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // TODO check return value

        printf("chunk %d complete\n", chunk_id);

        // FIXME BODGE
        chunk_id = OSAtomicIncrement32(args->current_chunk_id);
        chunk_id--;
    }

    printf("thread %d complete\n", args->thread_id);
    return 0;
}
#endif


