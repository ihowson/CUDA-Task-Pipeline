#include <assert.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <math_constants.h>
#include "helper_cuda.h"

// SMS=30 dbg=1 make

// This is OS X specific, unfortunately; haven't found a good cross-platform way to do this
// C++ <atomic> doesn't work on Mac for whatever reason
#include <libkern/OSAtomic.h>

// consider electricfence
// http://lh3lh3.users.sourceforge.net/memdebug.shtml
// http://thingsilearned.com/2007/09/30/electric-fence-on-os-x/

#define NUM_MIXTURE_COMPONENTS 2
#define M NUM_MIXTURE_COMPONENTS
// FIXME: should be set within the kernel func
#define N CHUNK_ENTRIES

// posterior probability for each component
typedef struct _posterior_t
{
    double component[NUM_MIXTURE_COMPONENTS];
} posterior_t;

int g_paronly = 0;

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
//#define SIMULTANEOUS_KERNELS 4


// TODO: this should be configurable and/or read from command line params
#define MAX_ITERATIONS 100

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

    // CUDA launch params
    int gridSize;
    int blockSize;
} thread_args_t;

// FIXME: 40kchunks is taking too long right now
//#define NUM_CHUNKS 40000
#define NUM_CHUNKS 4000
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


// density of the inverse gaussian distribution
__host__ __device__ double dinvgauss(double x, double mu, double lambda)
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

// FIXME: this should be somewhat dynamic depending on number of threads/data elements
#define ELEM_2 2048

/* 
    int NearestPowerOf2 (int n)
    {
    if (!n) return n;  //(0 == 2^0)
  
    int x = 1;
    while(x < n)
    {
        x <<= 1;
    }
    return x;
}
 */

// sum the values in temp; result goes into temp[0]
// temp must have BLOCK_SIZE elements (even if there are less threads)
// temp must be in __shared__ memory
// n is number of data elements (TODO: this is ignored for now)
// note that this DESTROYS the contents of temp
__device__ void sum_reduce(double *temp, unsigned n)
{
    int thread2;

    // FIXME: KLUDGE: hardcoded number of threads
    int nTotalThreads = ELEM_2;
    //int nTotalThreads = blockDim_2; // Total number of threads, rounded up to the next power of two
    // supposed to use blockDim.x to determine number of threads

    while (nTotalThreads > 1)
    {
        int halfPoint = (nTotalThreads >> 1); // divide by two
        // only the first half of the threads will be active.

        if (threadIdx.x < halfPoint)
        {
            thread2 = threadIdx.x + halfPoint;

            // Skipping the fictious threads blockDim.x ... blockDim_2-1
            if (thread2 < blockDim.x)
            {
                // Get the shared value stored by another thread
                temp[threadIdx.x] += temp[thread2];
            }
        }
        __syncthreads();

        // Reducing the binary tree size by two:
        nTotalThreads = halfPoint;
    }
}

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
            x_prob[m] = dinvgauss(x, params[m].mu, params[m].lambda);
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
            sum_reduce(temp, CHUNK_ENTRIES);

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
            sum_reduce(temp, CHUNK_ENTRIES);

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
            sum_reduce(temp, CHUNK_ENTRIES);

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
    iterations 
    loglike
    params
    */

    // TODO: also calculate membership probabilities and copy them back to global memory for collection by host


    // there's also https://code.google.com/p/stanford-cs193g-sp2010/source/browse/trunk/tutorials/sum_reduction.cu but it requires new kernel launches
    // this is also a nice simple alternative: http://stackoverflow.com/a/15162163/591483 - but does it work?
    ///// modified SUM REDUCTION from https://www.sharcnet.ca/help/index.php/CUDA_tips_and_tricks
    // better if we use CUB or something

    // implement E-step: calculate prior probabilities
    // implement E-step: calculate Q

    // implement M-step: calculate new alpha for each component
    // implement M-step: calculate new lambda for each component
    // implement M-step: calculate new mu for each component
    // implement E-step: calculate posterior probabilities (or do we do this for the new parameters? or maybe just when we converge)

}


    // TODO: calculate log-likelihood of each observation given the component parameters

    // e-step:
    // gamma.ll <- function(theta, z,lambda, k) -sum(z*log(dens(lambda,theta,k)))
    // dens1=dens(lambda,theta,k)
    // z=dens1/apply(dens1,1,sum)
    // posterior = z

    // dens <- function(lambda, theta, k){
    //     temp<-NULL
    //     alpha=theta[1:k]
    //     beta=theta[(k+1):(2*k)]
    //     for(j in 1:k){
    //      temp=cbind(temp,dgamma(x,shape=alpha[j],scale=beta[j]))  }
    //     temp=t(lambda*t(temp))
    //     temp
    // }




    /*
    // shared memory
    // the size is determined by the host application
    extern  __shared__  float sdata[];

    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;

    // read in input data from global memory
    sdata[tid] = g_chunk[tid];
    __syncthreads();
    */

// TODO: check syncthreads usage carefully

void print_time_elapsed(struct timeval start_time, struct timeval end_time)
{
    double elapsed_time = (1000000.0 * (end_time.tv_sec - start_time.tv_sec) + end_time.tv_usec - start_time.tv_usec) / 1000000.0;
    printf("%f seconds elapsed\n", elapsed_time);
}

int main(int argc, char **argv)
{
    struct timeval start_time, end_time;
    int num_gpus;

    for (unsigned i = 0; i < argc; i++)
    {
        if (0 == strcmp(argv[i], "--paronly"))
            g_paronly = 1;
    }

    checkCudaErrors(cudaGetDeviceCount(&num_gpus));

    assert(num_gpus == 1); // don't know what to do with multiple GPUs yet
    printf("There are %d GPUs\n", num_gpus);

    printf("Making up some data...\n");

    // For now, I'm assuming that the entire dataset resides within host RAM, in one single huge block

    // you might also consider unified memory: http://devblogs.nvidia.com/parallelforall/unified-memory-in-cuda-6/
    // probably lower performance, but easier programming model

    // this is pinned RAM, so bad things will happen if you allocate too much
    // (there's no paging; either the system will die or this process will be
    // OOM killed)
    // pinned RAM is a requirement for CUDA stream concurrency
    double *dataset;
    checkCudaErrors(cudaMallocHost(&dataset, DATASET_BYTES));

    // for each input observation, we generate a posterior probability
    posterior_t *posterior;
    checkCudaErrors(cudaMallocHost(&posterior, ALL_POSTERIOR_BYTES));

    printf("dataset is %lu bytes\n", DATASET_BYTES);

    // TODO you could fill it with inv gaussian data
    // fill it in with Uniform(0, 1)
    for (uint64_t i = 0; i < DATASET_ENTRIES; i++)
    {
        dataset[i] = (double)rand() / (double)INT_MAX;
    }

    printf("The first few entries look like %f %f %f.\n", dataset[0], dataset[1], dataset[2]);

    // http://stackoverflow.com/a/25010560/591483
    int blockSize;
    int minGridSize;

    // NOTE: update this - shared memory usage has changed
    // assume one thread per observation
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)inverse_gaussian_em_kernel, 0, N); 
    int gridSize = (N + blockSize - 1) / blockSize; 
    printf("gridSize=%d, blockSize=%d\n", gridSize, blockSize);

    assert(blockSize == BLOCK_SIZE);

    if (g_paronly == 0)
    {
        printf("CPU calculation is disabled; it's not implemented yet\n");
        /*
        // cpu-calculated results
        control_t cpu_controls[NUM_CHUNKS];

        gettimeofday(&start_time, 0);
        for (unsigned i = 0; i < NUM_CHUNKS; i++)
        {
            // double total = 0.0;

            for (unsigned j = 0; j < CHUNK_ENTRIES; j++)
            {
                unsigned global_index = i * CHUNK_ENTRIES + j;

                // posterior[global_index] += dataset[global_index];
                posterior[global_index] = dinvgauss(dataset[global_index], 1.0, 1.0);
            }

            // TODO you could calculate llik here
            // cpu_controls[i].sum = total;
        }
        gettimeofday(&end_time, 0);

        printf("CPU: ");
        print_time_elapsed(start_time, end_time);
        
        // sum up the first few chunks for later verification
        for (unsigned i = 0; i < 4; i++)
        {
            // printf("cpu: chunk %d sum is %f\n", i, cpu_controls[i].sum);
            // FIXME: hardcoded component numbers
            printf("cpu: chunk %d posterior is %f %f\n", i, posterior[i].component[0], posterior[i].component[1]);
        }
        */

        ///////////////////////////////////

        printf("Clearing posterior probabilities...\n");
        memset(posterior, 0, ALL_POSTERIOR_BYTES);

        ///////////////////////////////////

        // make space for control results
        control_t *controls = (control_t *)calloc(NUM_CHUNKS, sizeof(control_t));

        printf("Processing chunks sequentially\n");

        double *device_chunk;
        posterior_t *device_posterior;
        control_t *device_control;
        checkCudaErrors(cudaMalloc(&device_chunk, CHUNK_BYTES));
        checkCudaErrors(cudaMalloc(&device_posterior, POSTERIOR_CHUNK_BYTES));
        checkCudaErrors(cudaMalloc(&device_control, CONTROL_BYTES));

        gettimeofday(&start_time, 0);

        for (unsigned i = 0; i < NUM_CHUNKS; i++)
        {
            //printf("i = %d\n", i);
            // copy data to gpu
            checkCudaErrors(cudaMemcpy(device_chunk, &dataset[i * CHUNK_ENTRIES], CHUNK_BYTES, cudaMemcpyHostToDevice));

            // do work on gpu
            inverse_gaussian_em_kernel<<<gridSize, blockSize>>>(device_chunk, device_control, device_posterior);

            // copy results to host
            // TODO: combine control_t and the posterior chunk to cut down on small memcpy? Probably ends up being trivial as we only do this once for each EM+data chunk
            checkCudaErrors(cudaMemcpy(&(controls[i]), device_control, sizeof(control_t), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(&(posterior[i * CHUNK_ENTRIES]), device_posterior, POSTERIOR_CHUNK_BYTES, cudaMemcpyDeviceToHost));
        }

        gettimeofday(&end_time, 0);

        for (unsigned i = 0; i < 4; i++)
        {
            // printf("cuda seq: chunk %d sum is %f\n", i, controls[i].sum);
            // FIXME: hardcoded component numbers
            printf("cuda seq: posterior %d is %f %f\n", i, posterior[i].component[0], posterior[i].component[1]);
        }

        printf("sequential CUDA: ");
        print_time_elapsed(start_time, end_time);

        cudaFree(device_chunk);
        cudaFree(device_control);
        cudaFree(device_posterior);

        printf("Processed %i chunks\n", NUM_CHUNKS);

        ///////////////////////////////////

        printf("Clearing posterior probabilities...\n");
        memset(posterior, 0, ALL_POSTERIOR_BYTES);

        ///////////////////////////////////
    }

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
        args[i].gridSize = gridSize;
        args[i].blockSize = blockSize;

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

void *thread(void *void_args)
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


