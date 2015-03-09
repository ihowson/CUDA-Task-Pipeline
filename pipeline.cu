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


// one of these per chunk
typedef struct _output_t {
    unsigned iterations;
    double log_likelihood;

    // fitted distribution parameters
    double mu;
    double lambda;
} output_t;

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


typedef struct _thread_args_t
{
    unsigned thread_id;
    double *device_chunk; // this is a device pointer
    double *device_posterior; // this is a device pointer
    output_t *device_output; // this is a device pointer
    double *host_dataset; // full dataset stored on host
    double *host_posterior; // 1:1 posterior probabilities for each observation
    output_t *host_outputs; // outputs stored on host
    int *current_chunk_id; // pointer to shared chunk_id counter

    // CUDA launch params
    int gridSize;
    int blockSize;
} thread_args_t;

#define NUM_CHUNKS 40000
#define CHUNK_ENTRIES 2000
#define CHUNK_BYTES (CHUNK_ENTRIES * sizeof(double))
#define DATASET_ENTRIES (CHUNK_ENTRIES * NUM_CHUNKS)
#define DATASET_BYTES (sizeof(double) * DATASET_ENTRIES)

#define OUTPUT_BYTES (NUM_CHUNKS * sizeof(output_t))

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

__global__ void simple_kernel(double *g_chunk, output_t *g_output, double *g_posterior)
{
    // TODO pass in chunk size as an arg rather than assuming CHUNK_ENTRIES
    // g_chunk: pointer to the chunk that we will process. It could be modified in-place by the kernel.
    // g_output: pointer to output struct for this chunk
    // unsigned len = CHUNK_ENTRIES;

    unsigned t = threadIdx.x;
    // g_posterior[t] = (double)t;

    g_posterior[t] = dinvgauss(g_chunk[t], 1, 1);

    /*
    double *input = g_chunk;

    ///////////////////
    // Adapted from https://gist.github.com/wh5a/4424992
    //@@ Load a segment of the input vector into shared memory
    __shared__ double partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;
    if (start + t < len)
        partialSum[t] = input[start + t];
    else
        partialSum[t] = 0;
    if (start + BLOCK_SIZE + t < len)
        partialSum[BLOCK_SIZE + t] = g_chunk[start + BLOCK_SIZE + t];
    else
        partialSum[BLOCK_SIZE + t] = 0;
    //@@ Traverse the reduction tree
    for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
        __syncthreads();
        if (t < stride)
            partialSum[t] += partialSum[t+stride];
    }
    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
    if (t == 0)
    {
        g_output->sum = partialSum[0];
    }
    ///////////////////
    */

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



    // m-step: try https://github.com/jwetzl/CudaLBFGS
    // https://code.google.com/p/lbfgsb-on-gpu/
    // or http://en.wikipedia.org/wiki/Inverse_Gaussian_distribution#Maximum_likelihood - are these the right equations?


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
    double *posterior;
    checkCudaErrors(cudaMallocHost(&posterior, DATASET_BYTES));

    printf("dataset is %lu bytes\n", DATASET_BYTES);

    // TODO you could fill it with inv gaussian data
    // fill it in with Uniform(0, 1)
    for (uint64_t i = 0; i < DATASET_ENTRIES; i++)
    {
        dataset[i] = (double)rand() / (double)INT_MAX;
    }

    printf("The first few entries look like %f %f %f.\n", dataset[0], dataset[1], dataset[2]);

    // cpu-calculated results
    output_t cpu_outputs[NUM_CHUNKS];

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
        // cpu_outputs[i].sum = total;
    }
    gettimeofday(&end_time, 0);

    printf("CPU: ");
    print_time_elapsed(start_time, end_time);
    
    // sum up the first few chunks for later verification
    for (unsigned i = 0; i < 4; i++)
    {
        // printf("cpu: chunk %d sum is %f\n", i, cpu_outputs[i].sum);
        printf("cpu: chunk %d posterior is %f\n", i, posterior[i]);
    }

    ///////////////////////////////////

    printf("Clearing posterior probabilities...\n");
    memset(posterior, 0, DATASET_BYTES);

    ///////////////////////////////////

    // make space for output results
    output_t *outputs = (output_t *)calloc(NUM_CHUNKS, sizeof(output_t));

    printf("Processing chunks sequentially\n");

    double *device_chunk;
    double *device_posterior;
    output_t *device_output;
    checkCudaErrors(cudaMalloc(&device_chunk, CHUNK_BYTES));
    checkCudaErrors(cudaMalloc(&device_posterior, CHUNK_BYTES));
    checkCudaErrors(cudaMalloc(&device_output, OUTPUT_BYTES));

    // http://stackoverflow.com/a/25010560/591483
    int blockSize;
    int minGridSize;

    // NOTE: update this when shared memory usage changes
    // assume one thread per observation
    int N = CHUNK_ENTRIES;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void *)simple_kernel, 0, N); 
    int gridSize = (N + blockSize - 1) / blockSize; 
    printf("gridSize=%d, blockSize=%d\n", gridSize, blockSize);

    assert(blockSize == BLOCK_SIZE);

    gettimeofday(&start_time, 0);

    for (unsigned i = 0; i < NUM_CHUNKS; i++)
    {
        //printf("i = %d\n", i);
        // copy data to gpu
        checkCudaErrors(cudaMemcpy(device_chunk, &dataset[i * CHUNK_ENTRIES], CHUNK_BYTES, cudaMemcpyHostToDevice));

        // do work on gpu
        simple_kernel<<<gridSize, blockSize>>>(device_chunk, device_output, device_posterior);

        // copy results to host
        // TODO: combine output_t and the posterior chunk to cut down on small memcpy? Probably ends up being trivial as we only do this once for each EM+data chunk
        checkCudaErrors(cudaMemcpy(&(outputs[i]), device_output, sizeof(output_t), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&(posterior[i * CHUNK_ENTRIES]), device_posterior, CHUNK_BYTES, cudaMemcpyDeviceToHost));
    }

    gettimeofday(&end_time, 0);

    for (unsigned i = 0; i < 4; i++)
    {
        // printf("cuda seq: chunk %d sum is %f\n", i, outputs[i].sum);
        printf("cuda seq: posterior %d is %f\n", i, posterior[i]);
    }

    printf("sequential CUDA: ");
    print_time_elapsed(start_time, end_time);

    cudaFree(device_chunk);
    cudaFree(device_output);
    cudaFree(device_posterior);

    printf("Processed %i chunks\n", NUM_CHUNKS);

    ///////////////////////////////////

    printf("Clearing posterior probabilities...\n");
    memset(posterior, 0, DATASET_BYTES);

    ///////////////////////////////////

    int32_t current_chunk_id = 0;

    printf("Processing %d chunks simultaneously\n", SIMULTANEOUS_KERNELS);
    // fire off 16-32 threads; that'll make the logic simpler for you

    double *device_chunks[SIMULTANEOUS_KERNELS];
    output_t *device_outputs[SIMULTANEOUS_KERNELS];
    double *device_posteriors[SIMULTANEOUS_KERNELS];
    output_t host_outputs[NUM_CHUNKS];
    
    for (unsigned i = 0; i < SIMULTANEOUS_KERNELS; i++)
    {
        checkCudaErrors(cudaMalloc((void **)&(device_chunks[i]), CHUNK_BYTES));
        checkCudaErrors(cudaMalloc((void **)&(device_outputs[i]), OUTPUT_BYTES));
        checkCudaErrors(cudaMalloc((void **)&(device_posteriors[i]), CHUNK_BYTES));
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
        args[i].device_output = device_outputs[i];
        args[i].current_chunk_id = &current_chunk_id;
        args[i].host_dataset = dataset;
        args[i].host_outputs = host_outputs;
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
        // printf("cuda parallel: chunk %d sum is %f\n", i, host_outputs[i].sum);
        printf("cuda parallel: posterior %d is %f\n", i, posterior[i]);
    }

    //cudaFree(device_chunk);
    //cudaFree(device_output);
}

void *thread(void *void_args)
{
    thread_args_t *args = (thread_args_t *)void_args;
    cudaError_t error;
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
        double *host_posterior_ptr = &args->host_posterior[chunk_id * CHUNK_ENTRIES];

        // printf("thread %d chunk %d host_ptr %p\n", args->thread_id, chunk_id, host_chunk_ptr);

        cudaMemcpyAsync(args->device_chunk, host_chunk_ptr, CHUNK_BYTES, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        // do work on gpu
        simple_kernel<<<args->gridSize, args->blockSize, 0, stream>>>(args->device_chunk, args->device_output, args->device_posterior);
        cudaStreamSynchronize(stream);

        // These small (24 byte) copies are relatively inefficient. Eventually we'll have all of the posterior probabilities to copy back, so it might not matter.
        cudaMemcpyAsync(&(args->host_outputs[chunk_id]), args->device_output, sizeof(output_t), cudaMemcpyDeviceToHost, stream);
        // TODO: check that you can fire off two copies like this without synchronizing
        cudaMemcpyAsync(host_posterior_ptr, args->device_posterior, CHUNK_BYTES, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // TODO check return value

        // FIXME BODGE
        chunk_id = OSAtomicIncrement32(args->current_chunk_id);
        chunk_id--;
    }

    printf("thread %d complete\n", args->thread_id);
    return 0;
}


