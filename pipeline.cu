#include <assert.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include "helper_cuda.h"

// SMS=30 dbg=1 make

// This is OS X specific, unfortunately; haven't found a good cross-platform way to do this
// C++ <atomic> doesn't work on Mac for whatever reason
#include <libkern/OSAtomic.h>


typedef struct _output_t {
    double sum;
    double mean;
    double variance;
} output_t;

void *thread(void *void_args);
// __global__ void simple_kernel(double *g_chunk, output_t *g_output);

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
// #define SIMULTANEOUS_KERNELS 32
#define SIMULTANEOUS_KERNELS 4
//#define SIMULTANEOUS_KERNELS 2


typedef struct _thread_args_t
{
    unsigned thread_id;
    double *device_chunk; // this is a device pointer
    output_t *device_output; // this is a device pointer
    double *host_dataset; // full dataset stored on host
    output_t *host_outputs; // outputs stored on host
    int *current_chunk_id; // pointer to shared chunk_id counter
} thread_args_t;

#define NUM_CHUNKS 40000
#define CHUNK_ENTRIES 2000
#define CHUNK_BYTES (CHUNK_ENTRIES * sizeof(double))
#define DATASET_ENTRIES (CHUNK_ENTRIES * NUM_CHUNKS)
#define DATASET_BYTES (sizeof(double) * DATASET_ENTRIES)

#define OUTPUT_BYTES (NUM_CHUNKS * sizeof(output_t))

#define GRID 16
#define BLOCK 16

__global__ void simple_kernel(double *g_chunk)//, output_t *g_output)
{
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



    // perform some computations
    double foobar = 0.0;
    for (unsigned i = 0; i < 1000; i++)
    {
        foobar = foobar + 1 * 1.2;
        // sdata[tid] += (float) num_threads * sdata[tid];
    }
    __syncthreads();

    // write data to global memory
    // the g_output pointer is empty right now
    // g_output->mean = sdata[tid];
    // g_odata[tid] = sdata[tid];
}

int main(int argc, char **argv)
{
    int num_gpus;
    checkCudaErrors(cudaGetDeviceCount(&num_gpus));
    // cudaGetDeviceCount(&num_gpus);

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

    printf("dataset is %lu bytes\n", DATASET_BYTES);

    // printf("cuda error = %d\n", error);

    // fill it in with Uniform(0, 1)
    for (uint64_t i = 0; i < DATASET_ENTRIES; i++)
    {
        dataset[i] = (double)rand() / (double)INT_MAX;
    }

    printf("The first few entries look like %f %f %f.\n", dataset[0], dataset[1], dataset[2]);

    // make space for output results
    output_t *outputs = (output_t *)calloc(NUM_CHUNKS, sizeof(output_t));

    struct timeval start_time, end_time;
    double elapsed_time;
// #if 0 // DISABLED to make my life easier
    printf("Processing chunks sequentially\n");

    double *device_chunk;
    output_t *device_output;
    // TODO check errors
    checkCudaErrors(cudaMalloc(&device_chunk, CHUNK_BYTES));
    checkCudaErrors(cudaMalloc(&device_output, OUTPUT_BYTES));

    gettimeofday(&start_time, 0);

    for (unsigned i = 0; i < NUM_CHUNKS; i++)
    {
        //printf("i = %d\n", i);
        // copy data to gpu
        checkCudaErrors(cudaMemcpy(device_chunk, &dataset[i * CHUNK_ENTRIES], CHUNK_BYTES, cudaMemcpyHostToDevice));

        // do work on gpu
        simple_kernel<<<GRID, BLOCK>>>(device_chunk);//, device_output);

        // copy results to host
        checkCudaErrors(cudaMemcpy(&(outputs[i]), device_output, sizeof(output_t), cudaMemcpyDeviceToHost));
    }

    gettimeofday(&end_time, 0);
    elapsed_time = (1000000.0 * (end_time.tv_sec - start_time.tv_sec) + end_time.tv_usec - start_time.tv_usec) / 1000000.0;

    cudaFree(device_chunk);
    cudaFree(device_output);

    printf("Processed %i chunks in %fms\n", NUM_CHUNKS, elapsed_time);

    printf("TODO: dump a few outputs for verification\n");
// #endif



    ///////////////////////////////////

    int32_t current_chunk_id = 0;

    printf("Processing %d chunks simultaneously\n", SIMULTANEOUS_KERNELS);
    // fire off 16-32 threads; that'll make the logic simpler for you

    double *device_chunks[SIMULTANEOUS_KERNELS];
    output_t *device_outputs[SIMULTANEOUS_KERNELS];
    output_t host_outputs[NUM_CHUNKS];
    
    for (unsigned i = 0; i < SIMULTANEOUS_KERNELS; i++)
    {
        checkCudaErrors(cudaMalloc((void **)&(device_chunks[i]), CHUNK_BYTES));
        checkCudaErrors(cudaMalloc((void **)&(device_outputs[i]), OUTPUT_BYTES));
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
        args[i].device_output = device_outputs[i];
        args[i].current_chunk_id = &current_chunk_id;
        args[i].host_dataset = dataset;
        args[i].host_outputs = host_outputs;

        int rc = pthread_create(&threads[i], NULL, thread, (void *)&args[i]);
        assert(rc == 0);
    }

    // join threads
    for (unsigned i = 0; i < SIMULTANEOUS_KERNELS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // TODO wait for threads to join

    gettimeofday(&end_time, 0);
    elapsed_time = (1000000.0 * (end_time.tv_sec - start_time.tv_sec) + end_time.tv_usec - start_time.tv_usec) / 1000000.0;
    printf("Processed %i chunks in %fms\n", NUM_CHUNKS, elapsed_time);

    //cudaFree(device_chunk);
    //cudaFree(device_output);


}

void *thread(void *void_args)
{
    thread_args_t *args = (thread_args_t *)void_args;
    cudaError_t error;
    cudaStream_t stream;

    cudaSetDevice(0); // TODO: adjust when we have multiple GPUs; probably assign a new group of threads to each GPU

    error = cudaStreamCreate(&stream);

    if (cudaSuccess != error)
    {
        printf("error %d\n", error);
    } 
    assert(cudaSuccess == error);


    int32_t chunk_id = OSAtomicIncrement32(args->current_chunk_id);
    // TODO: I'm not convinced that this is working. 
    // TODO: we're definitely missing chunk 0; so bodge around it here
    chunk_id--;

    while (chunk_id < NUM_CHUNKS)
    {
        double *host_chunk_ptr = &args->host_dataset[chunk_id * CHUNK_ENTRIES];

        // printf("thread %d chunk %d host_ptr %p\n", args->thread_id, chunk_id, host_chunk_ptr);


        // printf("%d: pre copy\n", args->thread_id);
        cudaMemcpyAsync(args->device_chunk, host_chunk_ptr, CHUNK_BYTES, cudaMemcpyHostToDevice, stream);
        // printf("%d: pre-sync\n", args->thread_id);
        cudaStreamSynchronize(stream);
        // do work on gpu
        // printf("%d: launch\n", args->thread_id);
        // 2048 simultaneous launches?
        simple_kernel<<<GRID, BLOCK, 0, stream>>>(args->device_chunk);//, args->device_output);
        // simple_kernel<<< grid, threads, mem_size, ??? >>>(data, left, nright, depth+1);
        // kernel_call<<<dimGrid, dimBlock, 0>>>();
        cudaStreamSynchronize(stream);
        // printf("%d: sync\n", args->thread_id);
        // These small (16 byte) copies are relatively inefficient. Eventually we'll have all of the posterior probabilities to copy back, so it might not matter.
        cudaMemcpyAsync(&(args->host_outputs[chunk_id]), args->device_output, sizeof(output_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // TODO check return value


        // copy results to host
        //TODO 
        //cudaMemcpy(&(dataset[i * CHUNK_SIZE]), device_chunk, CHUNK_SIZE, cudaMemcpyDeviceToHost);
        /* 
        cudaMemcpyAsync ( dev1, host1, size, H2D, stream1 ) ;
        kernel2 <<< grid, block, 0, stream2 >>> ( …, dev2, … ) ;
        kernel3 <<< grid, block, 0, stream3 >>> ( …, dev3, … ) ;
        cudaMemcpyAsync ( host4, dev4, size, D2H, stream4 ) ;
        */
        // also check http://devblogs.nvidia.com/parallelforall/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
        //printf("TODO\n");

        //look at nvidia visual profiler 'nvvp' to visualise what is going on




        // FIXME BODGE
        chunk_id = OSAtomicIncrement32(args->current_chunk_id);
        chunk_id--;
    }

    printf("thread %d complete\n", args->thread_id);
    return 0;
}


