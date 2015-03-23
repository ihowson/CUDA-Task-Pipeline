// serial CUDA using Bulk
// we're going for CORRECTNESS right now and speed later

// most documentation seems to be in the GPU Tech Conf presentation:
// http://on-demand.gputechconf.com/gtc/2014/presentations/S4673-thrust-parallel-algorithms-bulk.pdf

#include <math_constants.h>
#include <iostream>

// NOTE: the order of includes seems to matter. Jarod's code puts bulk first, then thrust.
#include <thrust/device_vector.h>
#include <bulk/bulk.hpp>

// TODO: this is only used for checkCudaErrors - try to remove it
#include "helper_cuda.h"

#include "common.h"

__device__ double dinvgauss(double x, double mu, double lambda)
{
    // TODO would be nice to assert that x > 0 and lambda = 0

    double x_minus_mu = x - mu;
    return sqrt(lambda / (2 * CUDART_PI * pow(x, 3.0))) * exp((-lambda * x_minus_mu * x_minus_mu) / (2 * mu * mu * x));
    // http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.invgauss.html
    // invgauss.pdf(x, mu) = 1 / sqrt(2*pi*x**3) * exp(-(x-mu)**2/(2*x*mu**2))
}

struct member_prob
{
    /*
     * Calculate membership probability (TODO equation) for each observation.
     * 
     * params is an M-element array of component parameters
     * chunk is an N-element array of observations
     * member_prob is an MxN matrix of probabilities
     * sum_prob is a single value
     */
    __device__ void operator()(bulk::agent<> &self, invgauss_params_t *params, double *chunk, double *member_prob)
    {
        unsigned t = self.index();

        // This could be expressed more parallel, but we have plenty of parallelism already across tasks; this is much easier to understand
        // it is a little awkward that we're continually running for loops over the components; a matrix-like representation would be cleaner. We don't actually need matrix ops, though - we only do element-wise multiply and col/row sums.
        double weighted_prob[M]; // per-component weighted sum of probabilities
        for (unsigned m = 0; m < M; m++)
        {
            // calculate p(l|x_i, Theta^g)
            double x = chunk[m * N + t];
            double x_prob = dinvgauss(x, params[m].mu, params[m].lambda);
            weighted_prob[m] = params[m].alpha * x_prob;
        }

        double sum_prob = 0.0f;
        for (unsigned m = 0; m < M; m++)
        {
            sum_prob += weighted_prob[m];
        }

        for (unsigned m = 0; m < M; m++)
        {
            member_prob[m * N + t] = weighted_prob[m] / sum_prob;
        }
    }
};

// copied from https://github.com/jaredhoberock/bulk/blob/92b634e7f3def8c8e852e4bfcc4f6a4c74d0f465/sum.cu
// changed instances of 'int' to 'double'
struct sum_double
{
    __device__ void operator()(bulk::concurrent_group<> &g, thrust::device_ptr<double> data, thrust::device_ptr<double> result)
    {
        unsigned int n = g.size();

        // allocate some special memory that the group can use for fast communication
        double *s_data = static_cast<double*>(bulk::malloc(g, n * sizeof(int)));

        // the whole group cooperatively copies the data
        bulk::copy_n(g, data, n, s_data);

        while(n > 1)
        {
            unsigned int half_n = n / 2;

            if(g.this_exec.index() < half_n)
            {
                s_data[g.this_exec.index()] += s_data[n - g.this_exec.index() - 1];
            }

            // the group synchronizes after each update
            g.wait();

            n -= half_n;
        }

        if(g.this_exec.index() == 0)
        {
            *result = s_data[0];
        }

        // wait for agent 0 to store the result
        g.wait();

        // free the memory cooperatively
        bulk::free(g, s_data);
    }
};

// TODO Thrust provides these and other common functors like plus and multiplies in the file thrust/functional.h.
struct elementwise_multiply
{
    __device__ void operator()(bulk::agent<> &self, double *left, double *right, double *output)
    {
        unsigned t = self.index();
        output[t] = left[t] * right[t];
    }
};

struct subtract_scalar
{
    __device__ void operator()(bulk::agent<> &self, double *left, double right, double *output)
    {
        unsigned t = self.index();
        output[t] = left[t] - right;
    }
};

struct lambda_element
{
    __device__ void operator()(bulk::agent<> &self, double *chunk, double mu, double *dev_member_prob, double *temp)
    {
        unsigned t = self.index();
        double x = chunk[t];
        double x_minus_mu = x - mu;

        temp[t] = x_minus_mu * x_minus_mu * dev_member_prob[t] / (mu * mu * x);
    }
};

void serial_bulk(double *dataset)
{
    // persistent device memory
    // allocated out here to avoid cudaMalloc in inner loop
    thrust::device_vector<int> dev_chunka(N);
    thrust::device_vector<double> dev_chunk(N);
    thrust::device_vector<double> dev_posterior(N);
    thrust::device_vector<invgauss_params_t> dev_params(M);

    invgauss_params_t params_new[M];

    // starting parameters
    invgauss_params_t start_params[M];
    for (unsigned m = 0; m < M; m++)
    {
        start_params[m].mu = 0.99 + 0.02 * m;
        start_params[m].lambda = 1.0;
        start_params[m].alpha = 0.5;
    }

    thrust::device_vector<double> dev_member_prob(M * N);
    thrust::device_vector<double> dev_member_prob_times_x(M * N);
    thrust::device_vector<double> dev_member_prob_times_x_sum(M);
    thrust::device_vector<double> dev_x_minus_mu(N);
    thrust::device_vector<double> dev_temp(N);

    for (unsigned n = 0; n < NUM_CHUNKS; n++)
    {
        // init device parameters
        for (unsigned m = 0; m < M; m++)
        {
            params_new[m] = start_params[m];
        }

        // copy chunk to device
        double *chunk_host_ptr = &dataset[n * CHUNK_ENTRIES];
        dev_chunk.assign(chunk_host_ptr, chunk_host_ptr + CHUNK_BYTES);

        // run EM algorithm
        unsigned iteration = 0; // FIXME: nasty. Better to explicitly count.
        for (; iteration < MAX_ITERATIONS; iteration++)
        {
            dev_params.assign(params_new, params_new + M);

            //////// PROCESS CHUNK
            // generate member_prob (TODO equation) for each observation
            bulk::async(
                bulk::par(N),
                member_prob(),
                bulk::root.this_exec,
                thrust::raw_pointer_cast(dev_params.data()),
                thrust::raw_pointer_cast(dev_chunk.data()),
                thrust::raw_pointer_cast(dev_member_prob.data())).wait();


/////TEST ONLY
            // try to get a summation working
            // note that dev_x_minus_mu is undefined at this point
            thrust::device_vector<double> result(1);
            bulk::async(
                bulk::con(N),
                sum_double(),
                bulk::root,
                dev_x_minus_mu.data(),
                result.data());
#if 0
    having trouble getting summations to work, below
            // TODO: at this point, there are no interdependencies between components. You could process the whole thing in parallel and remove this for loop, if you wanted.
            for (unsigned m = 0; m < M; m++)
            {
                thrust::device_vector<double> dev_member_prob_sum(1);

                // column sum member_prob and put results in member_prob_sum
                // FIXME: this pointer arithmetic is not ideal
                // TODO: no reason we can't run all of these sums in parallel instead of using a for loop
                bulk::async(
                    bulk::con(N),
                    sum(),
                    bulk::root,
                    thrust::raw_pointer_cast(dev_member_prob.data()), i don't think this is right - it doesn't reindex for component m
                    thrust::raw_pointer_cast(dev_member_prob_sum.data()));

                // calculate new alpha
                params_new[m].alpha = dev_member_prob_sum[0] / N;

                // TODO: is there a better way to do this? In the raw C implementation, we can do the multiply on-GPU a moment before the sum and save a bunch of DRAM.
                // element-wise multiply member_prob by x (we will sum these to esimate new mu)
                bulk::async(
                    bulk::par(N),
                    elementwise_multiply(),
                    bulk::root.this_exec,
                    thrust::raw_pointer_cast(dev_chunk.data()),
                    thrust::raw_pointer_cast(dev_member_prob.data()),
                    thrust::raw_pointer_cast(dev_member_prob_times_x.data())).wait();

                // column sum member_prob_times_x
                bulk::async(
                    bulk::con(N),
                    sum(),
                    bulk::root,
                    thrust::raw_pointer_cast(dev_member_prob_times_x.data() + m * N),
                    thrust::raw_pointer_cast(dev_member_prob_times_x_sum.data() + m));

                // calculate new mu
                params_new[m].mu = dev_member_prob_times_x_sum[m] / dev_member_prob_sum[0];

                // copy current device parameters to host to feed into next kernel
                invgauss_params_t p = dev_params[m];

                // calculate new lambda (input to summation)
                bulk::async(bulk::par(N), lambda_element(), bulk::root.this_exec,
                    thrust::raw_pointer_cast(dev_chunk.data()),
                    p.mu,
                    thrust::raw_pointer_cast(dev_member_prob.data() + m),
                    thrust::raw_pointer_cast(dev_temp.data())).wait();

                // perform summation
                thrust::device_vector<double> sumresult(1);
                double sumtemp;
                bulk::async(
                    bulk::con(N),
                    sum(),
                    bulk::root,
                    thrust::raw_pointer_cast(dev_temp.data()),
                    thrust::raw_pointer_cast(sumresult.data()));
                sumtemp = sumresult[0];
                params_new[m].lambda = dev_member_prob_sum[0] / sumtemp;
            }
#endif

/*
                // TODO have we converged?
                if (t < M)
                {
                    params[t] = params_new[t];
                }
                */
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

            // calculate new mu

        */
        }


        // copy result to host
        printf("chunk %d\n", n);
        for (unsigned m = 0; m < M; m++)
        {
            invgauss_params_t params = dev_params[m];
            printf("\tcomponent %d: fit mu = %f, lambda = %f, alpha = %f\n", m, params.mu, params.lambda, params.alpha);
        }
        printf("\n");
    }
}
