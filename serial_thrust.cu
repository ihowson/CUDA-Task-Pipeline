// serial CUDA using Thrust

#include <math_constants.h>
#include <iostream>

#include <thrust/device_vector.h>

// TODO: this is only used for checkCudaErrors - try to remove it
#include "helper_cuda.h"

#include "common.h"

to test this, it might be best to load a CSV with test data and do an A/B comparison against the R implementation

__host__ __device__ double dinvgauss(double x, double mu, double lambda)
{
    // TODO would be nice to assert that x > 0 and lambda = 0

    double x_minus_mu = x - mu;
    return sqrt(lambda / (2 * CUDART_PI * pow(x, 3.0))) * exp((-lambda * x_minus_mu * x_minus_mu) / (2 * mu * mu * x));
    // http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.invgauss.html
    // invgauss.pdf(x, mu) = 1 / sqrt(2*pi*x**3) * exp(-(x-mu)**2/(2*x*mu**2))
}

#if 0
struct member_prob_functor
{
    /*
     * Calculate membership probability (TODO equation) for each observation.
     * 
     * x is a pointer to the first input x. We know that there are N 
     * params is an M-element array of component parameters
     * chunk is an N-element array of observations
     * member_prob is an MxN matrix of probabilities
     */
    __host__ __device__
    void operator()(const double *x, const invgauss_params_t& params) const
        // bulk::agent<> &self, invgauss_params_t *params, double *chunk, double *member_prob)
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
#endif

#if 0
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
#endif

/*
// arg1, arg2, result
struct x_prob_functor : public thrust::binary_function<invgauss_params_t, double, double>
{
    const invgauss_params_t params;

    __host__ __device__
    x_prob_functor(invgauss_params_t _params) : params(_params) {}

    __host__ __device__
    double operator()(const invgauss_params_t& params, const double& x) const { 
        return dinvgauss(x, params.mu, params.lambda);
    }
};
*/


// x.prob <- dinvgauss(x_expanded, mean=mu_expanded, shape=lambda_expanded)  # N x M matrix
// weighted.prob <- alpha_expanded * x.prob  # per-component weighted sum of probabilities (Nx2)
struct weighted_prob_functor
{
    const invgauss_params_t params;

    // __host__ __device__
    // TODO: is this implicitly copying from host memory?
    weighted_prob_functor(invgauss_params_t _params) : params(_params) {}

    __host__ __device__
    double operator()(const double& x) const
    { 
        return params.alpha * dinvgauss(x, params.mu, params.lambda);
    }
};

void serial_thrust(double *dataset)
{
    // persistent device memory
    // allocated out here to avoid cudaMalloc in main loop
    thrust::device_vector<double> dev_chunk(N);
    thrust::device_vector<invgauss_params_t> dev_params(M);
    thrust::device_vector<double> dev_posterior(N);

    // Lots of computations use 2D matrices.
    // - We don't use BLAS as there are no mat-mults, only elementwise mult/add
    // - We store in standard vectors to simplify expression
    // - We store columns grouped together as this is the common access pattern
    // - We often use for loops as (for the target dataset) N ~= 2000 and this is
    //   already more parallel than we have capacity for. We also have many
    //   datasets to run and the infrastructure to run tasks in parallel, which
    //   covers up many sins.

    thrust::device_vector<double> dev_x_prob(M * N);
    thrust::device_vector<double> dev_weighted_prob(M * N);
    thrust::device_vector<double> dev_sum_prob(N);
    thrust::device_vector<double> dev_member_prob(M * N);
    thrust::device_vector<double> dev_member_prob_times_x(M * N);

    // thrust::device_vector<double> dev_member_prob_sum(M);
    /*
    device_vector<double> dev_member_prob_times_x_sum(M);
    device_vector<double> dev_x_minus_mu(N);
    device_vector<double> dev_temp(N);
    */

    thrust::host_vector<invgauss_params_t> params_new(M);

    // starting parameters
    thrust::host_vector<invgauss_params_t> start_params(M);
    for (unsigned m = 0; m < M; m++)
    {
        start_params[m].mu = 0.99 + 0.02 * m;
        start_params[m].lambda = 1.0;
        start_params[m].alpha = 0.5;
    }

    // FIXME FIXME: we're getting a crash on the last 8; presumably we have an out-of-range memory access
    for (unsigned n = 0; n < NUM_CHUNKS - 10; n++)
    // for (unsigned n = 0; n < NUM_CHUNKS; n++)
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
            dev_params = params_new;

            //////// PROCESS CHUNK

            thrust::fill(dev_sum_prob.begin(), dev_sum_prob.end(), 0.0);

            // x.prob <- dinvgauss(x_expanded, mean=mu_expanded, shape=lambda_expanded)  # N x M matrix
            // and
            // weighted.prob <- alpha_expanded * x.prob  # per-component weighted sum of probabilities (Nx2)
            // combined
            // could be parallelised further but there's no need (see design tradeoffs)
            for (int m = 0; m < M; m++)
            {
                thrust::transform(
                    dev_chunk.begin(), dev_chunk.end(), // input
                    dev_weighted_prob.begin() + m * N, // output
                    // TODO: is this implicitly copying from host memory?
                    weighted_prob_functor(dev_params[m])); // operation
 
                // NOTE: this is cleaner and faster in raw C; we can fuse the addition with other operations rather than launching new kernels
                // you might be able to achieve this with thrust::device and/or thrust::for_each
                // sum.prob <- rowSums(weighted.prob)  # second line of the T function from wikipedia (Nx1)
                thrust::transform(
                    dev_weighted_prob.begin() + m * N, dev_weighted_prob.begin() + m * N + N, // input 1
                    dev_sum_prob.begin(), // input 2
                    dev_sum_prob.begin(), // output
                    thrust::plus<double>());
            }

            // all M iterations must run before sum_prob is correct
            // member.prob <- weighted.prob / sum.prob
            for (int m = 0; m < M; m++)
            {
                thrust::transform(
                    dev_weighted_prob.begin() + m * N, dev_weighted_prob.end() + m * N + N, // input 1
                    dev_sum_prob.begin(), // input 2
                    dev_member_prob.begin() + m * N, // output
                    thrust::divides<double>()); // operation
            }

            // there are no more interdepencies between components at this point, so the rest could run completely in parallel rather than using a for loop
            for (int m = 0; m < M; m++)
            {
                // member.prob.sum <- colSums(member.prob)
                // TODO you could use a transform_reduce here to eliminate the above 'divides' kernel launch
                double member_prob_sum = thrust::reduce(
                    dev_member_prob.begin(), dev_member_prob.end(), 
                    0.0, 
                    thrust::plus<double>());

                // alpha.new <- member.prob.sum / N  # should be 1x2 matrix
                params_new[m].alpha = member_prob_sum / N;

                //// mu.new <- colSums(x * member.prob) / member.prob.sum  # should be 1x2 matrix

                // TODO: this could run outside the for loop, but easier to express in here
                // calculate (x * member.prob)
                thrust::transform(
                    dev_chunk.begin() + m * N, dev_chunk.end() + m * N + N, // input 1
                    dev_member_prob.begin() + m * N, // input 2
                    dev_member_prob_times_x.begin() + m * N, // output
                    thrust::multiplies<double>()); // operation

                // calculate colSums
                double colSum = thrust::reduce(
                    dev_member_prob_times_x.begin(), dev_member_prob_times_x.end(), 
                    0.0, 
                    thrust::plus<double>());
                params_new[m].mu = colSum / member_prob_sum;

                // lambda.new <- member.prob.sum / colSums(((x_expanded - mu_expanded) ^ 2 * member.prob) / (mu_expanded ^ 2 * x_expanded))
                // TODO
            }

            printf("iteration %d\n", iteration);
            for (unsigned m = 0; m < M; m++)
            {
                printf("\tcomp %d: alpha=%f, mu=%f, lambda=%f\n", m, params_new[m].alpha, params_new[m].mu, params_new[m].lambda);
            }

            /*
            // TODO have we converged?
            if (t < M)
            {
                params[t] = params_new[t];
            }
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
