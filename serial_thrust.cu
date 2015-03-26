// serial CUDA using Thrust

#include <math_constants.h>
#include <iostream>

#include <thrust/device_vector.h>

// TODO: this is only used for checkCudaErrors - try to remove it
#include "helper_cuda.h"

#include "common.h"

// to test this, it might be best to load a CSV with test data and do an A/B comparison against the R implementation

__host__ __device__ double dinvgauss(double x, double mu, double lambda)
{
    // TODO would be nice to assert that x > 0 and lambda = 0

    double x_minus_mu = x - mu;
    return sqrt(lambda / (2 * CUDART_PI * pow(x, 3.0))) * exp((-lambda * x_minus_mu * x_minus_mu) / (2 * mu * mu * x));
    // http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.invgauss.html
    // invgauss.pdf(x, mu) = 1 / sqrt(2*pi*x**3) * exp(-(x-mu)**2/(2*x*mu**2))
}

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
    // for (unsigned n = 0; n < NUM_CHUNKS - 10; n++)
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
