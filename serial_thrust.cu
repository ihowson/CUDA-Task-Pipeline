// serial CUDA using Thrust

#include <math_constants.h>
#include <iostream>

#include <thrust/device_vector.h>

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

// ((x_expanded - mu_expanded) ^ 2 * member.prob) / (mu_expanded ^ 2 * x_expanded)
struct sum_term_functor
{
    invgauss_params_t params;
    const double *chunk;
    const double *member_prob;

    sum_term_functor(
        thrust::device_vector<double> _chunk,
        thrust::device_ptr<double> _member_prob,
        invgauss_params_t _params)
    {
        chunk = thrust::raw_pointer_cast(_chunk.data());
        // chunk = thrust::raw_pointer_cast(_chunk);
        member_prob = thrust::raw_pointer_cast(_member_prob);
        params = _params;
    }

    __host__ __device__
    double operator()(const int& i) const
    { 
        double x = chunk[i];
        double x_minus_mu = x - params.mu;
        return (x_minus_mu * x_minus_mu * member_prob[i]) / (params.mu * params.mu * x);
    }
};

void dump_array(const char *msg, thrust::device_vector<double>& array, int offset)
{
    printf("%s: ", msg);
    for (int i = offset; i < offset + 4; i++)
    {
        printf("%lf ", (double)array[i]);
    }

    printf("\n");
}

void dump_array(const char *msg, thrust::device_vector<double>& array)
{
    dump_array(msg, array, 0);
}

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

    // this would normally be an M * N array, but we only process N entries at a time and so reuse this space
    thrust::device_vector<double> dev_member_prob_times_x(N);
    thrust::device_vector<double> dev_sum_term(M * N);

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

                // FIXME: the weighted_probs are close, but not identical to the R implementation
 
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
                    dev_weighted_prob.begin() + m * N, dev_weighted_prob.begin() + m * N + N, // input 1
                    dev_sum_prob.begin(), // input 2
                    dev_member_prob.begin() + m * N, // output
                    thrust::divides<double>()); // operation
            }

            // there are no more interdepencies between components at this point, so the rest could run completely in parallel rather than using a for loop
            for (int m = 0; m < M; m++)
            {
                unsigned vec_begin = m * N; // offset from the start of 2D matrices stored as 1D vector
                unsigned vec_end = vec_begin + N; // offset of the end

                // member.prob.sum <- colSums(member.prob)
                // TODO you could use a transform_reduce here to eliminate the above 'divides' kernel launch
                double member_prob_sum = thrust::reduce(
                    dev_member_prob.begin() + vec_begin, dev_member_prob.begin() + vec_end, 
                    0.0, 
                    thrust::plus<double>());

                // alpha.new <- member.prob.sum / N  # should be 1x2 matrix
                params_new[m].alpha = member_prob_sum / N;

                //// mu.new <- colSums(x * member.prob) / member.prob.sum  # should be 1x2 matrix

                // TODO: this could run outside the for loop, but easier to express in here
                // calculate (x * member.prob)
                thrust::transform(
                    dev_chunk.begin() + vec_begin, dev_chunk.begin() + vec_end, // input 1
                    dev_member_prob.begin() + vec_begin, // input 2
                    dev_member_prob_times_x.begin(), // output
                    thrust::multiplies<double>()); // operation

                // calculate colSums
                double colSum = thrust::reduce(
                    dev_member_prob_times_x.begin(), dev_member_prob_times_x.end(), 
                    0.0, 
                    thrust::plus<double>());
                params_new[m].mu = colSum / member_prob_sum;

                //// lambda.new <- member.prob.sum / colSums(((x_expanded - mu_expanded) ^ 2 * member.prob) / (mu_expanded ^ 2 * x_expanded))

                // calculate term to colSums
                // we iterate over x *and* member.prob - the easiest thing to do is to pass in indices and index within the functor

                // TODO: making a mockery of type safety
                thrust::device_ptr<double> member_prob_ptr = &(*(dev_member_prob.begin() + vec_begin));
                thrust::transform(
                    thrust::make_counting_iterator(0), thrust::make_counting_iterator(N), // input
                    dev_sum_term.begin() + vec_begin, // output
                    // sum_term_functor(dev_chunk, member_prob_ptr, dev_params[m])); // operation
                    sum_term_functor(dev_chunk, member_prob_ptr, dev_params[m])); // operation
                double lambda_colSum = thrust::reduce(
                    dev_sum_term.begin() + vec_begin, dev_sum_term.begin() + vec_end,
                    0.0, 
                    thrust::plus<double>());
                params_new[m].lambda = member_prob_sum / lambda_colSum;
            }

            printf("iteration %d\n", iteration);
            for (unsigned m = 0; m < M; m++)
            {
                invgauss_params_t *p = &params_new[m];
                printf("\tcomp %d: alpha=%lf, mu=%lf, lambda=%lf\n", m, p->alpha, p->mu, p->lambda);
            }

            for (unsigned m = 0; m < M; m++)
            {
                invgauss_params_t *p = &params_new[m];
                if (p->mu < 0.0 || p->lambda < 0.0 || p->alpha < 0.0 || p->alpha > 1.0 || isnan(p->mu) || isnan(p->lambda) || isnan(p->alpha))
                {
                    printf("ABORT: parameters out of range");
                    exit(0);
                }
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
