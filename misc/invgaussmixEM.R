# EM implementation

# FIXME we assume two components

invgaussmixEM <- function(x) {
    # init
    max.iters <- 100  # TODO move this to function arg
    iterations <- 0
    N <- length(x)

    # TODO probably need to init these a little differently (random init) so they don't follow the exact same path
    mu <- matrix(c(1.01, 0.99), nrow=1)
    lambda <- matrix(c(1.0, 1.0), nrow=1)
    alpha <- matrix(c(0.5, 0.5), nrow=1)  # mixing components
    epsilon <- 0.000001
    diff <- 1
    log.lik <- epsilon + 1

    x_expanded <- matrix(x, nrow=N, ncol=2, byrow=FALSE)
    # need an initial value for llik here - start at infinity?

    # algorithm based off http://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm#Gaussian_mixture

    while (diff > epsilon && iterations <= max.iters) {
        iterations <- iterations + 1

        # TODO: this might be faster/better expressed using rep_len
        mu_expanded <- matrix(mu, nrow=N, ncol=2, byrow=TRUE)
        lambda_expanded <- matrix(lambda, nrow=N, ncol=2, byrow=TRUE)
        alpha_expanded <- matrix(alpha, nrow=N, ncol=2, byrow=TRUE)

        # E-step: calculate Q

        # calculate p(l|x_i, Theta^g)
        x.prob <- dinvgauss(x_expanded, mean=mu_expanded, shape=lambda_expanded)  # N x 2 matrix

        # have we converged to within epsilon?
        # we do this up here and not at the end as we're doing parts of the calculation anyway
        # loglik = sum_N(log(sum_M(alpha_m * p_m(x))))
        log.lik.old <- log.lik
        log.lik <- sum(log(rowSums(alpha_expanded * x.prob)))
        # cat('log-likelihood is ')
        # cat(log.lik)
        # cat('\n')
        # TODO this abs isn't ideal - shouldn't be necessary
        # TODO also catch if log-lik increases, which shouldn't happen (supposed to flag non-convergence and then try again with new initial conditions)
        diff <- abs(log.lik.old - log.lik)
        if (diff < epsilon) {
            break
        }

        # NOTE probably don't need to actually do this - wikipedia says it's optional

        # more E-step?
        # z=dens1/apply(dens1,1,sum)

        # then come up with new parameter estimates using the equations that you derived

        # we're just going to repeat the M step max.iters times and track how the parameters change over time
        # we will implement fast termination conditions later

        # membership probabilities
        weighted.prob <- alpha_expanded * x.prob  # per-component weighted sum of probabilities (Nx2)
        sum.prob <- rowSums(weighted.prob)  # second line of the T function from wikipedia (Nx1)
        member.prob <- weighted.prob / sum.prob

        # we've got components across columns and observations down rows, so we do all of the summations simultaneously on both 

        member.prob.sum <- colSums(member.prob)
        alpha.new <- member.prob.sum / N  # should be 1x2 matrix
        mu.new <- colSums(x * member.prob) / member.prob.sum  # should be 1x2 matrix
        lambda.new <- member.prob.sum / colSums(((x_expanded - mu_expanded) ^ 2 * member.prob) / (mu_expanded ^ 2 * x_expanded))


        mu <- mu.new
        lambda <- lambda.new
        alpha <- alpha.new

        # new.obs.ll <- sum(log(apply(dens(lambda.hat, theta.hat, k),1,sum)))
        # diff <- new.obs.ll-old.obs.ll
        # old.obs.ll <- new.obs.ll

        # LATER: it might make sense to try this algorithm using RGPU or something - it would be ideal to be able to plug in R code to your CUDA code. tHis might be a halfway option taht is fast enough
    }

    result <- list(x=x, alpha=alpha, mu=mu, lambda=lambda)  # TODO add llik here
    class(result) <- "mixEM"
    result
}