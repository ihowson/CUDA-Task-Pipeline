# Perform EM fitting 'repeats' times with random initial parameters and choose
# the best (highest likelihood) solution.

num.components = 2

source('invgaussmixEM.R')

# Maximum likelihood estimate of model parameters for data x
invgaussMaximumLikelihood <- function(x) {
    # from http://en.wikipedia.org/wiki/Inverse_Gaussian_distribution#Maximum_likelihood
    mu <- mean(x)
    lambda <- 1 / (1 / length(x) * sum(1 / x - 1 / mu))

    result <- list(mu=mu, lambda=lambda)

    return(result)
}

random_start_em <- function(x, repeats=100) {
    # RANDOMISED INITIALISATION
    # For j attempts, sample a few items from the dataset. Calculate the
    # maximum likelihood parameters and use those as initial parameters for a
    # mixture component attempt.

    best.fit = NULL

    # for j replicates
    for (j in 1:repeats) {
        initials = list()

        # come up with initial conditions
        for (m in 1:num.components) {
            # take a subset of the data
            # modified from http://stackoverflow.com/a/19861866/591483
            items <- sample(1:length(x), 3)
            randomsubset <- x[items]

            ml = invgaussMaximumLikelihood(randomsubset)
            ml$alpha = 0.5  # start with equal mixing proportions

            initials[[m]] = ml
        }

        # perform the fit
        fit <- invgaussmixEM(x, initials=initials)
        # print(paste0('fit llik: ', fit$llik, 'alpha: ', fit$alpha, ', lambda=', fit$lambda, ', mu=', fit$mu))

        if (is.null(best.fit)) {
            best.fit = fit
        } else if (fit$llik > best.fit$llik) {
            # print(paste0('found better ', fit$llik))
            best.fit = fit
        }
    }

    return(best.fit)
}
