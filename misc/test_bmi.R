#!/usr/bin/env Rscript

source('invgaussmixEM.R')

require(statmod)
require(mixsmsn)

# Prototype of inverse Gaussian mixture models fit with EM algorithm

num.components = 2

data('bmi')
x <- bmi$bmi
# x <- rinvgauss(100, mean=4, shape=0.5)

# RANDOMISED INITIALISATION
# for j attempts, sample a few items from the dataset. calculate the maximum likelihood parameters and use those as initial parameters for a mixture component attempt

# Maximum likelihood estimate of model parameters for data x
invgaussMaximumLikelihood <- function(x) {
    # from http://en.wikipedia.org/wiki/Inverse_Gaussian_distribution#Maximum_likelihood

    mu <- mean(x)
    lambda <- 1 / (1 / length(x) * sum(1 / x - 1 / mu))
    # lambda <- (1 / length(x) * sum(1 / x - 1 / mu))

    result <- list(mu=mu, lambda=lambda)

    # print(paste0('estimates mu=', mu, ', lambda=', lambda))

    return(result)
}

# for j replicates
for (j in 1:100) {
# for (j in 1:5) {
    initials = list()

    # come up with initial conditions
    for (m in 1:num.components) {
        # take a subset of the data
        # modified from http://stackoverflow.com/a/19861866/591483
        items <- sample(1:length(x), 3)
        randomsubset <- x[items]

        ml = invgaussMaximumLikelihood(randomsubset)
        ml$alpha = 0.5  # start with equal mixing proportions

        # print(initials)

        initials[[m]] = ml
    }

    # print(paste0('initials = ', initials))

    # perform the fit
    # fit <- invgaussmixEM(x)
    fit <- invgaussmixEM(x, initials=initials)
    print(paste0('fit alpha: ', fit$alpha, ', lambda=', fit$lambda, ', mu=', fit$mu))

    # plot the density function over the histogram
    # modified from http://www.statmethods.net/graphs/density.html
    h <- hist(x, breaks=40, col="red", main="BMI")
    xfit <- seq(min(x), max(x), length=40)
    yfit <- fit$alpha[1] * dinvgauss(xfit, mean=fit$mu[1], shape=fit$lambda[1]) + fit$alpha[2] * dinvgauss(xfit, mean=fit$mu[2], shape=fit$lambda[2])
    yfit <- yfit * diff(h$mids[1:2]) * length(x) 
    lines(xfit, yfit, col="blue", lwd=2)

    abort()
}



