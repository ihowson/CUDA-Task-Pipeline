#!/usr/bin/env Rscript

source('invgaussmixEM.R')

require(statmod)

# Prototype of inverse Gaussian mixture models fit with EM algorithm

x <- as.numeric(as.matrix(read.table('../test.data')))

for (i in 1:100) {
    fit <- invgaussmixEM(x)

    cat(sprintf('chunk %d: ', i))
    cat('fit alpha: ')
    cat(fit$alpha)
    cat(', lambda: ')
    cat(fit$lambda)
    cat(', mu: ')
    cat(fit$mu)
    cat('\n')
}

