#!/usr/bin/env Rscript

source('invgaussmixEM.R')

require(statmod)

# Prototype of inverse Gaussian mixture models fit with EM algorithm

# load test data from file

# TODO: blech - surely there's a cleaner way?
x <- as.numeric(as.matrix(read.table('../test.data')))

fit <- invgaussmixEM(x)

cat('fit alpha: ')
cat(fit$alpha)
cat(', lambda: ')
cat(fit$lambda)
cat(', mu: ')
cat(fit$mu)
cat('\n')

