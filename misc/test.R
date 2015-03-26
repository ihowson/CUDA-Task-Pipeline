#!/usr/bin/env Rscript

require(statmod)

# Prototype of inverse Gaussian mixture models fit with EM algorithm

# generate test data - borrowed from http://stats.stackexchange.com/a/70881/16399
N <- 2000

components <- sample(1:2, prob=c(0.3, 0.7), size=N, replace=TRUE)
mus <- c(1, 3)
lambdas <- c(30, 0.2)

x <- rinvgauss(n=N, mean=mus[components], shape=lambdas[components])

fit <- invgaussmixEM(x)
