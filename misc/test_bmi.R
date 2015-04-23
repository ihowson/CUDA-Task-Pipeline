#!/usr/bin/env Rscript

# TODO why is this needed?
source('random_start_em.R')

# require(statmod)
library(mixsmsn)

# Prototype of inverse Gaussian mixture models fit with EM algorithm

num.components = 2

data('bmi')
x <- bmi$bmi

fit <- random_start_em(x)

print(paste0('fit llik: ', fit$llik, 'alpha: ', fit$alpha, ', lambda=', fit$lambda, ', mu=', fit$mu))

# plot the density function over the histogram
# modified from http://www.statmethods.net/graphs/density.html
h <- hist(x, breaks=40, col="red", main="BMI")
xfit <- seq(min(x), max(x), length=40)
yfit <- fit$alpha[1] * dinvgauss(xfit, mean=fit$mu[1], shape=fit$lambda[1]) + fit$alpha[2] * dinvgauss(xfit, mean=fit$mu[2], shape=fit$lambda[2])
yfit <- yfit * diff(h$mids[1:2]) * length(x) 
lines(xfit, yfit, col="blue", lwd=2)
