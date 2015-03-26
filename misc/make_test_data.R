#!/usr/bin/env Rscript

# Generates a chunk of input data
# i.e. 2000 observations from a 2-component inverse Gaussian mixture model
# writes it to an output file, one observation per line

require(statmod)

# generate test data
N <- 2000
components <- sample(1:2, prob=c(0.3, 0.7), size=N, replace=TRUE)
mus <- c(1, 3)
lambdas <- c(30, 0.2)
x <- rinvgauss(n=N, mean=mus[components], shape=lambdas[components])

# write to file
write.table(x, file='../test.data', row.names=FALSE, col.names=FALSE)
