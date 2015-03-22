# set mu = 1, lambda = 1, alpha = 0.5/0.5
# do a grid search across mu=-3..3 and lambda=-1 .. 1
# at each point, run em, plot a point showing 'quality of fit', i.e. within what percentage are the fit parameters to reality?
# possibly plot the percentage error of the *worst* fit parameter

source('invgaussmixEM.R')

require(statmod)
require(ggplot2)

mu.vals <- seq(-2.5, 3, by=0.5)
lambda.vals <- seq(-0.8, 0.5, by=0.1)
alpha.vals <- c(0.5, 0.5)

N <- 2000  # generate samples of 2000 entries

mu1 <- 3
lambda1 <- 1

# make space for results
qual.matrix <- matrix(0, nrow=(length(mu.vals) * length(lambda.vals)), ncol=3)
row <- 1

for (mu.offset in mu.vals) {
    for (lambda.offset in lambda.vals) {
        # set up args
        mus <- c(mu1, mu1 + mu.offset)
        lambdas <- c(lambda1, lambda1 + lambda.offset)
        components <- sample(1:2, prob=alpha.vals, size=N, replace=TRUE)

        # generate sample data
        x <- rinvgauss(n=N, mean=mus[components], shape=lambdas[components])

        # fit it
        fit <- invgaussmixEM(x)

        # evaluate quality of fit
        cat('we fit mu = ')
        cat(fit$mu)
        cat(', (should be ')
        cat(mus)
        cat(', lambda = ')
        cat(fit$lambda)
        cat(', (should be ')
        cat(lambdas)
        cat(', alpha = ')
        cat(fit$alpha)
        cat(', (should be ')
        cat(alpha.vals)
        cat(')\n')

        #q <- 1 - (abs((fit$alpha[1] - 0.5)) / 0.5)

        #fit.mus <- fit$mu
        #smaller.fit.mu <- min(fit.mus)
        #smaller.actual.mu <- min(mus)
        #q <- 1 - (abs((smaller.fit.mu - smaller.actual.mu)) / smaller.actual.mu)

        fit.lambdas <- fit$lambda
        smaller.fit.lambda <- min(fit.lambdas)
        smaller.actual.lambda <- min(lambdas)
        q <- 1 - (abs((smaller.fit.lambda - smaller.actual.lambda)) / smaller.actual.lambda)
        
        # for now, we're just going to evaluate quality of the alpha params (because it's easier to calculate and interpret)
        # TODO this should take the worst-case of all parameters (though think about lambda, because it's not a linear relation)
        qual.matrix[row, 1] <- mu.offset
        qual.matrix[row, 2] <- lambda.offset
        qual.matrix[row, 3] <- q

        row <- row + 1
    }
}

qual.df=data.frame(mu=qual.matrix[, 1], lambda=qual.matrix[, 2], qual=qual.matrix[, 3])

# plot qual
# reformat for ggplot

#pp <- function (n) {
    #df <- expand.grid(x=mu.vals, y=lambda.vals)
    #df$r <- sqrt(df$x^2 + df$y^2)
    #df$z <- cos(df$r^2)*exp(-df$r/6)
    #df
#}
#print(pp)
#plot(ggplot(pp200, aes(x=x, y=y, fill=z)) + geom(raster()))

plot(
    ggplot(qual.df, aes(mu, lambda))
     + geom_point(aes(colour=qual), size=15.0)
     + scale_colour_gradient(low='red', high='green')
)
