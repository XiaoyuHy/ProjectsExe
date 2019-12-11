## grid define
library(RandomFields)
# nx <- ny <- 50
# x <- seq(0, 1, l=nx)
# y <- seq(0, 1, l=ny)
nx <- ny <- 1000
x <- seq(-12., -3., l=nx)
y <- seq(-6.5, 3, l=ny)
xy <- as.matrix(expand.grid(x, y))
N <- nx * ny

# ## create covariance matrix
# A <- exp(-as.matrix(dist(xy) ^ 2))
# 
# ## standard Cholesky method
# A1 <- A
# diag(A1) <- diag(A1) + 1e-7
# cholA1 <- chol(A1)
# Y1 <- crossprod(cholA1, rnorm(N))[,1]
# Y1 <- matrix(Y1, nrow=nx, ncol=ny)
# 
# ## rank-deficient pivoted Cholesky method
# R <- chol(A, pivot=TRUE) # creates a warning, but that's fine
# piv <- order(attr(R, "pivot"))  ## reverse pivoting index
# r <- attr(R, "rank")  ## numerical rank
# V <- R[1:r, piv]
# Y2 <- crossprod(V, rnorm(r))[,1]
# Y2 <- matrix(Y2, nrow=nx, ncol=ny)

## RandomFields method
for (SEED in 204:204){
set.seed(SEED)
#mod <- RMgauss(var=1, scale=0.2)
scale_par = 2.0
alpha_par = 1.8
mod <- RMstable(alpha =alpha_par,var=1, scale=scale_par)
Y3 <- RFsimulate(mod, x=x, y=y, spConform=FALSE)
rfSimData =  as.vector(as.matrix(Y3))
save(rfSimData,  file = paste("dataRsimulated/scale", formatC(scale_par, digits = 1, width = 3, format = "f", flag = "0"), "/delta", formatC(alpha_par, digits = 1, width = 3, format = "f", flag = "0"), "/rfSimDataSEED", SEED, ".RData", sep=""))
## plot different methods
# par(mfrow=c(1, 3))
# image(x, y, Y1)
# image(x, y, Y2)
#image(x, y, Y3)
}

model <- RMstable(alpha=2.0, var=1, scale=0.2)
x <- seq(0, 10, 0.02)
plot(model,xlim=c(0,1.5))
plot(RFsimulate(model, x=x))

test <- function(x){
  exp(-x^2/0.04)
}
y = test(x)
plot(x,y, type='l',xlim=c(0,1.5))
