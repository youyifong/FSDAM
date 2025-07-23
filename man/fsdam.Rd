\name{fsdam}
\alias{FSDAM}
\alias{fsdam}
\alias{plot.fsdam}
\title{
FS-DAM NLDR
}
\description{
Forward stepwise deep autoencoder-based monotone nonlinear dimension reduction.
}
\usage{
fsdam(dat, opt_numCode = ncol(dat), opt_seed = 1, opt_model = "n", opt_gpu = 0, 
opt_k = 100, opt_nEpochs = 10000, 
opt_constr = c("newpenalization", "constrained", "none"),
 opt_tuneParam = 10, opt_penfun = "mean", opt_ortho = 1, opt_earlystop = "no", 
 verbose = FALSE)

\method{plot}{fsdam} (x, which=c("mse", "history", "decoder.func", "scatterplot"),
 k=NULL, dim.1=NULL, dim.2=NULL, col.predict=2, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{dat}{
data frame.
}
  \item{opt_numCode}{
number of components to extract
}
  \item{opt_seed}{
seed for torch
}
  \item{opt_model}{
n for newpenalization
}
  \item{opt_gpu}{
zero-based index of gpu to be used among all gpus. If negative, then no gpu is used
}
  \item{opt_k}{
number of nodes in the coding/decoding layers
}
  \item{opt_nEpochs}{
number of epochs for training
}
  \item{opt_constr}{
constraint string
}
  \item{opt_tuneParam}{
tuning parameter for monotonicity penalty
}
  \item{opt_penfun}{
penalize sum or mean
}
  \item{opt_ortho}{
tuning parameter for orthogonality penalty
}
  \item{opt_earlystop}{
whether to stop early 
}
  \item{verbose}{
verbose
}
  \item{x}{
fsdam object
}
  \item{which}{
which
}
  \item{k}{
the component to plot}
  \item{dim.1}{
index of the first variable
}
  \item{dim.2}{
index of the second variable
}
  \item{col.predict}{
color of the predicted curve when which = scatterplot
}
  \item{\dots}{plotting arguments}
}
\details{

If using the default reticulate python virtual env, may need to install some python packages via:
reticulate::py_install("torch", pip = TRUE)
reticulate::py_install("numpy", pip = TRUE)
reticulate::py_install("pandas", pip = TRUE)

\cr
Alternatively, use a different python kernel by 
reticulate::use_python("/app/easybuild/software/Python/3.7.4-foss-2016b/bin/python") \cr

\cr
It is recommended that dat is scaled before calling fsdam.

}
\references{
Fong, Y, Xu, J. Multi-Stage Simultaneous Deep Autoencoder-based Monotone (MSS-DAM) Nonlinear Dimensionality Reduction Methods, Journal of Computational and Graphical Statistics, in press.
}
\examples{

\dontrun{
    
fit=fsdam(hvtn505tier1[1:100,-1], opt_numCode=2, verbose=TRUE)
fit
plot(fit,which="mse")
plot(fit,which="history")

}

}
