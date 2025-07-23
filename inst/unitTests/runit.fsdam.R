test.chngpt.test <- function() {

library("RUnit")
library("FSDAM")
suppressWarnings(RNGversion("3.5.0"))
RNGkind("Mersenne-Twister", "Inversion")    
tolerance=1e-6
verbose=0

# commented out only b/c it may not run due to python problem
#fit=fsdam(hvtn505tier1[1:100,-1], opt_numCode=2, opt_seed=1, opt_model="n", opt_k=10, opt_nEpochs=200, opt_constr="newpenalization", opt_tuneParam=10, opt_penfun="mean", opt_ortho=1, opt_earlystop="no", verbose=verbose)
#
#checkEqualsNumeric(fit$mse, c(1.140173, 1.039861), tolerance=tolerance) 
#checkEqualsNumeric(c(fit$history), c(2.498706, 2.254528, 1.660496, 1.491327), tolerance=tolerance) 
#checkEqualsNumeric(c(fit$code[1:4,]), c(0.3421956, 0.1124634, 0.4836745, 0.1541510, 0.3653351, 0.5179433, 0.2704228, 0.2618758), tolerance=tolerance) 


}
