from MCMC import MCMC_softmax_proposal, naive_MCMC
import q_learning


n_sim = 10
versions = ["v" + str( x +1) for x in range(n_sim)]

for v in versions:
    MCMC_softmax_proposal('softmax-sampler-{}'.format(v))
    naive_MCMC('naive-sampler-{}'.format(v))
