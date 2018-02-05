import matplotlib
matplotlib.use('Agg')
from MCMC import MCMC_softmax_proposal, naive_MCMC
from q_learning import q_learning


n_sim = 100
versions = ["v" + str( x +1) for x in range(n_sim)]

for v in versions:
    MCMC_softmax_proposal('house-price-softmax-sampler-{}'.format(v),
                          targetName="train_average_house_price",
                          lmbda=0.005, f_sd=3, Tt=0.1)
    naive_MCMC('house-price-naive-sampler-{}'.format(v),
               targetName="train_average_house_price",
               lmbda=0.005, f_sd=3, Tt=0.1)
#    q_learning('house-price-q-learning-sampler-{}'.format(v),
#               targetName='train_average_house_price',
#               lmbd=0.005, f_sd=3, Tt=0.1)
