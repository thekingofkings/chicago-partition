import matplotlib
matplotlib.use('Agg')
from MCMC import MCMC_softmax_proposal, naive_MCMC
from q_learning import q_learning


n_sim = 10
versions = ["v" + str( x +1) for x in range(n_sim)]

for v in versions:
    MCMC_softmax_proposal('house-price-softmax-sampler-{}'.format(v),targetName="train_average_house_price")
    naive_MCMC('house-price-naive-sampler-{}'.format(v),targetName="train_average_house_price")
    q_learning('house-price-q-learning-sampler-{}'.format(v),targetName='train_average_house_price')
