from tract import Tract
from community_area import CommunityArea
from MCMC import naive_MCMC, MCMC_softmax_proposal
from q_learning import q_learning

class ParamSensitivity(object):

    def __init__(self,project_name, task, max_m, min_m, plot):
        self.project_name = project_name
        self.task = task
        self.max_m = max_m
        self.min_m = min_m
        self.plot = plot

    def get_target(self, task):

        if task == 'crime':
            return 'total'
        elif task == 'house_price':
            return 'train_average_house_price'
        else:
            raise ValueError("Task must be either 'crime' or 'train_average_house_price'")

    def run_sim(self):

        m_grid = range(self.min_m, self.max_m+1)
        m_grid = sorted(m_grid, reverse=True)

        for i in m_grid:

            # Estimate models here
            if i == (self.max_m):
                print "Initializing {} regions".format(self.max_m)
                Tract.createAllTracts()
                Tract.generateFeatures(2011)
                CommunityArea.createAllCAs(Tract.tracts)
            else:
                print "Randomly combining {} regions into {} regions...".format(i + 1, i)
                CommunityArea.rand_init_communities(i)
                print "Dimensions of updated design matrix: {}".format(CommunityArea.features.shape)


            if self.plot:
                CommunityArea.visualizeCAs(fname='{}-{}.png'.format(self.project_name,i),
                                           labels=True, iter_cnt=i)
            pred_target = self.get_target(self.task)
            # estimate naive MCMC
            naive_MCMC('{}-naive-{}'.format(self.project_name,i), targetName=pred_target,
                       lmbda=0.005, f_sd=5, Tt=0.1, init_ca=False)
            # estimate MCMC with softmax proposal
            MCMC_softmax_proposal('{}-softmax-{}'.format(self.project_name,i), targetName=pred_target,
                                  lmbda=0.005, f_sd=5, Tt=0.1, init_ca=False)
            # MCMC with proposal from DQN
            q_learning("{}-q_learning-{}".format(self.project_name,i),
                       targetName=pred_target,
                       lmbd=0.005, f_sd=5, Tt=0.1, init_ca=False)




if __name__ == '__main__':

    crime_sim = ParamSensitivity(project_name='sensitivity-study-crime', task='crime',
                                 max_m=77, min_m=5, plot=True)

    house_price_sim = ParamSensitivity(project_name='sensitivity-study-houseprice',
                                        task='house_price', max_m=77, min_m=5, plot=True)

    crime_sim.run_sim()
    house_price_sim.run_sim()