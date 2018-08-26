from tract import Tract
from community_area import CommunityArea
from MCMC import naive_MCMC, MCMC_softmax_proposal, writeSimulationOutput
from q_learning import q_learning
from regression import NB_regression_evaluation
import numpy as np

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

    def get_target_cluster(self, task):
        tract_task_y_map = {'house_price': 'test_price', 'crime': 'total'}
        ca_task_y_map = {'house_price': 'test_average_house_price', 'crime': 'total'}
        y_tract = tract_task_y_map[task]
        y_ca = ca_task_y_map[task]

        return y_tract, y_ca

    def init_tracts(self):
        Tract.createAllTracts()
        Tract.generateFeatures(2011)



    def get_community_structure(self,m,prev_m):
        print "Initializing {} regions".format(self.max_m)
        CommunityArea.createAllCAs(Tract.tracts)
        if m < self.max_m:
            print "Randomly combining {} regions into {} regions...".format(prev_m, m)
            CommunityArea.rand_init_communities(m)
            print "Dimensions of updated design matrix: {}".format(CommunityArea.features.shape)

    def get_grid(self):
        m_grid = range(self.min_m, self.max_m+1)
        m_grid = sorted(m_grid, reverse=True)

        return m_grid

    def naive_mcmc_run(self, iter=None):
        self.init_tracts()
        pred_target = self.get_target(self.task)
        m_grid = self.get_grid()

        for i, m in enumerate(m_grid):
            self.get_community_structure(m=m,prev_m=m_grid[i-1])
            # estimate naive MCMC
            if iter:
                fname = self.project_name + "-naive-{}.{}".format(m, iter)

            else:
                fname = self.project_name + "-naive-{}".format(m)
            naive_MCMC(fname, targetName=pred_target,
                       lmbda=0.005, f_sd=3, Tt=0.1, init_ca=False)

    def softmax_mcmc_run(self, iter=None):
        self.init_tracts()
        pred_target = self.get_target(self.task)
        m_grid = self.get_grid()

        for i, m in enumerate(m_grid):
            self.get_community_structure(m=m,prev_m=m_grid[i-1])
            # estimate MCMC with softmax proposal
            if iter:
                fname = self.project_name + "-softmax-{}.{}".format(m, iter)

            else:
                fname = self.project_name + "-softmax-{}".format(m)

            MCMC_softmax_proposal(fname, targetName=pred_target,
                                  lmbda=0.005, f_sd=3, Tt=0.1, init_ca=False)

    def dqn_mcmc_run(self, iter=None):
        self.init_tracts()
        pred_target = self.get_target(self.task)
        m_grid = self.get_grid()

        for i, m in enumerate(m_grid):
            self.get_community_structure(m=m,prev_m=m_grid[i-1])
            # MCMC with proposal from DQN
            if iter:
                fname = self.project_name + "-dqn-{}.{}".format(m, iter)

            else:
                fname = self.project_name + "-dqn-{}".format(m)

            q_learning(fname,
                       targetName=pred_target,
                       lmbd=0.005, f_sd=3, Tt=0.1, init_ca=False)


    def kmeans_run(self, iter=None):
        self.init_tracts()
        y_tract, y_ca = self.get_target_cluster(self.task)
        m_grid = self.get_grid()

        for i, m in enumerate(m_grid):
            self.get_community_structure(m=m,prev_m=m_grid[i-1])
            # kmeans
            Tract.kMeansClustering(cluster_X=True, cluster_y=True, y=y_tract)
            mae, rmse, mre = NB_regression_evaluation(CommunityArea.features, CommunityArea.featureNames, y_ca)

            if iter:
                fname = self.project_name + "-kmeans-{}.{}".format(m, iter)

            else:
                fname = self.project_name + "-kmeans-{}".format(m)

            writeSimulationOutput(project_name=fname,
                                  mae=mae,
                                  rmse=rmse,
                                  accept_rate=np.nan,
                                  n_iter_conv=m)


    def agglomerative_run(self, iter=None):
        self.init_tracts()
        y_tract, y_ca = self.get_target_cluster(self.task)
        m_grid = self.get_grid()

        for i, m in enumerate(m_grid):
            self.get_community_structure(m=m, prev_m=m_grid[i - 1])
            # kmeans
            Tract.agglomerativeClustering(cluster_X=True, cluster_y=True, y=y_tract)
            mae, rmse, mre = NB_regression_evaluation(CommunityArea.features, CommunityArea.featureNames, y_ca)

            if iter:
                fname = self.project_name + "-agglomerative-{}.{}".format(m, iter)

            else:
                fname = self.project_name + "-agglomerative-{}".format(m)

            writeSimulationOutput(project_name=fname,
                                  mae=mae,
                                  rmse=rmse,
                                  accept_rate=np.nan,
                                  n_iter_conv=m)

    def spectral_run(self, iter=None):
        self.init_tracts()
        y_tract, y_ca = self.get_target_cluster(self.task)
        m_grid = self.get_grid()

        for i, m in enumerate(m_grid):
            self.get_community_structure(m=m, prev_m=m_grid[i - 1])
            # kmeans
            Tract.spectralClustering(cluster_X=True, cluster_y=True, y=y_tract)
            mae, rmse, mre = NB_regression_evaluation(CommunityArea.features, CommunityArea.featureNames, y_ca)

            if iter:
                fname = self.project_name + "-spectral-{}.{}".format(m, iter)

            else:
                fname = self.project_name + "-spectral-{}".format(m)

            writeSimulationOutput(project_name=fname,
                                  mae=mae,
                                  rmse=rmse,
                                  accept_rate=np.nan,
                                  n_iter_conv=m)

    def run_all(self, n_iter):

        for i in range(1, n_iter+1):
            self.kmeans_run(i)
            self.agglomerative_run(i)
            self.spectral_run(i)
            self.naive_mcmc_run(i)
            self.softmax_mcmc_run(i)
            self.dqn_mcmc_run(i)



if __name__ == '__main__':


    crime_sim = ParamSensitivity(project_name='sensitivity-study-crime', task='crime',
                                 max_m=77, min_m=75, plot=True)

    crime_sim.run_all(n_iter=10)



    house_price_sim = ParamSensitivity(project_name='sensitivity-study-houseprice',
                                        task='house_price', max_m=77, min_m=5, plot=True)
    house_price_sim.run_all(n_iter=10)