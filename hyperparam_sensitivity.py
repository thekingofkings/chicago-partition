from tract import Tract
from community_area import CommunityArea
from MCMC import naive_MCMC, MCMC_softmax_proposal, writeSimulationOutput
from q_learning import q_learning
from regression import NB_regression_evaluation
import numpy as np
import pickle as pkl
import os

class ParamSensitivity(object):

    def __init__(self,project_name, task, max_m, min_m, plot):
        self.project_name = project_name
        self.task = task
        self.max_m = max_m
        self.min_m = min_m
        self.plot = plot
        self.pkl_dir = 'data/community_states'

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

    def dump_pickle_tract_data(self, m):
        if not os.path.exists(self.pkl_dir):
            os.makedirs(self.pkl_dir)

        tract_dict = Tract.get_tract_ca_dict()
        tract_features = Tract.features
        boundary_set = [x.id for x in Tract.boundarySet]
        data = (tract_dict, tract_features, boundary_set)
        f_name = "{}/tract-data-m-{}.p".format(self.pkl_dir, m)
        with open(f_name, 'wb') as f:
            pkl.dump(data, f)

    def load_pickle_tract_data(self, m):
        f_name = "{}/tract-data-m-{}.p".format(self.pkl_dir, m)
        with open(f_name, 'rb') as f:
            tract_dict, tract_features, boundary_list = pkl.load(f)

        for t_id, tract in Tract.tracts.items():
            tract.CA = tract_dict[t_id]

        boundary_set = set()
        for t_id in boundary_list:
            tract = Tract.tracts[t_id]
            boundary_set.add(tract)
        Tract.boundarySet = boundary_set

        Tract.features = tract_features



    def dump_pickle_ca_data(self, m):
        if not os.path.exists(self.pkl_dir):
            os.makedirs(self.pkl_dir)
        ca_dict = CommunityArea.get_ca_tract_dict()
        ca_feature_dict = CommunityArea.features_ca_dict
        ca_features = CommunityArea.features

        ca_poly = dict()
        for id, ca in CommunityArea.CAs.items():
            ca_poly[id] = ca.polygon


        data = (ca_dict, ca_feature_dict, ca_features, ca_poly)
        f_name = "{}/ca-data-m-{}.p".format(self.pkl_dir, m)
        with open(f_name, 'wb') as f:
            pkl.dump(data, f)

    def load_pickle_ca_data(self, m):
        f_name = "{}/ca-data-m-{}.p".format(self.pkl_dir, m)
        with open(f_name, 'rb') as f:
            ca_dict, ca_feature_dict, ca_features, ca_poly = pkl.load(f)

        CommunityArea.features = ca_features
        CommunityArea.features_ca_dict = ca_feature_dict


        for ca_id, ca in CommunityArea.CAs.items():
            if ca_id in ca_dict.keys():
                    # Update tract list
                    ca_tract_ids = ca_dict.get(ca_id, [])
                    for t_id in ca_tract_ids:
                        ca.tracts[t_id] = Tract.tracts[t_id]

                    # update polygon shape
                    polygon = ca_poly.get(ca_id)
                    ca.polygon = polygon
            else:
                del CommunityArea.CAs[ca_id]

    def init_communities(self, m):
        if m == (self.max_m):
            print "Initializing {} regions".format(self.max_m)
            self.init_tracts()
            CommunityArea.createAllCAs(Tract.tracts)

        else:
            print "Loading community structure from file:: m={}".format(m)
            self.load_pickle_tract_data(m)
            self.load_pickle_ca_data(m)

        if self.plot:
            CommunityArea.visualizeCAs(fname='{}-{}.png'.format(self.project_name, m),
                                       labels=True, iter_cnt=m)

    def get_random_communities(self):
        """

        :param min_m:
        :param max_m: **Inclusive
        :return:
        """

        m_grid = self.get_grid()

        for i, m in enumerate(m_grid):

            # Estimate models here
            if m == (self.max_m):
                print "Initializing {} regions".format(self.max_m)
                self.init_tracts()
                CommunityArea.createAllCAs(Tract.tracts)
            else:
                print "Randomly combining {} regions into {} regions...".format(m_grid[i-1], m)
                CommunityArea.rand_init_communities(m)
                print "Dimensions of updated design matrix: {}".format(CommunityArea.features.shape)
                self.dump_pickle_tract_data(m=m)
                self.dump_pickle_ca_data(m=m)
                self.dump_pickle_ca_data(m=m)

            if self.plot:
                CommunityArea.visualizeCAs(fname='{}-{}.png'.format(self.project_name,m),
                                           labels=True, iter_cnt=m)

    def get_grid(self):
        m_grid = range(self.min_m, self.max_m+1)
        m_grid = sorted(m_grid, reverse=True)

        return m_grid

    def naive_mcmc_run(self, m, iter=None):
        print "Beginning Naive MCMC..."
        self.init_communities(m=m)
        pred_target = self.get_target(self.task)

        # estimate naive MCMC
        if iter:
            fname = self.project_name + "-naive-{}.{}".format(m, iter)

        else:
            fname = self.project_name + "-naive-{}".format(m)
        naive_MCMC(fname, targetName=pred_target,
                   lmbda=0.005, f_sd=3, Tt=0.1, init_ca=False)

    def softmax_mcmc_run(self, m, iter=None):
        print "Beginning Softmax MCMC..."
        self.init_communities(m=m)
        pred_target = self.get_target(self.task)
        # estimate MCMC with softmax proposal
        if iter:
            fname = self.project_name + "-softmax-{}.{}".format(m, iter)

        else:
            fname = self.project_name + "-softmax-{}".format(m)

        MCMC_softmax_proposal(fname, targetName=pred_target,
                              lmbda=0.005, f_sd=3, Tt=0.1, init_ca=False)

    def dqn_mcmc_run(self, m, iter=None):
        print "Beginning DQN..."
        self.init_communities(m=m)
        pred_target = self.get_target(self.task)
        # MCMC with proposal from DQN
        if iter:
            fname = self.project_name + "-dqn-{}.{}".format(m, iter)

        else:
            fname = self.project_name + "-dqn-{}".format(m)

        q_learning(fname,
                   targetName=pred_target,
                   lmbd=0.005, f_sd=3, Tt=0.1, init_ca=False)


    def kmeans_run(self, m, iter=None):
        print "Beginning Kmeans..."
        self.init_communities(m=m)
        y_tract, y_ca = self.get_target_cluster(self.task)

        Tract.kMeansClustering(cluster_X=True, cluster_y=True, y=y_tract)
        CommunityArea.createAllCAs(Tract.tracts)
        mae, rmse, mre = NB_regression_evaluation(CommunityArea.features,
                                                  CommunityArea.featureNames, y_ca)

        if iter:
            fname = self.project_name + "-kmeans-{}.{}".format(m, iter)

        else:
            fname = self.project_name + "-kmeans-{}".format(m)

        writeSimulationOutput(project_name=fname,
                              mae=mae,
                              rmse=rmse,
                              accept_rate=np.nan,
                              n_iter_conv=m)


    def agglomerative_run(self, m,iter=None):
        print "Beginning Agglomerative Clustering..."
        self.init_communities(m=m)
        y_tract, y_ca = self.get_target_cluster(self.task)
        Tract.agglomerativeClustering(cluster_X=True, cluster_y=True, y=y_tract)
        CommunityArea.createAllCAs(Tract.tracts)

        mae, rmse, mre = NB_regression_evaluation(CommunityArea.features.dropna(),
                                                  CommunityArea.featureNames,
                                                  y_ca)

        if iter:
            fname = self.project_name + "-agglomerative-{}.{}".format(m, iter)

        else:
            fname = self.project_name + "-agglomerative-{}".format(m)

        writeSimulationOutput(project_name=fname,
                              mae=mae,
                              rmse=rmse,
                              accept_rate=np.nan,
                              n_iter_conv=m)

    def spectral_run(self, m, iter=None):
        print "Beginning Spectral Clustering..."
        self.init_communities(m=m)
        y_tract, y_ca = self.get_target_cluster(self.task)

        Tract.spectralClustering(cluster_X=True, cluster_y=True, y=y_tract)
        CommunityArea.createAllCAs(Tract.tracts)
        mae, rmse, mre = NB_regression_evaluation(CommunityArea.features.dropna(),
                                                  CommunityArea.featureNames,
                                                  y_ca)

        if iter:
            fname = self.project_name + "-spectral-{}.{}".format(m, iter)

        else:
            fname = self.project_name + "-spectral-{}".format(m)

        writeSimulationOutput(project_name=fname,
                              mae=mae,
                              rmse=rmse,
                              accept_rate=np.nan,
                              n_iter_conv=m)

    def run_sim(self, n_iter, gen_ca=False):
        m_grid = self.get_grid()

        if gen_ca:
            self.get_random_communities()

        for m in m_grid:
            for i in range(1, n_iter+1):
                print "m = {}, i = {}".format(m, i)

                self.agglomerative_run(m, i)
                self.kmeans_run(m, i)
                self.spectral_run(m, i)
                self.naive_mcmc_run(m, i)
                self.softmax_mcmc_run(m, i)
                self.dqn_mcmc_run(m, i)



if __name__ == '__main__':


    crime_sim = ParamSensitivity(project_name='sensitivity-study-crime', task='crime',
                                 max_m=77, min_m=20, plot=True)

    crime_sim.run_sim(n_iter=10, gen_ca=False)


    house_price_sim = ParamSensitivity(project_name='sensitivity-study-houseprice',
                                        task='house_price', max_m=77, min_m=20, plot=False)
    house_price_sim.run_sim(n_iter=10, gen_ca=False)