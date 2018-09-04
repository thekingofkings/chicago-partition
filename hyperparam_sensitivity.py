from tract import Tract
from community_area import CommunityArea
from MCMC import naive_MCMC, MCMC_softmax_proposal, writeSimulationOutput
from q_learning import q_learning
from regression import NB_regression_evaluation
import numpy as np
import pickle as pkl
import os
import logging
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

class ParamSensitivity(object):

    def __init__(self,project_name, task, max_m, min_m, plot, f_sd=None, lmbda=None, T=None):
        self.project_name = project_name
        self.task = task
        self.max_m = max_m
        self.min_m = min_m
        self.plot = plot
        self.f_sd = f_sd
        self.lmbda = lmbda
        self.T = T
        self.pkl_dir = 'data/community_states'
        self.start_time = None
        self.end_time = None
        self.admin_boundary_m = 77

    def config_log(self):
        if not os.path.isdir('log'):
            os.mkdir('log')
        now = dt.datetime.now()
        self.start_time = now
        fname = 'log/sensitivity_study_{}.log'.format(now)
        logging.basicConfig(filename=fname, level=logging.INFO, filemode='w')
        logging.info('Starting sensitivity study: %s', now)

    def emit_log(self, msg):
        now = dt.datetime.now()
        msg = "\n\t{} --> ".format(now) + msg
        logging.info(msg)

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
        if m == (self.admin_boundary_m):
            print "Initializing {} regions".format(self.admin_boundary_m)
            self.init_tracts()
            CommunityArea.createAllCAs(Tract.tracts)

        else:
            print "Initializing {} regions".format(self.admin_boundary_m)
            self.init_tracts()
            CommunityArea.createAllCAs(Tract.tracts)
            print "Updating community structure from saved state:: m={}".format(m)
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
            if m == (self.admin_boundary_m):
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
        msg = "Beginning naice MCMC - {}.{}".format(m,iter)
        self.emit_log(msg)
        self.init_communities(m=m)
        pred_target = self.get_target(self.task)

        # estimate naive MCMC
        if iter:
            fname = self.project_name + "-naive-{}.{}".format(m, iter)

        else:
            fname = self.project_name + "-naive-{}".format(m)
        naive_MCMC(fname, targetName=pred_target,
                   lmbda=self.lmbda, f_sd=self.f_sd, Tt=self.T, init_ca=False)

    def softmax_mcmc_run(self, m, iter=None):
        msg = "Beginning Softmax MCMC - {}.{}".format(m,iter)
        self.emit_log(msg)
        self.init_communities(m=m)
        pred_target = self.get_target(self.task)
        # estimate MCMC with softmax proposal
        if iter:
            fname = self.project_name + "-softmax-{}.{}".format(m, iter)

        else:
            fname = self.project_name + "-softmax-{}".format(m)

        MCMC_softmax_proposal(fname, targetName=pred_target,
                              lmbda=self.lmbda, f_sd=self.f_sd, Tt=self.T, init_ca=False)

    def dqn_mcmc_run(self, m, iter=None):
        msg = "Beginning DQN MCMC - {}.{}".format(m,iter)
        self.emit_log(msg)
        self.init_communities(m=m)
        pred_target = self.get_target(self.task)
        # MCMC with proposal from DQN
        if iter:
            fname = self.project_name + "-dqn-{}.{}".format(m, iter)

        else:
            fname = self.project_name + "-dqn-{}".format(m)

        q_learning(fname,
                   targetName=pred_target,
                   lmbd=self.lmbda, f_sd=self.f_sd, Tt=self.T, init_ca=False)


    def kmeans_run(self, m, iter=None):
        msg = "Beginning KMeans Clustering - {}.{}".format(m,iter)
        self.emit_log(msg)
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
        msg = "Beginning Agglomerative Clustering - {}.{}".format(m,iter)
        self.emit_log(msg)
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
        msg = "Beginning Spectral Clustering - {}.{}".format(m,iter)
        self.emit_log(msg)
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

    def test_err(self):
        x = np.random.uniform(0,1,1)
        if x < .25:
            raise Exception("Testing error catching...")

    def run_sim(self, n_iter, gen_ca=False):
        self.config_log()
        m_grid = self.get_grid()

        if gen_ca:
            self.get_random_communities()

        for m in m_grid:
            for i in range(1, n_iter+1):
                prog = "m = {}, i = {}".format(m, i)
                print prog
                self.emit_log(prog)

                """try:
                    self.agglomerative_run(m, i)
                except:
                    logging.error(prog,exc_info=True)
                try:
                    self.kmeans_run(m, i)
                except:
                    logging.error(prog, exc_info=True)
                try:
                    self.spectral_run(m, i)
                except:
                    logging.error(prog, exc_info=True)
                try:
                    self.naive_mcmc_run(m, i)
                except:
                    logging.error(prog, exc_info=True)
                try:
                    self.softmax_mcmc_run(m, i)
                except:
                    logging.error(prog, exc_info=True)"""
                try:
                    self.dqn_mcmc_run(m, i)
                except:
                    logging.error(prog, exc_info=True)

        self.end_time = dt.datetime.now()
        msg = "Total running time: {}".format(self.end_time - self.start_time)
        self.emit_log(msg)


class ParamSensitivityPlotter(object):
    def __init__(self, project_name, max_m, min_m, task, n_iter, metric, mod_analysis_list=None):
        self.project_name = project_name
        self.max_m = max_m
        self.min_m = min_m
        self.task = task
        self.n_iter = n_iter
        self.metric = metric
        self.models = ['naive', 'softmax',
                       'dqn', 'kmeans', 'agglomerative', 'spectral']
        self.model_labels ={'naive': 'Naive',
                            'softmax': 'Softmax',
                            'dqn': 'DQN',
                            'kmeans': 'K-means',
                            'agglomerative': 'Agglomerative',
                            'spectral': 'Spectral'}
        if mod_analysis_list:
            self.mod_analysis_list = mod_analysis_list
        else:
            self.mod_analysis_list = self.models


    def get_task_str(self):
        if self.task == 'house_price':
            return 'houseprice'
        elif self.task == 'crime':
            return 'crime'
        else:
            raise ValueError('task: must be house_price or crime')

    def get_file_name(self, model, m, i):
        task_str = self.get_task_str()
        fname = "output/sensitivity-study-{}-{}-{}.{}-final-output.txt".format(task_str,
                                                                        model, m, i)
        return fname

    def get_results(self):
        results = list()

        for mod in self.models:
            for m in range(self.min_m, self.max_m+1):
                for i in range(1,self.n_iter+1):
                    fname = self.get_file_name(mod,m,i)
                    sim_result = self.get_final_output_file(fname)
                    row = [mod, m, i, sim_result]
                    results.append(row)

        results_df = pd.DataFrame(results, columns = ['model','m','i',self.metric])
        results_df.set_index(['model', 'm', 'i'], inplace=True)
        return results_df

    def get_final_output_file(self, fname):
        try:
            with open(fname, 'r') as f:
                for line in f:
                    line_split = line.split(":")
                    if line_split[0].strip() == self.metric:
                        result = float(line_split[1].strip())
                        break
                    else:
                        result = np.nan
            return result
        except IOError:
            result = np.nan
            return result

    def gen_plot(self):
        results = self.get_results()
        means = results.groupby(by=['model','m']).mean()
        std = results.groupby(by=['model','m']).std()

        print means
        print std

        plt.figure(figsize=(12,8))
        for mod in self.models:
            if mod in self.mod_analysis_list:
                arr = means.loc[mod]
                plt.plot(arr, label=self.model_labels[mod], linewidth=2.5, linestyle='-.')

        plt.xlim(self.max_m, self.min_m)
        plt.legend(loc='best')
        plt.title('Model Prediction Error by Number of Regions', fontsize=18)
        plt.xlabel('Number of Regions (m)', fontsize=15)
        plt.ylabel('Prediction Error ({})'.format(self.metric), fontsize=15)
        plt.savefig('plots/sensitivity-study-{}.pdf'.format(self.metric))



if __name__ == '__main__':


    #crime_sim = ParamSensitivity(project_name='sensitivity-study-crime', task='crime',
    #                             max_m=77, min_m=20, plot=True)

    #crime_sim.run_sim(n_iter=10, gen_ca=False)


    house_price_sim = ParamSensitivity(project_name='sensitivity-study-houseprice',
                                        task='house_price', max_m=77, min_m=40, plot=False,
                                       lmbda = 0.000001, f_sd=0.008, T=.01)
    house_price_sim.run_sim(n_iter=10, gen_ca=False)