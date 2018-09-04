from hyperparam_sensitivity import ParamSensitivityPlotter


hp = house_price_sim = ParamSensitivityPlotter(project_name='sensitivity-study-houseprice',
                                               task='house_price', max_m=77, min_m=40, n_iter=10,
                                               metric='mae')

hp.gen_plot()