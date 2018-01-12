import matplotlib.pyplot as plt
import numpy as np


def plotMcmcDiagnostics(error_array,variance_array,fname='mcmc-diagnostics'):
    x = range(len(error_array))
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(2, sharex=True,figsize=(12,8))
    axarr[0].plot(x, np.array(error_array))
    axarr[0].set_title('Mean Absolute Error')
    axarr[1].plot(x, np.array(variance_array))
    axarr[1].set_title('Population Variance (over communities)')

    plt.savefig(fname)
    plt.close()
    plt.clf()
