import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


def plot_learning_curve(time_steps_to_failure):
    # A log plot may show the convergence better, as the learning curve is
    # typically jagged even on convergence.
    log_tstf = np.log(time_steps_to_failure).reshape((1, time_steps_to_failure.shape[0]))
    plt.plot(log_tstf,'k')
 
    # Compute simple moving average.
    window = 50
    i = np.arange(window)
    w = np.ones(window) / window
    weights = sig.lfilter(w, 1, log_tstf)
    
    x1 = window / np.arange(1, log_tstf.shape[1]+1) - (window / 2)
    plot1 = plt.plot(x1[window:log_tstf.shape[1]], weights[window:log_tstf.shape[1]], 'r--', linewidth=2)

    return plot1
