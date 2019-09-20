import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_model(y_actual, yhat, fig_title):
    '''
    INPUTS:
    y_actual - np array of actual y values
    yhat - np array of estimated y values
    fig_title - string of title
    
    '''
    y_actual.sort()
    fig, ax = plt.subplots(figsize=(12,7))
    xs = np.linspace(0,len(y_actual),len(y_actual))
    # Plot Actual values
    ax.scatter( xs, y_actual, label="data",s=1)
    ax.set_xlabel("Observations Ordered by Incidence",fontsize = 14)
    ax.set_ylabel("Incidence Rate",fontsize = 14)
    ax.set_title(fig_title,fontsize = 18)
    
    # Plot Estimated values
    ax.plot(xs, yhat, c="red", label="model" )
    ax.legend()