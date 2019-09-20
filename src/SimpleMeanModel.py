import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Read data and find mean
    df = pd.read_csv('../data/main-df.csv')
    df = df[df.HIVincidence < 700]
    df_test = df.drop(columns='HIVincidence') # our X values
    incidence_mean = df['HIVincidence'].mean() # mean after removing outlier

    # Calculate MSE
    fhat = lambda X: np.ones(len(df_test))*incidence_mean
    yhat = fhat(df_test)
    mse = ((df['HIVincidence'] - yhat)**2).mean()
    rmse = np.sqrt(mse)
    incidence = df['HIVincidence'].sort_values()

    # Graph MSE
    fig, ax = plt.subplots(figsize=(12,7))
    xs = np.linspace(0,len(incidence),3139)
    ax.scatter( xs, incidence, label="data",s=1)
    ax.set_xlabel("Observations Ordered by Incidence",fontsize = 14)
    ax.set_ylabel("Incidence Rate",fontsize = 14)
    ax.set_title("MSE Using Mean",fontsize = 18)

    ax.plot( xs, yhat, c="red", label="model" )
    ax.legend()