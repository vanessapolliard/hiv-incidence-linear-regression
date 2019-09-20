# Forecasting HIV Infections

## EDA
Using the combined dataframe provided by Eric Logue's capstone project, we began by exploring the data values and correlations.

![Pairplot](https://github.com/vanessapolliard/regression-case-study/tree/Denver/images/sns_pairplot.png)

![EDA](https://github.com/vanessapolliard/regression-case-study/tree/Denver/images/eda.png)

![Diagnoses by State](https://github.com/vanessapolliard/regression-case-study/blob/Denver/images/HIV_diagnoses_by_state.png)

![Normalized Diagnoses by State](https://github.com/vanessapolliard/regression-case-study/blob/Denver/images/HIV_diagnoses_per_100000.png)

## Modeling

### Cleaning
We began the cleaning process by removing all rows in the data with null values. We also removed an outlier that had an incidence rate of over 700. Although approximately 2/3 of the county data had 0 incidence rates we decided to continue with these values and, time permitting, run regression models after removing the 0 values.

### Simple Model
As a baseline we began by calcualting RMSE if we used the mean HIV incidence rate. This gave us an RMSE of 9.3 to beat. Given that our mean incidence rate was 4 we didn't expect to have any difficulties generating a better model.

![Mean Model](https://github.com/vanessapolliard/regression-case-study/tree/Denver/images/mean_model.png)

### Regression Regularization

![CV Model Performance](https://github.com/vanessapolliard/regression-case-study/blob/Denver/images/model_performance_across_alphas.png)
![Ridge Coeffs](https://github.com/vanessapolliard/regression-case-study/blob/Denver/images/ridge_coefs.png)