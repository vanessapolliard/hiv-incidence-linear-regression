Regression Case Study
======================

In today's exercise you'll get a chance to try some of what you've learned about
supervised learning on a real-world problem.

The goal of the contest is to predict the sale price of a particular piece of
heavy equipment at auction based on it's usage, equipment type, and configuration.
The data is sourced from auction result postings and includes information on usage
and equipment configurations.

Evaluation
======================
The evaluation of your model will be based on Root Mean Squared Log Error. Which
is computed as follows: 

![](images/rmsle.png)

where *p<sub>i</sub>* are the predicted values and *a<sub>i</sub>* are the target
values.

See the code in score_model.py for details.

Setup
======================
Run
`pip install git+https://github.com/zipfian/performotron.git`

Data
======================
The data for this case study are in `./data`. Although there are both training and testing data sets,
the testing data set will only be utilized to evaluate your final model performance.  In other words, 
you should use cross-validation on the training data set to identify potential models, then score those models on 
the test data.

In order to score your model, you will need to 
output your predictions in the format specified in `data/median_benchmark.csv`. Then
you can submit your solution for evaluation using the command:

    python score_model.py data/median_benchmark.csv

Note that this will announce your score on Slack to everybody else, so you should
do this only when you feel you have a high-quality solution.



