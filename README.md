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
is computed as root mean of the log of the prediction+1 minus the log of the
target+1 squared. 

![rmlse](https://github.com/zipfian/regression-case-study/blob/master/images/rmlse.png)

See the code in score_model.py for details.

Setup
======================
Run
`pip install git+https://github.com/zipfian/performotron.git`

Data
======================
The data for this case study are in `./data`. There are training and testing data sets.
You will create scores on the test data only to evaluate your performance. You will
output your predictions in the format specified in `data/median_benchmark.csv`. Then
you can submit your solution for evaluation using the command:

    python score_model.py data/median_benchmark.csv

Note that this will announce your score on Slack to everybody else, so you should
do this only when you feel you have a high-quality solution.

To decide whether you have such a solution, you should do cross-validation using data
only from the training data.

