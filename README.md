# worldcup2018

This repository contains a Python code to run Monte Carlo simulations on upcoming World Cup 2018 matches, using eXtreme Gradient Boosting on historical team stats and match results from Fifa.com.

To run, type: python main.py

This will load the datasets, prepare them for the model, train the xgboost model, and run a number (default: 1000) of simulations of matches between all the teams on a given list. The output is the winner of each of the matches, and probability of the team winning.
