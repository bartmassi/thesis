# thesis
Code for manipulating a SQL database with experiment data. ACTIVELY USED AND UNDER DEVELOPMENT.

This code is for analyzing behavior collected from non-human primates performing an arithmetic task, 
that requires them to add and subtract small quantities. Data is stored in a sqlite database that
is not publically available. 

There are three main code libraries:
Helper - this contains various helper functions for analyzing this dataset. Some are related to querying the SQL database, others get information about each trialset.
Analyzer - this contains functions related to fitting models using maximum likelihood, regression, and other common analysis.
Plotter - this contains functions related to plotting code. 

Anything else is a script. Scripts that begin with 'test' are playgrounds for development and testing, or where exploratory analysis takes place. 

For most purposes, 'test.py' is the driver and contains multiple independent scripts for conducting analysis and visualizing data. In general, each code cell represents an independent script.
