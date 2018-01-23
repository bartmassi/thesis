This code is for analyzing behavior collected from non-human primates performing an arithmetic task, 
that requires them to add and subtract small quantities. Data is stored in a SQLite database that
is not publically available. THIS CODE IS UNDER ACTIVE DEVELOPMENT.

There are three main code libraries:

Helper - this contains various helper functions for analyzing this dataset. Some are related to querying the SQL database, others get information about each trialset.

Analyzer - this contains functions related to fitting models using maximum likelihood, regression, and other common analysis.

Plotter - this contains functions related to plotting code. 

Anything else is a script. Scripts that begin with 'TEST' are playgrounds for development and testing, or where exploratory analysis takes place. Thus, this code is messy, excessively segmented, and may not run.
