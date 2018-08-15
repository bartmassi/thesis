This code is for analyzing behavior collected from non-human primates performing an arithmetic task, 
that requires them to add and subtract small quantities. SQLite database with experimental data is not included. 

################################################################
################################################################
################################################################

There are three main function libraries:

Analyzer - this contains functions related to fitting models using maximum likelihood, regression, and other common analysis. Does not include rudimentary statistical analysis. This also contains information about process models that are fit.

Plotter - this contains functions related to plotting code. Format conforms to Lee Lab's standard. Mostly cosmetic, but includes some functionality for conveniently making certain types of plots (e.g., like the panelplots function).

Helper - this contains various helper functions for analyzing this dataset. Some are related to querying the SQL database, others get information about each experiment.


################################################################
################################################################
################################################################

There are two main scripts:

compute_stats: this contains statistical analysis of my dataset. Performs hypothesis testing, and regression-modeling. This builds up a list of outputs of specific tests, which are then written to a file.

make_figures: this contains analysis and plotting routines to make the figures that are included in my thesis. 

################################################################
################################################################
################################################################

Anything else is a script or for development purposes, and there is no guarantee that it will run. Scripts that begin with 'TEST' are playgrounds for development, and where exploratory analysis takes place. These files are intended for my use only.
