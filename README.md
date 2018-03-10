# ML homework 2: randomised optimisation

This repo is mostly a fork of [ABAGAIL](https://github.com/pushkar/ABAGAIL), 
with some code added for the homework inside the examples.

## File changed

 - src/opt/test/AbaloneTestStarcraft.java
 - src/opt/test/OptimizationTest.java
 
## Build

To compile this project you just need java compiler installed (tested with openjdk 1.8).

A makefile is provided to make build process easier, you just have to run `make` in the root directory.

## Running

The build process must have generated a `AbaloneTestStarcraft.class` and `OptimizationTest.class` inside the a bin directory.

You just have to run the one you need on your JVM.

 - AbaloneTestStarcraft: Run all the experiment on the neural network, this can take some hours / days.
 It's possible that you will need to use the `-Xmx` option to allow java to use more than 1 Gio of Ram.
 
 - OptimizationTest: Run all the experiment on different problems (some of them not discussed on the report).
 A full run might also require some hours / days to complete.
 
In both case the computation is parallelized on multiple threads
and result are writen to the OptimizationResults and Optimisation_Results respectively.

## Plotting

All the script needed to generate the plot are in the `plotter` directory.

They collect the data from a sqlite database: `plotter/problems.db` which need to be feed manually using with the csv results.