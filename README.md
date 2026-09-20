# Neural Network Training with Evolutionary Optimization

A project comparing **genetic algorithms, differential evolution, and particle swarm optimization** for training feed-forward neural networks, with backpropagation results as a baseline. The study examines how optimizer choice, network depth, and dataset characteristics affect classification and regression performance.

[Read the full paper](FinalPaper.pdf) · [Source code](src/main/)

## What is implemented

- **Genetic algorithm:** real-valued chromosomes, selection, crossover, mutation, and elitism.
- **Differential evolution:** population-based mutation, crossover, and selection of candidate network parameters.
- **Particle swarm optimization:** particle velocities, personal/global best positions, and velocity clamping.
- A shared feed-forward network representation and forward evaluation.
- Dataset preprocessing, min-max scaling, one-hot encoding, loss calculations, and cross-validation utilities.

The repository includes separate classes for the optimizers, network, preprocessing, and dataset-specific experiment drivers.

## Experimental study

The paper compares networks with **zero, one, and two hidden layers** across six datasets:

| Task | Datasets | Evaluation metric |
| --- | --- | --- |
| Classification | Wisconsin breast cancer, soybean, glass | 0/1 loss |
| Regression | Computer hardware, abalone, forest fires | Mean squared error |

The study describes stratified ten-fold cross-validation and hyperparameter tuning, and also examines average convergence rate where applicable. Backpropagation is described in more detail in the companion [NeuralNetBackprop project](https://github.com/michaeldowns203/NeuralNetBackprop).

## Run an experiment

Use a Java JDK with `javac` and `java` available. From the repository root:

```bash
  mkdir -p out
  javac -d out $(find src/main -name '*.java')
  cp -R src/resources/* out/
  java -cp out main.drivers.ComputerDriver
```

The inspected `ComputerDriver` reads `data/machine.data` and currently enables the differential-evolution configuration. Its genetic-algorithm and particle-swarm alternatives are present in a commented block. Review the selected optimizer and its parameters before running or comparing results.

The other dataset drivers include `AbaloneDriver`, `BreastDriver`, `ForestDriver`, `GlassDriver`, and `SoybeanDriver`, with additional variants for different experiment workflows. Parameters are configured in source.

## Repository guide

| File | Purpose |
| --- | --- |
| [`src/NeuralNetwork2.java`](src/main/utils/NeuralNetwork2.java) | Network representation and forward evaluation. |
| [`src/GA.java`](src/main/ga/GA.java), [`src/GAC.java`](src/main/ga/GAC.java) | Genetic-algorithm implementations. |
| [`src/DE.java`](src/main/de/DE.java) | Differential evolution. |
| [`src/PSO.java`](src/main/pso/PSO.java), [`src/PSOC.java`](src/main/pso/PSOC.java) | Particle-swarm implementations. |
| [`src/MinMaxScale.java`](src/main/utils/MinMaxScale.java), [`src/OneHotEncoder.java`](src/main/utils/OneHotEncoder.java) | Preprocessing. |
| [`src/LossFunctions.java`](src/main/utils/LossFunctions.java), [`src/TenFoldCrossValidation.java`](src/main/utils/TenFoldCrossValidation.java) | Evaluation utilities. |

## Limitations and reproducibility

There was a limited tuning budget: population-based hyperparameters were tuned for two-hidden-layer networks and reused for shallower architectures. Six datasets cannot establish general superiority across other tasks. Random initialization, stopping criteria, preprocessing, and fold construction also affect comparisons.

Before relying on fresh benchmark claims, validate the cross-validation utilities, ensure tuning/test sets are disjoint, and record the exact driver, parameters, and random seeds. The repository's experiment drivers should be treated as study code rather than a packaged, reproducible benchmark suite.

## Contributors

- **Michael Downs:** differential evolution and particle swarm optimization.
- **Max Hymer:** genetic algorithm.
