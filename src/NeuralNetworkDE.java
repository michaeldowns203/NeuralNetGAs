import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class NeuralNetworkDE {
    private int inputSize;
    private int[] hiddenLayerSizes;
    private int outputSize;
    private String activationType;
    private List<double[][]> weights;
    private List<double[]> biases;
    private Random rand;

    public NeuralNetworkDE(int inputSize, int[] hiddenLayerSizes, int outputSize, String activationType) {
        this.inputSize = inputSize;
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.outputSize = outputSize;
        this.activationType = activationType;
        rand = new Random();
        initializeWeights();
    }

    private void initializeWeights() {
        weights = new ArrayList<>();
        biases = new ArrayList<>();

        if (hiddenLayerSizes.length == 0) {
            weights.add(new double[inputSize][outputSize]);
            biases.add(new double[outputSize]);
        } else {
            weights.add(new double[inputSize][hiddenLayerSizes[0]]);
            biases.add(new double[hiddenLayerSizes[0]]);

            for (int i = 1; i < hiddenLayerSizes.length; i++) {
                weights.add(new double[hiddenLayerSizes[i - 1]][hiddenLayerSizes[i]]);
                biases.add(new double[hiddenLayerSizes[i]]);
            }

            weights.add(new double[hiddenLayerSizes[hiddenLayerSizes.length - 1]][outputSize]);
            biases.add(new double[outputSize]);
        }

        for (double[][] layerWeights : weights) {
            for (int i = 0; i < layerWeights.length; i++) {
                for (int j = 0; j < layerWeights[i].length; j++) {
                    layerWeights[i][j] = rand.nextGaussian() * Math.sqrt(1.0 / (layerWeights.length + layerWeights[0].length));
                }
            }
        }

        for (int i = 0; i < biases.size(); i++) {
            Arrays.fill(biases.get(i), rand.nextGaussian() * 0.01);
        }
    }

    // Sigmoid activation function
    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    // Softmax activation function
    private double[] softmax(double[] z) {
        double max = Arrays.stream(z).max().orElse(0.0);
        double sum = 0.0;
        double[] output = new double[z.length];

        for (int i = 0; i < z.length; i++) {
            output[i] = Math.exp(z[i] - max);
            sum += output[i];
        }

        for (int i = 0; i < z.length; i++) {
            output[i] /= sum;
        }

        return output;
    }

    public double[] forwardPass(double[] input) {
        double[] currentOutput = input;

        for (int i = 0; i < weights.size() - 1; i++) {
            double[] newOutput = new double[weights.get(i)[0].length];

            for (int j = 0; j < newOutput.length; j++) {
                double z = 0.0;
                for (int k = 0; k < currentOutput.length; k++) {
                    z += currentOutput[k] * weights.get(i)[k][j];
                }
                z += biases.get(i)[j];
                newOutput[j] = sigmoid(z);
            }

            currentOutput = newOutput;
        }

        double[] finalOutput = new double[outputSize];
        double[] z = new double[outputSize];

        for (int j = 0; j < outputSize; j++) {
            z[j] = 0.0;
            for (int k = 0; k < currentOutput.length; k++) {
                z[j] += currentOutput[k] * weights.get(weights.size() - 1)[k][j];
            }
            z[j] += biases.get(biases.size() - 1)[j];
        }

        if (activationType.equals("softmax")) {
            finalOutput = softmax(z);
        } else {
            finalOutput = z;
        }

        return finalOutput;
    }

    // Differential Evolution Training
    public void trainWithDifferentialEvolution(double[][] inputData, double[][] targetData, int populationSize, double crossoverRate, double differentialWeight, int maxGenerations) {
        List<Individual> population = initializePopulation(populationSize);

        for (int generation = 0; generation < maxGenerations; generation++) {
            List<Individual> newPopulation = new ArrayList<>();

            for (Individual target : population) {
                Individual mutant = mutate(population, target, differentialWeight);
                Individual trial = crossover(target, mutant, crossoverRate);
                evaluateFitness(trial, inputData, targetData);

                if (trial.fitness < target.fitness) {
                    newPopulation.add(trial);
                } else {
                    newPopulation.add(target);
                }
            }

            population = newPopulation;
            printBestIndividual(population, generation);
        }
    }

    private List<Individual> initializePopulation(int populationSize) {
        List<Individual> population = new ArrayList<>();

        for (int i = 0; i < populationSize; i++) {
            Individual individual = new Individual();
            individual.initializeRandomWeights(weights, biases, rand);
            population.add(individual);
        }

        return population;
    }

    private Individual mutate(List<Individual> population, Individual target, double differentialWeight) {
        Random rand = new Random();
        int a, b, c;
        do { a = rand.nextInt(population.size()); } while (a == population.indexOf(target));
        do { b = rand.nextInt(population.size()); } while (b == a || b == population.indexOf(target));
        do { c = rand.nextInt(population.size()); } while (c == b || c == a || c == population.indexOf(target));

        Individual indA = population.get(a);
        Individual indB = population.get(b);
        Individual indC = population.get(c);

        return indA.mutate(indB, indC, differentialWeight);
    }

    private Individual crossover(Individual target, Individual mutant, double crossoverRate) {
        return target.crossover(mutant, crossoverRate, rand);
    }

    private void evaluateFitness(Individual individual, double[][] inputData, double[][] targetData) {
        double totalLoss = 0.0;

        for (int i = 0; i < inputData.length; i++) {
            double[] predictedOutput = forwardPass(inputData[i]);
            double[] targetOutput = targetData[i];

            double loss = 0.0;
            for (int j = 0; j < targetOutput.length; j++) {
                loss += Math.pow(targetOutput[j] - predictedOutput[j], 2);
            }
            totalLoss += loss;
        }

        individual.fitness = totalLoss / inputData.length;
    }

    private void printBestIndividual(List<Individual> population, int generation) {
        Individual best = population.stream().min((a, b) -> Double.compare(a.fitness, b.fitness)).orElse(null);
        if (best != null) {
            System.out.println("Generation " + generation + " - Best Fitness: " + best.fitness);
        }
    }

    // Inner class to represent an individual in the population
    private class Individual {
        List<double[][]> weights;
        List<double[]> biases;
        double fitness;

        Individual() {
            weights = new ArrayList<>();
            biases = new ArrayList<>();
        }

        void initializeRandomWeights(List<double[][]> initialWeights, List<double[]> initialBiases, Random rand) {
            for (double[][] layerWeights : initialWeights) {
                double[][] newWeights = new double[layerWeights.length][layerWeights[0].length];
                for (int i = 0; i < layerWeights.length; i++) {
                    for (int j = 0; j < layerWeights[i].length; j++) {
                        newWeights[i][j] = rand.nextDouble() * 2 - 1;
                    }
                }
                weights.add(newWeights);
            }

            for (double[] layerBiases : initialBiases) {
                double[] newBiases = new double[layerBiases.length];
                for (int i = 0; i < layerBiases.length; i++) {
                    newBiases[i] = rand.nextDouble() * 2 - 1;
                }
                biases.add(newBiases);
            }
        }

        Individual mutate(Individual indB, Individual indC, double differentialWeight) {
            Individual mutant = new Individual();
            for (int i = 0; i < weights.size(); i++) {
                double[][] newWeights = new double[weights.get(i).length][weights.get(i)[0].length];
                for (int j = 0; j < newWeights.length; j++) {
                    for (int k = 0; k < newWeights[j].length; k++) {
                        newWeights[j][k] = weights.get(i)[j][k] + differentialWeight * (indB.weights.get(i)[j][k] - indC.weights.get(i)[j][k]);
                    }
                }
                mutant.weights.add(newWeights);

                double[] newBiases = new double[biases.get(i).length];
                for (int j = 0; j < newBiases.length; j++) {
                    newBiases[j] = biases.get(i)[j] + differentialWeight * (indB.biases.get(i)[j] - indC.biases.get(i)[j]);
                }
                mutant.biases.add(newBiases);
            }
            return mutant;
        }

        Individual crossover(Individual mutant, double crossoverRate, Random rand) {
            Individual trial = new Individual();
            for (int i = 0; i < weights.size(); i++) {
                double[][] newWeights = new double[weights.get(i).length][weights.get(i)[0].length];
                for (int j = 0; j < newWeights.length; j++) {
                    for (int k = 0; k < newWeights[j].length; k++) {
                        newWeights[j][k] = rand.nextDouble() < crossoverRate ? mutant.weights.get(i)[j][k] : weights.get(i)[j][k];
                    }
                }
                trial.weights.add(newWeights);

                double[] newBiases = new double[biases.get(i).length];
                for (int j = 0; j < newBiases.length; j++) {
                    newBiases[j] = rand.nextDouble() < crossoverRate ? mutant.biases.get(i)[j] : biases.get(i)[j];
                }
                trial.biases.add(newBiases);
            }
            return trial;
        }
    }
}

