import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class NeuralNetworkGAPrint {
    private int inputSize;
    private int[] hiddenLayerSizes;
    private int outputSize;
    private String activationType;
    private List<double[][]> weights;
    private List<double[]> biases;
    private Random rand;

    public NeuralNetworkGAPrint(int inputSize, int[] hiddenLayerSizes, int outputSize, String activationType) {
        this.inputSize = inputSize;
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.outputSize = outputSize;
        this.activationType = activationType;
        this.rand = new Random();

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
                    layerWeights[i][j] = rand.nextGaussian() * 0.1;
                }
            }
        }

        for (double[] layerBiases : biases) {
            for (int i = 0; i < layerBiases.length; i++) {
                layerBiases[i] = rand.nextGaussian() * 0.1;
            }
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

        for (int i = 0; i < weights.size(); i++) {
            double[] newOutput = new double[weights.get(i)[0].length];
            for (int j = 0; j < newOutput.length; j++) {
                double z = 0.0;
                for (int k = 0; k < currentOutput.length; k++) {
                    z += currentOutput[k] * weights.get(i)[k][j];
                }
                z += biases.get(i)[j];
                newOutput[j] = (i == weights.size() - 1 && activationType.equals("softmax")) ? z : sigmoid(z);
            }
            currentOutput = (i == weights.size() - 1 && activationType.equals("softmax")) ? softmax(newOutput) : newOutput;
        }

        return currentOutput;
    }

    // Genetic Algorithm
    public void trainGA(double[][] inputData, double[][] targetData, int populationSize, int generations, double mutationRate) {
        List<Chromosome> population = initializePopulation(populationSize);

        for (int gen = 0; gen < generations; gen++) {
            evaluatePopulation(population, inputData, targetData);
            population = selectNextGeneration(population, mutationRate);

            if (gen % 10 == 0) {
                System.out.println("Generation " + gen + ": Best fitness = " + population.get(0).fitness);
            }
        }

        // Set best weights and biases
        setWeightsAndBiases(population.get(0));
    }

    private List<Chromosome> initializePopulation(int populationSize) {
        List<Chromosome> population = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            Chromosome chromo = new Chromosome(weights, biases);
            population.add(chromo);
        }
        return population;
    }

    private void evaluatePopulation(List<Chromosome> population, double[][] inputData, double[][] targetData) {
        for (Chromosome chromo : population) {
            double totalLoss = 0.0;
            for (int i = 0; i < inputData.length; i++) {
                double[] predicted = chromo.forwardPass(inputData[i]);
                totalLoss += calculateLoss(targetData[i], predicted);
            }
            chromo.fitness = -totalLoss / inputData.length; // Negative loss for maximization
        }
        population.sort((a, b) -> Double.compare(b.fitness, a.fitness));
    }

    private List<Chromosome> selectNextGeneration(List<Chromosome> population, double mutationRate) {
        List<Chromosome> nextGeneration = new ArrayList<>();
        int eliteCount = (int) (0.1 * population.size()); // Keep top 10%
        nextGeneration.addAll(population.subList(0, eliteCount));

        Random rand = new Random();
        while (nextGeneration.size() < population.size()) {
            Chromosome parent1 = population.get(rand.nextInt(eliteCount));
            Chromosome parent2 = population.get(rand.nextInt(eliteCount));
            Chromosome offspring = parent1.crossover(parent2);
            offspring.mutate(mutationRate);
            nextGeneration.add(offspring);
        }
        return nextGeneration;
    }

    private double calculateLoss(double[] target, double[] predicted) {
        double loss = 0.0;
        for (int i = 0; i < target.length; i++) {
            loss += Math.pow(target[i] - predicted[i], 2);
        }
        return loss;
    }

    private void setWeightsAndBiases(Chromosome best) {
        this.weights = best.weights;
        this.biases = best.biases;
    }

    // Chromosome class for GA
    private class Chromosome {
        List<double[][]> weights;
        List<double[]> biases;
        double fitness;

        Chromosome(List<double[][]> weights, List<double[]> biases) {
            this.weights = cloneWeights(weights);
            this.biases = cloneBiases(biases);
        }

        double[] forwardPass(double[] input) {
            double[] currentOutput = input;
            for (int i = 0; i < weights.size(); i++) {
                double[] newOutput = new double[weights.get(i)[0].length];
                for (int j = 0; j < newOutput.length; j++) {
                    double z = 0.0;
                    for (int k = 0; k < currentOutput.length; k++) {
                        z += currentOutput[k] * weights.get(i)[k][j];
                    }
                    z += biases.get(i)[j];
                    newOutput[j] = (i == weights.size() - 1 && activationType.equals("softmax")) ? z : sigmoid(z);
                }
                currentOutput = (i == weights.size() - 1 && activationType.equals("softmax")) ? softmax(newOutput) : newOutput;
            }
            return currentOutput;
        }

        Chromosome crossover(Chromosome other) {
            Chromosome offspring = new Chromosome(this.weights, this.biases);
            for (int i = 0; i < weights.size(); i++) {
                for (int j = 0; j < weights.get(i).length; j++) {
                    for (int k = 0; k < weights.get(i)[j].length; k++) {
                        offspring.weights.get(i)[j][k] = (rand.nextBoolean()) ? this.weights.get(i)[j][k] : other.weights.get(i)[j][k];
                    }
                }
                for (int j = 0; j < biases.get(i).length; j++) {
                    offspring.biases.get(i)[j] = (rand.nextBoolean()) ? this.biases.get(i)[j] : other.biases.get(i)[j];
                }
            }
            return offspring;
        }

        void mutate(double mutationRate) {
            for (double[][] layerWeights : weights) {
                for (int i = 0; i < layerWeights.length; i++) {
                    for (int j = 0; j < layerWeights[i].length; j++) {
                        if (rand.nextDouble() < mutationRate) {
                            layerWeights[i][j] += rand.nextGaussian() * 0.1;
                        }
                    }
                }
            }
            for (double[] layerBiases : biases) {
                for (int i = 0; i < layerBiases.length; i++) {
                    if (rand.nextDouble() < mutationRate) {
                        layerBiases[i] += rand.nextGaussian() * 0.1;
                    }
                }
            }
        }
    }

    private List<double[][]> cloneWeights(List<double[][]> original) {
        List<double[][]> clone = new ArrayList<>();
        for (double[][] layer : original) {
            double[][] layerClone = new double[layer.length][];
            for (int i = 0; i < layer.length; i++) {
                layerClone[i] = layer[i].clone();
            }
            clone.add(layerClone);
        }
        return clone;
    }

    private List<double[]> cloneBiases(List<double[]> original) {
        List<double[]> clone = new ArrayList<>();
        for (double[] layer : original) {
            clone.add(layer.clone());
        }
        return clone;
    }
}
