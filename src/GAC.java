import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class GAC {
    private List<NeuralNetwork2> population;
    private List<Double> fitness; // Store fitness values for each individual
    private final int populationSize;
    private final double mutationRate;
    private final double crossoverRate;
    private final Random random;
    private List<Double> fitnessHistory;

    public GAC(int populationSize, double mutationRate, double crossoverRate) {
        this.populationSize = populationSize;
        this.mutationRate = mutationRate;
        this.crossoverRate = crossoverRate;
        this.random = new Random();
        this.population = new ArrayList<>();
        this.fitness = new ArrayList<>();
        this.fitnessHistory = new ArrayList<>();
    }

    // Initialize population with random individuals
    public void initializePopulation(int inputSize, int[] hiddenLayerSizes, int outputSize, String activationType) {
        for (int i = 0; i < populationSize; i++) {
            NeuralNetwork2 nn = new NeuralNetwork2(inputSize, hiddenLayerSizes, outputSize, activationType);
            population.add(nn);
            fitness.add(0.0); // Initialize fitness to 0
        }
    }

    // Convert a neural network's weights to a chromosome (array of real values)
    private double[] networkToChromosome(NeuralNetwork2 nn) {
        List<Double> chromosome = new ArrayList<>();
        for (double[][] layerWeights : nn.getWeights()) {
            for (double[] row : layerWeights) {
                for (double weight : row) {
                    chromosome.add(weight);
                }
            }
        }
        for (double[] biases : nn.getBiases()) {
            for (double bias : biases) {
                chromosome.add(bias);
            }
        }
        return chromosome.stream().mapToDouble(Double::doubleValue).toArray();
    }

    // Convert a chromosome back to a neural network's weights and biases
    private void chromosomeToNetwork(double[] chromosome, NeuralNetwork2 nn) {
        int index = 0;
        for (double[][] layerWeights : nn.getWeights()) {
            for (int i = 0; i < layerWeights.length; i++) {
                for (int j = 0; j < layerWeights[i].length; j++) {
                    layerWeights[i][j] = chromosome[index++];
                }
            }
        }
        for (double[] biases : nn.getBiases()) {
            for (int i = 0; i < biases.length; i++) {
                biases[i] = chromosome[index++];
            }
        }
    }

    // Evaluate the fitness of the population
    private void evaluateFitness(double[][] input, double[][] target) {
        for (int i = 0; i < population.size(); i++) {
            NeuralNetwork2 nn = population.get(i);
            double loss = evaluate(nn, input, target);
            fitness.set(i, -loss); // Use negative loss as fitness (higher fitness is better)
        }
    }

    // Cross-entropy loss calculation
    private double evaluate(NeuralNetwork2 nn, double[][] input, double[][] target) {
        double totalLoss = 0.0;
        for (int i = 0; i < input.length; i++) {
            double[] predicted = nn.forwardPass(input[i]); // Softmax output
            double[] actual = target[i]; // One-hot encoded target
            totalLoss += crossEntropyLoss(predicted, actual);
        }
        return totalLoss / input.length; // Average loss
    }

    private double crossEntropyLoss(double[] predicted, double[] actual) {
        double loss = 0.0;
        for (int i = 0; i < actual.length; i++) {
            // Avoid log(0) by adding a small constant
            loss += actual[i] * Math.log(predicted[i] + 1e-15);
        }
        return -loss;
    }

    // Perform roulette wheel selection
    private NeuralNetwork2 selectParent() {
        double totalFitness = fitness.stream().mapToDouble(Double::doubleValue).sum();
        double randomValue = random.nextDouble() * totalFitness;
        double cumulativeFitness = 0.0;

        for (int i = 0; i < population.size(); i++) {
            cumulativeFitness += fitness.get(i);
            if (cumulativeFitness >= randomValue) {
                return population.get(i);
            }
        }
        return population.get(population.size() - 1); // Fallback
    }

    // Perform uniform crossover
    private double[] crossover(double[] parent1, double[] parent2) {
        double[] child = new double[parent1.length];
        for (int i = 0; i < parent1.length; i++) {
            if (random.nextDouble() < crossoverRate) {
                child[i] = parent1[i];
            } else {
                child[i] = parent2[i];
            }
        }
        return child;
    }

    // Perform mutation
    private void mutate(double[] chromosome) {
        for (int i = 0; i < chromosome.length; i++) {
            if (random.nextDouble() < mutationRate) {
                chromosome[i] += random.nextGaussian() * 0.1; // Small mutation
            }
        }
    }

    public NeuralNetwork2 run(int inputSize, int[] hiddenLayerSizes, int outputSize, String activationType, double[][] input,
                              double[][] target, double convergenceThreshold, int patience) {
        initializePopulation(inputSize, hiddenLayerSizes, outputSize, activationType);

        double previousBestFitness = Double.NEGATIVE_INFINITY;
        int stableGenerations = 0;

        int generation = 0;
        while (stableGenerations < patience) {
            evaluateFitness(input, target);

            // Identify the best individual
            int bestIndex = fitness.indexOf(Collections.max(fitness));
            NeuralNetwork2 bestIndividual = population.get(bestIndex);

            double bestFitness = fitness.get(bestIndex);
            fitnessHistory.add(bestFitness); // Track best fitness in this generation
            System.out.println("Generation " + generation + ": Best Fitness = " + bestFitness);

            // Check for convergence
            if (Math.abs(bestFitness - previousBestFitness) < convergenceThreshold) {
                stableGenerations++;
            } else {
                stableGenerations = 0;
            }
            previousBestFitness = bestFitness;

            // Create new population with elitism
            List<NeuralNetwork2> newPopulation = new ArrayList<>();
            List<Double> newFitness = new ArrayList<>();

            // Add the best individual to the new population (elitism)
            newPopulation.add(bestIndividual);
            newFitness.add(fitness.get(bestIndex));

            for (int i = 1; i < populationSize / 2; i++) {
                NeuralNetwork2 parent1 = selectParent();
                NeuralNetwork2 parent2 = selectParent();

                double[] parent1Chromosome = networkToChromosome(parent1);
                double[] parent2Chromosome = networkToChromosome(parent2);

                double[] child1Chromosome = crossover(parent1Chromosome, parent2Chromosome);
                double[] child2Chromosome = crossover(parent1Chromosome, parent2Chromosome);

                mutate(child1Chromosome);
                mutate(child2Chromosome);

                NeuralNetwork2 child1 = new NeuralNetwork2(inputSize, hiddenLayerSizes, outputSize, activationType);
                NeuralNetwork2 child2 = new NeuralNetwork2(inputSize, hiddenLayerSizes, outputSize, activationType);

                chromosomeToNetwork(child1Chromosome, child1);
                chromosomeToNetwork(child2Chromosome, child2);

                newPopulation.add(child1);
                newFitness.add(0.0);
                newPopulation.add(child2);
                newFitness.add(0.0);
            }

            // Update population and fitness
            population = newPopulation;
            fitness = newFitness;
            evaluateFitness(input, target);

            generation++;
        }

        // Return the best individual
        int bestIndex = fitness.indexOf(Collections.max(fitness));
        return population.get(bestIndex);
    }

    // Method to calculate the average convergence rate
    public double getAverageConvergenceRate() {
        if (fitnessHistory.size() < 2) {
            return 0.0; // Not enough data to compute the rate
        }

        double totalRate = IntStream.range(1, fitnessHistory.size())
                .mapToDouble(i -> Math.abs(fitnessHistory.get(i) - fitnessHistory.get(i - 1)))
                .sum();

        return totalRate / (fitnessHistory.size() - 1); // Average rate of fitness change
    }
}
