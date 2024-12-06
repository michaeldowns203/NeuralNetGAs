import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DE {
    private int populationSize;
    private int maxNoImprovementGenerations;
    private double mutationFactor; // Typically 0.5 ≤ F ≤ 1
    private double crossoverRate; // Typically 0.7 ≤ CR ≤ 0.9
    private double tolerance;
    private NeuralNetwork2[] population; // Population of neural networks
    private NeuralNetwork2 bestIndividual;
    private double[] bestFitness;
    private Random random;

    public DE(int populationSize, int maxNoImprovementGenerations, double mutationFactor, double crossoverRate, double tolerance) {
        this.populationSize = populationSize;
        this.maxNoImprovementGenerations = maxNoImprovementGenerations;
        this.mutationFactor = mutationFactor;
        this.crossoverRate = crossoverRate;
        this.tolerance = tolerance;
        this.random = new Random();
    }

    public NeuralNetwork2 optimize(double[][] input, double[] target) {
        initializePopulation(input[0].length);

        int generation = 0;
        int noImprovementCount = 0;
        double previousBestFitness = -Double.MAX_VALUE;

        while (noImprovementCount < maxNoImprovementGenerations) {
            for (int i = 0; i < populationSize; i++) {
                NeuralNetwork2 trial = mutateAndCrossover(i);
                double trialFitness = evaluate(trial, input, target);

                if (trialFitness > bestFitness[i]) {
                    population[i] = trial;
                    bestFitness[i] = trialFitness;

                    // Update global best individual
                    if (trialFitness > evaluate(bestIndividual, input, target)) {
                        bestIndividual = trial;
                    }
                }
            }

            double currentBestFitness = evaluate(bestIndividual, input, target);
            System.out.println("Generation " + generation + " Best Fitness: " + currentBestFitness);

            if (Math.abs(previousBestFitness - currentBestFitness) < tolerance) {
                noImprovementCount++;
            } else {
                noImprovementCount = 0;
            }

            previousBestFitness = currentBestFitness;
            generation++;
        }

        return bestIndividual;
    }

    public NeuralNetwork2 optimize(double[][] input, double[][] target) {
        initializePopulation(input[0].length);

        int generation = 0;
        int noImprovementCount = 0;
        double previousBestFitness = -Double.MAX_VALUE;

        while (noImprovementCount < maxNoImprovementGenerations) {
            for (int i = 0; i < populationSize; i++) {
                NeuralNetwork2 trial = mutateAndCrossover(i);
                double trialFitness = evaluate(trial, input, target);

                if (trialFitness > bestFitness[i]) {
                    population[i] = trial;
                    bestFitness[i] = trialFitness;

                    // Update global best individual
                    if (trialFitness > evaluate(bestIndividual, input, target)) {
                        bestIndividual = trial;
                    }
                }
            }
            double currentBestFitness = evaluate(bestIndividual, input, target);
            System.out.println("Generation " + generation + " Best Fitness: " + currentBestFitness);

            if (Math.abs(previousBestFitness - currentBestFitness) < tolerance) {
                noImprovementCount++;
            } else {
                noImprovementCount = 0;
            }

            previousBestFitness = currentBestFitness;
            generation++;
        }

        return bestIndividual;
    }


    private void initializePopulation(int inputSize) {
        int[] hiddenLayerSizes = {4, 2};
        int outputSize = 1;
        String activationType = "linear";
        population = new NeuralNetwork2[populationSize];
        bestFitness = new double[populationSize];
        for (int i = 0; i < populationSize; i++) {
            population[i] = new NeuralNetwork2(inputSize, hiddenLayerSizes, outputSize, activationType);
            bestFitness[i] = -Double.MAX_VALUE;
        }
        bestIndividual = population[0];
    }

    private NeuralNetwork2 mutateAndCrossover(int targetIndex) {
        int a, b, c;
        do {
            a = random.nextInt(populationSize);
        } while (a == targetIndex);

        do {
            b = random.nextInt(populationSize);
        } while (b == targetIndex || b == a);

        do {
            c = random.nextInt(populationSize);
        } while (c == targetIndex || c == a || c == b);

        // Mutation and Crossover
        List<double[][]> x_a = population[a].getWeights();
        List<double[][]> x_b = population[b].getWeights();
        List<double[][]> x_c = population[c].getWeights();
        List<double[][]> v = new ArrayList<>();

        for (int layer = 0; layer < x_a.size(); layer++) {
            double[][] layerWeightsA = x_a.get(layer);
            double[][] layerWeightsB = x_b.get(layer);
            double[][] layerWeightsC = x_c.get(layer);
            double[][] mutatedLayer = new double[layerWeightsA.length][layerWeightsA[0].length];

            for (int i = 0; i < layerWeightsA.length; i++) {
                for (int j = 0; j < layerWeightsA[i].length; j++) {
                    mutatedLayer[i][j] = layerWeightsA[i][j] +
                            mutationFactor * (layerWeightsB[i][j] - layerWeightsC[i][j]);
                }
            }
            v.add(mutatedLayer);
        }

        List<double[][]> x_target = population[targetIndex].getWeights();
        List<double[][]> u = new ArrayList<>();

        for (int layer = 0; layer < v.size(); layer++) {
            double[][] mutatedLayer = v.get(layer);
            double[][] targetLayer = x_target.get(layer);
            double[][] crossoverLayer = new double[mutatedLayer.length][mutatedLayer[0].length];

            for (int i = 0; i < mutatedLayer.length; i++) {
                for (int j = 0; j < mutatedLayer[i].length; j++) {
                    if (random.nextDouble() < crossoverRate) {
                        crossoverLayer[i][j] = mutatedLayer[i][j];
                    } else {
                        crossoverLayer[i][j] = targetLayer[i][j];
                    }
                }
            }
            u.add(crossoverLayer);
        }

        NeuralNetwork2 trial = population[targetIndex].copy();
        trial.setWeights(u);
        return trial;
    }

    private double evaluate(NeuralNetwork2 nn, double[][] input, double[] target) {
        double error = 0.0;
        for (int i = 0; i < input.length; i++) {
            double predicted = nn.forwardPass(input[i])[0]; // Assuming single output
            error += Math.pow(predicted - target[i], 2);
        }
        return  -error / input.length; // Mean squared error
    }

    private double evaluate(NeuralNetwork2 nn, double[][] input, double[][] target) {
        double totalLoss = 0.0;

        for (int i = 0; i < input.length; i++) {
            double[] predicted = nn.forwardPass(input[i]); // Predicted probabilities
            double[] actual = target[i]; // Actual one-hot encoded target

            for (int j = 0; j < predicted.length; j++) {
                // Avoid log(0) by adding a small constant epsilon
                double epsilon = 1e-15;
                double predictedClamped = Math.max(epsilon, Math.min(1 - epsilon, predicted[j]));

                // Cross-entropy contribution for this output
                totalLoss -= actual[j] * Math.log(predictedClamped);
            }
        }

        // Return the average cross-entropy loss
        return -totalLoss / input.length;
    }

    public double getAverageConvergenceRate() {
        double totalConvergenceRate = 0.0;
        int convergenceCount = 0;
        double previousBestFitness = -Double.MAX_VALUE;

        for (int generation = 0; generation < bestFitness.length; generation++) {
            double currentBestFitness = bestFitness[generation];

            if (generation > 0) {
                double improvement = Math.abs(previousBestFitness - currentBestFitness);
                totalConvergenceRate += improvement;
                convergenceCount++;
            }

            previousBestFitness = currentBestFitness;
        }

        double averageConvergenceRate = (convergenceCount > 0) ? (totalConvergenceRate / convergenceCount) : 0.0;
        return averageConvergenceRate;
    }


}
