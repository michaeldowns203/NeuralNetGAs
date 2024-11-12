import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class NeuralNetworkPSOPrint {
    private int inputSize;
    private int[] hiddenLayerSizes;
    private int outputSize;
    private String activationType;

    private List<double[][]> weights;
    private List<double[]> biases;

    // PSO Parameters
    private int swarmSize = 30;
    private double inertiaWeight = 0.7;
    private double cognitiveCoefficient = 1.5;
    private double socialCoefficient = 1.5;
    private int maxIterations = 1000;

    private List<Particle> swarm;
    private double[] globalBestPosition;
    private double globalBestError = Double.MAX_VALUE;

    public NeuralNetworkPSOPrint(int inputSize, int[] hiddenLayerSizes, int outputSize, String activationType) {
        this.inputSize = inputSize;
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.outputSize = outputSize;
        this.activationType = activationType;

        initializeWeights();
        initializeSwarm();
    }

    private void initializeWeights() {
        Random rand = new Random();
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
            for (int j = 0; j < biases.get(i).length; j++) {
                biases.get(i)[j] = rand.nextGaussian() * 0.01;
            }
        }
    }

    private void initializeSwarm() {
        swarm = new ArrayList<>();
        for (int i = 0; i < swarmSize; i++) {
            Particle particle = new Particle(weights, biases);
            swarm.add(particle);
        }
        globalBestPosition = swarm.get(0).getPosition();
    }

    public double[] forwardPass(double[] input) {
        double[] currentOutput = input;

        if (hiddenLayerSizes.length == 0) {
            double[] finalOutput = new double[outputSize];
            double[] z = new double[outputSize];

            for (int j = 0; j < outputSize; j++) {
                z[j] = 0.0;
                for (int k = 0; k < inputSize; k++) {
                    z[j] += input[k] * weights.get(0)[k][j];
                }
                z[j] += biases.get(0)[j];
            }

            if (activationType.equals("softmax")) {
                finalOutput = softmax(z);
            } else {
                finalOutput = z;
            }

            return finalOutput;
        } else {
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
    }

    public void train(double[][] inputData, double[][] targetData) {
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            for (Particle particle : swarm) {
                double error = 0.0;
                for (int i = 0; i < inputData.length; i++) {
                    double[] predictedOutput = forwardPass(inputData[i]);
                    error += calculateError(targetData[i], predictedOutput);
                }
                error /= inputData.length;

                if (error < globalBestError) {
                    globalBestError = error;
                    globalBestPosition = particle.getPosition().clone();
                }

                particle.updatePersonalBest(error);
            }

            for (Particle particle : swarm) {
                particle.updateVelocity(globalBestPosition, inertiaWeight, cognitiveCoefficient, socialCoefficient);
                particle.updatePosition();
                particle.applyPositionToWeights(weights, biases);
            }

            System.out.printf("Iteration %d: Global Best Error = %.6f\n", iteration, globalBestError);
        }
    }

    private double calculateError(double[] target, double[] predicted) {
        double error = 0.0;
        for (int i = 0; i < target.length; i++) {
            error += Math.pow(target[i] - predicted[i], 2);
        }
        return error;
    }

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

    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }
}

