import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class NeuralNetwork2 {
    private int inputSize;
    private int[] hiddenLayerSizes;
    private int outputSize;
    private String activationType;

    private List<double[][]> weights;
    private List<double[]> biases;

    public NeuralNetwork2(int inputSize, int[] hiddenLayerSizes, int outputSize, String activationType) {
        this.inputSize = inputSize;
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.outputSize = outputSize;
        this.activationType = activationType;

        initializeWeights();
    }

    private void initializeWeights() {
        Random rand = new Random();
        weights = new ArrayList<>();
        biases = new ArrayList<>();

        if (hiddenLayerSizes.length == 0) {
            // No hidden layers, just connect input to output
            weights.add(new double[inputSize][outputSize]);
            biases.add(new double[outputSize]);
        } else {
            // Input to first hidden layer, hidden layers, and output
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
                biases.get(i)[j] = rand.nextGaussian() * 0.01;  // Small random values for bias
            }
        }
    }

    // Sigmoid activation function for hidden layers
    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    // Softmax activation function for classification
    private double[] softmax(double[] z) {
        // Find the maximum value in z to subtract for numerical stability
        double max = Arrays.stream(z).max().orElse(0.0);

        // Calculate the sum of the exponentials after adjusting for numerical stability
        double sum = 0.0;
        double[] output = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            output[i] = Math.exp(z[i] - max); // Subtract max for numerical stability
            sum += output[i];
        }

        // Normalize the outputs to get probabilities
        for (int i = 0; i < z.length; i++) {
            output[i] /= sum;
        }

        return output;
    }


    public double[] forwardPass(double[] input) {
        // Clear layerOutputs at the beginning of every forward pass
        layerOutputs.clear();

        // Set inputLayer if it's not already set
        if (inputLayer == null) {
            setInputLayer(input);
        }

        double[] currentOutput = input;

        // Store the input layer in layerOutputs
        storeLayerOutput(0, currentOutput);

        if (hiddenLayerSizes.length == 0) {
            // Directly go from input to output if no hidden layers
            double[] finalOutput = new double[outputSize];
            double[] z = new double[outputSize];

            for (int j = 0; j < outputSize; j++) {
                z[j] = 0.0;
                for (int k = 0; k < inputSize; k++) {
                    z[j] += input[k] * weights.get(0)[k][j]; // Only one weight matrix
                }
                z[j] += biases.get(0)[j];  // Only one bias set
            }

            // Apply activation
            if (activationType.equals("softmax")) {
                finalOutput = softmax(z);
            } else {
                finalOutput = z;  // For linear activation or regression
            }

            return finalOutput;
        }
        else {

            // Loop through hidden layers
            for (int i = 0; i < weights.size() - 1; i++) {
                int previousLayerSize = currentOutput.length;
                int currentLayerSize = weights.get(i)[0].length; // Number of neurons in the current layer

                double[] newOutput = new double[currentLayerSize]; // New output size

                // For each neuron in the current layer
                for (int j = 0; j < newOutput.length; j++) {
                    double z = 0.0;

                    // Sum up the weighted inputs from the previous layer's outputs (currentOutput)
                    for (int k = 0; k < currentOutput.length; k++) {
                        z += currentOutput[k] * weights.get(i)[k][j]; // Use k for previous layer, j for current layer
                    }

                    z += biases.get(i)[j]; // Add the bias term for neuron j in layer i
                    newOutput[j] = sigmoid(z); // Apply sigmoid activation
                }

                // Store the output of the current hidden layer
                storeLayerOutput(i + 1, newOutput);

                currentOutput = newOutput; // Update currentOutput to the new layer's output
            }
        }

        // Output layer activation (softmax or linear)
        double[] finalOutput = new double[outputSize];
        double[] z = new double[outputSize];

        // For the output layer, compute the weighted sum
        for (int j = 0; j < outputSize; j++) {
            z[j] = 0.0;
            for (int k = 0; k < currentOutput.length; k++) {
                z[j] += currentOutput[k] * weights.get(weights.size() - 1)[k][j]; // Last layer's weights
            }
            z[j] += biases.get(biases.size() - 1)[j];
        }

        // Apply softmax to the entire output vector if softmax is the activation type
        if (activationType.equals("softmax")) {
            finalOutput = softmax(z);
        } else {
            finalOutput = z; // For linear activation
        }

        // Store the output of the output layer
        storeLayerOutput(weights.size(), finalOutput);

        return finalOutput;
    }

    // Stores the input layer values when the network starts processing
    private double[] inputLayer;

    // Store the output of each layer after forward pass
    private List<double[]> layerOutputs = new ArrayList<>();

    // Helper to retrieve the input to the network
    private double[] getInputLayer() {
        return inputLayer;
    }

    // Helper to store input values to the network before the forward pass
    public void setInputLayer(double[] inputLayer) {
        this.inputLayer = inputLayer;
    }

    // Helper to get the output of the current layer
    private double[] outputLayer(int layerIdx) {
        return layerOutputs.get(layerIdx);
    }

    // Store the output of each layer after forward pass
    private void storeLayerOutput(int layerIdx, double[] output) {
        if (layerOutputs.size() > layerIdx) {
            layerOutputs.set(layerIdx, output);  // Update output if it already exists
        } else {
            layerOutputs.add(output);  // Add new output if it doesn't exist
        }
    }

    public List<double[][]> getWeights() {
        return weights;
    }

    public List<double[]> getBiases() {
        return biases;
    }

    public void setWeights(List<double[][]> weights) {
        this.weights = weights;
    }

    public void setBiases(List<double[]> biases) {
        this.biases = biases;
    }

    public NeuralNetwork2 copy() {
        NeuralNetwork2 copy = new NeuralNetwork2(this.inputSize, this.hiddenLayerSizes, this.outputSize, this.activationType);

        // Deep copy weights
        List<double[][]> copiedWeights = new ArrayList<>();
        for (double[][] layerWeights : this.weights) {
            double[][] layerCopy = new double[layerWeights.length][layerWeights[0].length];
            for (int i = 0; i < layerWeights.length; i++) {
                System.arraycopy(layerWeights[i], 0, layerCopy[i], 0, layerWeights[i].length);
            }
            copiedWeights.add(layerCopy);
        }
        copy.setWeights(copiedWeights);

        // Deep copy biases
        List<double[]> copiedBiases = new ArrayList<>();
        for (double[] layerBiases : this.biases) {
            double[] layerCopy = Arrays.copyOf(layerBiases, layerBiases.length);
            copiedBiases.add(layerCopy);
        }
        copy.biases = copiedBiases; // Directly assign, or use a setter if available

        return copy;
    }

}
