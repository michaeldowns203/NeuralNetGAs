import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class PSOPrint {
    private class Particle {
        List<double[][]> position; // Current weights
        List<double[][]> velocity;
        List<double[][]> bestPosition; // Personal best weights
        double bestFitness; // Personal best fitness

        Particle(List<double[][]> initialWeights, Random random) {
            position = copyWeightList(initialWeights);
            velocity = createZeroWeightList(initialWeights);
            bestPosition = copyWeightList(position);
            bestFitness = -Double.MAX_VALUE;

            // Randomize velocity
            for (int layer = 0; layer < velocity.size(); layer++) {
                double[][] velLayer = velocity.get(layer);
                for (int i = 0; i < velLayer.length; i++) {
                    for (int j = 0; j < velLayer[i].length; j++) {
                        velLayer[i][j] = random.nextDouble() * 0.1 - 0.05; // Small random velocity
                    }
                }
            }
        }
    }

    private final int numParticles;
    private final int maxIterations;
    private final double inertiaWeight;
    private final double cognitiveComponent; // Personal influence
    private final double socialComponent; // Group influence
    private final double vMax;

    private final NeuralNetwork2 neuralNetwork;
    private final double[][] inputs;
    private final double[] outputs;
    private Particle[] particles;
    private List<double[][]> globalBestPosition;
    private double globalBestFitness;

    public PSOPrint(NeuralNetwork2 neuralNetwork, double[][] inputs, double[] outputs,
               int numParticles, int maxIterations,
               double inertiaWeight, double cognitiveComponent, double socialComponent, double vMax) {
        this.neuralNetwork = neuralNetwork;
        this.inputs = inputs;
        this.outputs = outputs;
        this.numParticles = numParticles;
        this.maxIterations = maxIterations;
        this.inertiaWeight = inertiaWeight;
        this.cognitiveComponent = cognitiveComponent;
        this.socialComponent = socialComponent;
        this.vMax = vMax;
    }

    public List<double[][]> optimize() {
        List<double[][]> initialWeights = neuralNetwork.getWeights();
        particles = new Particle[numParticles];
        globalBestPosition = copyWeightList(initialWeights);
        globalBestFitness = -Double.MAX_VALUE;

        Random random = new Random();

        // Initialize particles
        for (int i = 0; i < numParticles; i++) {
            particles[i] = new Particle(initialWeights, random);
        }

        // Optimization loop
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            System.out.printf("Iteration %d:%n", iteration);

            for (Particle particle : particles) {
                // Set weights in neural network and evaluate fitness
                neuralNetwork.setWeights(particle.position);
                double fitness = evaluate(neuralNetwork, inputs, outputs);

                // Update personal best
                if (fitness > particle.bestFitness) {
                    particle.bestFitness = fitness;
                    particle.bestPosition = copyWeightList(particle.position);
                    System.out.printf("  Updated personal best: Fitness = %.6f%n", fitness);
                }

                // Update global best
                if (fitness > globalBestFitness) {
                    globalBestFitness = fitness;
                    globalBestPosition = copyWeightList(particle.position);
                    System.out.printf("  Updated global best: Fitness = %.6f%n", fitness);
                }
            }

            // Update particle velocities and positions
            for (Particle particle : particles) {
                for (int layer = 0; layer < particle.velocity.size(); layer++) {
                    double[][] velLayer = particle.velocity.get(layer);
                    double[][] posLayer = particle.position.get(layer);
                    double[][] bestLayer = particle.bestPosition.get(layer);
                    double[][] globalBestLayer = globalBestPosition.get(layer);

                    for (int i = 0; i < velLayer.length; i++) {
                        for (int j = 0; j < velLayer[i].length; j++) {
                            double r1 = random.nextDouble();
                            double r2 = random.nextDouble();

                            // Update velocity
                            double oldVelocity = velLayer[i][j];
                            velLayer[i][j] = inertiaWeight * velLayer[i][j]
                                    + cognitiveComponent * r1 * (bestLayer[i][j] - posLayer[i][j])
                                    + socialComponent * r2 * (globalBestLayer[i][j] - posLayer[i][j]);

                            // Apply velocity clamping
                            velLayer[i][j] = Math.max(-vMax, Math.min(velLayer[i][j], vMax));

                            // Update position
                            double oldPosition = posLayer[i][j];
                            posLayer[i][j] += velLayer[i][j];

                            // Print velocity and position updates
                            System.out.printf("    Layer %d, Neuron (%d,%d): Velocity updated from %.6f to %.6f, Position updated from %.6f to %.6f%n",
                                    layer, i, j, oldVelocity, velLayer[i][j], oldPosition, posLayer[i][j]);
                        }
                    }

                }
            }

            System.out.printf("Iteration %d: Best Fitness = %.6f%n%n", iteration, globalBestFitness);
        }

        return globalBestPosition;
    }

    // Helper to create a copy of a List<double[][]>
    private List<double[][]> copyWeightList(List<double[][]> original) {
        List<double[][]> copy = new ArrayList<>();
        for (double[][] layer : original) {
            double[][] layerCopy = new double[layer.length][];
            for (int i = 0; i < layer.length; i++) {
                layerCopy[i] = layer[i].clone();
            }
            copy.add(layerCopy);
        }
        return copy;
    }

    // Helper to create a zero-initialized List<double[][]> with the same structure as another
    private List<double[][]> createZeroWeightList(List<double[][]> structure) {
        List<double[][]> zeroList = new ArrayList<>();
        for (double[][] layer : structure) {
            double[][] zeroLayer = new double[layer.length][];
            for (int i = 0; i < layer.length; i++) {
                zeroLayer[i] = new double[layer[i].length];
            }
            zeroList.add(zeroLayer);
        }
        return zeroList;
    }

    private double evaluate(NeuralNetwork2 nn, double[][] input, double[] target) {
        double error = 0.0;
        for (int i = 0; i < input.length; i++) {
            double predicted = nn.forwardPass(input[i])[0]; // Assuming single output
            error += Math.pow(predicted - target[i], 2);
        }
        return -error / input.length; // - Mean squared error
    }
}
