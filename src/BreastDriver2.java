import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
//normal 10 fold
public class BreastDriver2 {

    public static void main(String[] args) throws IOException {
        String inputFile1 = "src/breast-cancer-wisconsin.data";
        try {
            FileInputStream fis = new FileInputStream(inputFile1);
            InputStreamReader isr = new InputStreamReader(fis);
            BufferedReader stdin = new BufferedReader(isr);

            // First, count the number of lines to determine the size of the lists
            int lineCount = 0;
            while (stdin.readLine() != null) {
                lineCount++;
            }

            // Reset the reader to the beginning of the file
            stdin.close();
            fis = new FileInputStream(inputFile1);
            isr = new InputStreamReader(fis);
            stdin = new BufferedReader(isr);

            // Initialize the lists
            List<List<Object>> dataset = new ArrayList<>();
            List<Object> labels = new ArrayList<>();

            String line;
            int lineNum = 0;

            // Read the file and fill the dataset
            while ((line = stdin.readLine()) != null) {
                String[] rawData = line.split(",");
                List<Object> row = new ArrayList<>();

                // Assign the label (last column)
                labels.add(Double.parseDouble(rawData[10]));

                // Fill the data row (columns 2 to 10)
                for (int i = 1; i < rawData.length - 1; i++) {
                    if (rawData[i].equals("?")) {
                        row.add((Math.random() * 10) + 1); // Handle missing values
                    } else {
                        row.add(Double.parseDouble(rawData[i]));
                    }
                }
                row.add(labels.get(lineNum)); // Add the label to the row
                dataset.add(row);
                lineNum++;
            }

            stdin.close();

            // Split the remaining dataset into stratified chunks
            List<List<List<Object>>> chunks = TenFoldCrossValidation.splitIntoStratifiedChunksC(dataset, 10);

            // Loss instance variables
            double total01loss = 0;
            double totalACR = 0;

            for (int i = 0; i < 10; i++) {
                List<List<Object>> trainingSet = new ArrayList<>();
                List<List<Double>> trainingData = new ArrayList<>();
                List<List<Double>> trainingLabels = new ArrayList<>();
                List<Double> predictedList = new ArrayList<>();
                List<Double> actualList = new ArrayList<>();

                List<List<Object>> testSet = chunks.get(i);

                int correctPredictions = 0;

                for (int j = 0; j < 10; j++) {
                    if (j != i) {
                        for (List<Object> row : chunks.get(j)) {
                            List<Object> all = new ArrayList<>();
                            for (int k = 0; k < row.size(); k++) {
                                all.add((Double) row.get(k));
                            }
                            trainingSet.add(all);
                        }
                    }
                }

                List<List<Double>> scaledTrainingData = MinMaxScale.minMaxScale(trainingSet);
                List<List<Double>> scaledTestData = MinMaxScale.minMaxScale(testSet);

                // Loop through the scaledTrainingData to extract features and labels
                for (int j = 0; j < scaledTrainingData.size(); j++) {
                    if (j != i) { // If excluding a specific chunk (e.g., for cross-validation)
                        List<Double> row = scaledTrainingData.get(j);
                        List<Double> features = new ArrayList<>(row.subList(0, row.size() - 1)); // All but the last element
                        Double label = row.get(row.size() - 1); // The last element as the label

                        trainingData.add(features); // Add features to trainingData
                        trainingLabels.add(Collections.singletonList(label)); // Add label to trainingLabels
                    }
                }

                double[][] trainInputs = new double[trainingData.size()][];
                double[][] trainOutputs = new double[trainingLabels.size()][];

                for (int t = 0; t < trainingData.size(); t++) {
                    trainInputs[t] = trainingData.get(t).stream().mapToDouble(Double::doubleValue).toArray();
                    trainOutputs[t] = trainingLabels.get(t).stream().mapToDouble(Double::doubleValue).toArray();
                }

                double[][] trainOutputsOHE = OneHotEncoder.oneHotEncode(trainOutputs);

                double[][] testInputs = new double[scaledTestData.size()][];
                for (int t = 0; t < scaledTestData.size(); t++) {
                    testInputs[t] = scaledTestData.get(t).subList(0, scaledTestData.get(t).size() - 1)
                            .stream().mapToDouble(Double::doubleValue).toArray();
                }

                int inputSize = trainInputs[0].length;
                int[] hiddenLayerSizes = {6,4};
                int outputSize = 2;
                String activationType = "softmax";

                /*
                int populationSize = 50;
                double mutationRate = 0.05;
                double crossoverRate = 0.9;
                double tolerance = 0.0001;
                int patience = 50;
                GAC ga = new GAC(populationSize, mutationRate, crossoverRate);
                ga.initializePopulation(inputSize, hiddenLayerSizes, outputSize, activationType);
                NeuralNetwork2 nn = ga.run(inputSize, hiddenLayerSizes, outputSize, activationType, trainInputs, trainOutputsOHE, tolerance, patience);
                */

                int numParticles = 100;
                int maxIterations = 200;
                double inertiaWeight = 1.0;
                double cognitiveComponent = 2.5;
                double socialComponent = 2.0;
                double vMax = 0.1;
                NeuralNetwork2 nn2 = new NeuralNetwork2(inputSize, hiddenLayerSizes, outputSize, activationType);
                PSOC pso = new PSOC(nn2, trainInputs, trainOutputsOHE, numParticles, maxIterations, inertiaWeight, cognitiveComponent, socialComponent, vMax);
                List <double[][]> weights = pso.optimize();
                NeuralNetwork2 nn = new NeuralNetwork2(inputSize, hiddenLayerSizes, outputSize, activationType);
                nn.setWeights(weights);

                /*
                int populationSize = 100;
                int maxNoImprovementGenerations = 20; //lower this probably
                double mutationFactor = 0.5;
                double crossoverRate = 0.9;
                double tolerance = 0.0001;
                DE de = new DE(populationSize, maxNoImprovementGenerations, mutationFactor, crossoverRate, tolerance);

                NeuralNetwork2 nn = de.optimize(trainInputs, trainOutputsOHE);
                //remember to change values in de algorithm (hidden layer sizes, softmax, num outputs)
                 */


                for (int t = 0; t < testInputs.length; t++) {
                    double[] prediction = nn.forwardPass(testInputs[t]);
                    double actual = scaledTestData.get(t).get(scaledTestData.get(t).size() - 1);

                    if (prediction[0] > 0.5)
                        predictedList.add(0.0);
                    else
                        predictedList.add(1.0);

                    actualList.add(actual);

                    System.out.printf("Test Instance: %s | Predicted: %.4f | Actual: %.4f%n",
                            Arrays.toString(testInputs[t]), predictedList.get(t), actual);

                    if (predictedList.get(t) == (actual)) {
                        correctPredictions++;
                    }
                }
                // Calculate 0/1 loss
                double loss01 = 1.0 - (double) correctPredictions / testSet.size();
                total01loss += loss01;
                System.out.printf("Fold %d 0/1 loss: %.4f%n", i+1, loss01);

                //double acrFold = de.getAverageConvergenceRate();
                //totalACR += acrFold;
                //System.out.printf("Fold %d Average Convergence Rate: %.4f%n", i+1,  acrFold);
            }

            double AACR = totalACR / 10;
            System.out.printf("Average Convergence Rate across all epochs across 10 folds: %.4f%n", AACR);

            double average01loss = total01loss / 10;
            System.out.printf("Average 0/1 Loss: %.4f%n", average01loss);
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }

}


