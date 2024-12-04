import java.util.*;

public class TenFoldCrossValidation {

    public static List<List<Object>> extractTenPercentR(List<List<Object>> dataset) {
        dataset.sort(Comparator.comparingDouble(row -> (Double) row.get(row.size() - 1)));
        List<List<List<Object>>> groups = new ArrayList<>();
        int groupSize = 10;

        for (int i = 0; i < dataset.size(); i += groupSize) {
            int end = Math.min(i + groupSize, dataset.size());
            groups.add(new ArrayList<>(dataset.subList(i, end)));
        }

        List<List<Object>> tuningData = new ArrayList<>();
        int tuningSize = (int) Math.ceil(dataset.size() * 0.1);

        for (int fold = 0; fold < groupSize; fold++) {
            for (int i = fold; i < groups.size(); i += groupSize) {
                if (i < groups.size() && fold < groups.get(i).size()) {
                    tuningData.add(groups.get(i).get(fold));
                }
                if (tuningData.size() >= tuningSize) {
                    break;
                }
            }
            if (tuningData.size() >= tuningSize) {
                break;
            }
        }
        return tuningData.subList(0, Math.min(tuningData.size(), tuningSize));
    }

    public static List<List<List<Object>>> splitIntoStratifiedChunksR10(List<List<Object>> dataset, int numChunks) {
        dataset.sort(Comparator.comparingDouble(row -> (Double) row.get(row.size() - 1)));
        List<List<Object>> tuningData = extractTenPercentR(dataset);
        List<List<Object>> remainingData = new ArrayList<>(dataset.subList(tuningData.size(), dataset.size()));

        List<List<List<Object>>> chunks = new ArrayList<>();
        for (int i = 0; i < numChunks; i++) {
            chunks.add(new ArrayList<>());
        }

        int groupSize = 10;
        List<List<Object>> group = new ArrayList<>();

        for (int i = 0; i < remainingData.size(); i++) {
            group.add(remainingData.get(i));
            if (group.size() == groupSize || i == remainingData.size() - 1) {
                for (int j = 0; j < group.size(); j++) {
                    int chunkIndex = j % numChunks;
                    chunks.get(chunkIndex).add(group.get(j));
                }
                group.clear();
            }
        }
        return chunks;
    }

    public static List<List<List<Object>>> splitIntoStratifiedChunksR(List<List<Object>> dataset, int numChunks) {
        // Sort the dataset based on the response value (the last element in each list)
        dataset.sort(Comparator.comparingDouble(row -> (Double) row.get(row.size() - 1)));

        // Create the chunks for each fold
        List<List<List<Object>>> chunks = new ArrayList<>();
        for (int i = 0; i < numChunks; i++) {
            chunks.add(new ArrayList<>());
        }

        // Break the remaining dataset into groups of 10 consecutive examples
        int groupSize = 10;
        List<List<Object>> group = new ArrayList<>();

        // Distribute each item into the corresponding chunk
        for (int i = 0; i < dataset.size(); i++) {
            group.add(dataset.get(i));  // Add item to the group

            // Once the group reaches the group size, distribute it across the chunks
            if (group.size() == groupSize || i == dataset.size() - 1) {
                for (int j = 0; j < group.size(); j++) {
                    int chunkIndex = j % numChunks;
                    chunks.get(chunkIndex).add(group.get(j));  // Distribute across the chunks
                }
                group.clear();  // Reset the group for the next batch
            }
        }
        return chunks;
    }

    public static List<List<Object>> extractTenPercentC(List<List<Object>> dataset) {
        // Create a map to hold instances of each class
        Map<String, List<List<Object>>> classMap = new HashMap<>();

        // Populate the class map
        for (List<Object> row : dataset) {
            String label = row.get(row.size() - 1).toString();
            classMap.putIfAbsent(label, new ArrayList<>());
            classMap.get(label).add(row);
        }

        List<List<Object>> removedInstances = new ArrayList<>();

        // Extract 10% of instances while maintaining class proportions
        for (List<List<Object>> classInstances : classMap.values()) {
            Random random = new Random(123);
            Collections.shuffle(classInstances, random); // Shuffle instances within each class

            // Determine the number of instances to remove (10%)
            int numToRemove = (int) (classInstances.size() * 0.1);

            // Extract the instances and add them to the removed list
            removedInstances.addAll(classInstances.subList(0, numToRemove));

            // Retain the remaining instances in the class instances list
            classInstances.subList(0, numToRemove).clear(); // Remove the extracted instances
        }
        return removedInstances;
    }

    public static List<List<List<Object>>> splitIntoStratifiedChunksC10(List<List<Object>> dataset, int numChunks) {
        // Extract 10% of the dataset
        List<List<Object>> removedInstances = extractTenPercentC(dataset);

        // Create a map to hold instances of each class
        Map<String, List<List<Object>>> classMap = new HashMap<>();

        // Populate the class map with the instances of each class
        for (List<Object> row : dataset) {
            String label = row.get(row.size() - 1).toString();
            classMap.putIfAbsent(label, new ArrayList<>());
            classMap.get(label).add(row);
        }

        // Create chunks for stratified sampling
        List<List<List<Object>>> chunks = new ArrayList<>();
        for (int i = 0; i < numChunks; i++) {
            chunks.add(new ArrayList<>());
        }

        // Distribute instances into chunks while maintaining class proportions
        for (Map.Entry<String, List<List<Object>>> entry : classMap.entrySet()) {
            List<List<Object>> classInstances = entry.getValue();

            // Shuffle instances within each class for randomness
            Collections.shuffle(classInstances, new Random(123));

            // Distribute instances among chunks
            int currentChunk = 0;
            for (List<Object> instance : classInstances) {
                chunks.get(currentChunk).add(instance);
                currentChunk = (currentChunk + 1) % numChunks;
            }

            // Handle small classes: Ensure every class contributes to at least one chunk
            if (classInstances.size() < numChunks) {
                for (int i = 0; i < classInstances.size(); i++) {
                    chunks.get(i).add(classInstances.get(i));
                }
            }
        }

        return chunks;
    }

    public static List<List<List<Object>>> splitIntoStratifiedChunksC(List<List<Object>> dataset, int numChunks) {
        // Create a map to hold instances of each class
        Map<String, List<List<Object>>> classMap = new HashMap<>();

        // Populate the class map with the instances of each class
        for (List<Object> row : dataset) {
            String label = row.get(row.size() - 1).toString();
            classMap.putIfAbsent(label, new ArrayList<>());
            classMap.get(label).add(row);
        }

        // Create chunks for stratified sampling
        List<List<List<Object>>> chunks = new ArrayList<>();
        for (int i = 0; i < numChunks; i++) {
            chunks.add(new ArrayList<>());
        }

        // Distribute instances into chunks while maintaining class proportions
        for (Map.Entry<String, List<List<Object>>> entry : classMap.entrySet()) {
            List<List<Object>> classInstances = entry.getValue();

            // Shuffle instances within each class for randomness
            Collections.shuffle(classInstances, new Random(123));

            // Distribute instances among chunks
            int currentChunk = 0;
            for (List<Object> instance : classInstances) {
                chunks.get(currentChunk).add(instance);
                currentChunk = (currentChunk + 1) % numChunks;
            }

            // Handle small classes: Ensure every class contributes to at least one chunk
            if (classInstances.size() < numChunks) {
                for (int i = 0; i < classInstances.size(); i++) {
                    chunks.get(i).add(classInstances.get(i));
                }
            }
        }

        return chunks;
    }
}
