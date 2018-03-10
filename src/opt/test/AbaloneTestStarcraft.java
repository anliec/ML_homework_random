package opt.test;

import func.nn.Layer;
import func.nn.Link;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;
import util.linalg.DenseVector;

import java.util.*;
import java.io.*;
import java.text.*;
import java.util.concurrent.Semaphore;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class AbaloneTestStarcraft implements Runnable{

    public static class OA{
        public String name;
        public OptimizationAlgorithm algorithm;
        public BackPropagationNetwork nn;
        public NeuralNetworkOptimizationProblem pb;
        public double[] errors;
        public double[] test_errors;
        public long[] times;
        public String res = "";
        public Thread t;
        public OA(String oa_name, OptimizationAlgorithm opt, BackPropagationNetwork bpn, NeuralNetworkOptimizationProblem nnpb){
            name = oa_name;
            algorithm = opt;
            nn = bpn;
            pb = nnpb;
        }
    }

    public Thread t;
    private static Semaphore runSem = new Semaphore(4);

	private static String outputDir = "./OptimizationResults";
    private static int inputLayer = 72, outputLayer = 200, trainingIterations = 10000;
    private static Instance[] instances = initializeInstances("data/starcraft_x_train.csv", "data/starcraft_y_train.csv");
    private static Instance[] test_instances = initializeInstances("data/starcraft_x_test.csv", "data/starcraft_y_test.csv");
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);
    private static DataSet test_set = new DataSet(test_instances);

    private static DecimalFormat df = new DecimalFormat("0.000");

    private static String outFileParticule = "";

    private OA trainingOa;

    AbaloneTestStarcraft(OA oa){
        trainingOa = oa;
    }


    public static void main(String[] args) throws InterruptedException {
        boolean computeRHC = true, computeSA = true, computeGA = true;
        if(args.length >= 2) {
            System.out.println(args[1] + " selected");
            switch (args[1]) {
                case "RHC":
                    computeSA = false;
                    computeGA = false;
                    break;
                case "SA":
                    computeRHC = false;
                    computeGA = false;
                    break;
                case "GA":
                    computeRHC = false;
                    computeSA = false;
                    break;
            }
        }
        if(args.length >= 1)
            outFileParticule = args[0];
        LinkedList<OA> oa_list = new LinkedList<>();

        BackPropagationNetwork network;
        NeuralNetworkOptimizationProblem nnop;

        LinkedList<Integer[]> hiddenLayerSizes = new LinkedList<>();
//        hiddenLayerSizes.add(new Integer[] {});
//        hiddenLayerSizes.add(new Integer[] {5});
//        hiddenLayerSizes.add(new Integer[] {10});
//        hiddenLayerSizes.add(new Integer[] {15});
//        hiddenLayerSizes.add(new Integer[] {25});
//        hiddenLayerSizes.add(new Integer[] {35});
//        hiddenLayerSizes.add(new Integer[] {20, 10});
        hiddenLayerSizes.add(new Integer[] {30, 10});

//        int count=0;

        // for each size of NN generate a model to train
        for(Integer[] hiddenLayerShape : hiddenLayerSizes) {
            int[] nnLayer = new int[2 + hiddenLayerShape.length];
            nnLayer[0] = inputLayer;
            for (int i = 1; i < 1 + hiddenLayerShape.length; i++)
                nnLayer[i] = hiddenLayerShape[i - 1];
            nnLayer[1 + hiddenLayerShape.length] = outputLayer;

            //RHC
            if(computeRHC) {
                network = factory.createClassificationNetwork(nnLayer);
                nnop = new NeuralNetworkOptimizationProblem(set, network, measure);
                oa_list.add(new OA("RHC",
                        new RandomizedHillClimbing(nnop),
                        network,
                        nnop));
//                count++;
            }

            //SA
            if (computeSA){
                for (double it = 50; it < 200.0; it += 50.0) {
                    for (double cooling = 0.5; cooling < 1.0; cooling += 0.1) {
                        network = factory.createClassificationNetwork(nnLayer);
                        nnop = new NeuralNetworkOptimizationProblem(set, network, measure);
                        oa_list.add(new OA("SA",
                                new SimulatedAnnealing(it, cooling, nnop),
                                network,
                                nnop));
//                        count++;
                    }
                }
            }

            //GA
            if(computeGA) {
                for (int pop = 50; pop < 700; pop += 200) {
                    for (int toMate = 30; toMate < (pop/2); toMate += 100) {
                        for (int toMutate = 10; toMutate < (toMate/2); toMutate += 50) {
                            network = factory.createClassificationNetwork(nnLayer);
                            nnop = new NeuralNetworkOptimizationProblem(set, network, measure);
                            oa_list.add(new OA("GA",
                                    new StandardGeneticAlgorithm(pop, toMate, toMutate, nnop),
                                    network,
                                    nnop));
//                            count++;
                        }
                    }
                }
            }
        }

//        System.out.println("Trainning "+count+" different instances");
        System.out.println("Trainning "+oa_list.size()+" different instances");

        for(OA oa : oa_list) {
            AbaloneTestStarcraft trainner = new AbaloneTestStarcraft(oa);
            trainner.start();
        }
        for(OA oa : oa_list){
            oa.t.join();

            //write errors to csv file
            StringBuilder sb = new StringBuilder();
            StringBuilder baseLine = new StringBuilder();
            baseLine.append(oa.name);
            baseLine.append(",");
            for(Layer l : (ArrayList<Layer>) oa.nn.getHiddenLayers()){
                baseLine.append(" ");
                baseLine.append(l.getNodeCount() - 1);
            }
            baseLine.append(", ");
            if (oa.algorithm instanceof SimulatedAnnealing){
                SimulatedAnnealing alg = (SimulatedAnnealing) oa.algorithm;
                baseLine.append(alg.getStartT());
                baseLine.append(", ");
                baseLine.append(alg.getCooling());
                baseLine.append(", 0, 0, 0, ");
            }
            else if (oa.algorithm instanceof StandardGeneticAlgorithm){
                baseLine.append("0 , 0, ");
                StandardGeneticAlgorithm alg = (StandardGeneticAlgorithm) oa.algorithm;
                baseLine.append(alg.getPopulationSize());
                baseLine.append(", ");
                baseLine.append(alg.getToMate());
                baseLine.append(", ");
                baseLine.append(alg.getToMutate());
                baseLine.append(", ");
            }
            else{
                baseLine.append("0, 0, 0, 0, 0, ");
            }
            String line_start = baseLine.toString();
            for (int i=0 ; i < oa.errors.length ; i++) {
                sb.append(line_start);
                sb.append(i);
                sb.append(", ");
                sb.append(oa.errors[i]);
                sb.append(", ");
                sb.append(oa.test_errors[i]);
                sb.append(", ");
                sb.append(oa.times[i]);
                sb.append('\n');
            }
            Utils.writeOutputToFile(outputDir, "StarcraftTest" + outFileParticule + ".csv", oa.res);
            Utils.writeOutputToFile(outputDir, "StarcraftTestErrors" + outFileParticule + ".csv", sb.toString());
        }
        System.out.println("Finished !");
    }

    
    private void train(OA oa) {
        oa.errors = new double[trainingIterations];
        oa.times = new long[trainingIterations];
        oa.test_errors = new double[trainingIterations];

        long old_time = 0;
        for(int i = 0; i < trainingIterations; i++) {
            long start = System.nanoTime();
            oa.algorithm.train();
            old_time += System.nanoTime() - start;
            oa.times[i] = old_time;

            int predicted, actual;
            int correct = 0;
            for (Instance ins : instances) {
                oa.nn.setInputValues(ins.getData());
                oa.nn.run();

                actual = ins.getLabel().getData().argMax();
                predicted = oa.nn.getOutputValues().argMax();

                if (actual == predicted)
                    correct++;
            }
            oa.errors[i] = ((double) correct) / ((double) instances.length);

            correct = 0;
            for (Instance ins : test_instances) {
                oa.nn.setInputValues(ins.getData());
                oa.nn.run();

                actual = ins.getLabel().getData().argMax();
                predicted = oa.nn.getOutputValues().argMax();

                if (actual == predicted)
                    correct++;
            }
            oa.test_errors[i] = ((double) correct) / ((double) test_instances.length);
        }
    }

    private static Instance[] initializeInstances(String x_value_path, String y_value_path) {
        LinkedList<Instance> instances = new LinkedList<>();
        try {
//            FileReader y_train_file = new FileReader("data/starcraft_y_train.csv");
//            FileReader x_train_file = new FileReader("data/starcraft_x_train.csv");
            FileReader y_train_file = new FileReader(y_value_path);
            FileReader x_train_file = new FileReader(x_value_path);
            BufferedReader y_train_br = new BufferedReader(y_train_file);
            BufferedReader x_train_br = new BufferedReader(x_train_file);

            String y_train_line = y_train_br.readLine();
            String x_train_line = x_train_br.readLine();

            while (y_train_line != null && x_train_line != null){
                Scanner x_train_sc = new Scanner(x_train_line);
                x_train_sc.useDelimiter(",");
                double x_train_labels[] = new double[inputLayer];
                int i=0;
                while (x_train_sc.hasNext()){
                    x_train_labels[i] = Double.parseDouble(x_train_sc.next());
                    i++;
                }
                assert i == inputLayer;

                Scanner y_train_sc = new Scanner(x_train_line);
                y_train_sc.useDelimiter(",");
                double y_train_labels[] = new double[outputLayer];
                i=0;
                while (y_train_sc.hasNext()){
                    y_train_labels[i] = Double.parseDouble(y_train_sc.next());
                    assert y_train_labels[i] == 0.0 || y_train_labels[i] == 1.0;
                    i++;
                }
                assert i == outputLayer;

                instances.add(new Instance(new DenseVector(x_train_labels), new Instance(y_train_labels)));

                y_train_line = y_train_br.readLine();
                x_train_line = x_train_br.readLine();
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        // convert linked list to array
        Instance[] instances_array = new Instance[instances.size()];
        instances_array = instances.toArray(instances_array);

//        for(Instance i : instances_array){
//            assert i.getLabel().getData().sum() == 1.0;
//        }

        return instances_array;
    }

    @Override
    public void run() {
        try {
            runSem.acquire();
            double start = System.currentTimeMillis(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(trainingOa);
            end = System.currentTimeMillis();
            trainingTime = end - start;
            trainingTime /= 1000;

            Instance optimalInstance = trainingOa.algorithm.getOptimal();
            trainingOa.nn.setWeights(optimalInstance.getData());

            int predicted, actual;
            start = System.currentTimeMillis();
            for (Instance ins : instances) {
                trainingOa.nn.setInputValues(ins.getData());
                trainingOa.nn.run();

                actual = ins.getLabel().getData().argMax();
                predicted = trainingOa.nn.getOutputValues().argMax();

                if (actual == predicted)
                    correct++;
                else
                    incorrect++;
            }
            end = System.currentTimeMillis();
            testingTime = end - start;
            testingTime /= 1000;

            trainingOa.res = "\nResults for " + trainingOa.name + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct / (correct + incorrect) * 100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
            runSem.release();
        }
        catch (InterruptedException e){
            e.printStackTrace();
        }
    }

    public void start () {
        if (t == null)
        {
            t = new Thread (this);
            t.start ();
            trainingOa.t = t;
        }
    }
}
