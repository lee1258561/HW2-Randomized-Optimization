package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;
import func.nn.activation.DifferentiableActivationFunction;
import func.nn.activation.HyperbolicTangentSigmoid;
import func.nn.activation.LinearActivationFunction;
import func.nn.activation.LogisticSigmoid;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class MNISTTest {
    private static Instance[] instances, validationInstances;
    private static double valSplit = 0.2;
    private static int[] layers = new int[] {115, 16, 16, 1};
    private static ErrorMeasure measure = new SumOfSquaresError();
    private static DataSet set;
    private static String results = "";
    private static DecimalFormat df = new DecimalFormat("0.000");
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static int[] oaIters = {50000, 50000, 5000};
    private static int population = 500, mate = 75, mutate = 150, RHCRestart = 6;
    private static double temp = 1E11, coolRate = .95;

    private static boolean verbose = true; 

    public static void main(String[] args) {
        initializeInstances();
        set = new DataSet(instances);
        if (args.length == 0){
            for (int i = 0; i < oaNames.length; i++){
                run(oaNames[i]);
            }
        }else if (args.length == 1){
            run(args[0]);
        }else{
            System.out.println("ERROR: Invalid Argument");
        }
        System.out.println(results);
    }

    private static void run(String oaName){
        BackPropagationNetwork network = factory.createClassificationNetwork(layers, new LogisticSigmoid());
        NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(set, network, measure);

        int iters, restart = 1;
        FileWriter restartWriter;
        PrintWriter restartPrinter;
        if (oaName.equals("RHC")){
            restart = RHCRestart;
        }
        try{
            restartWriter = new FileWriter("./restart" + Integer.toString(restart) + "_" 
                                                    + oaName + "_" + Integer.toString(oaIters[0]) + ".log");
        }catch(IOException ex){
            System.out.println("IOException Caught!!!");
            return;
        }
        restartPrinter = new PrintWriter(restartWriter);   


        for (int i = 0; i < restart; i++){
            String parameterString = "";
            OptimizationAlgorithm oa;
            if (oaName.equals("RHC")){
                System.out.println(Integer.toString(RHCRestart));
                oa = new RandomizedHillClimbing(nnop);
                parameterString = parameterString + "_RESTART" + Integer.toString(RHCRestart);
                iters = oaIters[0];
            }else if (oaName.equals("SA")){
                System.out.println(Double.toString(temp) + " " + Double.toString(coolRate));
                oa = new SimulatedAnnealing(temp, coolRate, nnop);
                parameterString = parameterString + "_TEMP" + Double.toString(temp)
                                                  + "_COOL" + Double.toString(coolRate);
                iters = oaIters[1];
            }else if (oaName.equals("GA")){
                System.out.println("" + population + " " + mate + " " + mutate);
                oa = new StandardGeneticAlgorithm(population, mate, mutate, nnop);
                parameterString = parameterString + "_POP" + Integer.toString(population)
                                                  + "_MAT" + Integer.toString(mate)
                                                  + "_MU" + Integer.toString(mutate);
                iters = oaIters[2];
            }else{
                System.out.println("ERROR: Unknown Optimization Algorithm");
                return;
            }

            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0, error = 0;
            train(oa, network, oaName, iters, parameterString); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa.getOptimal();
            network.setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance example = instances[j], output = new Instance(network.getOutputValues());
                output.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                output = output.getLabel();
                error += measure.value(output, example);

                actual = Double.parseDouble(instances[j].getLabel().toString());
                predicted = Double.parseDouble(network.getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            String trainingAcc = df.format(correct/(correct+incorrect)*100);

            correct = 0;
            predicted = 0;
            for(int j = 0; j < validationInstances.length; j++) {
                network.setInputValues(validationInstances[j].getData());
                network.run();

                actual = Double.parseDouble(validationInstances[j].getLabel().toString());
                predicted = Double.parseDouble(network.getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
            }

            String valAcc = df.format(correct/(correct+incorrect)*100);

            
            String eval = "\nResults for " + oaName + ": \nCorrectly classified " + correct + " instances." +
                          "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                          + trainingAcc + "Validation Accuracy: " + valAcc + "%\nTraining time: " + df.format(trainingTime)
                          + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
            results += eval;

            FileWriter fileWriter;
            try{
                fileWriter = new FileWriter("./results" + oaName + "_" + Integer.toString(iters) + parameterString + ".log");
            }catch(IOException ex){
                System.out.println("IOException Caught!!!");
                return;
            }
            PrintWriter printWriter = new PrintWriter(fileWriter);
            printWriter.println(eval);
            printWriter.close();
            

            if (oaName.equals("RHC")){
                restartPrinter.println(df.format(error) + "," + trainingAcc + "," + valAcc);
            }
        }
        restartPrinter.close();
        if (!oaName.equals("RHC")){
            File file = new File("./restart" + Integer.toString(restart) + "_" 
                                            + oaName + "_" + Integer.toString(oaIters[0]) + ".log");
            file.delete();
        }
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int iters, String parameterString) {
        if (verbose){
            System.out.println("\nError results for " + oaName + "\n---------------------------");
        }
        String fileName = oaName + "_" + Integer.toString(iters) + parameterString + ".log";
        FileWriter fileWriter;
        try{
            fileWriter = new FileWriter(fileName);
        }catch(IOException ex){
            System.out.println("IOException Caught!!!");
            return;
        }
        PrintWriter printWriter = new PrintWriter(fileWriter);

        for(int i = 0; i < iters; i++) {
            oa.train();

            double error = 0, correct = 0, incorrect = 0, predicted, actual;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance example = instances[j], output = new Instance(network.getOutputValues());
                output.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                output = output.getLabel();
                error += measure.value(output, example);

                actual = Double.parseDouble(instances[j].getLabel().toString());
                predicted = Double.parseDouble(network.getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
            }

            String trainingAcc = df.format(correct/(correct+incorrect)*100);

            correct = 0;
            predicted = 0;
            for(int j = 0; j < validationInstances.length; j++) {
                network.setInputValues(validationInstances[j].getData());
                network.run();

                actual = Double.parseDouble(validationInstances[j].getLabel().toString());
                predicted = Double.parseDouble(network.getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
            }

            String valAcc = df.format(correct/(correct+incorrect)*100);
            if (verbose){
                System.out.println("<" + oaName + " Iter: " + Integer.toString(i) + "> error: " + df.format(error) + ", train acc: " + trainingAcc + ", validation acc: " + valAcc);
            }
            printWriter.println(df.format(error) + "," + trainingAcc + "," + valAcc);
        }
        printWriter.close();
    }

    private static void initializeInstances() {

        double[][][] attributes = new double[2000][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/MNIST_4_9_size-1000_PCA.csv")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[115]; // 7 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 115; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
            if (br.readLine() != null){
                System.out.println("WARNING: Not expected line");
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        instances = new Instance[(int)((double) attributes.length * (1 - valSplit))];

        System.out.println("" + instances.length);
        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }
        validationInstances = new Instance[(int)((double) attributes.length * valSplit)];
        for(int i = instances.length; i < attributes.length; i++) {
            validationInstances[i - instances.length] = new Instance(attributes[i][0]);
            validationInstances[i - instances.length].setLabel(new Instance(attributes[i][1][0]));
        }


        //return instances;
    }
}
