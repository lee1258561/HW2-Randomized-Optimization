package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import java.util.*;
import java.io.*;
import java.text.*;
import opt.OptimizationAlgorithm;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    /** The n value */
    private static final int N = 20;
    private static DecimalFormat df = new DecimalFormat("0.000");
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();   
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp); 

        double start;
        //FixedIterationTrainer fit;     
        // fit = new FixedIterationTrainer(rhc, 20000);
        // fit.train();
        start = System.nanoTime();
        fixedIterationTraining(rhc, 1000, "TSP_RHC.log");
        System.out.println(ef.value(rhc.getOptimal()));
        System.out.println("Training Time: " + (System.nanoTime() - start) / Math.pow(10,9));
        
        
        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
        // fit = new FixedIterationTrainer(sa, 20000);
        // fit.train();
        start = System.nanoTime();
        fixedIterationTraining(sa, 1000, "TSP_SA.log");
        System.out.println(ef.value(sa.getOptimal()));
        System.out.println("Training Time: " + (System.nanoTime() - start) / Math.pow(10,9));
        
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
        // fit = new FixedIterationTrainer(ga, 1000);
        // fit.train();
        start = System.nanoTime();
        fixedIterationTraining(ga, 1000, "TSP_GA.log");
        System.out.println(ef.value(ga.getOptimal()));
        System.out.println("Training Time: " + (System.nanoTime() - start) / Math.pow(10,9));
        
        // for mimic we use a sort encoding
        ef = new TravelingSalesmanSortEvaluationFunction(points);
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        odd = new  DiscreteUniformDistribution(ranges);
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        MIMIC mimic = new MIMIC(200, 100, pop);
        // fit = new FixedIterationTrainer(mimic, 1000);
        // fit.train();
        start = System.nanoTime();
        fixedIterationTraining(mimic, 1000, "TSP_MIMIC.log");
        System.out.println(ef.value(mimic.getOptimal()));
        System.out.println("Training Time: " + (System.nanoTime() - start) / Math.pow(10,9));

        
    }

    private static double fixedIterationTraining(OptimizationAlgorithm trainer, int iterations, String outputFile){
        FileWriter fileWriter;
        try{
            fileWriter = new FileWriter(outputFile);
        }catch(IOException ex){
            System.out.println("IOException Caught!!!");
            return 0.0;
        }
        PrintWriter printWriter = new PrintWriter(fileWriter);
        double sum = 0;
        for (int i = 0; i < iterations; i++) {
            sum += trainer.train();
            double fitness = trainer.getOptimizationProblem().value(trainer.getOptimal());
            printWriter.println(df.format(fitness));
        }
        printWriter.close();
        return sum / iterations;
    }
}
