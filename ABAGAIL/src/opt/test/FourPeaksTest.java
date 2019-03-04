package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import java.util.*;
import java.io.*;
import java.text.*;
import opt.OptimizationAlgorithm;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The n value */
    private static final int N = 100;
    /** The t value */
    private static final int T = 6;
    private static DecimalFormat df = new DecimalFormat("0.000");
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        double start;
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 20000);
        fit.train();
        start = System.nanoTime();
        fixedIterationTraining(rhc, 20000, "FP_RHC.log");

        System.out.println("RHC: " + ef.value(rhc.getOptimal()));
        System.out.println("Training Time: " + (System.nanoTime() - start) / Math.pow(10,9));
        
        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
        // fit = new FixedIterationTrainer(sa, 20000);
        // fit.train();
        start = System.nanoTime();
        fixedIterationTraining(sa, 20000, "FP_SA.log");
        System.out.println("SA: " + ef.value(sa.getOptimal()));
        System.out.println("Training Time: " + (System.nanoTime() - start) / Math.pow(10,9));
        
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 50, gap);
        // fit = new FixedIterationTrainer(ga, 1000);
        // fit.train();
        start = System.nanoTime();
        fixedIterationTraining(ga, 20000, "FP_GA.log");
        System.out.println("GA: " + ef.value(ga.getOptimal()));
        System.out.println("Training Time: " + (System.nanoTime() - start) / Math.pow(10,9));
        
        MIMIC mimic = new MIMIC(200, 20, pop);
        // fit = new FixedIterationTrainer(mimic, 1000);
        // fit.train();
        start = System.nanoTime();
        fixedIterationTraining(mimic, 10000, "FP_MIMIC.log");
        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));
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
