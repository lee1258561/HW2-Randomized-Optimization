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
 * A test using the flip flop evaluation function
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FlipFlopTest {
    /** The n value */
    private static final int N = 100;
    private static DecimalFormat df = new DecimalFormat("0.000");
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FlipFlopEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        // FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 20000);
        // fit.train();
        fixedIterationTraining(rhc, 20000, "FF_RHC.log");
        System.out.println(ef.value(rhc.getOptimal()));
        
        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
        // fit = new FixedIterationTrainer(sa, 20000);
        // fit.train();
        fixedIterationTraining(sa, 20000, "FF_SA.log");
        System.out.println(ef.value(sa.getOptimal()));
        
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 20, gap);
        // fit = new FixedIterationTrainer(ga, 1000);
        // fit.train();
        fixedIterationTraining(ga, 1000, "FF_GA.log");
        System.out.println(ef.value(ga.getOptimal()));
        
        MIMIC mimic = new MIMIC(200, 100, pop);
        // fit = new FixedIterationTrainer(mimic, 1000);
        // fit.train();
        fixedIterationTraining(mimic, 1000, "FF_MIMIC.log");
        System.out.println(ef.value(mimic.getOptimal()));
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
