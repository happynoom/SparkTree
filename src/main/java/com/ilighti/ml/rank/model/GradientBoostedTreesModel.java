package com.ilighti.ml.rank.model;

import com.github.fommil.netlib.BLAS;

import java.io.Serializable;

/**
 * Created by rain on 16-8-9.
 */
public class GradientBoostedTreesModel implements Serializable {
    private Algo algo;
    private DecisionTreeModel[] trees;
    private double[] treeWeights;
    private EnsembleCombiningStrategy strategy;

    public GradientBoostedTreesModel(Algo algo, DecisionTreeModel[] trees, double[] treeWeights, EnsembleCombiningStrategy strategy) {
        this.algo = algo;
        this.trees = trees;
        this.treeWeights = treeWeights;
        this.strategy = strategy;
    }

    public static GradientBoostedTreesModel of(org.apache.spark.mllib.tree.model.GradientBoostedTreesModel model) {
        DecisionTreeModel[] decisionTreeModels = new DecisionTreeModel[model.numTrees()];
        for(int i=0; i<model.numTrees(); i++) {
            decisionTreeModels[i] = DecisionTreeModel.of(model.trees()[i]);
        }
        Algo algo = Algo.fromString(model.algo().toString());
        double[] treeWeights = model.treeWeights();
        EnsembleCombiningStrategy strategy = EnsembleCombiningStrategy.fromString(model.combiningStrategy().toString());
        return new GradientBoostedTreesModel(algo, decisionTreeModels, treeWeights, strategy);
    }

    public Algo getAlgo() {
        return algo;
    }

    public void setAlgo(Algo algo) {
        this.algo = algo;
    }

    public DecisionTreeModel[] getTrees() {
        return trees;
    }

    public void setTrees(DecisionTreeModel[] trees) {
        this.trees = trees;
    }

    public double[] getTreeWeights() {
        return treeWeights;
    }

    public void setTreeWeights(double[] treeWeights) {
        this.treeWeights = treeWeights;
    }

    public EnsembleCombiningStrategy getStrategy() {
        return strategy;
    }

    public void setStrategy(EnsembleCombiningStrategy strategy) {
        this.strategy = strategy;
    }

    public Double predictBySumming(Vector feature) {
        int numTrees = trees.length;
        double[] treePredictions = new double[numTrees];
        int k = 0;
        for(DecisionTreeModel tree : trees) {
            treePredictions[k++] = tree.predict(feature);
        }
        BLAS bias = com.github.fommil.netlib.BLAS.getInstance();
        return bias.ddot(numTrees, treePredictions, 1, treeWeights, 1);
    }
}
