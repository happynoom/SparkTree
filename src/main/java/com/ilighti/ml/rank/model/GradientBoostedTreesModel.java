package com.ilighti.ml.rank.model;

import com.github.fommil.netlib.BLAS;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

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

    public Double predictByVoting(Vector feature) {
        Map<Integer, Double> votes = new HashMap<Integer, Double>();
        int k = 0;
        for(DecisionTreeModel tree : trees) {
            Integer prediction = (int)tree.predict(feature);
            if(votes.containsKey(prediction)) {
                votes.put(prediction, votes.get(prediction) + treeWeights[k]);
            } else {
                votes.put(prediction, treeWeights[k]);
            }
            k++;
        }
        //max voted label
        Integer label = 0;
        Double maxVotes = Double.valueOf(-Float.MIN_VALUE);
        for(Map.Entry<Integer, Double> entry : votes.entrySet()) {
            if(maxVotes < entry.getValue()) {
                maxVotes = entry.getValue();
                label = entry.getKey();
            }
        }
        return label.doubleValue();
    }

    private double sumOfWeights() {
        double sum = 0.;
        for(double w : treeWeights) {
            sum += w;
        }
        return sum;
    }

    public Double predict(Vector feature) {
        if (algo.equals(Algo.Regression) && strategy.equals(EnsembleCombiningStrategy.Sum)) {
            return predictBySumming(feature);
        } else if (algo.equals(Algo.Regression) && strategy.equals(EnsembleCombiningStrategy.Average)) {
            return predictBySumming(feature) / sumOfWeights();
        } else if (algo.equals(Algo.Classification) && strategy.equals(EnsembleCombiningStrategy.Sum)) {
            Double prediction = predictBySumming(feature);
            return prediction > 0. ? 1. : 0.;
        } else if (algo.equals(Algo.Classification) && strategy.equals(EnsembleCombiningStrategy.Vote)) {
            return predictByVoting(feature);
        } else {
            throw new RuntimeException("TreeEnsembleModel given unsupported (" + algo + "," + strategy + ")");
        }
    }

    public Double[] predict(Vector[] features) {
        Double[] ret = new Double[features.length];
        for (int i = 0; i < features.length; i++) {
            ret[i] = predict(features[i]);
        }
        return ret;
    }
}
