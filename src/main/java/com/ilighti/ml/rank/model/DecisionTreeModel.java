package com.ilighti.ml.rank.model;

import java.io.Serializable;

/**
 * Created by rain on 16-8-9.
 */
public class DecisionTreeModel implements Serializable {
    private Node topNode;
    private Algo algo;

    public DecisionTreeModel(Node topNode, Algo algo) {
        this.topNode = topNode;
        this.algo = algo;
    }

    public static DecisionTreeModel of(org.apache.spark.mllib.tree.model.DecisionTreeModel model) {
        return new DecisionTreeModel(Node.of(model.topNode()), Algo.fromString(model.algo().toString()));
    }

    public Node getTopNode() {
        return topNode;
    }

    public void setTopNode(Node topNode) {
        this.topNode = topNode;
    }

    public Algo getAlgo() {
        return algo;
    }

    public void setAlgo(Algo algo) {
        this.algo = algo;
    }

    public double predict(Vector feature) {
        return topNode.predict(feature);
    }
}
