package com.ilighti.ml.rank.model;

import java.io.Serializable;

/**
 * Created by rain on 16-8-9.
 */
public class Node  implements Serializable {
    private Integer id;
    private Predict predict;
    private Double impurity;
    private Boolean isLeaf;
    private Split split;
    private Node leftNode;
    private Node rightNode;
    private InformationGainStats stats;

    public Node(Integer id, Predict predict, Double impurity, Boolean isLeaf, Split split, Node leftNode, Node rightNode, InformationGainStats stats) {
        this.id = id;
        this.predict = predict;
        this.impurity = impurity;
        this.isLeaf = isLeaf;
        this.split = split;
        this.leftNode = leftNode;
        this.rightNode = rightNode;
        this.stats = stats;
    }

    public static Node of(org.apache.spark.mllib.tree.model.Node node) {
        InformationGainStats st = null;
        if(node.stats().isDefined()) {
            st = InformationGainStats.of(node.stats().get());
        }
        if(node.isLeaf()) {
            return new Node(node.id(), new Predict(node.predict().predict(), node.predict().prob()), node.impurity(), node.isLeaf(),
                    null, null, null, st);
        } else {
            return new Node(node.id(), new Predict(node.predict().predict(), node.predict().prob()), node.impurity(), node.isLeaf(),
                    Split.of(node.split().get()), of(node.leftNode().get()), of(node.rightNode().get()), st);
        }
    }

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public Predict getPredict() {
        return predict;
    }

    public void setPredict(Predict predict) {
        this.predict = predict;
    }

    public Double getImpurity() {
        return impurity;
    }

    public void setImpurity(Double impurity) {
        this.impurity = impurity;
    }

    public Boolean getLeaf() {
        return isLeaf;
    }

    public void setLeaf(Boolean leaf) {
        isLeaf = leaf;
    }

    public Split getSplit() {
        return split;
    }

    public void setSplit(Split split) {
        this.split = split;
    }

    public Node getLeftNode() {
        return leftNode;
    }

    public void setLeftNode(Node leftNode) {
        this.leftNode = leftNode;
    }

    public Node getRightNode() {
        return rightNode;
    }

    public void setRightNode(Node rightNode) {
        this.rightNode = rightNode;
    }

    public InformationGainStats getStats() {
        return stats;
    }

    public void setStats(InformationGainStats stats) {
        this.stats = stats;
    }

    public double predict(Vector feature) {
        if(isLeaf) {
            return predict.getPredict();
        }
        if(split.getFeatureType() == FeatureType.Continuous) {
            if(feature.getIndexedValue(split.getFeature())<=split.getThreshold()) {
                return leftNode.predict(feature);
            } else {
                return rightNode.predict(feature);
            }
        } else {
            if(split.getCategories().contains(feature.getIndexedValue(split.getFeature()))) {
                return leftNode.predict(feature);
            } else {
                return rightNode.predict(feature);
            }
        }
    }
}
