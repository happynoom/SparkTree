package com.ilighti.ml.rank.model;

import java.io.Serializable;

/**
 * Created by rain on 16-8-9.
 */
public class Predict implements Serializable {
    private Double predict;
    private Double prob;

    public Predict(Double predict, Double prob) {
        this.predict = predict;
        this.prob = prob;
    }

    public static Predict of(org.apache.spark.mllib.tree.model.Predict pred) {
        return new Predict(pred.predict(), pred.prob());
    }

    public Double getPredict() {
        return predict;
    }

    public void setPredict(Double predict) {
        this.predict = predict;
    }

    public Double getProb() {
        return prob;
    }

    public void setProb(Double prob) {
        this.prob = prob;
    }
}
