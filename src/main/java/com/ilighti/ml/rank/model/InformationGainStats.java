package com.ilighti.ml.rank.model;

import java.io.Serializable;

/**
 * Created by rain on 16-8-9.
 */
public class InformationGainStats implements Serializable {
    private Double gain;
    private Double impurity;
    private Double leftImpurity;
    private Double rightImpurity;
    private Predict leftPredict;
    private Predict rightPredict;

    public InformationGainStats(Double gain, Double impurity, Double leftImpurity, Double rightImpurity, Predict leftPredict, Predict rightPredict) {
        this.gain = gain;
        this.impurity = impurity;
        this.leftImpurity = leftImpurity;
        this.rightImpurity = rightImpurity;
        this.leftPredict = leftPredict;
        this.rightPredict = rightPredict;
    }

    public static InformationGainStats of(org.apache.spark.mllib.tree.model.InformationGainStats stats) {
        return new InformationGainStats(stats.gain(), stats.impurity(), stats.leftImpurity(), stats.rightImpurity(),
                Predict.of(stats.leftPredict()), Predict.of(stats.rightPredict()));
    }

    public Double getGain() {
        return gain;
    }

    public void setGain(Double gain) {
        this.gain = gain;
    }

    public Double getImpurity() {
        return impurity;
    }

    public void setImpurity(Double impurity) {
        this.impurity = impurity;
    }

    public Double getLeftImpurity() {
        return leftImpurity;
    }

    public void setLeftImpurity(Double leftImpurity) {
        this.leftImpurity = leftImpurity;
    }

    public Double getRightImpurity() {
        return rightImpurity;
    }

    public void setRightImpurity(Double rightImpurity) {
        this.rightImpurity = rightImpurity;
    }

    public Predict getLeftPredict() {
        return leftPredict;
    }

    public void setLeftPredict(Predict leftPredict) {
        this.leftPredict = leftPredict;
    }

    public Predict getRightPredict() {
        return rightPredict;
    }

    public void setRightPredict(Predict rightPredict) {
        this.rightPredict = rightPredict;
    }
}
