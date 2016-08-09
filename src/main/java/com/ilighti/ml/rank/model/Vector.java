package com.ilighti.ml.rank.model;

import java.io.Serializable;

/**
 * Created by rain on 16-8-9.
 */
public class Vector implements Serializable {
    private int[] indices;
    private double[] values;

    public int length() {
        return indices.length;
    }

    public Vector(int[] indices, double[] values) {
        this.indices = indices;
        this.values = values;
    }

    public static Vector of(org.apache.spark.mllib.linalg.SparseVector feature) {
        return new Vector(feature.indices(), feature.values());
    }

    public int[] getIndices() {
        return indices;
    }

    public void setIndices(int[] indices) {
        this.indices = indices;
    }

    public double[] getValues() {
        return values;
    }

    public void setValues(double[] values) {
        this.values = values;
    }

    public double getIndexedValue(int index) {
        for(int i=0;i<indices.length;i++) {
            if(index == indices[i]) {
                return values[i];
            }
        }
        return 0d;
    }
}
