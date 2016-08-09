package com.ilighti.ml.rank.model;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by rain on 16-8-9.
 */
public class Split implements Serializable {
    private Integer feature;
    private Double threshold;
    private FeatureType featureType;
    private List<Double> categories;

    public Split(Integer feature, Double threshold, FeatureType featureType, List<Double> categories) {
        this.feature = feature;
        this.threshold = threshold;
        this.featureType = featureType;
        this.categories = categories;
    }

    public static Split of(org.apache.spark.mllib.tree.model.Split split) {
        List<Double> cate = null;
        if(split.featureType() == org.apache.spark.mllib.tree.configuration.FeatureType.Categorical()) {
            cate = new ArrayList<Double>();
            scala.collection.Iterator<Object> iter = split.categories().iterator();
            while(iter.hasNext()) {
                Double value = (Double) iter.next();
                cate.add(value);
            }
        }
        return new Split(split.feature(), split.threshold(), FeatureType.fromString(split.featureType().toString()), cate);
    }

    public Integer getFeature() {
        return feature;
    }

    public void setFeature(Integer feature) {
        this.feature = feature;
    }

    public Double getThreshold() {
        return threshold;
    }

    public void setThreshold(Double threshold) {
        this.threshold = threshold;
    }

    public FeatureType getFeatureType() {
        return featureType;
    }

    public void setFeatureType(FeatureType featureType) {
        this.featureType = featureType;
    }

    public List<Double> getCategories() {
        return categories;
    }

    public void setCategories(List<Double> categories) {
        this.categories = categories;
    }
}
