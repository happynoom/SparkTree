package com.ilighti.ml.rank.model;

/**
 * Created by rain on 16-8-9.
 */
public enum FeatureType {
    Continuous("Continuous"), Categorical("Categorical");
    private String name;
    FeatureType(String name) {
        this.name = name;
    }
    public static FeatureType fromString(String name) {
        if(name.toLowerCase().equals("continuous")) {
            return Continuous;
        } else if(name.toLowerCase().equals("categorical")) {
            return Categorical;
        }
        return null;
    }
    public String toString() {
        return name;
    }
}
