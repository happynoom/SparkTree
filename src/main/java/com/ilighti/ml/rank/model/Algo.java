package com.ilighti.ml.rank.model;

/**
 * Created by rain on 16-8-9.
 */
public enum Algo {
    Classification("Classification"), Regression("Regression");
    private String name;
    Algo(String name) {
        this.name = name;
    }
    public static Algo fromString(String name) {
        if(name.toLowerCase().equals("classification")) {
            return Classification;
        } else if(name.toLowerCase().equals("regression")) {
            return Regression;
        }
        return null;
    }
    public String toString() {
        return name;
    }
}
