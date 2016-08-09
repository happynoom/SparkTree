package com.ilighti.ml.rank.model;

/**
 * Created by rain on 16-8-9.
 */
public enum EnsembleCombiningStrategy {
    Average("Average"), Sum("Sum"), Vote("Vote");
    private String name;
    EnsembleCombiningStrategy(String name) {
        this.name = name;
    }
    public static EnsembleCombiningStrategy fromString(String name) {
        if(name.toLowerCase().equals("average")) {
            return Average;
        } else if(name.toLowerCase().equals("sum")) {
            return Sum;
        } else if(name.toLowerCase().equals("vote")) {
            return Vote;
        }
        return null;
    }
    public String toString() {
        return name;
    }
}
