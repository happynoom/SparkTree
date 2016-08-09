package com.ilighti.ml.app;

import com.google.gson.Gson;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.GradientBoostedTrees;
import org.apache.spark.mllib.tree.configuration.BoostingStrategy;
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel;
import org.apache.spark.mllib.util.MLUtils;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by rain on 16-7-6.
 */
public class AnchorCtrEstimate {
    public static void main(String[] args) {
        // $example on$
        SparkConf sparkConf = new SparkConf().setMaster("local[2]").setAppName("AnchorCtrEstimate");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        // Load and parse the data file.
        String datapath = "/home/rain/Projects/liblinear/daily_feature.train.28";
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();
        // Split the data into training and test sets (30% held out for testing)
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

        // Train a GradientBoostedTrees model.
        // The defaultParams for Regression use SquaredError by default.
        BoostingStrategy boostingStrategy = BoostingStrategy.defaultParams("Regression");
        boostingStrategy.setNumIterations(3); // Note: Use more iterations in practice.
        boostingStrategy.getTreeStrategy().setMaxDepth(5);
        // Empty categoricalFeaturesInfo indicates all features are continuous.
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        boostingStrategy.treeStrategy().setCategoricalFeaturesInfo(categoricalFeaturesInfo);

        final GradientBoostedTreesModel model =
                GradientBoostedTrees.train(trainingData, boostingStrategy);

        // Evaluate model on test instances and compute test error
        JavaPairRDD<Double, Double> predictionAndLabel =
                testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
                    @Override
                    public Tuple2<Double, Double> call(LabeledPoint p) {
                        return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
                    }
                });
        Double testMSE =
                predictionAndLabel.map(new Function<Tuple2<Double, Double>, Double>() {
                    @Override
                    public Double call(Tuple2<Double, Double> pl) {
                        Double diff = pl._1() - pl._2();
                        return diff * diff;
                    }
                }).reduce(new Function2<Double, Double, Double>() {
                    @Override
                    public Double call(Double a, Double b) {
                        return a + b;
                    }
                }) / data.count();
        System.out.println("Test Mean Squared Error: " + testMSE);
        System.out.println("Learned regression GBT model:\n" + model.toDebugString());

        // Save and load model
        model.save(jsc.sc(), "myGradientBoostingRegressionModel");
        GradientBoostedTreesModel sameModel = GradientBoostedTreesModel.load(jsc.sc(),
                "myGradientBoostingRegressionModel");
        final com.ilighti.ml.rank.model.GradientBoostedTreesModel jmodel = com.ilighti.ml.rank.model.GradientBoostedTreesModel.of(sameModel);
        System.out.println(new Gson().toJson(jmodel));

        JavaPairRDD<Double, Double> postTransformPredictAndLabel = testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
            @Override
            public Tuple2<Double, Double> call(LabeledPoint p) {
                return new Tuple2<Double, Double>(jmodel.predictBySumming(com.ilighti.ml.rank.model.Vector.of(p.features().toSparse())), p.label());
            }
        });
        Double postTransformTestMSE =
                postTransformPredictAndLabel.map(new Function<Tuple2<Double, Double>, Double>() {
                    @Override
                    public Double call(Tuple2<Double, Double> pl) {
                        Double diff = pl._1() - pl._2();
                        return diff * diff;
                    }
                }).reduce(new Function2<Double, Double, Double>() {
                    @Override
                    public Double call(Double a, Double b) {
                        return a + b;
                    }
                }) / data.count();
        System.out.println("Post Transform Mean Squared Error: " + postTransformTestMSE);
        // $example off$
    }
}