package com.goal98.gpurecommendation;


import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.example.netflix.NetflixDataModel;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.junit.Test;

import java.io.File;
import java.util.List;

public class NetflixDataTest {

    @Test
    public void testRecommendation() throws Exception {

        DataModel dataModel = new NetflixDataModel(new File("E:\\netflix\\download"), true);

        final GpuALSWRFactorizer factorizer = new GpuALSWRFactorizer(dataModel, 3, 0.1, 1);

        RMSRecommenderEvaluator evaluator = new RMSRecommenderEvaluator();
        evaluator.evaluate(new RecommenderBuilder() {
            @Override
            public Recommender buildRecommender(DataModel dataModel) throws TasteException {
                return new SVDRecommender(dataModel, factorizer);
            }
        }, null, dataModel, 0.85, 1.0);


    }

}
