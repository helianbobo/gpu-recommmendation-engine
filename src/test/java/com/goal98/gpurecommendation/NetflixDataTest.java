package com.goal98.gpurecommendation;


import org.apache.mahout.cf.taste.example.netflix.NetflixDataModel;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.junit.Test;

import java.io.File;
import java.util.List;

public class NetflixDataTest {

    @Test
    public void testRecommendation() throws Exception {

        DataModel dataModel = new NetflixDataModel(new File("E:\\netflix\\download"), true);

        SVDRecommender svdRecommender = new SVDRecommender(dataModel, new GpuALSWRFactorizer(dataModel, 3, 0.1, 1));
        List<RecommendedItem> recommendedItems = svdRecommender.recommend(1488844, 10);

        for (int i = 0; i < recommendedItems.size(); i++) {
            RecommendedItem item = recommendedItems.get(i);
            System.out.println("item = " + item);
        }

    }

}
