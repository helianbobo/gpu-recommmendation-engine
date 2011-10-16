package com.goal98.gpurecommendation;


import com.amd.aparapi.Kernel;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.recommender.svd.ALSWRFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.AbstractFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.Factorization;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.als.AlternateLeastSquaresSolver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.rmi.runtime.Log;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;

public class GpuALSWRFactorizer extends ALSWRFactorizer {

    private final DataModel dataModel;


    private final int numFeatures;

    private final double lambda;

    private final int numIterations;

    private static final Logger log = LoggerFactory.getLogger(ALSWRFactorizer.class);

    public GpuALSWRFactorizer(DataModel dataModel, int numFeatures, double lambda, int numIterations) throws TasteException {
        super(dataModel, numFeatures, lambda, numIterations);
        this.dataModel = dataModel;
        this.numFeatures = numFeatures;
        this.lambda = lambda;
        this.numIterations = numIterations;

    }

    static class ALSWRKernel extends Kernel {

        double[] features;
        int numFeatures;

        ALSWRKernel(double[] features, int numFeatures) {
            this.features = features;
            this.numFeatures = numFeatures;
        }

        @Override
        public void run() {
            int globalId = getGlobalId();
            final int index = globalId * numFeatures;
            features[index] = features[index] * 2;
        }
    }

    @Override
    public Factorization factorize() throws TasteException {
        log.info("starting to compute the factorization...");
        final AlternateLeastSquaresSolver solver = new AlternateLeastSquaresSolver();
        final Kernel.EXECUTION_MODE executionMode = Kernel.EXECUTION_MODE.GPU;

        final int numItems = dataModel.getNumItems();
        final int numUsers = dataModel.getNumUsers();

        final double[] featuresM;
        final double[] featuresU;

        log.info("features initialization ...");
        Random random = RandomUtils.getRandom();
        featuresM = new double[numItems * this.numFeatures];
        LongPrimitiveIterator itemIDsIterator = this.dataModel.getItemIDs();
        while (itemIDsIterator.hasNext()) {
            long itemID = itemIDsIterator.nextLong();
            int itemIDIndex = itemIndex(itemID);
            featuresM[itemIDIndex * numFeatures] = averateRating(itemID);
            for (int feature = 1; feature < this.numFeatures; feature++) {
                featuresM[itemIDIndex * numFeatures + feature] = random.nextDouble() * 0.1;
            }
        }
        featuresU = new double[numUsers * this.numFeatures];


        log.info("start kernelM...");
        calculateFeatureExplicit(executionMode, numItems, featuresM);

        calculateFeatureExplicit(executionMode, numUsers, featuresU);

        /*log.info("start kernelU...");
        Kernel kernelU = new ALSWRKernel(featuresU, numFeatures);
        kernelU.setExecutionMode(executionMode);

        kernelU.execute(numUsers);
        kernelU.dispose();*/


        log.info("finished computation of the factorization...");

        double[][] M = new double[numItems][numFeatures];
        double[][] U = new double[numUsers][numFeatures];

        buildMatrix(U, featuresU);
        buildMatrix(M, featuresM);
        return createFactorization(U, M);
    }

    private void calculateFeature(Kernel.EXECUTION_MODE executionMode, int num, double[] features) {
        Kernel kernel = new ALSWRKernel(features, numFeatures);
        kernel.setExecutionMode(executionMode);
        kernel.execute(num);
        kernel.dispose();
    }

    private void calculateFeatureExplicit(Kernel.EXECUTION_MODE executionMode, int num, final double[] features) {

        final int[] numFeaturesArray = new int[]{numFeatures};

        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int globalId = getGlobalId();
                final int index = globalId * numFeaturesArray[0];

                if(index + numFeaturesArray[0] > features.length) return;

                for (int i = 0; i < numFeaturesArray[0]; i++) {
                    features[index + i] = features[index + i] * 2;
                }

            }
        };
        kernel.setExecutionMode(executionMode);
        kernel.setExplicit(true);

        kernel.put(features);
        kernel.put(numFeaturesArray);

        kernel.execute(findExeSize(num));

        kernel.get(features);

        kernel.dispose();
    }

    private int findExeSize(int num){

        log.debug("-- findExeSize({})", num);

        int result = 0;
        int i = 10;
        while (result < num){
            result = (int)Math.pow(2, i++);
        }


        log.debug("<< findExeSize return {}", result);

        return result;

    }

    private void buildMatrix(double[][] u, double[] features) {
        for (int i = 0; i < features.length; i++) {
            double featureValue = features[i];
            u[i / numFeatures][i % numFeatures] = featureValue;

        }
    }

    private double averateRating(long itemID) throws TasteException {
        PreferenceArray prefs = dataModel.getPreferencesForItem(itemID);
        RunningAverage avg = new FullRunningAverage();
        for (Preference pref : prefs) {
            avg.addDatum(pref.getValue());
        }
        return avg.getAverage();
    }


    static class Features {

        private final DataModel dataModel;
        private final int numFeatures;

        private final double[] M;
        private final double[] U;

        Features(GpuALSWRFactorizer factorizer) throws TasteException {
            this.dataModel = factorizer.dataModel;
            this.numFeatures = factorizer.numFeatures;
            Random random = RandomUtils.getRandom();
            M = new double[this.dataModel.getNumItems() * this.numFeatures];
            LongPrimitiveIterator itemIDsIterator = this.dataModel.getItemIDs();
            while (itemIDsIterator.hasNext()) {
                long itemID = itemIDsIterator.nextLong();
                int itemIDIndex = factorizer.itemIndex(itemID);
                M[itemIDIndex * numFeatures] = averateRating(itemID);
                for (int feature = 1; feature < this.numFeatures; feature++) {
                    M[itemIDIndex * numFeatures + feature] = random.nextDouble() * 0.1;
                }
            }
            U = new double[this.dataModel.getNumUsers() * this.numFeatures];
        }

        double[] getM() {
            return M;
        }

        double[] getU() {
            return U;
        }

        DenseVector getUserFeatureColumn(int index) {
            double[] values = new double[numFeatures];
            for (int i = 0; i < numFeatures; i++) {
                values[i] = U[index * numFeatures + i];

            }
            return new DenseVector(values);
        }

        DenseVector getItemFeatureColumn(int index) {
            double[] values = new double[numFeatures];
            for (int i = 0; i < numFeatures; i++) {
                values[i] = M[index * numFeatures + i];

            }
            return new DenseVector(values);
        }

        void setFeatureColumnInU(int idIndex, Vector vector) {
            setFeatureColumn(U, idIndex, vector);
        }

        void setFeatureColumnInM(int idIndex, Vector vector) {
            setFeatureColumn(M, idIndex, vector);
        }

        protected void setFeatureColumn(double[] matrix, int idIndex, Vector vector) {
            for (int feature = 0; feature < numFeatures; feature++) {
                matrix[idIndex * numFeatures + feature] = vector.get(feature);
            }
        }

        protected double averateRating(long itemID) throws TasteException {
            PreferenceArray prefs = dataModel.getPreferencesForItem(itemID);
            RunningAverage avg = new FullRunningAverage();
            for (Preference pref : prefs) {
                avg.addDatum(pref.getValue());
            }
            return avg.getAverage();
        }
    }
}
