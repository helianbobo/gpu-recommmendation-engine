package com.goal98.gpurecommendation;


import com.amd.aparapi.Kernel;

import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class Tanimoto {

    public static void main(String[] args) {
        int size = 1000000;
        final int [] setA = new int[1000000];
        final int [] setB = new int[size];
        final byte [] result = new byte[size];

        Random random = new Random();


        initSet(setA, random);
        initSet(setB, random);

//        hash( setA, setB);
        doKernel(size, setA, setB, result);

    }

    private static void hash(int[] setA, int[] setB) {
        long start = System.currentTimeMillis();

        Set<Integer> set = new HashSet<Integer>(1000000);
        for (int i = 0; i < setB.length; i++) {
            int i1 = setB[i];
            set.add(i1);
        }

        int intersection = 0;

        for (int i = 0; i < setA.length; i++) {
            int i1 = setA[i];
            if(set.contains(i1)){
                intersection++;
            }
        }

        System.out.println("intersection = " + intersection);
        System.out.println("System.currentTimeMillis() - start = " + (System.currentTimeMillis() - start));
    }

    private static void doKernel(int size, final int[] setA, final int[] setB, final byte[] result) {
        Kernel kernel = new Kernel(){
            @Override
            public void run() {
                int gid = getGlobalId();
                for (int i = 0; i < setA.length; i++) {
                    if(setB[gid] == setA[i]){
                        result[gid] = 1;
                    }
                }
            }
        };

        kernel.setExecutionMode(Kernel.EXECUTION_MODE.CPU);
        kernel.setExplicit(true);
        kernel.put(setA);
        kernel.put(setB);
        kernel.put(result);

        kernel.execute(size);

        kernel.get(result);

        System.out.println("kernel.getExecutionTime() = " + kernel.getExecutionTime());

        kernel.dispose();

        int intersection = 0;
        for (int i = 0; i < result.length; i++) {
            if(result[i] == 1)
                intersection ++;
        }

        System.out.println("intersection = " + intersection);
    }

    private static void initSet(int[] setB, Random random) {
        for (int i = 0; i < setB.length; i++) {
            setB[i] = Math.abs(random.nextInt()%10000000);
        }
    }

}
