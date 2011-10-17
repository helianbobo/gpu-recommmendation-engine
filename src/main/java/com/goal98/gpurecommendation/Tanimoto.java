package com.goal98.gpurecommendation;


import com.amd.aparapi.Kernel;

import java.util.*;

public class Tanimoto {

    public static void main(String[] args) {
        int size = 1000;
        int items = 512;

        int[] data = new int[size*items];

        initData(size, items, data);

        int [] instructions = new int[items * items * 4];

        initInstructions(size, items, instructions);



        final int[] result = new int[items*items];

        doKernel(items, data, instructions, result, Kernel.EXECUTION_MODE.JTP);
        doKernel(items, data, instructions, result, Kernel.EXECUTION_MODE.CPU);
        doKernel(items, data, instructions, result, Kernel.EXECUTION_MODE.GPU);

    }

    private static void initData(int size, int items, int[] data) {
        for (int i = 0; i < items; i++) {
            int[] array = new int[size];
            initSet(array);
            System.arraycopy(array, 0, data, i*size, size);
        }
    }

    private static void initInstructions(int size, int items, int[] instructions) {
        int instructionsCount = 0;
        for (int i = 0; i < items; i++) {
            for (int j = 0; j < items; j++) {
                instructions[instructionsCount*4] = i * size;
                instructions[instructionsCount*4 + 1] = size;
                instructions[instructionsCount*4 + 2] = j * size;
                instructions[instructionsCount*4 + 3] = size;
                instructionsCount++;
            }
        }
    }

    private static void hash(int[] setA, int[] setB) {
        long start = System.currentTimeMillis();

        Set<Integer> set = new HashSet<Integer>();
        for (int i = 0; i < setB.length; i++) {
            int i1 = setB[i];
            set.add(i1);
        }

        int intersection = 0;

        for (int i = 0; i < setA.length; i++) {
            int i1 = setA[i];
            if (set.contains(i1)) {
                intersection++;
            }
        }

        System.out.println("intersection = " + intersection);
        System.out.println("System.currentTimeMillis() - start = " + (System.currentTimeMillis() - start));
    }

    private static void binaryserarch(int[] setA, int[] setB) {
        long start = System.currentTimeMillis();
        int intersection = 0;

        for (int i = 0; i < setA.length; i++) {
            int i1 = setA[i];

            if (contains(setB, i1)) {
                intersection++;
            }
        }

        System.out.println("intersection = " + intersection);
        System.out.println("System.currentTimeMillis() - start = " + (System.currentTimeMillis() - start));
    }

    private static boolean contains(int []array, int num) {
        int low = 0;
        int high = array.length - 1;

        while (low <= high) {
            int mid = (low + high) >>> 1;
            int midVal = array[mid];

            if (midVal < num)
                low = mid + 1;
            else if (midVal > num)
                high = mid - 1;
            else
                return true; // key found
        }
        return false;  // key not found
    }

    private static void doKernel(int items, final int[] data, final int[] instructions, final int[] result, Kernel.EXECUTION_MODE mode) {
        int pass = 16;
        int perpass = items * items / pass;

        final int [] perpassArray = new int[]{perpass};

        Kernel kernel = new Kernel() {
            @Override
            public void run() {

                int perpass = perpassArray[0];
                int passid = getPassId();
                int gid = getGlobalId() + perpass*passid;


                int startA = instructions[gid];
                int lenghtA = instructions[gid+1];
                int startB = instructions[gid+2];
                int lenghtB = instructions[gid+3];

                int intersection = 0;

                for(int i = startA; i < startA + lenghtA; i++){
                    int key = data[i];
                    if (contains(key, startB, lenghtB))
                        intersection++;
                }
                result[gid] = intersection;


            }

            private boolean contains(int num, int start, int length) {
                int low = start;
                int high = start + length - 1;

                while (low <= high) {
                    int mid = (low + high) >>> 1;
                    int midVal = data[mid];

                    if (midVal < num)
                        low = mid + 1;
                    else if (midVal > num)
                        high = mid - 1;
                    else
                        return true; // key found
                }
                return false;  // key not found
            }

        };

        kernel.setExecutionMode(mode);
//        kernel.setExplicit(true);
//        kernel.put(data);
//        kernel.put(instructions);
//        kernel.put(result);

        kernel.execute(perpass, pass);

//        kernel.get(result);

        System.out.println("kernel.getExecutionTime() = " + kernel.getExecutionTime());
        System.out.println("kernel.getConversionTime() = " + kernel.getConversionTime());
        final long actualTime = kernel.getExecutionTime() - kernel.getConversionTime();
        System.out.println("actualTime = " + actualTime);

        kernel.dispose();
    }

    private static void initSet(int[] set) {
        Random random = new Random();
        int shift = Math.abs(random.nextInt()%10000);

        for (int i = 0; i < set.length; i++) {
            set[i] = i + shift;

        }
    }

}
