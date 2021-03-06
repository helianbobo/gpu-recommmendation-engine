import com.amd.aparapi.Kernel;
import org.apache.mahout.common.RandomUtils;

import java.nio.ByteBuffer;


public class MurmurHash {

    public static void main(String[] _args) {


        final int size = 24;

        hash((int) Math.pow(2, size), Kernel.EXECUTION_MODE.JTP);


        for (int i = 0; i < 3; i++) {
            hash((int) Math.pow(2, size), Kernel.EXECUTION_MODE.GPU);
        }
    }


    private static void hash(int size, Kernel.EXECUTION_MODE mode) {
        System.out.println("\n\nsize = " + size);

        int numInt = 1;
        int length = numInt * 4;
        final int[] lengthArray = new int[]{length};

        final byte[] values = new byte[size * length];

        for (int i = 0; i < size; i++) {
            ByteBuffer buffer = ByteBuffer.allocate(length);

            for (int j = 0; j < numInt; j++) {
                buffer.putInt(i);
            }

            byte[] bytes = buffer.array();

            System.arraycopy(bytes, 0, values, i * length, length);


        }

        final long[] squares = new long[size];

        Kernel kernel = new Kernel() {

            public static final int MAX_INT_SMALLER_TWIN_PRIME = 2147482949;

            @Override
            public void run() {
                int gid = getGlobalId();
                squares[gid] = hash(gid, 1,10,100);

            }

            private int hash(int gid, int seedA, int seedB, int seedC) {
                long hashValue = 31;
                final int length = lengthArray[0];
                final int offset = gid * length;

//                final float s = sqrt(100) * asin(10)*tan(30)*sqrt(101) * asin(11)*tan(31);

                for (int i = 0; i < length; i++){

                    long byteVal = values[offset + i];
                    hashValue *= seedA * (byteVal >> 4);
                    hashValue += seedB * byteVal + seedC;

                }

                return abs((int) (hashValue % MAX_INT_SMALLER_TWIN_PRIME));
            }
        };

        kernel.setExecutionMode(mode);
        kernel.setExplicit(true);
        kernel.put(values);
        kernel.put(lengthArray);
        kernel.execute(size);
        kernel.get(squares);

        System.out.println("duration = " + kernel.getExecutionTime());
        System.out.println("kernel.getConversionTime() = " + kernel.getConversionTime());

        kernel.dispose();


//        murmurHash(size, mode, lengthArray, values);


    }

    private static void murmurHash(int size, Kernel.EXECUTION_MODE mode, final int[] lengthArray, final byte[] values) {
        /** Output array which will be populated with square values of corresponding input array elements. */
        final long[] squares = new long[size];

        final int[] seeds = new int[]{1000};

        /** Aparapi Kernel which computes squares of input array elements and populates them in corresponding elements of
         * output array.
         **/
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int gid = getGlobalId();
                squares[gid] = murmurHash(gid);

            }

            private long murmurHash(int gid) {

                int length = lengthArray[0];

                int offset = gid * length;

                int seed = seeds[0];
                long m64 = 0xc6a4a7935bd1e995L;
                int r64 = 47;

                long h64 = (seed & 0xffffffffL) ^ (m64 * length);

                int lenLongs = length >> 3;

                for (int i = 0; i < lenLongs; ++i) {
                    int i_8 = i << 3;

                    long k64 =
                            ((long) values[offset + i_8 + 0] & 0xff) + (((long) values[offset + i_8 + 1] & 0xff) << 8) +
                                    (((long) values[offset + i_8 + 2] & 0xff) << 16) + (((long) values[offset + i_8 + 3] & 0xff) << 24) +
                                    (((long) values[offset + i_8 + 4] & 0xff) << 32) + (((long) values[offset + i_8 + 5] & 0xff) << 40) +
                                    (((long) values[offset + i_8 + 6] & 0xff) << 48) + (((long) values[offset + i_8 + 7] & 0xff) << 56);

                    k64 *= m64;
                    k64 ^= k64 >>> r64;
                    k64 *= m64;

                    h64 ^= k64;
                    h64 *= m64;
                }

                int rem = length & 0x7;

                if (rem == 0) {

                } else if (rem == 1) {

                    h64 ^= (long) values[offset + length - rem];
                    h64 *= m64;
                } else {
                    h64 ^= (long) values[offset + length - 1] << (8 * (rem - 1));
                }


                h64 ^= h64 >>> r64;
                h64 *= m64;
                h64 ^= h64 >>> r64;

                return h64;

            }
        };

        for (int i = 0; i < 1; ++i) {
            doHash(size, mode, lengthArray, values, squares, seeds, kernel);
        }
    }

    private static void doHash(int size, Kernel.EXECUTION_MODE mode, int[] lengthArray, byte[] values, long[] squares, int[] seeds, Kernel kernel) {
        kernel.setExplicit(true);
        kernel.put(lengthArray);
        kernel.put(values);
        kernel.put(squares);
        kernel.put(seeds);

        kernel.setExecutionMode(mode);


        // Execute Kernel.
        kernel.execute(size);

        kernel.get(squares);

        System.out.println("duration = " + kernel.getExecutionTime());
        System.out.println("kernel.getConversionTime() = " + kernel.getConversionTime());

        // Report target execution mode: GPU or JTP (Java Thread Pool).
        System.out.println("Execution mode=" + kernel.getExecutionMode());

        // Display computed square values.
        /*for (int i = 0; i < size; i++) {
           System.out.printf("%6.0f %8.0f\n", values[i], squares[i]);
        }*/

        System.out.printf("%d %d\n", 1, squares[0]);
        System.out.printf("%d %d\n", 2, squares[1]);
        System.out.printf("%d %d\n", 3, squares[2]);
        System.out.printf("%d %d\n", size, squares[size - 1]);

        // Dispose Kernel resources.
        kernel.dispose();
    }

}
