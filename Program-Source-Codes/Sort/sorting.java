package sorting_algo;

import java.util.Random;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.util.Arrays;

public class sorting {

	public static void bubbleSort(int[] ar) {
		for (int i = (ar.length - 1); i >= 0; i--) {
			for (int j = 1; j <= i; j++) {
				if (ar[j - 1] > ar[j]) {
					int temp = ar[j - 1];
					ar[j - 1] = ar[j];
					ar[j] = temp;
				}
			}
		}
	}

	public static void selectionSort(int[] ar) {
		for (int i = 0; i < ar.length - 1; i++) {
			int min = i;
			for (int j = i + 1; j < ar.length; j++)
				if (ar[j] < ar[min])
					min = j;
			int temp = ar[i];
			ar[i] = ar[min];
			ar[min] = temp;
		}
	}

	public static void insertionSort(int[] ar) {
		for (int i = 1; i < ar.length; i++) {
			int index = ar[i];
			int j = i;
			while (j > 0 && ar[j - 1] > index) {
				ar[j] = ar[j - 1];
				j--;
			}
			ar[j] = index;
		}
	}

	public static void bucketSort(int[] a, int maxVal) {
		int[] bucket = new int[maxVal + 1];

		for (int i = 0; i < bucket.length; i++) {
			bucket[i] = 0;
		}

		for (int i = 0; i < a.length; i++) {
			bucket[a[i]]++;
		}

		int outPos = 0;
		for (int i = 0; i < bucket.length; i++) {
			for (int j = 0; j < bucket[i]; j++) {
				a[outPos++] = i;
			}
		}
	}

	public static long time_bubbleSort(int[] A) {
		long start = System.nanoTime();
		bubbleSort(A);
		long end = System.nanoTime();
		long duration = end - start;
		return duration;
	}

	public static long time_selectionSort(int[] A) {
		long start = System.nanoTime();
		selectionSort(A);
		long end = System.nanoTime();
		long duration = end - start;
		return duration;
	}

	public static long time_insertionSort(int[] A) {
		long start = System.nanoTime();
		insertionSort(A);
		long end = System.nanoTime();
		long duration = end - start;
		return duration;
	}

	public static long time_bucketSort(int[] A, int max) {
		long start = System.nanoTime();
		bucketSort(A, max);
		long end = System.nanoTime();
		long duration = end - start;
		return duration;
	}

	// Return timing arrays for each sorting function (times in ns are stored in the
	// arrays passed as parameters)
	public static void get_timing(int lo, int hi, int random_factor, int[][] inputs, long[] bubble_arr, long[] selection_arr,
			long[] insertion_arr, long[] bucket_arr, long[] merge_arr, long[] quick_arr) {
		for (int i = lo; i < hi; i++) {
			for (int k = 0; k < random_factor;k++) 
			{
				// Copy of the input row to be passed to the sorting functions
				// This has to be done because sorting modifies the input array
				int[] A = new int[hi];
				int max = -1;

				// Row index
				int idx = (i - lo)*random_factor + k;

				for (int j = 0; j < i; j++) {
					Random ran = new Random();
					inputs[idx][j] = ran.nextInt(i);
					if (inputs[idx][j] > max)
						max = inputs[idx][j];
				}

				// Copy row
				A = Arrays.copyOf(inputs[idx], inputs[idx].length);

				long bubble = time_bubbleSort(A);
				long selection = time_selectionSort(A);
				long insertion = time_insertionSort(A);
				long bucket = time_bucketSort(A, max);

				MergerSort m = new MergerSort();
				long merge = m.time_mergeSort(A);

				QuickSort q = new QuickSort();
				long quick = q.time_quickSort(A);
				//System.out.printf("%d %d %d %d %d %d\n", bubble, selection, insertion, bucket, merge, quick);
				
				// Update timing arrays
				bubble_arr[idx] = bubble;
				selection_arr[idx] = selection;
				insertion_arr[idx] = insertion;
				bucket_arr[idx] = bucket;
				merge_arr[idx] = merge;
				quick_arr[idx] = quick;
			}
			System.out.printf("%d \n", i);
		}
	}

	public static byte[] getByteArray(long[] arr) {
		// Set up a ByteBuffer called longBuffer
		ByteBuffer longBuffer = ByteBuffer.allocate(8 * arr.length); // 8 bytes in a long
		longBuffer.order(ByteOrder.LITTLE_ENDIAN); // Java's default is big-endian

		// Copy longs from longArray into longBuffer as bytes
		for (int i = 0; i < arr.length; i++) {
			longBuffer.putLong(arr[i]);
		}

		// Convert the ByteBuffer to a byte array and return it
		byte[] byteArray = longBuffer.array();
		return byteArray;
	}

	// Polymorphic function variant of the above method (special case to handle
	// matrix inputs)
	public static byte[] getByteArray(int[][] mat) {
		int iMax = mat.length;
		int jMax = mat[0].length;

		// Set up a ByteBuffer called intBuffer
		ByteBuffer intBuffer = ByteBuffer.allocate(4 * iMax * jMax); // 4 bytes in an int
		intBuffer.order(ByteOrder.LITTLE_ENDIAN); // Java's default is big-endian

		// Copy ints from intArray into intBuffer as bytes
		for (int i = 0; i < iMax; i++) {
			for (int j = 0; j < jMax; j++) {
				intBuffer.putInt(mat[i][j]);
			}
		}

		// Convert the ByteBuffer to a byte array and return it
		byte[] byteArray = intBuffer.array();
		return byteArray;
	}

	public static void store(byte[] arr, String fileName){
		String path = new String("./data/");
		path = path + fileName;

		File f = new File(path);
		
		try {
			if (f.createNewFile()) {
				Files.write(new File(path).toPath(), arr);
			} else {
				Files.write(new File(path).toPath(), arr);
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
   
   public static void main(String[] args)
   {
		// Set range
		int lo = 100;
		int hi = 1000;

		// Set random factor
		// This is the number of times we samples at a particular array size
		int random_factor = 25;

		// Setup arrays to store timing data
		int arr_size = (hi-lo)*random_factor;
		long[] bubble_arr = new long[arr_size];
		long[] selection_arr = new long[arr_size];
		long[] insertion_arr = new long[arr_size];
		long[] bucket_arr = new long[arr_size];
		long[] merge_arr = new long[arr_size];
		long[] quick_arr = new long[arr_size];

		// Setup matrix to store the arrays used as input 
		int[][] inputs = new int[arr_size][hi];

		// Get timing info
		get_timing(lo, hi, random_factor, inputs, bubble_arr, selection_arr, insertion_arr, bucket_arr, merge_arr, quick_arr);
		
		System.out.println(bubble_arr.length);
		// Setup byte arrays for storage (in binary format) and eventual use with Numpy
		byte[] byte_bubble = getByteArray(bubble_arr);
		byte[] byte_selection = getByteArray(selection_arr);
		byte[] byte_insertion = getByteArray(insertion_arr);
		byte[] byte_bucket = getByteArray(bucket_arr);
		byte[] byte_merge = getByteArray(merge_arr);
		byte[] byte_quick = getByteArray(quick_arr);
		byte[] byte_inputs = getByteArray(inputs);

		// Store byte arrays 
		store(byte_bubble, "data_bubble");
		store(byte_selection, "data_selection");
		store(byte_insertion, "data_insertion");
		store(byte_bucket, "data_bucket");
		store(byte_merge, "data_merge");
		store(byte_quick, "data_quick");
		store(byte_inputs, "data_inputs");
	}
	
}


