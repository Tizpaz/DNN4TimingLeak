package sorting_algo;

// QuickSort with iterations instead of recursive calls
class QuickSort 
{ 
	/* This function takes last element as pivot, 
	places the pivot element at its correct 
	position in sorted array, and places all 
	smaller (smaller than pivot) to left of 
	pivot and all greater elements to right 
	of pivot */
	public static int partition(int arr[], int low, int high) 
	{ 
		int pivot = arr[high]; 
		int i = (low-1); // index of smaller element 
		for (int j=low; j<=high-1; j++) 
		{ 
			// If current element is smaller than or 
			// equal to pivot 
			if (arr[j] <= pivot) 
			{ 
				i++; 

				// swap arr[i] and arr[j] 
				int temp = arr[i]; 
				arr[i] = arr[j]; 
				arr[j] = temp; 
			} 
		} 

		// swap arr[i+1] and arr[high] (or pivot) 
		int temp = arr[i+1]; 
		arr[i+1] = arr[high]; 
		arr[high] = temp; 

		return i+1; 
	} 

	/* The main function that implements QuickSort() 
	arr[] --> Array to be sorted, 
	low --> Starting index, 
	high --> Ending index */
	public static void qSort(int arr[], int l, int h) 
	{ 
		// Create an auxiliary stack 
		int[] stack = new int[h-l+1]; 

		// initialize top of stack 
		int top = -1; 

		// push initial values of l and h to stack 
		stack[++top] = l; 
		stack[++top] = h; 

		// Keep popping from stack while is not empty 
		while (top >= 0) 
		{ 
			// Pop h and l 
			h = stack[top--]; 
			l = stack[top--]; 

			// Set pivot element at its correct position 
			// in sorted array 
			int p = partition(arr, l, h); 

			// If there are elements on left side of pivot, 
			// then push left side to stack 
			if (p-1 > l) 
			{ 
				stack[++top] = l; 
				stack[++top] = p - 1; 
			} 

			// If there are elements on right side of pivot, 
			// then push right side to stack 
			if (p+1 < h) 
			{ 
				stack[++top] = p + 1; 
				stack[++top] = h; 
			} 
		} 
	} 

	public static void quickSort(int arr[])
	{
		qSort(arr, 0, arr.length - 1);
	}

	public static long time_quickSort(int[] A)
    {
        long start = System.nanoTime();
		quickSort(A);
		long end = System.nanoTime();
		long duration = end - start;
		return duration;
    }
}