/*  First Name: Aditya Subramanian
 *  Last Name: Muralidaran
 *  UBIT Name: adityasu
 */
#include<omp.h>


template <typename T, typename Op>
void omp_scan(int n, const T* in, T* out, Op op) 
{
	int p = 1;

	/* Calculating total number of processors */
	#pragma omp parallel
	{
		p = omp_get_num_threads();
	}
	
	/* k is the total number of groups to which
	   the input n is divided */
	int k = n/p;

	/* Calculating the prefix of each group in sequential 
	   manner and in parallel for all k groups */
	#pragma omp parallel for schedule(auto)
	for(int i=0; i<n; i=i+k)
	{	
		out[i] = in[i];
		for(int j = i+1; j<(i+k) && j<n; j++)
		{
			out[j] = op(in[j], out[j-1]);
		}
	}

	/* Calculating the prefix of last element in each group 
	   in sequential manner */ 
	for(int i = (2*k)-1; i<n; i=i+k)
	{
		out[i] = op(out[i], out[i-k]);
	}
	
	/* Broadcasting the last value of every group to elements 
	   of next group in parallel */
	#pragma omp parallel for schedule(auto)
	for(int i = k-1; i<n; i=i+k)
	{
		int j = i+1;
		while(j<n && j<(i+k))
		{
			out[j] = op(out[j], out[i]);
			j++;
		}
	}

} // omp_scan




