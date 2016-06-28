/* Project that takes signal and calculates averaged power spectrum using NVIDIA
   CUDA GPGPU programming. 

   Function main() tests empty data sets to check if the kernels 
   (GPU-based algorithms), allocations, and data transfers are working properly.

   The central power of this program is from the kernels below tied together nicely
   in the process_vector() function.

   periodically wake up, 
   scan the directory, 
   read in data from file, 
   and process...

   Juha Vierinen (x@mit.edu)
*/

// header file for the plasmaline project
#include "spectrometer.h"
#include <sys/stat.h>

#define N_BLOCKS 16       // each block covers a unique frequency bin
#define N_THREADS 256      // each thread goes through all spectra

// debug output files
//#define WRITE_INPUT 1
//#define DEBUG_Z_OUT 1
#define TIMING_PRINT 1
/*
  create Blackmann-Harris window function
 */
void blackmann_harris( float* pOut, unsigned int num )
{
  const float a0      = 0.35875f;
  const float a1      = 0.48829f;
  const float a2      = 0.14128f;
  const float a3      = 0.01168f;

  unsigned int idx    = 0;
  while( idx < num )
  {
    pOut[idx]   = a0 - (a1 * cosf( (2.0f * M_PI * idx) / (num - 1) )) + (a2 * cosf( (4.0f * M_PI * idx) / (num - 1) )) - (a3 * cosf( (6.0f * M_PI * idx) / (num - 1) ));
    idx++;
  }
}

/*
  Kernel for performing an averaged power spectrum
  *z a complex interleaved single precision float vector with n_spectra*spectrum_length elements
     each spectrum is concatenated into a long vector
     z = | spec_0 | spec_1 | ... | spec_n |
     *spectrum output spectrum
 */
__global__ void square_and_accumulate_sum(cufftComplex *z, float *spectrum, int n_spectra, int spectrum_length)
{
  // each block processes an independent spectrum bin to avoid multiple threads in multiple blocks 
  // writing to the same memory
  for(int spec_idx=blockIdx.x; spec_idx < n_spectra ; spec_idx++)
  {
    for(int freq_idx=threadIdx.x; freq_idx < spectrum_length ; freq_idx+=N_THREADS)
    {
      int idx=spec_idx*spectrum_length + freq_idx;
      spectrum[freq_idx] += z[idx].x*z[idx].x + z[idx].y*z[idx].y;
    }
  }
}

/*
  convert uint16_t data vector into single precision floating point
  also apply window function *w
 */
__global__ void short_to_float(uint16_t *ds, float *df, float *w, int n_spectra, int spectrum_length)
{
  for(int spec_idx=blockIdx.x; spec_idx < n_spectra ; spec_idx+=N_BLOCKS)
  {
    for(int freq_idx=threadIdx.x; freq_idx < spectrum_length ; freq_idx+=N_THREADS)
    {
      int idx=spec_idx*spectrum_length + freq_idx;
      df[idx] = w[freq_idx]*(((float)ds[idx])/65532.0 - 0.5);
    }
  }
}

int main(int argc, char **argv) 
{
    // some example parameters
    int spectrum_length = 8192*2;
    int n_spectra=2440/2; // for some reason a too large n_spectra causes numerical overflow
    int data_length = n_spectra*spectrum_length;
    struct stat st;
    uint16_t *r_in;
    FILE *in;

    if(argc > 1)
    {
      stat(argv[1],&st);
      int size=st.st_size;
      printf("%s %d\n",argv[1],size);
      if(size > spectrum_length*2)
      {
	printf("reading data %d\n",size/2);
	data_length=size/2;
	n_spectra=data_length/spectrum_length;
	r_in = (uint16_t *)malloc(data_length*sizeof(uint16_t));
	in=(FILE *)fopen(argv[1],"r");
	fread(r_in,sizeof(uint16_t),data_length,in);
	fclose(in);
      }
      else
	exit(0);
    }else{
      // real valued input signal as short ints
      r_in = (uint16_t *)malloc(data_length*sizeof(uint16_t));
    }

    // test signal
    for(int ti=0; ti<spectrum_length; ti++)
    {
      r_in[ti]=(uint16_t) (sinf(2.0*M_PI*10.0*(float)ti/((float)spectrum_length))*256.0 + (0.5*65532.0) );
    }

    for(int i=1; i<n_spectra; i++)
    {
	for(int ti=0; ti<spectrum_length; ti++) {
	  r_in[i*spectrum_length + ti]=r_in[ti];
	}
    }

#ifdef WRITE_INPUT 
    FILE *out;
    out=fopen("in.bin","w");
    fwrite(r_in,sizeof(uint16_t),10.0*spectrum_length,out);
    fclose(out);
    out=fopen("w.bin","w");
    fwrite(window,sizeof(float),spectrum_length,out);
    fclose(out);
#endif

    spectrometer_data *d;
    d=new_spectrometer_data(data_length,spectrum_length);
    process_vector(r_in, d);

    free_spectrometer_data(d);
    free(r_in);
    return 0;
}

extern "C" spectrometer_data *new_spectrometer_data(int data_length, int spectrum_length)
{
  spectrometer_data *d;
  int n_spectra;
  n_spectra=data_length/spectrum_length;

  d=(spectrometer_data *)malloc(sizeof(spectrometer_data));

  // result on the cpu
  d->spectrum = (float *)malloc( (spectrum_length/2+1) * sizeof(float));

  d->data_length=data_length;
  d->spectrum_length=spectrum_length;
  d->n_spectra=n_spectra;

  d->window = (float *)malloc(spectrum_length*sizeof(float));
  blackmann_harris(d->window,spectrum_length);

  // allocating device memory to the above pointers
  // reserve extra for +1 in place transforms
  if (cudaMalloc((void **) &d->d_in, sizeof(cufftComplex)*n_spectra*(spectrum_length/2 + 1)) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to allocate input data vector\n");
    exit(EXIT_FAILURE);
  }
  if (cudaMalloc((void **) &d->d_z_out, sizeof(cufftComplex)*n_spectra*(spectrum_length/2 + 1)) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to allocate input data vector\n");
    exit(EXIT_FAILURE);
  }

  if (cudaMalloc((void **) &d->d_window, sizeof(float)*spectrum_length) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to allocate input data vector\n");
    exit(EXIT_FAILURE);
  }

  // in-place seems to have a bug that causes the end to be garbled.
  //  d->d_z_out =(cufftComplex *) d->d_in;

  if (cudaMalloc((void **) &d->ds_in,sizeof(uint16_t)*data_length) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to allocate input data vector\n");
    exit(EXIT_FAILURE);
  }

  if (cudaMalloc((void **) &d->d_spectrum,sizeof(float)*(spectrum_length/2+1))
      != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Failed to allocate spectrum\n");
    exit(EXIT_FAILURE);
  }

  // initializing 1D FFT plan, this will tell cufft execution how to operate
  // cufft is well optimized and will run with different parameters than above
  //    cufftHandle plan;
  if (cufftPlan1d(&d->plan, spectrum_length, CUFFT_R2C, n_spectra) != CUFFT_SUCCESS) 
  {
    fprintf(stderr, "CUFFT error: Plan creation failed\n");
    exit(EXIT_FAILURE);
  }

  // copy window to device
  if (cudaMemcpy(d->d_window, d->window, sizeof(float)*spectrum_length, cudaMemcpyHostToDevice) != cudaSuccess)
  {
    fprintf(stderr, "Cuda error: Memory copy failed, window function HtD\n");
    exit(EXIT_FAILURE);
  }
  return(d);
}

extern "C" void free_spectrometer_data(spectrometer_data *d)
{
  if (cudaFree(d->d_in) != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to free input\n");
    exit(EXIT_FAILURE);
  }
  free(d->window);
  free(d->spectrum);
  if (cudaFree(d->d_window) != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to free window function\n");
    exit(EXIT_FAILURE);
  }
  if (cudaFree(d->d_z_out) != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to free input\n");
    exit(EXIT_FAILURE);
  }
  if (cudaFree(d->ds_in) != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to free input\n");
    exit(EXIT_FAILURE);
  }

  if (cudaFree(d->d_spectrum) != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to free spec\n");
    exit(EXIT_FAILURE);
  }
  if (cufftDestroy(d->plan) != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: Failed to destroy plan\n");
    exit(EXIT_FAILURE);
  }
  free(d);
}


/* This is the primary function in this program, meant to be embedded into other
**  programs. The transmit signal, tx, should be complex conjugated prior to use here.
**  The float types for tx and echo are useful when this function is embedded; the extern
**  "C" is also here for that purpose. Process_echoes sets up and runs the kernels on
**  the GPU, complex_mult, a 1D FFT, and a spectrum accumulation. Host spectrum is not
**  freed, so as to be taken and analyzed.
*/
//    process_vector((float *)z_in, data_length, spectrum, spectrum_length);


extern "C" void process_vector(uint16_t *d_in, spectrometer_data *d)
{
    int n_spectra, data_length, spectrum_length;
    FILE *out;

    n_spectra=d->n_spectra;
    data_length=d->data_length;
    spectrum_length=d->spectrum_length;

#ifdef DEBUG_Z_OUT
    // debug out
    cufftComplex *z_out = (cufftComplex *)malloc( n_spectra*(spectrum_length/2 + 1)*sizeof(cufftComplex));
#endif
    // ensure empty device spectrum
    if (cudaMemset(d->d_spectrum, 0, sizeof(float)*(spectrum_length/2 + 1)) != cudaSuccess)
    {
      fprintf(stderr, "Cuda error: Failed to zero device spectrum\n");
        exit(EXIT_FAILURE);
    }

#ifdef TIMING_PRINT
    // execution timing, done with CPU
    clock_t start, end;
    start=clock();
#endif
    
    // copy mem to device
    if (cudaMemcpy(d->ds_in, d_in, sizeof(uint16_t)*data_length, cudaMemcpyHostToDevice) != cudaSuccess)
    {
      fprintf(stderr, "Cuda error: Memory copy failed, tx HtD\n");
      exit(EXIT_FAILURE);
    }
    
    // convert datatype using GPU
    short_to_float<<< N_BLOCKS, N_THREADS >>>(d->ds_in, d->d_in, d->d_window, n_spectra, spectrum_length);

    // cufft kernel execution
    if (cufftExecR2C(d->plan, (float *)d->d_in, (cufftComplex *)d->d_z_out)
	!= CUFFT_SUCCESS)
    {
      fprintf(stderr, "CUFFT error: ExecC2C Forward failed\n");
      exit(EXIT_FAILURE);
    }

    // copying device resultant spectrum to host, now able to be manipulated
    // debug 
#ifdef DEBUG_Z_OUT
    cudaMemcpy(z_out, d->d_z_out, sizeof(cufftComplex)*n_spectra*(spectrum_length/2 + 1 ), cudaMemcpyDeviceToHost);
    out=fopen("z_out.bin","w");
    fwrite(z_out,sizeof(cufftComplex),n_spectra*(spectrum_length/2 + 1),out);
    fclose(out);
#endif

    // this needs to be faster:
    square_and_accumulate_sum<<< 1, N_THREADS >>>(d->d_z_out, d->d_spectrum, n_spectra, spectrum_length/2+1);
    if (cudaGetLastError() != cudaSuccess) {
       fprintf(stderr, "Cuda error: Kernel failure, square_and_accumulate_sum\n");
       exit(EXIT_FAILURE);
    }

    // copying device resultant spectrum to host, now able to be manipulated
    if (cudaMemcpy(d->spectrum, d->d_spectrum, sizeof(float) * spectrum_length/2,
        cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: Memory copy failed, spectrum DtH\n");
        exit(EXIT_FAILURE);
    }

    out=fopen("spec.bin","w");
    fwrite(d->spectrum,sizeof(float),spectrum_length/2,out);
    fclose(out);

    // execution timing and comparison to real-time data collection speed
#ifdef TIMING_PRINT
    end=clock();
    double dt = ((double) (end-start))/CLOCKS_PER_SEC;
    printf("\rTime elapsed %f s / %f data points %1.3f speed ratio", dt, (double)data_length/400e6,
           ((double)data_length/400e6)  / dt);
#endif

}
