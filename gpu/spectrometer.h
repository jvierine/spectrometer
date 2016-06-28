#ifndef PLASMA_LINE
#define PLASMA_LINE

// system includes
#include <stdio.h>
#include <time.h>

// CUDA includes
#include <cuComplex.h>
#include <cufft.h>
#include <stdint.h>

typedef struct spectrometer_data_str {
  float *d_in;
  uint16_t *ds_in;
  cufftComplex *d_z_out;
  float *d_spectrum;
  float *spectrum;
  float *d_window;
  float *window;
  cufftHandle plan;
  int spectrum_length;
  int data_length;
  int n_spectra;
} spectrometer_data;

extern "C" void process_vector(uint16_t *d_in, spectrometer_data *d);
extern "C" spectrometer_data *new_spectrometer_data(int data_length, int spectrum_length);
extern "C" void free_spectrometer_data(spectrometer_data *d);
__global__ void short_to_float(uint8_t *ds, float *df, float *w, int n_spectra, int spectrum_length);

__global__ void square_and_accumulate_sum(cufftComplex *z, float *spectrum);
extern "C" void blackmann_harris( float* pOut, unsigned int num );

#endif


