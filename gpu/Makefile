CC = nvcc
CFLAGS = 
SO_CFLAGS = --ptxas-options=-v --compiler-options '-fPIC' -shared
LDFLAGS = -lcufft

spectrometer: spectrometer.cu
	$(CC) $(CFLAGS) $(LDFLAGS) spectrometer.cu -o $@
libspectrometer.so: spetrometer.cu
	$(CC) $(SO_CFLAGS) $(LDFLAGS) spectrometer.cu -o $@
