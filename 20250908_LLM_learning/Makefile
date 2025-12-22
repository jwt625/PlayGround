# Makefile for NVML programs

CC = gcc
CFLAGS = -Wall -Wextra -O2
NVML_INCLUDE = /usr/local/cuda/include
NVML_LIB = /usr/local/cuda/lib64
LDFLAGS = -L$(NVML_LIB) -lnvidia-ml

# Targets
all: test_nvml_fields monitor_gpu_v2

test_nvml_fields: test_nvml_fields.c
	$(CC) $(CFLAGS) -I$(NVML_INCLUDE) -o $@ $< $(LDFLAGS)

monitor_gpu_v2: monitor_gpu_v2.c
	$(CC) $(CFLAGS) -I$(NVML_INCLUDE) -o $@ $< $(LDFLAGS)

clean:
	rm -f test_nvml_fields monitor_gpu_v2

.PHONY: all clean

