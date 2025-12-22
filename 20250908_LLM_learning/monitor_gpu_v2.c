/*
 * High-resolution GPU power and temperature monitoring using NVML Field Values API.
 * C implementation - uses proper nvmlDeviceGetTemperatureV (non-deprecated).
 * Logs metrics at ~10ms resolution to CSV file.
 *
 * Key improvements over Python version:
 * - Uses nvmlDeviceGetTemperatureV() - recommended, non-deprecated API
 * - Uses nvmlDeviceGetFieldValues() with NVML_FI_DEV_POWER_INSTANT for instant power
 * - Also logs NVML_FI_DEV_POWER_AVERAGE for comparison (1-sec averaged on H100)
 * - Lower overhead for high-frequency polling (100+ Hz capable)
 * - No GIL contention
 *
 * Usage: ./monitor_gpu_v2 [interval_ms] [output_file]
 *   interval_ms: sampling interval in milliseconds (default: 10)
 *   output_file: CSV output file (default: auto-generated with timestamp)
 */

#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>

// Global flag for graceful shutdown
static volatile int running = 1;

void signal_handler(int sig) {
    (void)sig;
    printf("\nStopping monitoring...\n");
    running = 0;
}

// Get current timestamp in microseconds
long long get_timestamp_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000000LL + tv.tv_usec;
}

// Format timestamp as YYYY-MM-DD HH:MM:SS.mmm
void format_timestamp(char *buffer, size_t size) {
    struct timeval tv;
    struct tm *tm_info;
    
    gettimeofday(&tv, NULL);
    tm_info = localtime(&tv.tv_sec);
    
    strftime(buffer, size, "%Y-%m-%d %H:%M:%S", tm_info);
    sprintf(buffer + strlen(buffer), ".%03ld", tv.tv_usec / 1000);
}

typedef struct {
    double power_instant_w;
    double power_avg_w;
    int temp_c;
    int valid;
} GpuMetrics;

int get_gpu_metrics(nvmlDevice_t device, GpuMetrics *metrics) {
    nvmlReturn_t result;
    
    // Initialize field values for power queries
    nvmlFieldValue_t field_values[2] = {0};
    field_values[0].fieldId = NVML_FI_DEV_POWER_INSTANT;
    field_values[0].scopeId = 0;
    field_values[1].fieldId = NVML_FI_DEV_POWER_AVERAGE;
    field_values[1].scopeId = 0;
    
    // Query power fields
    result = nvmlDeviceGetFieldValues(device, 2, field_values);
    if (result != NVML_SUCCESS) {
        metrics->valid = 0;
        return -1;
    }
    
    // Extract power values
    metrics->valid = 1;
    if (field_values[0].nvmlReturn == NVML_SUCCESS) {
        metrics->power_instant_w = field_values[0].value.uiVal / 1000.0;
    } else {
        metrics->valid = 0;
    }
    
    if (field_values[1].nvmlReturn == NVML_SUCCESS) {
        metrics->power_avg_w = field_values[1].value.uiVal / 1000.0;
    } else {
        metrics->valid = 0;
    }
    
    // Get temperature using recommended API
    nvmlTemperature_v1_t temp_v1;
    temp_v1.version = nvmlTemperature_v1;
    temp_v1.sensorType = NVML_TEMPERATURE_GPU;
    result = nvmlDeviceGetTemperatureV(device, (nvmlTemperature_t*)&temp_v1);
    if (result == NVML_SUCCESS) {
        metrics->temp_c = temp_v1.temperature;
    } else {
        metrics->valid = 0;
    }
    
    return metrics->valid ? 0 : -1;
}

int main(int argc, char *argv[]) {
    int interval_ms = 10;  // default 10ms
    char output_file[256] = {0};
    
    // Parse command line arguments
    if (argc > 1) {
        interval_ms = atoi(argv[1]);
        if (interval_ms <= 0) {
            fprintf(stderr, "Invalid interval: %s\n", argv[1]);
            return 1;
        }
    }
    
    if (argc > 2) {
        strncpy(output_file, argv[2], sizeof(output_file) - 1);
    } else {
        // Auto-generate filename with timestamp
        time_t now = time(NULL);
        struct tm *tm_info = localtime(&now);
        strftime(output_file, sizeof(output_file), "gpu_metrics_v2_%Y%m%d_%H%M%S.csv", tm_info);
    }
    
    double interval_sec = interval_ms / 1000.0;
    
    printf("Starting GPU monitoring (v2 - Field Values API, C implementation):\n");
    printf("  Sampling interval: %dms (%.1f Hz)\n", interval_ms, 1000.0 / interval_ms);
    printf("  Output file: %s\n", output_file);
    printf("  Power: INSTANT + AVERAGE (1-sec) for comparison\n");
    printf("  Temperature: nvmlDeviceGetTemperatureV (non-deprecated)\n");
    printf("  Press Ctrl+C to stop\n\n");
    
    // Setup signal handler
    signal(SIGINT, signal_handler);

    // Initialize NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Failed to initialize NVML: %s\n", nvmlErrorString(result));
        return 1;
    }

    // Get GPU handles
    unsigned int num_gpus;
    result = nvmlDeviceGetCount(&num_gpus);
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Failed to get device count: %s\n", nvmlErrorString(result));
        nvmlShutdown();
        return 1;
    }

    printf("Detected %u GPU(s)\n", num_gpus);

    nvmlDevice_t *devices = malloc(num_gpus * sizeof(nvmlDevice_t));
    if (!devices) {
        fprintf(stderr, "Failed to allocate memory for device handles\n");
        nvmlShutdown();
        return 1;
    }

    // Get device handles and print names
    for (unsigned int i = 0; i < num_gpus; i++) {
        result = nvmlDeviceGetHandleByIndex(i, &devices[i]);
        if (result != NVML_SUCCESS) {
            fprintf(stderr, "Failed to get handle for GPU %u: %s\n", i, nvmlErrorString(result));
            free(devices);
            nvmlShutdown();
            return 1;
        }

        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        result = nvmlDeviceGetName(devices[i], name, NVML_DEVICE_NAME_BUFFER_SIZE);
        if (result == NVML_SUCCESS) {
            printf("  GPU %u: %s\n", i, name);
        }
    }
    printf("\n");

    // Open CSV file
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open output file: %s\n", output_file);
        free(devices);
        nvmlShutdown();
        return 1;
    }

    // Write CSV header
    fprintf(fp, "timestamp,elapsed_sec");
    for (unsigned int i = 0; i < num_gpus; i++) {
        fprintf(fp, ",gpu%u_power_instant_w,gpu%u_power_avg_w,gpu%u_temp_c", i, i, i);
    }
    fprintf(fp, "\n");
    fflush(fp);

    // Monitoring loop
    long long start_time_us = get_timestamp_us();
    unsigned long sample_count = 0;

    while (running) {
        long long loop_start_us = get_timestamp_us();

        char timestamp[32];
        format_timestamp(timestamp, sizeof(timestamp));
        double elapsed_sec = (loop_start_us - start_time_us) / 1000000.0;

        fprintf(fp, "%s,%.3f", timestamp, elapsed_sec);

        // Query each GPU
        for (unsigned int i = 0; i < num_gpus; i++) {
            GpuMetrics metrics;
            if (get_gpu_metrics(devices[i], &metrics) == 0) {
                fprintf(fp, ",%.2f,%.2f,%d",
                        metrics.power_instant_w,
                        metrics.power_avg_w,
                        metrics.temp_c);
            } else {
                fprintf(fp, ",N/A,N/A,N/A");
            }
        }

        fprintf(fp, "\n");
        fflush(fp);

        sample_count++;

        // Print status every 100 samples
        if (sample_count % 100 == 0) {
            double actual_rate = sample_count / elapsed_sec;
            printf("Samples: %lu, Actual rate: %.1f Hz, Elapsed: %.1fs\n",
                   sample_count, actual_rate, elapsed_sec);
        }

        // Sleep for remaining time to maintain interval
        long long loop_end_us = get_timestamp_us();
        long long loop_duration_us = loop_end_us - loop_start_us;
        long long sleep_us = (long long)(interval_sec * 1000000.0) - loop_duration_us;

        if (sleep_us > 0) {
            usleep(sleep_us);
        }
    }

    // Cleanup
    fclose(fp);

    // Final statistics
    long long end_time_us = get_timestamp_us();
    double total_time = (end_time_us - start_time_us) / 1000000.0;
    double actual_rate = sample_count / total_time;

    printf("\nMonitoring stopped:\n");
    printf("  Total samples: %lu\n", sample_count);
    printf("  Total time: %.2fs\n", total_time);
    printf("  Actual sampling rate: %.2f Hz\n", actual_rate);
    printf("  Data saved to: %s\n", output_file);

    free(devices);
    nvmlShutdown();

    return 0;
}

