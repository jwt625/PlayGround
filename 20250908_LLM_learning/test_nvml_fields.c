/*
 * NVML Field Values API Test - C Implementation
 * Tests legacy and new NVML APIs to verify availability and correctness.
 * Equivalent to test_nvml_fields.py
 */

#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>

void test_legacy_api(nvmlDevice_t device, int gpu_id) {
    printf("\n=== GPU %d: Testing Legacy API ===\n", gpu_id);
    
    // Test legacy power API
    unsigned int power_mw;
    nvmlReturn_t result = nvmlDeviceGetPowerUsage(device, &power_mw);
    if (result == NVML_SUCCESS) {
        printf("  Power (legacy): %.2f W\n", power_mw / 1000.0);
    } else {
        printf("  Power (legacy) FAILED: %s\n", nvmlErrorString(result));
    }
    
    // Test legacy temperature API (deprecated)
    unsigned int temp;
    result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
    if (result == NVML_SUCCESS) {
        printf("  Temperature (legacy, deprecated): %u C\n", temp);
    } else {
        printf("  Temperature (legacy) FAILED: %s\n", nvmlErrorString(result));
    }
}

void test_temperature_api(nvmlDevice_t device, int gpu_id) {
    printf("\n=== GPU %d: Testing Temperature APIs ===\n", gpu_id);
    
    // Test deprecated nvmlDeviceGetTemperature
    unsigned int temp;
    nvmlReturn_t result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
    if (result == NVML_SUCCESS) {
        printf("  nvmlDeviceGetTemperature (deprecated): %u C\n", temp);
    } else {
        printf("  nvmlDeviceGetTemperature FAILED: %s\n", nvmlErrorString(result));
    }
    
    // Test new nvmlDeviceGetTemperatureV (recommended API)
    printf("\n  Testing nvmlDeviceGetTemperatureV (recommended):\n");
    nvmlTemperature_v1_t temp_v1;
    temp_v1.version = nvmlTemperature_v1;
    temp_v1.sensorType = NVML_TEMPERATURE_GPU;
    result = nvmlDeviceGetTemperatureV(device, (nvmlTemperature_t*)&temp_v1);
    if (result == NVML_SUCCESS) {
        printf("    GPU Temperature: %d C\n", temp_v1.temperature);
    } else {
        printf("    nvmlDeviceGetTemperatureV FAILED: %s\n", nvmlErrorString(result));
    }
    
    // Test memory temperature via Field Values API
    printf("\n  Trying temperature via Field Values API:\n");
    nvmlFieldValue_t field_values[1] = {0};
    field_values[0].fieldId = NVML_FI_DEV_MEMORY_TEMP;
    field_values[0].scopeId = 0;

    result = nvmlDeviceGetFieldValues(device, 1, field_values);
    if (result == NVML_SUCCESS) {
        if (field_values[0].nvmlReturn == NVML_SUCCESS) {
            printf("    NVML_FI_DEV_MEMORY_TEMP = %d\n", NVML_FI_DEV_MEMORY_TEMP);
            printf("    Memory Temperature: %u C\n", field_values[0].value.uiVal);
        } else {
            printf("    Memory temp query FAILED: %s\n", 
                   nvmlErrorString(field_values[0].nvmlReturn));
        }
    } else {
        printf("    Field Values API FAILED: %s\n", nvmlErrorString(result));
    }
}

void test_field_values_api(nvmlDevice_t device, int gpu_id) {
    printf("\n=== GPU %d: Testing Field Values API ===\n", gpu_id);

    // Query power fields
    nvmlFieldValue_t field_values[2] = {0};
    field_values[0].fieldId = NVML_FI_DEV_POWER_INSTANT;
    field_values[0].scopeId = 0;
    field_values[1].fieldId = NVML_FI_DEV_POWER_AVERAGE;
    field_values[1].scopeId = 0;
    
    printf("  Querying field IDs:\n");
    printf("    NVML_FI_DEV_POWER_INSTANT = %d\n", NVML_FI_DEV_POWER_INSTANT);
    printf("    NVML_FI_DEV_POWER_AVERAGE = %d\n", NVML_FI_DEV_POWER_AVERAGE);
    
    nvmlReturn_t result = nvmlDeviceGetFieldValues(device, 2, field_values);
    if (result == NVML_SUCCESS) {
        printf("  nvmlDeviceGetFieldValues: SUCCESS\n");
        printf("  Number of results: 2\n");
        
        for (int i = 0; i < 2; i++) {
            const char* field_name = (field_values[i].fieldId == NVML_FI_DEV_POWER_INSTANT) 
                                     ? "INSTANT" : "AVERAGE";
            printf("\n  Result %d - Power %s:\n", i, field_name);
            printf("    fieldId: %d\n", field_values[i].fieldId);
            printf("    nvmlReturn: %d (%s)\n", field_values[i].nvmlReturn,
                   field_values[i].nvmlReturn == NVML_SUCCESS ? "SUCCESS" : "FAILED");
            printf("    timestamp: %lld\n", (long long)field_values[i].timestamp);
            printf("    latencyUsec: %lld\n", (long long)field_values[i].latencyUsec);
            printf("    valueType: %d\n", field_values[i].valueType);
            
            if (field_values[i].nvmlReturn == NVML_SUCCESS) {
                unsigned int power_mw = field_values[i].value.uiVal;
                printf("    Power: %u mW = %.2f W\n", power_mw, power_mw / 1000.0);
            }
        }
    } else {
        printf("  Field Values API test FAILED: %s\n", nvmlErrorString(result));
    }
    
    // Test alternative: total energy consumption
    printf("\n  Alternative power APIs:\n");
    unsigned long long total_energy;
    result = nvmlDeviceGetTotalEnergyConsumption(device, &total_energy);
    if (result == NVML_SUCCESS) {
        printf("    nvmlDeviceGetTotalEnergyConsumption: %llu mJ\n", total_energy);
    } else {
        printf("    nvmlDeviceGetTotalEnergyConsumption: ERROR - %s\n", 
               nvmlErrorString(result));
    }
}

int main(void) {
    printf("Initializing NVML...\n");
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        return 1;
    }
    printf("NVML initialized successfully\n");
    
    unsigned int device_count;
    result = nvmlDeviceGetCount(&device_count);
    if (result != NVML_SUCCESS) {
        printf("Failed to get device count: %s\n", nvmlErrorString(result));
        nvmlShutdown();
        return 1;
    }
    printf("\nDetected %u GPU(s)\n", device_count);

    for (unsigned int i = 0; i < device_count; i++) {
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (result != NVML_SUCCESS) {
            printf("Failed to get handle for GPU %u: %s\n", i, nvmlErrorString(result));
            continue;
        }

        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        if (result == NVML_SUCCESS) {
            printf("\nGPU %u: %s\n", i, name);
        } else {
            printf("\nGPU %u: (name unavailable)\n", i);
        }

        // Test all APIs
        test_legacy_api(device, i);
        test_temperature_api(device, i);
        test_field_values_api(device, i);
    }

    nvmlShutdown();
    printf("\n\nNVML shutdown successfully\n");

    return 0;
}

