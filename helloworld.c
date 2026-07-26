#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "xil_printf.h"
#include "mnist_data.h"

// Network Dimensions
#define IMG_W 28
#define CONV1_F 8
#define CONV1_OUT 26 // 28 - 3 + 1
#define POOL1_OUT 13 // 26 / 2
#define DENSE_IN (POOL1_OUT * POOL1_OUT * CONV1_F) // 13 * 13 * 8 = 1352
#define NUM_CLASSES 10

// Static Memory Allocation
float conv1_weights[CONV1_F][3][3];
float dense_weights[NUM_CLASSES][DENSE_IN];
float dense_biases[NUM_CLASSES];

// Intermediate activations
float conv1_out[CONV1_F][CONV1_OUT][CONV1_OUT];
float pool1_out[CONV1_F][POOL1_OUT][POOL1_OUT];
float dense_out[NUM_CLASSES];
float softmax_out[NUM_CLASSES];

// Tracking indices for MaxPool backpropagation
int pool1_max_idx_i[CONV1_F][POOL1_OUT][POOL1_OUT];
int pool1_max_idx_j[CONV1_F][POOL1_OUT][POOL1_OUT];

// Arrays for gradients
float d_conv1_out[CONV1_F][CONV1_OUT][CONV1_OUT];
float d_pool1_out[CONV1_F][POOL1_OUT][POOL1_OUT];

//Initialization 
float rand_float() {
    return ((float)rand() / (float)(RAND_MAX)) - 0.5f;
}

float rand_normal(float mean, float stddev) {
   float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    
    // 6.283185307f is 2 * PI
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(6.283185307f * u2);
    
    return (z0 * stddev) + mean;
}

void init_network() {
    float conv1_stddev = sqrtf(1.0f / 9.0f); 
    float dense_stddev = sqrtf(1.0f / (float)DENSE_IN);
    //initializing the 8 3x3 filters
    for(int f=0; f<CONV1_F; f++) {
        for(int i=0; i<3; i++) {
            for(int j=0; j<3; j++) {
                conv1_weights[f][i][j] = rand_normal(0.0f, conv1_stddev); 
            }
        }
    }
    
    for(int c=0; c<NUM_CLASSES; c++) {
        dense_biases[c] = 0.0f;
        for(int i=0; i<DENSE_IN; i++) {
            dense_weights[c][i] = rand_normal(0.0f, dense_stddev);
        }
    }
}


void forward_conv1(const float* image) {
    for (int f = 0; f < CONV1_F; f++) {
        for (int i = 0; i < CONV1_OUT; i++) {
            for (int j = 0; j < CONV1_OUT; j++) {
                float sum = 0.0f;
                for (int ki = 0; ki < 3; ki++) {
                    for (int kj = 0; kj < 3; kj++) {
                        int img_idx = (i + ki) * IMG_W + (j + kj);
                        sum += image[img_idx] * conv1_weights[f][ki][kj];
                    }
                }
                conv1_out[f][i][j] = (sum > 0) ? sum : 0;
            }
        }
    }
}

void forward_maxpool1() {
    for (int f = 0; f < CONV1_F; f++) {
        for (int i = 0; i < POOL1_OUT; i++) {
            for (int j = 0; j < POOL1_OUT; j++) {
                float max_val = -9999.0f;
                int best_pi = 0, best_pj = 0;
                
                for (int pi = 0; pi < 2; pi++) {
                    for (int pj = 0; pj < 2; pj++) {
                        float val = conv1_out[f][i*2 + pi][j*2 + pj];
                        if (val > max_val) {
                            max_val = val;
                            best_pi = pi;
                            best_pj = pj;
                        }
                    }
                }
                pool1_out[f][i][j] = max_val;
                pool1_max_idx_i[f][i][j] = best_pi;
                pool1_max_idx_j[f][i][j] = best_pj;
            }
        }
    }
}

void forward_dense_and_softmax() {
    float max_logit = -9999.0f;
    float exp_sum = 0.0f;

    for (int c = 0; c < NUM_CLASSES; c++) {
        float sum = dense_biases[c];
        int flat_idx = 0;
        for (int f = 0; f < CONV1_F; f++) {
            for (int i = 0; i < POOL1_OUT; i++) {
                for (int j = 0; j < POOL1_OUT; j++) {
                    sum += pool1_out[f][i][j] * dense_weights[c][flat_idx];
                    flat_idx++;
                }
            }
        }
        dense_out[c] = sum;
        if (sum > max_logit) max_logit = sum;
    }

    for (int c = 0; c < NUM_CLASSES; c++) {
        softmax_out[c] = expf(dense_out[c] - max_logit);
        exp_sum += softmax_out[c];
    }
    for (int c = 0; c < NUM_CLASSES; c++) {
        softmax_out[c] /= exp_sum;
    }
}

void backward_pass(const float* image, int target, float lr) {
    //Calculate Softmax Error
    float d_dense_out[NUM_CLASSES];
    for(int c=0; c<NUM_CLASSES; c++) {
        //Error = predicted_probability - actual_outcome
        d_dense_out[c] = softmax_out[c] - (c == target ? 1.0f : 0.0f);
    }

    // Reset gradient arrays to zero
    memset(d_pool1_out, 0, sizeof(d_pool1_out));
    memset(d_conv1_out, 0, sizeof(d_conv1_out));

    //Backpropagate through Dense Layer 
    for(int c=0; c<NUM_CLASSES; c++) {
        dense_biases[c] -= lr * d_dense_out[c]; //Update bias
        
        int flat_idx = 0;
        for(int f=0; f<CONV1_F; f++){
            for(int i=0; i<POOL1_OUT; i++){
                for(int j=0; j<POOL1_OUT; j++){
                    float pool_val = pool1_out[f][i][j];
                    //Pass gradient down to pooling layer
                    d_pool1_out[f][i][j] += d_dense_out[c] * dense_weights[c][flat_idx];
                    //Update the weight
                    dense_weights[c][flat_idx] -= lr * d_dense_out[c] * pool_val;
                    flat_idx++;
                }
            }
        }
    }

    //Backpropagate through MaxPool
    for(int f=0; f<CONV1_F; f++){
        for(int i=0; i<POOL1_OUT; i++){
            for(int j=0; j<POOL1_OUT; j++){
                int pi = pool1_max_idx_i[f][i][j];
                int pj = pool1_max_idx_j[f][i][j];
                float original_val = conv1_out[f][i*2 + pi][j*2 + pj];
                
                //Only pass error back if val > 0
                if(original_val > 0.0f) {
                    d_conv1_out[f][i*2 + pi][j*2 + pj] += d_pool1_out[f][i][j];
                }
            }
        }
    }

    //Backpropagate through Conv1 (Update Conv Filters)
    for (int f = 0; f < CONV1_F; f++) {
        for (int i = 0; i < CONV1_OUT; i++) {
            for (int j = 0; j < CONV1_OUT; j++) {
                float d_out = d_conv1_out[f][i][j];
                if (d_out == 0.0f) continue;
                
                for (int ki = 0; ki < 3; ki++) {
                    for (int kj = 0; kj < 3; kj++) {
                        int img_idx = (i + ki) * IMG_W + (j + kj);
                        conv1_weights[f][ki][kj] -= lr * d_out * image[img_idx];
                    }
                }
            }
        }
    }
}


void run_interactive_inference() {
    uint8_t raw_image[IMG_W * IMG_W];
    float input_image[IMG_W * IMG_W];
    
    xil_printf("\r\n--- Entering Interactive Inference Mode ---\r\n");
    xil_printf("READY\n"); // Signal the laptop that the board is listening

    while(1) {
        // Block and wait to receive exactly 784 bytes from the laptop
        for(int i = 0; i < (IMG_W * IMG_W); i++) {
            // getchar() waits for UART data from the micro-USB cable
            raw_image[i] = getchar(); 
            // Normalize to 0.0 - 1.0
            input_image[i] = (float)raw_image[i] / 255.0f;
        }

        // Run the forward pass with the received image
        forward_conv1(input_image);
        forward_maxpool1();
        forward_dense_and_softmax();

        // Find the highest probability prediction
        int prediction = 0;
        float max_prob = softmax_out[0];
        for(int c = 1; c < NUM_CLASSES; c++){
            if(softmax_out[c] > max_prob){
                max_prob = softmax_out[c];
                prediction = c;
            }
        }
        xil_printf("%d\n", prediction);
    }
}

// --- Main Execution ---

int main() {
    xil_printf("\r\n--- Zynq Z-7010 Bare-Metal CNN Training ---\r\n");
    
    init_network();
    xil_printf("Network Weights Initialized.\r\n");
    xil_printf("Starting Training Loop over %d samples...\r\n", NUM_SAMPLES);

    float learning_rate = 0.005f;
    int correct_predictions = 0;
    float total_loss = 0.0f;
    for (int epoch = 1; epoch <= 3; epoch++) {
        xil_printf("=== Epoch %d ===\r\n", epoch);
        
        for (int step = 0; step < NUM_SAMPLES; step++) {
            // Forward Pass
            forward_conv1(mnist_images[step]);
            forward_maxpool1();
            forward_dense_and_softmax();

            int target = mnist_labels[step];
            
            int prediction = 0;
            float max_prob = softmax_out[0];
            for(int c = 1; c < NUM_CLASSES; c++){
                if(softmax_out[c] > max_prob){
                    max_prob = softmax_out[c];
                    prediction = c;
                }
            }

            if (prediction == target) correct_predictions++;
            total_loss += -logf(softmax_out[target] + 1e-7f);

            //Learning Step
            backward_pass(mnist_images[step], target, learning_rate);

            if ((step + 1) % 50 == 0) {
                int accuracy = (correct_predictions * 100) / 50;
                xil_printf("Step %d | Loss: %d.%03d | Accuracy: %d%%\r\n", 
                            step + 1, 
                            (int)(total_loss/50), 
                            (int)((total_loss/50 - (int)(total_loss/50)) * 1000), 
                            accuracy);
                correct_predictions = 0;
                total_loss = 0.0f;
            }
        }
    }

    xil_printf("Training Complete!\r\n");
    run_interactive_inference();
    
    return 0;
}