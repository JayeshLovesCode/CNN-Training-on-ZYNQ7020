#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "xil_printf.h"
#include "mnist_data.h"

#include "xparameters.h"
#include "xaxidma.h"
#include "xil_cache.h"
#include "xil_io.h"

// DMA Instances
XAxiDma dma_0; // pixels out, result back
XAxiDma dma_1; // weights out

// --- Network Dimensions ---
#define IMG_W       28
#define CONV1_F     8
#define CONV1_OUT   26                                  // 28 - 3 + 1
#define POOL1_OUT   13                                  // 26 / 2
#define DENSE_IN    (POOL1_OUT * POOL1_OUT * CONV1_F)    // 1352
#define NUM_CLASSES 10

// --- Train / test split over the 500 samples in mnist_data.h ---
#define TOTAL_USED  500
#define TRAIN_N     400
#define TEST_N      (TOTAL_USED - TRAIN_N)

#define EPOCHS      30
#define LR          0.5f
#define CONV_W_MAG  4.0f    // random conv weights is drawn from this

// --- Weights and activations ---
uint8_t conv1_weights[CONV1_F][3][3]__attribute__((aligned(64)));
uint8_t dense_weights[NUM_CLASSES][DENSE_IN]__attribute__((aligned(64)));
float   dense_biases[NUM_CLASSES];

uint8_t conv1_out[CONV1_F][CONV1_OUT][CONV1_OUT]__attribute__((aligned(64)));
uint8_t pool1_out[CONV1_F][POOL1_OUT][POOL1_OUT]__attribute__((aligned(64)));

// --- Training state ---
uint8_t feat_cache[TOTAL_USED][DENSE_IN]__attribute__((aligned(64)));
float   fc_w[NUM_CLASSES][DENSE_IN];
float   fc_b[NUM_CLASSES];
float   e4m3_lut[256];
float   feat_norm   = 1.0f;
float   dense_scale = 1.0f;

//E4M3 helpers

uint8_t float_to_e4m3(float val) {
    if (val == 0.0f) return 0x00;

    union { float f; uint32_t i; } u;
    u.f = val;

    uint8_t sign = (u.i >> 31) & 0x01;
    int exp      = ((u.i >> 23) & 0xFF) - 127;
    int mantissa = (u.i >> 20) & 0x07;

    int e4m3_exp = exp + 7;

    if (e4m3_exp <= 0)  return sign << 7;                       
    if (e4m3_exp >= 15) return (sign << 7) | (14 << 3) | 7;      
    return (sign << 7) | (e4m3_exp << 3) | mantissa;
}

float e4m3_to_float(uint8_t v) {
    if ((v & 0x7F) == 0) return 0.0f;
    int sign = (v >> 7) & 0x01;
    int exp  = (v >> 3) & 0x0F;
    int mant = v & 0x07;
    float m = 1.0f + (float)mant / 8.0f;
    float result = ldexpf(m, exp - 7);
    return sign ? -result : result;
}

void build_lut(void) {
    for (int i = 0; i < 256; i++) e4m3_lut[i] = e4m3_to_float((uint8_t)i);
}

// Random conv filters, large enough that products clear E4M3's 2^-6 floor
void init_conv_random(void) {
    for (int f = 0; f < CONV1_F; f++)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                float r = (((float)rand() / (float)RAND_MAX) - 0.5f) * 2.0f * CONV_W_MAG;
                conv1_weights[f][i][j] = float_to_e4m3(r);
            }
}

int init_hardware_dmas(void) {
    int status;
    XAxiDma_Config *cfg_0, *cfg_1;

    cfg_0 = XAxiDma_LookupConfig(XPAR_XAXIDMA_0_BASEADDR);
    if (!cfg_0) return XST_FAILURE;
    status = XAxiDma_CfgInitialize(&dma_0, cfg_0);
    if (status != XST_SUCCESS) return XST_FAILURE;

    cfg_1 = XAxiDma_LookupConfig(XPAR_XAXIDMA_1_BASEADDR);
    if (!cfg_1) return XST_FAILURE;
    status = XAxiDma_CfgInitialize(&dma_1, cfg_1);
    if (status != XST_SUCCESS) return XST_FAILURE;

    XAxiDma_Reset(&dma_0);
    while (XAxiDma_ResetIsDone(&dma_0) != 1) {}
    XAxiDma_Reset(&dma_1);
    while (XAxiDma_ResetIsDone(&dma_1) != 1) {}

    XAxiDma_IntrDisable(&dma_0, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
    XAxiDma_IntrDisable(&dma_0, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);
    XAxiDma_IntrDisable(&dma_1, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
    XAxiDma_IntrDisable(&dma_1, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

    return XST_SUCCESS;
}

static void dump_dma(const char *name, UINTPTR base) {
    u32 mm2s_sr = Xil_In32(base + 0x04);
    u32 s2mm_sr = Xil_In32(base + 0x34);

    xil_printf("%s MM2S: SR=0x%08X halt=%d idle=%d slvErr=%d decErr=%d SA=0x%08X LEN=%d\r\n",
        name, mm2s_sr, (int)(mm2s_sr & 1), (int)((mm2s_sr >> 1) & 1),
        (int)((mm2s_sr >> 5) & 1), (int)((mm2s_sr >> 6) & 1),
        (unsigned)Xil_In32(base + 0x18), (int)Xil_In32(base + 0x28));

    xil_printf("%s S2MM: SR=0x%08X halt=%d idle=%d slvErr=%d decErr=%d DA=0x%08X LEN=%d\r\n",
        name, s2mm_sr, (int)(s2mm_sr & 1), (int)((s2mm_sr >> 1) & 1),
        (int)((s2mm_sr >> 5) & 1), (int)((s2mm_sr >> 6) & 1),
        (unsigned)Xil_In32(base + 0x48), (int)Xil_In32(base + 0x58));
}

uint8_t hardware_dot_product(uint8_t *pixels, uint8_t *weights, int length) {
    static uint8_t rx_buffer[32] __attribute__((aligned(64)));
    rx_buffer[0] = 0;

    Xil_DCacheFlushRange((UINTPTR)pixels, length);
    Xil_DCacheFlushRange((UINTPTR)weights, length);
    Xil_DCacheFlushRange((UINTPTR)rx_buffer, 32);

    Xil_Out32(XPAR_XFP8_DOT_PRODUCT_0_BASEADDR + 0x10, (u32)length);
    Xil_Out32(XPAR_XFP8_DOT_PRODUCT_0_BASEADDR + 0x00, 0x01);   // ap_start

    int s1 = XAxiDma_SimpleTransfer(&dma_0, (UINTPTR)rx_buffer, 32,    XAXIDMA_DEVICE_TO_DMA);
    int s2 = XAxiDma_SimpleTransfer(&dma_0, (UINTPTR)pixels,   length, XAXIDMA_DMA_TO_DEVICE);
    int s3 = XAxiDma_SimpleTransfer(&dma_1, (UINTPTR)weights,  length, XAXIDMA_DMA_TO_DEVICE);

    if (s1 || s2 || s3) {
        xil_printf("TRANSFER REJECTED: rx=%d tx0=%d tx1=%d (len=%d)\r\n", s1, s2, s3, length);
        return 0;
    }

    int timeout = 100000 + length * 10000;
    while ((XAxiDma_Busy(&dma_0, XAXIDMA_DMA_TO_DEVICE) ||
            XAxiDma_Busy(&dma_1, XAXIDMA_DMA_TO_DEVICE) ||
            XAxiDma_Busy(&dma_0, XAXIDMA_DEVICE_TO_DMA)) && timeout > 0) {
        timeout--;
    }

    if (timeout == 0) {
        xil_printf("\r\n--- TIMEOUT (len=%d) ---\r\n", length);
        dump_dma("DMA0", XPAR_XAXIDMA_0_BASEADDR);
        dump_dma("DMA1", XPAR_XAXIDMA_1_BASEADDR);
        return 0;
    }

    Xil_DCacheInvalidateRange((UINTPTR)rx_buffer, 32);
    return rx_buffer[0];
}

void forward_conv1(const uint8_t *image) {
    static uint8_t window[16]       __attribute__((aligned(64)));
    static uint8_t flat_weights[16] __attribute__((aligned(64)));

    for (int f = 0; f < CONV1_F; f++) {
        int idx = 0;
        for (int ki = 0; ki < 3; ki++)
            for (int kj = 0; kj < 3; kj++)
                flat_weights[idx++] = conv1_weights[f][ki][kj];
        for (int k = 9; k < 16; k++) flat_weights[k] = 0x00;

        for (int i = 0; i < CONV1_OUT; i++) {
            for (int j = 0; j < CONV1_OUT; j++) {
                idx = 0;
                for (int ki = 0; ki < 3; ki++)
                    for (int kj = 0; kj < 3; kj++)
                        window[idx++] = image[(i + ki) * IMG_W + (j + kj)];
                for (int k = 9; k < 16; k++) window[k] = 0x00;

                uint8_t r = hardware_dot_product(window, flat_weights, 16);
                conv1_out[f][i][j] = (r & 0x80) ? 0x00 : r;     // ReLU
            }
        }
    }
}

void forward_maxpool1(void) {
    for (int f = 0; f < CONV1_F; f++)
        for (int i = 0; i < POOL1_OUT; i++)
            for (int j = 0; j < POOL1_OUT; j++) {
                uint8_t best_b = conv1_out[f][i * 2][j * 2];
                float   best_v = e4m3_lut[best_b];
                for (int pi = 0; pi < 2; pi++)
                    for (int pj = 0; pj < 2; pj++) {
                        uint8_t b = conv1_out[f][i * 2 + pi][j * 2 + pj];
                        if (e4m3_lut[b] > best_v) { best_v = e4m3_lut[b]; best_b = b; }
                    }
                pool1_out[f][i][j] = best_b;
            }
}

void extract_all_features(void) {
    static uint8_t img[IMG_SIZE] __attribute__((aligned(64)));

    for (int n = 0; n < TOTAL_USED; n++) {
        for (int p = 0; p < IMG_SIZE; p++)
            img[p] = float_to_e4m3(mnist_images[n][p]);

        forward_conv1(img);
        forward_maxpool1();
        memcpy(feat_cache[n], pool1_out, DENSE_IN);

        if ((n + 1) % 25 == 0)
            xil_printf("  features %d / %d\r\n", n + 1, TOTAL_USED);
    }
}

void compute_feat_norm(void) {
    double acc = 0.0;
    for (int n = 0; n < TRAIN_N; n++) {
        double s = 0.0;
        for (int j = 0; j < DENSE_IN; j++) {
            float v = e4m3_lut[feat_cache[n][j]];
            s += (double)v * (double)v;
        }
        acc += s;
    }
    feat_norm = sqrtf((float)(acc / (double)TRAIN_N));
    if (feat_norm < 1e-6f) feat_norm = 1.0f;
    xil_printf("feat_norm = %d\r\n", (int)feat_norm);
}

void train_dense(void) {
    static float a[DENSE_IN];

    for (int c = 0; c < NUM_CLASSES; c++) {
        fc_b[c] = 0.0f;
        for (int j = 0; j < DENSE_IN; j++) fc_w[c][j] = 0.0f;
    }

    for (int ep = 0; ep < EPOCHS; ep++) {
        int correct = 0;

        for (int n = 0; n < TRAIN_N; n++) {
            for (int j = 0; j < DENSE_IN; j++)
                a[j] = e4m3_lut[feat_cache[n][j]] / feat_norm;

            // forward
            float logit[NUM_CLASSES], mx = -1e30f;
            for (int c = 0; c < NUM_CLASSES; c++) {
                float s = fc_b[c];
                const float *w = fc_w[c];
                for (int j = 0; j < DENSE_IN; j++)
                    if (a[j] != 0.0f) s += w[j] * a[j];
                logit[c] = s;
                if (s > mx) mx = s;
            }

            // softmax
            float sm[NUM_CLASSES], sum = 0.0f;
            for (int c = 0; c < NUM_CLASSES; c++) { sm[c] = expf(logit[c] - mx); sum += sm[c]; }
            for (int c = 0; c < NUM_CLASSES; c++) sm[c] /= sum;

            int y = mnist_labels[n];
            int pred = 0;
            for (int c = 1; c < NUM_CLASSES; c++) if (logit[c] > logit[pred]) pred = c;
            if (pred == y) correct++;

            for (int c = 0; c < NUM_CLASSES; c++) {
                float g = LR * (sm[c] - ((c == y) ? 1.0f : 0.0f));
                if (g == 0.0f) continue;
                float *w = fc_w[c];
                fc_b[c] -= g;
                for (int j = 0; j < DENSE_IN; j++)
                    if (a[j] != 0.0f) w[j] -= g * a[j];
            }
        }

        int acc = (correct * 1000) / TRAIN_N;
        xil_printf("  epoch %2d / %d   train accuracy %d.%d%%\r\n",
                   ep + 1, EPOCHS, acc / 10, acc % 10);
    }
}

// Choose a power-of-two scale, then encode the trained weights to E4M3.
// Two constraints: individual weights must stay inside E4M3's range, and the
// accumulated sum must stay under the 256 saturation guard in fixed_to_e4m3().
void quantize_dense(void) {
    float wmax = 0.0f;
    for (int c = 0; c < NUM_CLASSES; c++)
        for (int j = 0; j < DENSE_IN; j++) {
            float e = fc_w[c][j] / feat_norm;
            float m = (e < 0.0f) ? -e : e;
            if (m > wmax) wmax = m;
        }

    float lmax = 0.0f;
    for (int n = 0; n < TRAIN_N; n++)
        for (int c = 0; c < NUM_CLASSES; c++) {
            float s = 0.0f;
            for (int j = 0; j < DENSE_IN; j++) {
                uint8_t b = feat_cache[n][j];
                if (b) s += (fc_w[c][j] / feat_norm) * e4m3_lut[b];
            }
            float m = (s < 0.0f) ? -s : s;
            if (m > lmax) lmax = m;
        }

    float sa = 4.0f  / ((wmax > 1e-9f) ? wmax : 1e-9f);// per-weight range
    float sb = 64.0f / ((lmax > 1e-9f) ? lmax : 1e-9f);// accumulator range
    float s  = (sa < sb) ? sa : sb;

    dense_scale = 1.0f;
    while (dense_scale * 2.0f <= s) dense_scale *= 2.0f;
    while (dense_scale > s && dense_scale > 1.0f / 65536.0f) dense_scale *= 0.5f;

    int nz = 0;
    for (int c = 0; c < NUM_CLASSES; c++) {
        dense_biases[c] = fc_b[c];
        for (int j = 0; j < DENSE_IN; j++) {
            dense_weights[c][j] = float_to_e4m3((fc_w[c][j] / feat_norm) * dense_scale);
            if (dense_weights[c][j] & 0x7F) nz++;
        }
    }

    xil_printf("dense_scale = %d   nonzero dense weights %d / %d\r\n",
               (int)dense_scale, nz, NUM_CLASSES * DENSE_IN);
}

void evaluate_hw(void) {
    static uint8_t feat[DENSE_IN] __attribute__((aligned(64)));
    int correct = 0;
    int hist[NUM_CLASSES];
    for (int c = 0; c < NUM_CLASSES; c++) hist[c] = 0;

    for (int n = TRAIN_N; n < TOTAL_USED; n++) {
        memcpy(feat, feat_cache[n], DENSE_IN);

        int   pred = 0;
        float best = -1e30f;
        for (int c = 0; c < NUM_CLASSES; c++) {
            uint8_t r = hardware_dot_product(feat, dense_weights[c], DENSE_IN);
            float logit = e4m3_to_float(r) / dense_scale + dense_biases[c];
            if (logit > best) { best = logit; pred = c; }
        }

        hist[pred]++;
        if (pred == mnist_labels[n]) correct++;

        int done = n - TRAIN_N + 1;
        if (done <= 10)
            xil_printf("  sample %3d: predicted %d, actual %d  %s\r\n",
                       n, pred, mnist_labels[n],
                       (pred == mnist_labels[n]) ? "OK" : "x");
    }

    int pct = (correct * 1000) / TEST_N;
    xil_printf("\r\n========================================\r\n");
    xil_printf(" TEST ACCURACY: %d.%d%%   (%d / %d)\r\n",
               pct / 10, pct % 10, correct, TEST_N);
    xil_printf(" samples %d..%d were never trained on\r\n", TRAIN_N, TOTAL_USED - 1);
    xil_printf("========================================\r\n");

    xil_printf("Prediction distribution:\r\n");
    for (int c = 0; c < NUM_CLASSES; c++)
        xil_printf("  class %d: %d\r\n", c, hist[c]);
}

int main(void) {
    xil_printf("\r\n--- Zynq Z-7020 FP8 CNN: train + evaluate ---\r\n");

    build_lut();

    if (init_hardware_dmas() != XST_SUCCESS) {
        xil_printf("DMA Initialization Failed!\r\n");
        return -1;
    }
    xil_printf("DMAs Initialized.\r\n");

    static uint8_t sp[16] __attribute__((aligned(64)));
    static uint8_t sw[16] __attribute__((aligned(64)));
    for (int i = 0; i < 16; i++) { sp[i] = 0x38; sw[i] = 0x38; }
    uint8_t r = hardware_dot_product(sp, sw, 16);
    xil_printf("Sanity: 0x%02X (expect 0x58)\r\n", r);
    if (r != 0x58) {
        xil_printf("Sanity FAILED - stopping.\r\n");
        return -1;
    }

    srand(12345);
    init_conv_random();
    xil_printf("Conv filters: random, fixed (only the dense layer is trained).\r\n");

    xil_printf("\r\n[1/4] Extracting features on hardware for %d images...\r\n", TOTAL_USED);
    extract_all_features();

    xil_printf("\r\n[2/4] Computing feature normalisation...\r\n");
    compute_feat_norm();

    xil_printf("\r\n[3/4] Training dense layer on samples 0..%d...\r\n", TRAIN_N - 1);
    train_dense();

    quantize_dense();

    xil_printf("\r\n[4/4] Hardware inference on unseen samples %d..%d...\r\n",
               TRAIN_N, TOTAL_USED - 1);
    evaluate_hw();

    xil_printf("\r\nExecution Complete.\r\n");
    return 0;
}