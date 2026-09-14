#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>

typedef ap_uint<8> fp8_e4m3;
typedef ap_fixed<24,12,AP_TRN,AP_SAT> acc_t;
typedef ap_axiu<8, 0, 0, 0> axis_t;

fp8_e4m3 mult_fp8(fp8_e4m3 a, fp8_e4m3 b) {
    #pragma HLS INLINE
    ap_uint<1> sign_a = a[7];
    ap_uint<4> exp_a  = a.range(6, 3);
    ap_uint<3> mant_a = a.range(2, 0);

    ap_uint<1> sign_b = b[7];
    ap_uint<4> exp_b  = b.range(6, 3);
    ap_uint<3> mant_b = b.range(2, 0);

    // Make subnormals and zero to 0
    if (exp_a == 0 || exp_b == 0) return 0;

    // Sign: XOR the sign bits (Negative * Negative = Positive)
    ap_uint<1> sign_res = sign_a ^ sign_b;

    // Exponent: Add exponents and subtract 7
    ap_int<6> exp_res = exp_a + exp_b - 7;

    // Mantissa: Add the 1 to the front so of the form 1.xxx
    ap_uint<4> m_a_full = (1 << 3) | mant_a;
    ap_uint<4> m_b_full = (1 << 3) | mant_b;

    ap_uint<8> m_mult = m_a_full * m_b_full;

    // NORMALIZATION
    ap_uint<3> mant_res;
    
    // If the top bit [7] is 1, the result is >= 2.0 
    if (m_mult[7] == 1) { 
        exp_res += 1;                  // Shift decimal point right
        mant_res = m_mult.range(6, 4); // Grab the top 3 fractional bits
    } else {                           // Result is < 2.0 (e.g., 01.xxxxxx)
        mant_res = m_mult.range(5, 3); 
    }

    fp8_e4m3 result;
    result[7] = sign_res;
    
    // Check for overflow/underflow 
    if (exp_res <= 0) {
        result.range(6, 3) = 0;
        result.range(2, 0) = 0; // Underflow to 0
    } else if (exp_res >= 15) {
        result.range(6, 3) = 15;
        result.range(2, 0) = 6; // Overflow to Max Value 
    } else {
        result.range(6, 3) = exp_res(3, 0);
        result.range(2, 0) = mant_res;
    }

    return result;
}

acc_t e4m3_to_fixed(fp8_e4m3 v) {
    #pragma HLS INLINE
    ap_uint<1> sign = v[7];
    ap_uint<4> exp  = v.range(6,3);
    ap_uint<3> mant = v.range(2,0);

    if (exp == 0) return (acc_t)0;

    // Value = 1.mant x 2^(exp-7);  1.mant as an integer is (8+mant)/8
    ap_uint<4> m = (ap_uint<4>)(8 + mant);
    int shift = (int)exp - 7 - 3;      // -3 undoes the /8

    acc_t val = (acc_t)m;
    if (shift > 0)      val = val << shift;
    else if (shift < 0) val = val >> (-shift);

    return sign ? (acc_t)(-val) : val;
}

fp8_e4m3 fixed_to_e4m3(acc_t v) {
    #pragma HLS INLINE
    bool neg = (v < 0);
    acc_t a = neg ? (acc_t)(-v) : v;

    if (a == 0) return 0;
    // Above 2^8 is outside E4M3's exponent range - clamp to max (matches float_to_e4m3)
    if (a >= (acc_t)256) {
        fp8_e4m3 sat;
        sat[7] = neg ? 1 : 0;
        sat.range(6,3) = 14;
        sat.range(2,0) = 7;
        return sat;
    }
    // Find the exponent: largest e with 2^e <= a, searched over E4M3's range
    int e = -7;
    for (int k = 7; k >= -6; k--) {
        #pragma HLS UNROLL
        acc_t p = (k >= 0) ? (acc_t)(1 << k) : (acc_t)(acc_t(1) >> (-k));
        if (a >= p) { e = k; break; }
    }

    // Normalize to 1.mant, extract 3 mantissa bits
    acc_t norm = (e >= 0) ? (acc_t)(a >> e) : (acc_t)(a << (-e));   // in [1,2)
    acc_t frac = norm - (acc_t)1;
    ap_uint<3> mant = (ap_uint<3>)(int)(frac * 8);

    int biased = e + 7;
    fp8_e4m3 out;
    out[7] = neg ? 1 : 0;
    if (biased <= 0)       { out.range(6,0) = 0; }
    else if (biased >= 15) { out.range(6,3) = 14; out.range(2,0) = 7; }
    else                   { out.range(6,3) = biased; out.range(2,0) = mant; }
    return out;
}

void fp8_dot_product(hls::stream<axis_t>& pixel_stream,
                     hls::stream<axis_t>& weight_stream,
                     hls::stream<axis_t>& output_stream,
                     int array_length) {

    #pragma HLS INTERFACE axis port=pixel_stream
    #pragma HLS INTERFACE axis port=weight_stream
    #pragma HLS INTERFACE axis port=output_stream
    #pragma HLS INTERFACE s_axilite port=array_length bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    acc_t running_sum = 0;

    for (int i = 0; i < array_length; i++) {
        #pragma HLS PIPELINE
        axis_t p_pkt = pixel_stream.read();
        axis_t w_pkt = weight_stream.read();

        // Multiply in E4M3 (cheap, and matches your existing unit)
        fp8_e4m3 prod = mult_fp8(p_pkt.data, w_pkt.data);

        // Accumulate at full precision
        running_sum += e4m3_to_fixed(prod);
    }

    axis_t final_packet;
    final_packet.data = fixed_to_e4m3(running_sum);
    final_packet.last = 1;
    final_packet.keep = 1;
    final_packet.strb = 1;

    output_stream.write(final_packet);
}
