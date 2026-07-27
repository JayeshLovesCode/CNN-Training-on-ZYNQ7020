#include <ap_int.h>

typedef ap_uint<8> fp8_e4m3;

fp8_e4m3 mult_fp8(fp8_e4m3 a, fp8_e4m3 b) {
    ap_uint<1> sign_a = a[7];
    ap_uint<4> exp_a  = a.range(6, 3);
    ap_uint<3> mant_a = a.range(2, 0);

    ap_uint<1> sign_b = b[7];
    ap_uint<4> exp_b  = b.range(6, 3);
    ap_uint<3> mant_b = b.range(2, 0);

    //Zero case
    if ((exp_a == 0 && mant_a == 0) || (exp_b == 0 && mant_b == 0)) {
        return 0; 
    }

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
        result.range(2, 0) = 7; // Overflow to Max Value (NaN/Inf handling)
    } else {
        result.range(6, 3) = exp_res(3, 0);
        result.range(2, 0) = mant_res;
    }

    return result;
}