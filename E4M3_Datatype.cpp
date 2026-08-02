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

    // Mantissa: Add the 1 to the front 
    ap_uint<4> m_a_full = (1 << 3) | mant_a;
    ap_uint<4> m_b_full = (1 << 3) | mant_b;

    ap_uint<8> m_mult = m_a_full * m_b_full;

    // NORMALIZATION
    ap_uint<3> mant_res;
    
    // If the top bit [7] is 1, the result is >= 2.0 
    if (m_mult[7] == 1) { 
        exp_res += 1;                  // Shift decimal point right
        mant_res = m_mult.range(6, 4); // Grab the top 3 fractional bits
    } else {                           // Result is < 2.0
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
        result.range(2, 0) = 7; // Overflow to Max Value 
    } else {
        result.range(6, 3) = exp_res(3, 0);
        result.range(2, 0) = mant_res;
    }

    return result;
}

fp8_e4m3 add_fp8(fp8_e4m3 a, fp8_e4m3 b) {
    if (a == 0) return b;
    if (b == 0) return a;

    ap_uint<1> sign_a = a[7];
    ap_uint<4> exp_a  = a.range(6, 3);
    ap_uint<4> mant_a = (1 << 3) | a.range(2, 0); // Add implicit leading 1

    ap_uint<1> sign_b = b[7];
    ap_uint<4> exp_b  = b.range(6, 3);
    ap_uint<4> mant_b = (1 << 3) | b.range(2, 0);

    ap_uint<4> final_exp = exp_a;
    ap_uint<1> final_sign = sign_a;
    ap_int<6>  aligned_a = mant_a; 
    ap_int<6>  aligned_b = mant_b;

    if (exp_a > exp_b) {
        ap_uint<4> diff = exp_a - exp_b;
        aligned_b = mant_b >> diff; 
    } else if (exp_b > exp_a) {
        ap_uint<4> diff = exp_b - exp_a;
        aligned_a = mant_a >> diff;
        final_exp = exp_b;
        final_sign = sign_b;
    } else {
        // If exponents match, the sign depends on the larger mantissa
        if (mant_b > mant_a) final_sign = sign_b;
    }

    // Apply signs to the aligned mantissas
    if (sign_a == 1) aligned_a = -aligned_a;
    if (sign_b == 1) aligned_b = -aligned_b;

    ap_int<7> sum = aligned_a + aligned_b;

    if (sum == 0) return 0;

    // Check if result is negative and get absolute value 
    if (sum < 0) {
        final_sign = 1;
        sum = -sum;
    } else {
        final_sign = 0;
    }

    ap_uint<6> abs_sum = sum;
    
    if (abs_sum & (1 << 4)) { // Sum overflowed 
        abs_sum >>= 1;
        final_exp += 1;
    } else {
        // Shift left until the 3rd bit is a 1 
        while ((abs_sum & (1 << 3)) == 0 && final_exp > 0) {
            abs_sum <<= 1;
            final_exp -= 1;
        }
    }

    fp8_e4m3 result;
    result[7] = final_sign;
    
    // Check underflow/overflow
    if (final_exp <= 0) return 0;
    if (final_exp >= 15) {
        result.range(6, 3) = 15;
        result.range(2, 0) = 7;
        return result;
    }

    result.range(6, 3) = final_exp;
    result.range(2, 0) = abs_sum.range(2, 0); 
    
    return result;
}

fp8_e4m3 mac_fp8(fp8_e4m3 pixel, fp8_e4m3 weight, fp8_e4m3 current_sum) {
    fp8_e4m3 mult_result = mult_fp8(pixel, weight);
    fp8_e4m3 new_sum = add_fp8(current_sum, mult_result);
    return new_sum;
}
