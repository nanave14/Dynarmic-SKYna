/* This file is part of the dynarmic project.
 * Copyright (c) 2021 MerryMage
 * SPDX-License-Identifier: 0BSD
 */
//! Rust implementations of the x86 reciprocal-square-root benchmark routines.
//! SSE operations keep each floating-point step separate, without FMA contraction.
//! Uses the caller's MXCSR rounding/denormal mode, like the original routines.
//! The existing Dynarmic test oracle supplies `_rsqrt_fallback` for denormals.
//! Register-allocation and branch-layout benchmarks are now Rust compiler choices;
//! these entry points preserve result algorithms, not original instruction timing.

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::*;
    const MASK: u32 = 0xffff8000;
    const MIN_NORMAL: u32 = 0x00800000;
    const QUIET_NAN: u32 = 0x7fc00000;
    extern "C" { fn _rsqrt_fallback(value: u32) -> u32; }

    fn input(value: u32) -> __m128 {
        // SSE2 is guaranteed on x86_64; no AVX or SSE4.1 CPU requirement remains.
        unsafe { _mm_castsi128_ps(_mm_cvtsi32_si128(value as i32)) }
    }
    fn bits(value: __m128) -> u32 {
        unsafe { _mm_cvtsi128_si32(_mm_castps_si128(value)) as u32 }
    }
    fn normalized(value: u32) -> u32 { (value & MASK) | 0x8000 }
    fn round(value: u32) -> u32 { value.wrapping_add(0x4000) & MASK }
    fn ordinary(value: u32) -> bool {
        unsafe { _mm_ucomigt_ss(input(value), input(MIN_NORMAL)) != 0 }
    }
    fn fallback(value: u32) -> u32 { unsafe { _rsqrt_fallback(value) } }
    fn exceptional(value: u32) -> u32 {
        let absolute = value & 0x7fffffff;
        if absolute > 0x7f800000 { return value | 0x00400000; }
        if unsafe { _mm_ucomieq_ss(input(value), _mm_setzero_ps()) != 0 } { return value | 0x7f800000; }
        if value & 0x80000000 != 0 { return QUIET_NAN; }
        if absolute == 0x7f800000 { return 0; }
        fallback(value)
    }
    fn full(value: u32) -> u32 {
        let n = normalized(value);
        if !ordinary(n) { return exceptional(value); }
        unsafe { round(bits(_mm_div_ss(_mm_set_ss(1.0), _mm_sqrt_ss(input(n))))) }
    }
    fn non_branch(value: u32, canonical_negative: bool) -> u32 {
        let n = normalized(value);
        if ordinary(n) { return full(value); }
        if value < MIN_NORMAL { return fallback(value); }
        let estimate = unsafe { bits(_mm_rsqrt_ss(input(value))) };
        if canonical_negative && value & 0x80000000 != 0 { return QUIET_NAN; }
        if estimate & 0x7fffffff > 0x7f800000 {
            if value & 0x7fffffff > 0x7f800000 {
                // The old nb/nb2 NaN path returned uninitialized EAX. Return the
                // quieted input payload instead, matching full's defined policy.
                return value | 0x00400000;
            }
            return QUIET_NAN;
        }
        estimate
    }
    #[no_mangle]
    pub extern "C" fn rsqrt_inaccurate(value: u32) -> u32 { unsafe { bits(_mm_rsqrt_ss(input(value))) } }
    #[no_mangle]
    pub extern "C" fn rsqrt_full(value: u32) -> u32 { full(value) }
    #[no_mangle]
    pub extern "C" fn rsqrt_full_gpr(value: u32) -> u32 { full(value) }
    #[no_mangle]
    pub extern "C" fn rsqrt_full_nb(value: u32) -> u32 { non_branch(value, false) }
    #[no_mangle]
    pub extern "C" fn rsqrt_full_nb2(value: u32) -> u32 { non_branch(value, false) }
    #[no_mangle]
    pub extern "C" fn rsqrt_full_nb_gpr(value: u32) -> u32 { non_branch(value, true) }
    #[no_mangle]
    pub extern "C" fn rsqrt_newton(value: u32) -> u32 {
        let n = normalized(value);
        if !ordinary(n) { return exceptional(value); }
        unsafe {
            let x = input(n);
            let estimate = _mm_rsqrt_ps(x);
            let half_x = _mm_mul_ss(x, _mm_set_ss(0.5));
            let squared = _mm_mul_ss(estimate, estimate);
            let correction = _mm_sub_ss(_mm_set_ss(1.5), _mm_mul_ss(squared, half_x));
            round(bits(_mm_mul_ss(correction, estimate)))
        }
    }
    #[no_mangle]
    pub extern "C" fn rsqrt_hack(value: u32) -> u32 {
        let n = normalized(value);
        if !ordinary(n) { return exceptional(value); }
        let estimate = unsafe { bits(_mm_rsqrt_ps(input(n))) };
        let shifted = value.wrapping_shl(8);
        let sign_mask = ((shifted as i32) >> 31) as u32;
        let comparison = if shifted.wrapping_add(0x60000000) as i32 > 0x3c000000 { u32::MAX } else { 0 };
        let correction = 0x47ffu32.wrapping_sub(comparison ^ sign_mask);
        estimate.wrapping_add(correction) & MASK
    }
}
#[cfg(target_arch = "x86_64")]
pub use x86::*;
