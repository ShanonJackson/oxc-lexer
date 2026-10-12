//! Comment scans, 32 bytes at a time.
//!
//! Unlike the searches in `find`, these do real work with the mask after the test - masking off
//! the body before a terminator, walking the `@` candidates one bit at a time - so they take the
//! 32-bit mask `mm32` builds from a pair of chunks and run the same bit arithmetic as the AVX2
//! backend, rather than `find`'s cheaper-to-reduce nibble mask.

use std::arch::aarch64::*;

use crate::pipeline::chunk::{load128, mm32, veq};

use super::common::lic_verify_at;

pub unsafe fn scan_block_comment(src: *const u8, n: usize, mut i: usize) -> (usize, bool, i64) {
    let mut saw_nl = false;
    let mut lic_q: i64 = -1;
    while i + 32 <= n {
        let v0 = load128(src, i);
        let v1 = load128(src, i + 16);
        // The `/` of a `*/` straddling the chunk boundary is found by the overlapping load, which
        // reads at most 1 byte past `n` into the caller-guaranteed pad.
        let w0 = load128(src, i + 1);
        let w1 = load128(src, i + 17);
        let term = mm32(
            vandq_u8(veq(v0, b'*'), veq(w0, b'/')),
            vandq_u8(veq(v1, b'*'), veq(w1, b'/')),
        );
        let nl = if saw_nl {
            0
        } else {
            mm32(
                vorrq_u8(veq(v0, b'\n'), veq(v0, b'\r')),
                vorrq_u8(veq(v1, b'\n'), veq(v1, b'\r')),
            )
        };
        let at = if lic_q < 0 { mm32(veq(v0, b'@'), veq(v1, b'@')) } else { 0 };
        if term != 0 {
            let j = term.trailing_zeros() as usize;
            let bodymask: u32 = if j > 0 { (1u32 << j) - 1 } else { 0 };
            saw_nl |= (nl & bodymask) != 0;
            if lic_q < 0 {
                let am = at & bodymask;
                if am != 0 {
                    lic_q = lic_first(src, i, am);
                }
            }
            return (i + j + 1, saw_nl, lic_q);
        }
        saw_nl |= nl != 0;
        if lic_q < 0 && at != 0 {
            lic_q = lic_first(src, i, at);
        }
        i += 32;
    }
    while i + 1 < n {
        let c = *src.add(i);
        if c == b'*' && *src.add(i + 1) == b'/' {
            return (i + 1, saw_nl, lic_q);
        }
        if c == b'\n' || c == b'\r' {
            saw_nl = true;
        }
        if c == b'@' && lic_q < 0 && lic_verify_at(src, i) {
            lic_q = i as i64;
        }
        i += 1;
    }
    (n, saw_nl, lic_q)
}

/// Scan a `//` line comment to its LineTerminator (LF, CR, or LS/PS),
/// tracking the first `@license` / `@preserve` position for comment
/// metadata. A 0xE2 that isn't LS/PS (typographic punctuation in prose) is
/// cleared from the mask and the scan continues; the LS/PS confirm can read
/// the pad, which never matches 0x80.
pub unsafe fn scan_line_comment(src: *const u8, n: usize, mut i: usize) -> (usize, i64) {
    let mut lic_q: i64 = -1;
    while i + 32 <= n {
        let v0 = load128(src, i);
        let v1 = load128(src, i + 16);
        let termv = |v: uint8x16_t| vorrq_u8(vorrq_u8(veq(v, b'\n'), veq(v, b'\r')), veq(v, 0xE2));
        let mut term = mm32(termv(v0), termv(v1));
        let at = mm32(veq(v0, b'@'), veq(v1, b'@'));
        while term != 0 {
            let t = term.trailing_zeros() as usize;
            let c = *src.add(i + t);
            if c == 0xE2
                && !(*src.add(i + t + 1) == 0x80
                    && (*src.add(i + t + 2) == 0xA8 || *src.add(i + t + 2) == 0xA9))
            {
                term &= term - 1; // not LS/PS: clear and keep scanning
                continue;
            }
            if lic_q < 0 {
                let am = at & if t > 0 { (1u32 << t) - 1 } else { 0 };
                if am != 0 {
                    lic_q = lic_first(src, i, am);
                }
            }
            return (i + t, lic_q);
        }
        if lic_q < 0 && at != 0 {
            lic_q = lic_first(src, i, at);
        }
        i += 32;
    }
    while i < n {
        let c = *src.add(i);
        if c == b'\n' || c == b'\r' {
            return (i, lic_q);
        }
        if c == 0xE2
            && *src.add(i + 1) == 0x80
            && (*src.add(i + 2) == 0xA8 || *src.add(i + 2) == 0xA9)
        {
            return (i, lic_q);
        }
        if c == b'@' && lic_q < 0 && lic_verify_at(src, i) {
            lic_q = i as i64;
        }
        i += 1;
    }
    (n, lic_q)
}

#[inline(always)]
unsafe fn lic_first(src: *const u8, base: usize, mut am: u32) -> i64 {
    while am != 0 {
        let q = base + am.trailing_zeros() as usize;
        am &= am - 1;
        if lic_verify_at(src, q) {
            return q as i64;
        }
    }
    -1
}
