use std::arch::x86_64::*;

use crate::pipeline::chunk::{load128, mm128, veq128};

#[inline]
pub unsafe fn find1_sse2(src: *const u8, n: usize, mut i: usize, a: u8) -> usize {
    while i + 16 <= n {
        let m = mm128(veq128(load128(src, i), a));
        if m != 0 {
            return i + m.trailing_zeros() as usize;
        }
        i += 16;
    }
    while i < n {
        if *src.add(i) == a {
            return i;
        }
        i += 1;
    }
    n
}

#[inline]
pub unsafe fn find2_sse2(src: *const u8, n: usize, mut i: usize, a: u8, b: u8) -> usize {
    while i + 16 <= n {
        let v = load128(src, i);
        let m = mm128(_mm_or_si128(veq128(v, a), veq128(v, b)));
        if m != 0 {
            return i + m.trailing_zeros() as usize;
        }
        i += 16;
    }
    while i < n {
        let c = *src.add(i);
        if c == a || c == b {
            return i;
        }
        i += 1;
    }
    n
}

#[inline]
pub unsafe fn find3_sse2(src: *const u8, n: usize, mut i: usize, a: u8, b: u8, c: u8) -> usize {
    while i + 16 <= n {
        let v = load128(src, i);
        let m = mm128(_mm_or_si128(_mm_or_si128(veq128(v, a), veq128(v, b)), veq128(v, c)));
        if m != 0 {
            return i + m.trailing_zeros() as usize;
        }
        i += 16;
    }
    while i < n {
        let ch = *src.add(i);
        if ch == a || ch == b || ch == c {
            return i;
        }
        i += 1;
    }
    n
}

#[inline]
pub unsafe fn find4_sse2(
    src: *const u8,
    n: usize,
    mut i: usize,
    a: u8,
    b: u8,
    c: u8,
    d: u8,
) -> usize {
    while i + 16 <= n {
        let v = load128(src, i);
        let m = mm128(_mm_or_si128(
            _mm_or_si128(veq128(v, a), veq128(v, b)),
            _mm_or_si128(veq128(v, c), veq128(v, d)),
        ));
        if m != 0 {
            return i + m.trailing_zeros() as usize;
        }
        i += 16;
    }
    while i < n {
        let x = *src.add(i);
        if x == a || x == b || x == c || x == d {
            return i;
        }
        i += 1;
    }
    n
}

macro_rules! define_find_function_sse2 {
    ($(#[$attr:meta])* $name:ident: $($needle:expr),+ $(,)?) => {
        $(#[$attr])*
        #[inline]
        pub unsafe fn $name(src: *const u8, n: usize, mut i: usize) -> usize {
            use std::arch::x86_64::_mm_or_si128;
            use crate::pipeline::{find::sse2::vor, chunk::{load128, veq128, mm128}};

            while i + 16 <= n {
                let v = load128(src, i);
                let m = mm128(vor!($(veq128(v, $needle)),+));
                if m != 0 {
                    return i + m.trailing_zeros() as usize;
                }
                i += 16;
            }
            while i < n {
                let c = *src.add(i);
                if $(c == $needle)||+ {
                    return i;
                }
                i += 1;
            }
            n
        }
    };
}
pub(super) use define_find_function_sse2;

/// OR-fold of byte compares, associated as a tree.
macro_rules! vor {
    ($a:expr) => { $a };
    ($a:expr, $b:expr) => { _mm_or_si128($a, $b) };
    ($a:expr, $b:expr, $($rest:expr),+) => {
        _mm_or_si128(vor!($a, $b), vor!($($rest),+))
    };
}
pub(super) use vor;

/// Bits of the 64 bytes at base that are brackets: bit i for byte base + i.
#[inline]
pub fn bracket_bits_sse2(src: &[u8], base: usize) -> u64 {
    let block = &src[base..base + 64];
    let mut out = 0u64;
    for q in 0..4 {
        // SAFETY: block holds 64 bytes, so a 16-byte load at offset 0, 16, 32 or 48 stays inside it.
        let mask = unsafe {
            let v = load128(block.as_ptr(), q * 16);
            let m = vor!(
                veq128(v, b'('),
                veq128(v, b')'),
                veq128(v, b'['),
                veq128(v, b']'),
                veq128(v, b'{'),
                veq128(v, b'}')
            );
            mm128(m)
        };
        out |= u64::from(mask) << (q * 16);
    }
    out
}
