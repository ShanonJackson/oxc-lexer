//! Byte searches, 16 bytes at a time.
//!
//! These run over the interior of a literal or a comment and stop at the first hit, so they are
//! paced by how fast one chunk's answer reaches the branch, not by how many bytes a chunk holds.
//! `nib` is the cheap reduction for that - 2 instructions where a full 32-bit `mm32` takes 6 -
//! and 16-byte chunks keep the common short literal down to a single iteration. The nibble it
//! gives each lane means the first hit is at `trailing_zeros() >> 2`.

use std::arch::aarch64::*;

use crate::pipeline::chunk::{load128, mm64, nib, veq};

#[inline]
pub unsafe fn find1(src: *const u8, n: usize, mut i: usize, a: u8) -> usize {
    while i + 16 <= n {
        let m = nib(veq(load128(src, i), a));
        if m != 0 {
            return i + (m.trailing_zeros() >> 2) as usize;
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
pub unsafe fn find2(src: *const u8, n: usize, mut i: usize, a: u8, b: u8) -> usize {
    while i + 16 <= n {
        let v = load128(src, i);
        let m = nib(vorrq_u8(veq(v, a), veq(v, b)));
        if m != 0 {
            return i + (m.trailing_zeros() >> 2) as usize;
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
pub unsafe fn find3(src: *const u8, n: usize, mut i: usize, a: u8, b: u8, c: u8) -> usize {
    while i + 16 <= n {
        let v = load128(src, i);
        let m = nib(vorrq_u8(vorrq_u8(veq(v, a), veq(v, b)), veq(v, c)));
        if m != 0 {
            return i + (m.trailing_zeros() >> 2) as usize;
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
pub unsafe fn find4(src: *const u8, n: usize, mut i: usize, a: u8, b: u8, c: u8, d: u8) -> usize {
    while i + 16 <= n {
        let v = load128(src, i);
        let m = nib(vorrq_u8(vorrq_u8(veq(v, a), veq(v, b)), vorrq_u8(veq(v, c), veq(v, d))));
        if m != 0 {
            return i + (m.trailing_zeros() >> 2) as usize;
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

macro_rules! define_find_function {
    ($(#[$attr:meta])* $name:ident: $($needle:expr),+ $(,)?) => {
        $(#[$attr])*
        #[inline]
        pub unsafe fn $name(src: *const u8, n: usize, mut i: usize) -> usize {
            use std::arch::aarch64::vorrq_u8;
            use crate::pipeline::{find::neon::vor, chunk::{load128, veq, nib}};

            while i + 16 <= n {
                let v = load128(src, i);
                let m = nib(vor!($(veq(v, $needle)),+));
                if m != 0 {
                    return i + (m.trailing_zeros() >> 2) as usize;
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
pub(super) use define_find_function;

/// OR-fold of `CMEQ` results, associated as a tree.
macro_rules! vor {
    ($a:expr) => { $a };
    ($a:expr, $b:expr) => { vorrq_u8($a, $b) };
    ($a:expr, $b:expr, $($rest:expr),+) => {
        vorrq_u8(vor!($a, $b), vor!($($rest),+))
    };
}
pub(super) use vor;

/// Bits of the 64 bytes at `base` that are brackets (`(){}[]`): bit `i` for byte `base + i`.
#[inline]
pub fn bracket_bits(src: &[u8], base: usize) -> u64 {
    let block = &src[base..base + 64];

    // A whole block at once, so `mm64`'s reduction is amortized over all 64 bytes rather than
    // paid per chunk the way the searches above pay `nib`.
    // SAFETY: `block` holds 64 bytes, so the four 16-byte loads stay inside it; the OR-folds and
    // the reduction touch no memory and need only `neon`, which this module's `#[cfg]` guarantees.
    unsafe {
        let brackets = |off: usize| {
            let v = load128(block.as_ptr(), off);
            vor!(veq(v, b'('), veq(v, b')'), veq(v, b'['), veq(v, b']'), veq(v, b'{'), veq(v, b'}'))
        };
        mm64(brackets(0), brackets(16), brackets(32), brackets(48))
    }
}
