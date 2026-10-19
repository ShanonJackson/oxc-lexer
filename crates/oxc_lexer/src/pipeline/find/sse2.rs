use std::arch::x86_64::*;

use crate::pipeline::chunk::{load128, mm128, veq128};

/// First byte at or after i that is one of needles, or n.
#[inline(always)]
pub unsafe fn find<const K: usize>(
    src: *const u8,
    n: usize,
    mut i: usize,
    needles: [u8; K],
) -> usize {
    while i + 16 <= n {
        let m = mm128(any_of(load128(src, i), needles));
        if m != 0 {
            return i + m.trailing_zeros() as usize;
        }
        i += 16;
    }
    while i < n {
        if needles.contains(&*src.add(i)) {
            return i;
        }
        i += 1;
    }
    n
}

// The compares are OR-folded pairwise, level by level, so the tree stays shallow for 7 needles.
#[inline(always)]
unsafe fn any_of<const K: usize>(v: __m128i, needles: [u8; K]) -> __m128i {
    let mut m = needles.map(|c| veq128(v, c));
    let mut len = K;
    while len > 1 {
        let half = len / 2;
        for j in 0..half {
            m[j] = _mm_or_si128(m[2 * j], m[2 * j + 1]);
        }
        if len % 2 == 1 {
            m[half] = m[len - 1];
        }
        len = half + len % 2;
    }
    m[0]
}

/// Bits of the 64 bytes at base that are brackets: bit i for byte base + i.
#[inline]
pub fn bracket_bits(src: &[u8], base: usize) -> u64 {
    let block = &src[base..base + 64];
    let mut out = 0u64;
    for q in 0..4 {
        // SAFETY: block holds 64 bytes, so a 16-byte load at offset 0, 16, 32 or 48 stays inside it.
        let mask = unsafe {
            let v = load128(block.as_ptr(), q * 16);
            mm128(any_of(v, [b'(', b')', b'[', b']', b'{', b'}']))
        };
        out |= u64::from(mask) << (q * 16);
    }
    out
}
