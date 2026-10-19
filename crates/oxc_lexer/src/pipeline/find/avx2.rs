use std::arch::x86_64::*;

use crate::pipeline::chunk::{load256, mm, veq};

/// First byte at or after i that is one of needles, or n.
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn find<const K: usize>(
    src: *const u8,
    n: usize,
    mut i: usize,
    needles: [u8; K],
) -> usize {
    while i + 32 <= n {
        let m = mm(any_of(load256(src, i), needles));
        if m != 0 {
            return i + m.trailing_zeros() as usize;
        }
        i += 32;
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
unsafe fn any_of<const K: usize>(v: __m256i, needles: [u8; K]) -> __m256i {
    let mut m = needles.map(|c| veq(v, c));
    let mut len = K;
    while len > 1 {
        let half = len / 2;
        for j in 0..half {
            m[j] = _mm256_or_si256(m[2 * j], m[2 * j + 1]);
        }
        if len % 2 == 1 {
            m[half] = m[len - 1];
        }
        len = half + len % 2;
    }
    m[0]
}
