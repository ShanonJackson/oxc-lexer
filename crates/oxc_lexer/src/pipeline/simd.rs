use crate::pipeline::{
    cpu::{Cpu, Generic},
    find, scan,
};

#[cfg(target_arch = "x86_64")]
use crate::pipeline::cpu::Avx2;
#[cfg(target_arch = "aarch64")]
use crate::pipeline::cpu::Neon;

/// The byte-search leaves, one body per feature set, so carve gets the AVX2 bodies inlined in its AVX2 copy.
pub(super) trait Simd: Cpu {
    #[inline(always)]
    unsafe fn find<const K: usize>(src: *const u8, n: usize, i: usize, needles: [u8; K]) -> usize {
        find::find(src, n, i, needles)
    }

    #[inline(always)]
    unsafe fn scan_block_comment(src: *const u8, n: usize, i: usize) -> (usize, bool, i64) {
        scan::scan_block_comment(src, n, i)
    }

    #[inline(always)]
    unsafe fn scan_line_comment(src: *const u8, n: usize, i: usize) -> (usize, i64) {
        scan::scan_line_comment(src, n, i)
    }
}

impl Simd for Generic {}

#[cfg(target_arch = "aarch64")]
impl Simd for Neon {}

// An Avx2 only exists when detect found the features these bodies use, which the unsafe calls rely on.
#[cfg(target_arch = "x86_64")]
impl Simd for Avx2 {
    #[inline(always)]
    unsafe fn find<const K: usize>(src: *const u8, n: usize, i: usize, needles: [u8; K]) -> usize {
        find::avx2::find(src, n, i, needles)
    }

    #[inline(always)]
    unsafe fn scan_block_comment(src: *const u8, n: usize, i: usize) -> (usize, bool, i64) {
        scan::avx2::scan_block_comment(src, n, i)
    }

    #[inline(always)]
    unsafe fn scan_line_comment(src: *const u8, n: usize, i: usize) -> (usize, i64) {
        scan::avx2::scan_line_comment(src, n, i)
    }
}
