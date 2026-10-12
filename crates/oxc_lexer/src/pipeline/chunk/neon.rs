use std::arch::aarch64::*;

/// Load 128 bits starting at byte `i` as a `uint8x16_t`.
///
/// The load is unaligned, so neither `src` nor `i` has any alignment requirement.
///
/// # SAFETY
///
/// `src.add(i)` must be valid for reads of 16 bytes.
#[inline(always)]
pub unsafe fn load128(src: *const u8, i: usize) -> uint8x16_t {
    vld1q_u8(src.add(i))
}

/// Compare every byte of chunk `v` against `c`.
///
/// Returns a `uint8x16_t` where each of the 16 lanes is `0xFF`
/// where the lane in `v` equals `c`, and `0x00` where it does not.
#[inline(always)]
pub fn veq(v: uint8x16_t, c: u8) -> uint8x16_t {
    // SAFETY: These intrinsics touch no memory and require only the `neon` target feature,
    // which this module's `#[cfg]` guarantees
    unsafe { vceqq_u8(v, vdupq_n_u8(c)) }
}

/// Gather the top bit of each byte of chunks `a` and `b` into a 32-bit mask.
///
/// Bit N of the result is the top bit of byte N of the 32 bytes `a` then `b`.
///
/// AArch64 has no `vpmovmskb`, so the bits are summed into place instead: AND each lane with its
/// own bit value, then let `ADDP` fold the lanes together, halving the live byte count each time
/// until one 32-bit lane holds the whole mask.
///
/// Every lane of `a` and `b` must be `0xFF` or `0x00` - a mask straight out of a compare, or out
/// of a `CMTST` bit test - or the sums carry into each other and the result is garbage.
#[inline(always)]
pub fn mm32(a: uint8x16_t, b: uint8x16_t) -> u32 {
    // SAFETY: These intrinsics touch no memory and require only the `neon` target feature,
    // which this module's `#[cfg]` guarantees
    unsafe {
        let bits = lane_bits();
        // 32 lanes -> 16 pair sums (2 bits each): `a`'s in bytes 0-7, `b`'s in bytes 8-15
        let s = vpaddq_u8(vandq_u8(a, bits), vandq_u8(b, bits));
        // -> 8 nibble sums -> 4 byte sums: [a lo, a hi, b lo, b hi]
        let s = vpaddq_u8(s, s);
        let s = vpaddq_u8(s, s);
        vgetq_lane_u32(vreinterpretq_u32_u8(s), 0)
    }
}

/// Gather the top bit of each byte of chunks `a`, `b`, `c` and `d` into a 64-bit mask.
///
/// Bit N of the result is the top bit of byte N of the 64 bytes `a`, `b`, `c`, `d` in order,
/// which is exactly one block of the pipeline's bitmaps. See [`mm32`] for how the reduction
/// works and what the lanes have to hold.
#[inline(always)]
pub fn mm64(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t, d: uint8x16_t) -> u64 {
    // SAFETY: These intrinsics touch no memory and require only the `neon` target feature,
    // which this module's `#[cfg]` guarantees
    unsafe {
        let bits = lane_bits();
        let ab = vpaddq_u8(vandq_u8(a, bits), vandq_u8(b, bits));
        let cd = vpaddq_u8(vandq_u8(c, bits), vandq_u8(d, bits));
        // 64 lanes -> 32 pair sums -> 16 nibble sums -> 8 byte sums, all in lane 0
        let s = vpaddq_u8(ab, cd);
        vgetq_lane_u64(vreinterpretq_u64_u8(vpaddq_u8(s, s)), 0)
    }
}

/// Gather each byte of chunk `v` into a nibble of a 64-bit mask.
///
/// Nibble N of the result is `0xF` where byte N of `v` is `0xFF` and `0` where it is `0x00`, so
/// `trailing_zeros() >> 2` is the index of the first set lane, and the mask is zero only if no
/// lane is set. Every lane of `v` must be `0xFF` or `0x00`.
///
/// This is 2 instructions against [`mm32`]'s 6, which is what the search loops want: they test
/// one chunk and branch out of the loop, so they pay the reduction's latency on every chunk
/// instead of amortizing it over a whole block the way the bitmap builders do.
#[inline(always)]
pub fn nib(v: uint8x16_t) -> u64 {
    // SAFETY: These intrinsics touch no memory and require only the `neon` target feature,
    // which this module's `#[cfg]` guarantees
    unsafe {
        // Each 16-bit lane holds a pair of mask bytes; `SHRN` by 4 narrows the pair to its nibbles
        let packed = vshrn_n_u16::<4>(vreinterpretq_u16_u8(v));
        vget_lane_u64(vreinterpret_u64_u8(packed), 0)
    }
}

/// Load 16 bytes of a lookup table as a `uint8x16_t`.
#[inline(always)]
pub fn load_lut(lut: &[u8; 16]) -> uint8x16_t {
    // SAFETY: `lut` is 16 bytes, so the load stays inside it
    unsafe { vld1q_u8(lut.as_ptr()) }
}

/// One bit per lane, for the pairwise-add reduction in [`mm32`] and [`mm64`].
///
/// The pattern repeats every 8 bytes because `ADDP` reduces within 8-byte halves, so each half
/// contributes one byte of the result.
#[inline(always)]
fn lane_bits() -> uint8x16_t {
    const BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    load_lut(&BITS)
}
