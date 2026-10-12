use std::arch::aarch64::*;

use oxc_span::Span;

use crate::{lanes::Lanes, token::tk};

use crate::pipeline::chunk::{load128, mm32, mm64, veq};

use super::{
    common::{emit_value, invalid_diags},
    pair_luts::get_pair_luts,
};

/// Turn the token-start bitmap into a list of token positions and kinds, 16 bits at a time.
///
/// The LUT scheme is `pair_luts`'; this reads the tables exactly as the AVX2 backend does. `TBL`
/// stands in for `vpshufb` with no adjustment, because a control byte of `0x80` is out of range
/// for a 16-byte table and so zeroes its lane either way.
pub(super) unsafe fn compress_blocks(
    st: *const u64,
    kind: *const u8,
    b0: usize,
    b1: usize,
    starts: *mut u32,
    kinds: *mut u8,
) -> usize {
    // SAFETY: `init_pair_luts` has always been called in `Lexer::new` on SIMD builds
    let pair_luts = unsafe { get_pair_luts() };
    let lut0z = pair_luts.lut0z.as_ptr().cast::<u8>();
    let lutpad = pair_luts.lutpad.as_ptr().cast::<u8>();

    let mut m = 0usize;
    for b in b0..b1 {
        let mword = *st.add(b);
        if mword == 0 {
            continue;
        }
        let base = b * 64;
        let o0: u32 = 0;
        let o1 = o0 + (mword & 0xffff).count_ones();
        let o2 = o1 + ((mword >> 16) & 0xffff).count_ones();
        let o3 = o2 + ((mword >> 32) & 0xffff).count_ones();
        let o4 = o3 + (mword >> 48).count_ones();
        let mut basev = vdupq_n_u32(base as u32);
        macro_rules! pair {
            ($p:expr, $off:expr) => {{
                let sub0 = ((mword >> (16 * $p)) & 0xff) as usize;
                let sub1 = ((mword >> (16 * $p + 8)) & 0xff) as usize;
                let pc0 = sub0.count_ones() as usize;
                // Loading a D register zeroes the top half of the Q register, which is what puts
                // `sub1`'s offsets in the clear when the two rows are OR-ed together.
                let row0 = vcombine_u8(vld1_u8(lut0z.add(sub0 * 8)), vdup_n_u8(0));
                let row1 = vld1q_u8(lutpad.add(sub1 * 32 + 8 - pc0));
                let ctrl = vorrq_u8(row0, row1);
                let kw = vld1q_u8(kind.add(base + 16 * $p));
                // Widen the 16 control bytes to `u32` and add the group's base position. Lanes
                // past this group's token count hold the `0x80` padding, so they come out as
                // junk positions - the next group's store lands on top of them.
                let lo = vmovl_u8(vget_low_u8(ctrl));
                let hi = vmovl_high_u8(ctrl);
                let at = starts.add(m + $off as usize);
                vst1q_u32(at, vaddq_u32(basev, vmovl_u16(vget_low_u16(lo))));
                vst1q_u32(at.add(4), vaddq_u32(basev, vmovl_high_u16(lo)));
                vst1q_u32(at.add(8), vaddq_u32(basev, vmovl_u16(vget_low_u16(hi))));
                vst1q_u32(at.add(12), vaddq_u32(basev, vmovl_high_u16(hi)));
                basev = vaddq_u32(basev, vdupq_n_u32(16));
                vqtbl1q_u8(kw, ctrl)
            }};
        }
        let ka = pair!(0, o0);
        let kb = pair!(1, o1);
        vst1q_u8(kinds.add(m + o0 as usize), ka);
        vst1q_u8(kinds.add(m + o1 as usize), kb);
        let kc = pair!(2, o2);
        let kd = pair!(3, o3);
        vst1q_u8(kinds.add(m + o2 as usize), kc);
        vst1q_u8(kinds.add(m + o3 as usize), kd);
        m += o4 as usize;
    }
    m
}

pub(super) unsafe fn lanes_post(
    src: &[u8],
    out_kinds: *const u8,
    out_spans: *const Span,
    m: usize,
    nn: u32,
    lanes: &mut Lanes,
) {
    // The kinds whose text the value lanes still have to read.
    let hits = |v: uint8x16_t| {
        vorrq_u8(
            vorrq_u8(veq(v, tk!(Number)), veq(v, tk!(BigInt))),
            vorrq_u8(veq(v, tk!(IdentEscaped)), veq(v, tk!(PrivateIdentEscaped))),
        )
    };
    // `tk!(Invalid)` is the byte-class default for stray/control bytes and
    // reaches the output as a 1-byte token. Track "any seen" alongside the
    // value sweep; localize cold.
    let mut inv = vdupq_n_u8(0);
    let mut i = 0usize;
    while i + 64 <= m {
        let v0 = load128(out_kinds, i);
        let v1 = load128(out_kinds, i + 16);
        let v2 = load128(out_kinds, i + 32);
        let v3 = load128(out_kinds, i + 48);
        inv = vorrq_u8(
            inv,
            vorrq_u8(
                vorrq_u8(veq(v0, tk!(Invalid)), veq(v1, tk!(Invalid))),
                vorrq_u8(veq(v2, tk!(Invalid)), veq(v3, tk!(Invalid))),
            ),
        );
        let mut mask = mm64(hits(v0), hits(v1), hits(v2), hits(v3));
        while mask != 0 {
            emit_value(src, out_kinds, out_spans, i + mask.trailing_zeros() as usize, lanes);
            mask &= mask - 1;
        }
        i += 64;
    }
    let mut inv_dirty = vmaxvq_u8(inv) != 0;
    while i < m {
        // Reads up to 15 bytes past `m`, into the arena's PAD. Both masks are clamped to `rem`
        // below, so nothing read from there can reach the output. The AVX2 backend's tail reads
        // the same padding, 31 bytes of it, so this needs no capacity the caller isn't already
        // required to supply.
        let v = load128(out_kinds, i);
        let zero = vdupq_n_u8(0);
        let mut mask = mm32(hits(v), zero);
        let mut invm = mm32(veq(v, tk!(Invalid)), zero);
        let rem = m - i;
        if rem < 16 {
            mask &= (1u32 << rem) - 1;
            invm &= (1u32 << rem) - 1;
        }
        inv_dirty |= invm != 0;
        while mask != 0 {
            emit_value(src, out_kinds, out_spans, i + mask.trailing_zeros() as usize, lanes);
            mask &= mask - 1;
        }
        i += 16;
    }
    if inv_dirty {
        invalid_diags(src, out_kinds, out_spans, m, nn, lanes);
    }
}
