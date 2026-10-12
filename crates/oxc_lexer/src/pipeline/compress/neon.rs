use std::arch::aarch64::*;

use oxc_span::Span;

use crate::{
    lanes::Lanes,
    token::{TRIVIA_MAX, TRIVIA_MIN, tk},
};

use crate::pipeline::chunk::{load128, load_lut, mm32, mm64, veq};

use super::{
    bcompact::BCOMPACT,
    common::{build_spans as build_spans_scalar, emit_value, invalid_diags},
    pair_luts::get_pair_luts,
};

/// For each 4-bit "keep" mask, the byte indices that gather the kept spans out of a 4-span
/// window, two spans at a time.
///
/// A `Span` is 8 bytes, so span `k` of the window is bytes `8k..8k+7`; `TBL` reads a 2-register
/// window as 32 indexable bytes. `LO` takes the first two kept spans and `HI` the next two, so
/// between them they move any subset into place with two lookups. Unused lanes are `0x80`, out
/// of range for `TBL` and so zeroed - the next group's store covers them.
static SPAN_LO: [[u8; 16]; 16] = span_ctrl(0);
static SPAN_HI: [[u8; 16]; 16] = span_ctrl(2);

const fn span_ctrl(skip: usize) -> [[u8; 16]; 16] {
    let mut t = [[0x80u8; 16]; 16];
    let mut mask = 0usize;
    while mask < 16 {
        // Walk the kept spans, and emit the two this half is responsible for.
        let (mut span, mut kept) = (0usize, 0usize);
        while span < 4 {
            if mask & (1 << span) != 0 {
                if kept >= skip && kept < skip + 2 {
                    let out = (kept - skip) * 8;
                    let mut byte = 0usize;
                    while byte < 8 {
                        t[mask][out + byte] = (span * 8 + byte) as u8;
                        byte += 1;
                    }
                }
                kept += 1;
            }
            span += 1;
        }
        mask += 1;
    }
    t
}

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

/// Write the staged `(pos, kind)` pairs out as spans, dropping trivia, 8 at a time.
///
/// The scalar form in `common` is paced by its `w` chain: one conditional increment per staged
/// token, which is also the address every store depends on, so it cannot go faster than a cycle
/// a token however wide the machine is. Handling eight at a time turns that into one increment
/// per eight, and the compaction that replaces it is a pair of table lookups.
pub(super) unsafe fn build_spans(
    stage_kind: *const u8,
    stage_pos: *const u32,
    m: usize,
    spans: *mut Span,
    sig_kinds: *mut u8,
) -> usize {
    let sp = spans.cast::<u64>();
    let mut w = 0usize;
    let mut j = 0usize;

    while j + 8 <= m {
        // Which of the eight survive: in the trivia range, and not the hashbang.
        let k8 = vld1_u8(stage_kind.add(j));
        let triv = vceq_u8(
            vqsub_u8(vsub_u8(k8, vdup_n_u8(TRIVIA_MIN)), vdup_n_u8(TRIVIA_MAX - TRIVIA_MIN)),
            vdup_n_u8(0),
        );
        let keep = vorr_u8(vmvn_u8(triv), vceq_u8(k8, vdup_n_u8(tk!(Hashbang))));
        // One bit per token. `ADDV` sums the lanes' bit values, which is the mask.
        const BITS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
        let sig = usize::from(vaddv_u8(vand_u8(keep, vld1_u8(BITS.as_ptr()))));

        // A token's span runs to the next token's start, so the spans are the overlapping `u32`
        // pairs of `stage_pos`. Zipping the window against itself shifted by one builds them.
        if sig == 0xff {
            // Nothing dropped, so both compactions would be the identity. Minified input stages
            // almost no trivia - better than 95% of its groups land here - and unminified input
            // reaches this stage too cheaply for the mispredicts to show.
            spans8(sp.add(w), stage_pos.add(j));
            vst1_u8(sig_kinds.add(w), k8);
            w += 8;
        } else {
            let lo4 = sig & 0xf;
            put4(sp.add(w), stage_pos.add(j), lo4);
            put4(sp.add(w + lo4.count_ones() as usize), stage_pos.add(j + 4), sig >> 4);
            // The kinds compact with the same mask, over a window of 8 rather than 4.
            let kq = vcombine_u8(k8, vdup_n_u8(0));
            vst1_u8(sig_kinds.add(w), vget_low_u8(vqtbl1q_u8(kq, load_lut(&BCOMPACT[sig]))));
            w += sig.count_ones() as usize;
        }
        j += 8;
    }

    w + build_spans_scalar(
        stage_kind.add(j),
        stage_pos.add(j),
        m - j,
        spans.add(w),
        sig_kinds.add(w),
    )
}

/// Write the eight spans starting at `pos` to `dst`, keeping all of them.
#[inline(always)]
unsafe fn spans8(dst: *mut u64, pos: *const u32) {
    for g in 0..2 {
        let v0 = vld1q_u32(pos.add(g * 4));
        let v1 = vld1q_u32(pos.add(g * 4 + 1));
        vst1q_u32(dst.add(g * 4).cast::<u32>(), vzip1q_u32(v0, v1));
        vst1q_u32(dst.add(g * 4 + 2).cast::<u32>(), vzip2q_u32(v0, v1));
    }
}

/// Compact the four spans starting at `pos` by `mask`, writing them at `dst`.
///
/// Writes four spans' worth either way; the lanes past `mask`'s population count are zero, and
/// the next write lands on top of them. That is the same overshoot the AVX2 backend has, and
/// what the `PAD` on the caller's token buffers covers.
#[inline(always)]
unsafe fn put4(dst: *mut u64, pos: *const u32, mask: usize) {
    let v0 = vld1q_u32(pos);
    let v1 = vld1q_u32(pos.add(1));
    let win = uint8x16x2_t(
        vreinterpretq_u8_u32(vzip1q_u32(v0, v1)),
        vreinterpretq_u8_u32(vzip2q_u32(v0, v1)),
    );
    vst1q_u8(dst.cast::<u8>(), vqtbl2q_u8(win, load_lut(&SPAN_LO[mask])));
    vst1q_u8(dst.add(2).cast::<u8>(), vqtbl2q_u8(win, load_lut(&SPAN_HI[mask])));
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
