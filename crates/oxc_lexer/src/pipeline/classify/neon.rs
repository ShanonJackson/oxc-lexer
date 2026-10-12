use std::arch::aarch64::*;

use crate::token::tk;

use crate::pipeline::chunk::{load128, load_lut, mm64, veq};

use super::luts::{MERGED_LUTS, WORD_LUTS};
use super::punct1_luts::{PH_A, PH_B, PH_T0, PH_T1};

/// The lookup tables, held in registers across the block loop.
///
/// `TBL` indexes all 16 bytes of its table and zeros an index of 16 or more, where `vpshufb`
/// takes the index's low nibble and zeros on its top bit. Both nibble lookups land on the same
/// answer anyway: the low-nibble index is masked to `0x0F` for free, because `classify_chunk`
/// needs that nibble regardless, and `vshrq_n_u8` yields a high nibble of 0-15 which is always
/// in range. Neither `hi` table has an entry for a high nibble of 8-15, so a non-ASCII byte ANDs
/// to zero on its own - the same "bytes >= 0x80 get no bits" the AVX2 backend gets out of
/// `vpshufb` zeroing on the top bit.
struct Luts {
    wb_lo: uint8x16_t,
    wb_hi: uint8x16_t,
    mrg_lo: uint8x16_t,
    mrg_hi: uint8x16_t,
    ph_a: uint8x16_t,
    ph_b: uint8x16_t,
    /// `PH_T0` and `PH_T1` as one 32-entry table. `TBL` indexes both halves in a single lookup,
    /// so the perfect hash needs no blend on bit 4 the way the AVX2 backend's does.
    ph_t: uint8x16x2_t,
}

/// What one 16-byte chunk contributes: a lane per bitmap, `0xFF` where the byte is in that
/// class and `0x00` where it is not, plus its 16 `kind` bytes.
///
/// `mm64` gathers four chunks' worth of a lane into one 64-bit bitmap word, so every field has
/// to be a compare result and not, say, a masked-off bit.
struct Classified {
    word: uint8x16_t,
    ws: uint8x16_t,
    digit: uint8x16_t,
    dot: uint8x16_t,
    kwinit: uint8x16_t,
    opch: uint8x16_t,
    misc: uint8x16_t,
    kind: uint8x16_t,
}

pub(super) unsafe fn classify_impl(
    ts: bool,
    src: *const u8,
    n: usize,
    word: *mut u64,
    st: *mut u64,
    kwinit: *mut u64,
    opch: *mut u64,
    digit: *mut u64,
    dot: *mut u64,
    misc: *mut u64,
    kind: *mut u8,
) {
    // The merged LUT variants differ only in the keyword-initial bits; the
    // selection happens once, outside the loop.
    let mrg_lo = if ts { MERGED_LUTS.lo_ts } else { MERGED_LUTS.lo };
    let luts = Luts {
        wb_lo: load_lut(&WORD_LUTS.lo),
        wb_hi: load_lut(&WORD_LUTS.hi),
        mrg_lo: load_lut(&mrg_lo),
        mrg_hi: load_lut(&MERGED_LUTS.hi),
        ph_a: load_lut(&PH_A),
        ph_b: load_lut(&PH_B),
        ph_t: uint8x16x2_t(load_lut(&PH_T0), load_lut(&PH_T1)),
    };

    let mut cw: u64 = 0;
    let mut cs: u64 = 0;
    let mut i = 0usize;
    let mut b = 0usize;
    // Process ceil(n/64) blocks. When n is not a multiple of 64 the final
    // block overreads up to 63 bytes into the caller-guaranteed zeroed PAD
    // and is masked below - this replaces the byte-at-a-time scalar tail,
    // which cost ~18 cyc per tail byte (up to ~1.1k cyc when n mod 64 is
    // near 63) and dominated small-file lexing.
    let nb_ceil = n.div_ceil(64);
    while b < nb_ceil {
        let c0 = classify_chunk(load128(src, i), &luts);
        let c1 = classify_chunk(load128(src, i + 16), &luts);
        let c2 = classify_chunk(load128(src, i + 32), &luts);
        let c3 = classify_chunk(load128(src, i + 48), &luts);

        vst1q_u8(kind.add(i), c0.kind);
        vst1q_u8(kind.add(i + 16), c1.kind);
        vst1q_u8(kind.add(i + 32), c2.kind);
        vst1q_u8(kind.add(i + 48), c3.kind);

        let wordm = mm64(c0.word, c1.word, c2.word, c3.word);
        let wsm = mm64(c0.ws, c1.ws, c2.ws, c3.ws);
        *word.add(b) = wordm;
        *digit.add(b) = mm64(c0.digit, c1.digit, c2.digit, c3.digit);
        *dot.add(b) = mm64(c0.dot, c1.dot, c2.dot, c3.dot);
        *kwinit.add(b) = mm64(c0.kwinit, c1.kwinit, c2.kwinit, c3.kwinit);
        *opch.add(b) = mm64(c0.opch, c1.opch, c2.opch, c3.opch);
        *misc.add(b) = mm64(c0.misc, c1.misc, c2.misc, c3.misc);

        let wprev = (wordm << 1) | cw;
        let sprev = (wsm << 1) | cs;
        cw = wordm >> 63;
        cs = wsm >> 63;
        *st.add(b) = (wsm & !sprev) | (wordm & !wprev) | (!wsm & !wordm);

        i += 64;
        b += 1;
    }
    // Mask the overread positions [n, ceil) out of the last block's bitmaps.
    // Zero PAD bytes classify as `st`=1 (non-word, non-ws => token start),
    // which would otherwise mint spurious tokens past `n` in `compress`; the
    // other six bitmaps are already 0 for a zero byte, but masking all seven
    // makes the last word bit-identical to the old scalar tail's output (real
    // bits [0, rem), zeros above) regardless of LUT contents. `kind` past `n`
    // is never read - `compress` only visits masked `st` starts - so it needs
    // no fixup.
    let rem = n & 63;
    if rem != 0 {
        let last = nb_ceil - 1;
        let m = (1u64 << rem) - 1;
        *word.add(last) &= m;
        *st.add(last) &= m;
        *kwinit.add(last) &= m;
        *opch.add(last) &= m;
        *digit.add(last) &= m;
        *dot.add(last) &= m;
        *misc.add(last) &= m;
    }
}

#[inline(always)]
unsafe fn classify_chunk(v: uint8x16_t, l: &Luts) -> Classified {
    let lo = vandq_u8(v, vdupq_n_u8(0x0f));
    let hn = vshrq_n_u8::<4>(v);

    // `tb` answers "identifier char / whitespace / digit", `t` answers
    // "keyword-initial / operator char / dot". See `luts` for the bit layout.
    let tb = vandq_u8(vqtbl1q_u8(l.wb_lo, lo), vqtbl1q_u8(l.wb_hi, hn));
    let t = vandq_u8(vqtbl1q_u8(l.mrg_lo, lo), vqtbl1q_u8(l.mrg_hi, hn));

    // Each class test is `(bits & mask) != 0`, which `vtstq_u8` states directly and LLVM lowers
    // per case: a single-bit test at the top of the byte becomes `CMLT #0`, a contiguous group at
    // the top becomes one unsigned compare, and the rest become `AND` plus a compare against zero
    // whose inverse folds into the consumer. All of them keep the all-ones/all-zeros lanes `mm64`
    // needs, which is what rules out masking the bits out and testing the result directly.
    let na = vtstq_u8(v, vdupq_n_u8(0x80));
    // Non-ASCII bytes count as identifier chars in `word` and are folded
    // into `misc` so `misc_pre` can re-classify Unicode whitespace.
    let word = vorrq_u8(vtstq_u8(tb, vdupq_n_u8(0x3f)), na);
    let ws = vtstq_u8(tb, vdupq_n_u8(0xc0));
    let digit = vtstq_u8(tb, vdupq_n_u8(0x02));
    let misc = vorrq_u8(vorrq_u8(veq(v, b'#'), veq(v, b'\\')), na);

    let mut kind = punct1(v, lo, hn, l);
    kind = vbslq_u8(ws, vdupq_n_u8(tk!(Whitespace)), kind);
    kind = vbslq_u8(word, vdupq_n_u8(tk!(Ident)), kind);
    kind = vbslq_u8(digit, vdupq_n_u8(tk!(Number)), kind);

    Classified {
        word,
        ws,
        digit,
        dot: vtstq_u8(t, vdupq_n_u8(0x80)),
        kwinit: vtstq_u8(t, vdupq_n_u8(0x03)),
        opch: vtstq_u8(t, vdupq_n_u8(0x3c)),
        misc,
        kind,
    }
}

/// The single-byte punctuator kind of each lane, via the `punct1_luts` perfect hash.
///
/// Bytes outside 0x20 - 0x7F get `Invalid` without consulting the hash. A byte >= 0x80 still
/// hashes to some slot - `lo` keeps its low nibble and `PH_B` has no entry for its high nibble,
/// so `h` is `PH_A[lo]`, still under 32 and still in range for `ph_t` - but reads as negative
/// here, so the blend discards whatever that slot held. Identifier chars and whitespace can land
/// in any slot too; `classify_chunk` overwrites their kind afterwards.
#[inline(always)]
unsafe fn punct1(v: uint8x16_t, lo: uint8x16_t, hn: uint8x16_t, l: &Luts) -> uint8x16_t {
    let h = veorq_u8(vqtbl1q_u8(l.ph_a, lo), vqtbl1q_u8(l.ph_b, hn));
    let ord = vqtbl2q_u8(l.ph_t, h);
    let ctl = vcltq_s8(vreinterpretq_s8_u8(v), vdupq_n_s8(0x20));
    vbslq_u8(ctl, vdupq_n_u8(tk!(Invalid)), ord)
}
