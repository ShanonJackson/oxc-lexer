use oxc_span::Span;

use crate::{
    error::DiagCode,
    lanes::Lanes,
    token::{is_trivia_byte, tk},
};

/// Write the staged `(pos, kind)` pairs out as spans, dropping trivia.
///
/// A token's span runs to the next token's start, so the staged positions are read as overlapping
/// pairs - which is why there is always one more position staged than there are tokens to emit.
///
/// This is the scalar form, used as-is by the NEON and scalar backends and as the AVX2 backend's
/// tail. Vectorizing it means compacting at `Span` granularity, which AVX2 gets from a single
/// `vpermd`; on AArch64 the equivalent is a `TBL` over a 4-register table and a control LUT per
/// mask, which has not yet earned its keep against how cheap this loop already is.
#[inline(always)]
pub(super) unsafe fn build_spans(
    stage_kind: *const u8,
    stage_pos: *const u32,
    m: usize,
    spans: *mut Span,
    sig_kinds: *mut u8,
) -> usize {
    let sp = spans.cast::<u64>();
    let mut w = 0usize;
    for j in 0..m {
        let k = *stage_kind.add(j);
        *sp.add(w) = stage_pos.add(j).cast::<u64>().read_unaligned();
        *sig_kinds.add(w) = k;
        w += usize::from(!is_trivia_byte(k) || k == tk!(Hashbang));
    }
    w
}

#[inline(always)]
pub(super) unsafe fn emit_value(
    src: &[u8],
    out_kinds: *const u8,
    out_spans: *const Span,
    j: usize,
    lanes: &mut Lanes,
) {
    let sp = *out_spans.add(j);
    let s = sp.start as usize;
    let e = sp.end as usize;
    let k = *out_kinds.add(j);
    if k < tk!(IdentEscaped) {
        lanes.push_number_swar(src, s, e);
    } else if k == tk!(IdentEscaped) {
        lanes.push_atom(src, s, e);
    } else {
        lanes.push_atom(src, s + 1, e);
    }
}

#[cold]
pub(super) unsafe fn invalid_diags(
    src: &[u8],
    out_kinds: *const u8,
    out_spans: *const Span,
    m: usize,
    nn: u32,
    lanes: &mut Lanes,
) {
    let char_after = |cs: u32| -> u32 {
        if cs >= nn {
            return 0;
        }
        let b = src[cs as usize];
        let l: u32 = if b < 0x80 {
            1
        } else if b >= 0xF0 {
            4
        } else if b >= 0xE0 {
            3
        } else if b >= 0xC0 {
            2
        } else {
            1
        };
        l.min(nn - cs)
    };
    for j in 0..m {
        if *out_kinds.add(j) == tk!(Invalid) {
            let sp = *out_spans.add(j);
            let (s, e) = (sp.start, sp.end);
            let b0 = src[s as usize];
            if b0 == b'\\' {
                lanes.push_diag(s + 1, char_after(s + 1), DiagCode::InvalidIdentifierEscape);
            } else if b0 == b'#' {
                lanes.push_diag(s + 1, char_after(s + 1), DiagCode::UnexpectedCharacter);
            } else {
                lanes.push_diag(s, e - s, DiagCode::UnexpectedCharacter);
            }
        }
    }
}
