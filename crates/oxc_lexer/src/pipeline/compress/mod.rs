use oxc_span::Span;

use crate::{
    lanes::Lanes,
    token::{SPAN_SENTINELS, tk},
};

mod common;

#[cfg(test)]
mod tests;

crate::backend! {
    avx2 {
        mod avx2;
        use avx2::{build_spans, compress_blocks, lanes_post};
    }
    neon {
        mod neon;
        use neon::{build_spans, compress_blocks, lanes_post};
    }
    // Both vector backends read token positions, and compact kinds, out of the same LUTs.
    simd {
        mod bcompact;
        mod pair_luts;
        pub(super) use pair_luts::init_pair_luts;
    }
    generic {
        mod generic;
        use common::build_spans;
        use generic::{compress_blocks, lanes_post};
        pub(super) use generic::init_pair_luts;
    }
}

pub static SUB_NS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];
pub static STAGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

const STAGE_BLOCKS: usize = 32;
pub const STAGE_CAP: usize = STAGE_BLOCKS * 64 + 128;

#[inline(never)]
pub unsafe fn compress(
    src: &[u8],
    n: usize,
    nb: usize,
    st: *const u64,
    kind: *const u8,
    stage_pos: *mut u32,
    stage_kind: *mut u8,
    out_kinds: *mut u8,
    out_spans: *mut Span,
    lanes: &mut Lanes,
) -> usize {
    let mut c = 0usize;
    let mut w = 0usize;
    let mut b = 0usize;
    while b < nb {
        let b1 = (b + STAGE_BLOCKS).min(nb);
        let __t = std::time::Instant::now();
        c += compress_blocks(st, kind, b, b1, stage_pos.add(c), stage_kind.add(c));
        SUB_NS[0].fetch_add(__t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        b = b1;
        if c > 1 {
            STAGED.fetch_add((c - 1) as u64, std::sync::atomic::Ordering::Relaxed);
            let __t = std::time::Instant::now();
            w += build_spans(stage_kind, stage_pos, c - 1, out_spans.add(w), out_kinds.add(w));
            SUB_NS[1]
                .fetch_add(__t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            *stage_pos = *stage_pos.add(c - 1);
            *stage_kind = *stage_kind.add(c - 1);
            c = 1;
        }
    }
    if c == 1 {
        *stage_pos.add(1) = n as u32;
        w += build_spans(stage_kind, stage_pos, 1, out_spans.add(w), out_kinds.add(w));
    }
    write_sentinels(n as u32, out_spans.add(w), out_kinds.add(w));
    let __t = std::time::Instant::now();
    lanes_post(src, out_kinds, out_spans, w, n as u32, lanes);
    SUB_NS[2].fetch_add(__t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    w
}

pub unsafe fn write_sentinels(n: u32, spans: *mut Span, sig_kinds: *mut u8) {
    let eof = u64::from(n) | (u64::from(n) << 32);
    for s in 0..SPAN_SENTINELS {
        *spans.cast::<u64>().add(s) = eof;
        *sig_kinds.add(s) = tk!(Eof);
    }
}
