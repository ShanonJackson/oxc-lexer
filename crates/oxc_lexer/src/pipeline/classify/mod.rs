crate::backend! {
    avx2    { mod avx2;    use avx2::classify_impl; }
    neon    { mod neon;    use neon::classify_impl; }
    generic { mod generic; use generic::classify_impl; }

    // The nibble tables both vector backends look bytes up in. Their own tests check them
    // against the scalar predicates, so a scalar build still compiles them under `cfg(test)`.
    simd_or_test { mod luts; mod punct1_luts; }

    // `PUNCT1` is where the scalar backend reads single-byte punctuator kinds from, and the
    // oracle `punct1_luts`' test checks the vector backends' perfect hash against.
    generic_or_test { mod punct1; }
}

#[inline(never)]
pub(super) unsafe fn classify(
    ts: bool,
    src: *const u8,
    n: usize,
    nb: usize,
    word: *mut u64,
    st: *mut u64,
    kwinit: *mut u64,
    opch: *mut u64,
    digit: *mut u64,
    dot: *mut u64,
    misc: *mut u64,
    kind: *mut u8,
) {
    classify_impl(ts, src, n, word, st, kwinit, opch, digit, dot, misc, kind);

    *word.add(nb) = 0;
    *st.add(nb) = 0;
    *kwinit.add(nb) = 0;
    *opch.add(nb) = 0;
    *digit.add(nb) = 0;
    *dot.add(nb) = 0;
    *misc.add(nb) = 0;
}
