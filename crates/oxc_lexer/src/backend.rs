//! Compile-time selection of the lexer's SIMD backend.
//!
//! The per-byte kernels - `classify`, `compress`, `find` and `scan` - each ship in three forms,
//! and a build compiles exactly one of them:
//!
//! | Backend   | Selected when                            | Block worked in     |
//! |-----------|------------------------------------------|---------------------|
//! | `avx2`    | `x86_64` with `avx2` + `bmi2` + `popcnt` | 2 x 256-bit vectors |
//! | `neon`    | `aarch64`, always                        | 4 x 128-bit vectors |
//! | `generic` | everywhere else                          | 8 x 64-bit SWAR     |
//!
//! All three are bit-exact in their output, and all three are tested: see the `*-lexer`,
//! `*-lexer-simd` and `*-lexer-generic` recipes in the justfile.
//!
//! Neither `avx2`, `bmi2` nor `popcnt` is in the `x86_64` baseline on any platform, so the AVX2
//! core has to be asked for with explicit `-C target-feature` flags (`.cargo/lexer-simd.toml`).
//! NEON, by contrast, is mandatory in the AArch64 baseline - `target_feature = "neon"` is on for
//! every `aarch64-*` target - so the NEON core needs no flags and is simply what `aarch64` builds.
//!
//! # Adding to a stage
//!
//! The predicates are written here and nowhere else. A stage names the backends and lets
//! [`backend!`] attach the matching `#[cfg]` to each arm's items:
//!
//! ```ignore
//! crate::backend! {
//!     avx2    { mod avx2;    use avx2::classify_impl; }
//!     neon    { mod neon;    use neon::classify_impl; }
//!     generic { mod generic; use generic::classify_impl; }
//! }
//! ```
//!
//! Every arm is optional. Besides the three backends there are three unions of them, for the
//! tables and helpers that don't line up one-to-one with a backend:
//!
//! * `simd` - both vector backends, for what they share and the scalar one has no use for.
//! * `simd_or_test` - and in test builds, for a shared table whose own tests check it against
//!   the scalar definitions.
//! * `generic_or_test` - the scalar backend, and in test builds, for a scalar table that the
//!   vector backends replace with a LUT but still check themselves against.
//!
//! Keeping the predicates in one place is also what makes the move to runtime dispatch a local
//! change: the stage modules name backends, not target features, so they don't need to change
//! when the choice between them stops being made at compile time.

/// Attach the `#[cfg]` for each backend to that arm's items. See [the module docs](self).
macro_rules! backend {
    (
        $(avx2            { $($avx2:item)*    })?
        $(neon            { $($neon:item)*    })?
        $(simd            { $($simd:item)*    })?
        $(generic         { $($generic:item)* })?
        $(simd_or_test    { $($st:item)*      })?
        $(generic_or_test { $($gt:item)*      })?
    ) => {
        $($(
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx2",
                target_feature = "bmi2",
                target_feature = "popcnt"
            ))]
            $avx2
        )*)?
        $($(
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            $neon
        )*)?
        $($(
            #[cfg(any(
                all(
                    target_arch = "x86_64",
                    target_feature = "avx2",
                    target_feature = "bmi2",
                    target_feature = "popcnt"
                ),
                all(target_arch = "aarch64", target_feature = "neon"),
            ))]
            $simd
        )*)?
        $($(
            #[cfg(not(any(
                all(
                    target_arch = "x86_64",
                    target_feature = "avx2",
                    target_feature = "bmi2",
                    target_feature = "popcnt"
                ),
                all(target_arch = "aarch64", target_feature = "neon"),
            )))]
            $generic
        )*)?
        $($(
            #[cfg(any(
                test,
                all(
                    target_arch = "x86_64",
                    target_feature = "avx2",
                    target_feature = "bmi2",
                    target_feature = "popcnt"
                ),
                all(target_arch = "aarch64", target_feature = "neon"),
            ))]
            $st
        )*)?
        $($(
            #[cfg(any(
                test,
                not(any(
                    all(
                        target_arch = "x86_64",
                        target_feature = "avx2",
                        target_feature = "bmi2",
                        target_feature = "popcnt"
                    ),
                    all(target_arch = "aarch64", target_feature = "neon"),
                ))
            ))]
            $gt
        )*)?
    };
}
pub(crate) use backend;

/// Name of the backend this build compiled, one of `avx2`, `neon` or `scalar`.
///
/// Printed by the benchmark and conformance harnesses, so a run reports which of the three it
/// actually exercised rather than which one it meant to.
pub const BACKEND: &str = {
    backend! {
        avx2    { const NAME: &str = "avx2"; }
        neon    { const NAME: &str = "neon"; }
        generic { const NAME: &str = "scalar"; }
    }
    NAME
};

/// `true` if a SIMD core is compiled in, `false` if the scalar fallback is.
///
/// Used by CI to ensure it's testing the implementation it thinks it is.
/// [`BACKEND`] names which of the two SIMD cores it got.
pub const IS_SIMD: bool = {
    backend! {
        simd    { const SIMD: bool = true; }
        generic { const SIMD: bool = false; }
    }
    SIMD
};
