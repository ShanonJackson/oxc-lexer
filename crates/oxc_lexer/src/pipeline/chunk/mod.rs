//! The vector primitives the stages share: load a chunk, compare it against a byte, and gather
//! the result into a bitmask. One module per backend, selected by [`mod@crate::backend`].

crate::backend! {
    avx2    { mod avx2;    pub(super) use avx2::*; }
    neon    { mod neon;    pub(super) use neon::*; }
    generic { mod generic; pub(super) use generic::*; }
}
