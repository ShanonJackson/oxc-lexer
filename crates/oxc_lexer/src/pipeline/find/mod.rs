#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
mod sse2;
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
pub(super) use sse2::{bracket_bits, find};

#[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
mod generic;
#[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
pub(super) use generic::{bracket_bits, find};

mod common;
pub(super) use common::{
    JSX_TAG, JSX_TEXT, OPENER, OPENER_JSX5, OPENER_JSX7, OPENER6, REGEX, TMPL,
    find_line_terminator, unicode_ws_len,
};
