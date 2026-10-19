#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
mod sse2;
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
use sse2::define_find_function_sse2 as define_find_function;
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
pub(super) use sse2::{
    bracket_bits_sse2 as bracket_bits, find1_sse2 as find1, find2_sse2 as find2,
    find3_sse2 as find3, find4_sse2 as find4,
};

#[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
mod generic;
#[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
use generic::define_find_function_generic as define_find_function;
#[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
pub(super) use generic::{
    bracket_bits_generic as bracket_bits, find1_generic as find1, find2_generic as find2,
    find3_generic as find3, find4_generic as find4,
};

mod common;
pub(super) use common::{
    find_jsx_tag, find_jsx_text, find_line_terminator, find_opener, find_opener_jsx5,
    find_opener_jsx7, find_opener6, find_regex, find_tmpl, unicode_ws_len,
};
