#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
mod sse2;
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
pub(super) use sse2::{
    scan_block_comment_sse2 as scan_block_comment, scan_line_comment_sse2 as scan_line_comment,
};

#[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
mod generic;
#[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
pub(super) use generic::{
    scan_block_comment_generic as scan_block_comment,
    scan_line_comment_generic as scan_line_comment,
};

mod common;
pub(super) use common::{scan_ident_esc, scan_number, scan_quoted, scan_regex, scan_tmpl_text};
