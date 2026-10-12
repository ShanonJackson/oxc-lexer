crate::backend! {
    avx2    { mod avx2;    pub(super) use avx2::{scan_block_comment, scan_line_comment}; }
    neon    { mod neon;    pub(super) use neon::{scan_block_comment, scan_line_comment}; }
    generic { mod generic; pub(super) use generic::{scan_block_comment, scan_line_comment}; }
}

mod common;
pub(super) use common::{scan_ident_esc, scan_number, scan_quoted, scan_regex, scan_tmpl_text};
