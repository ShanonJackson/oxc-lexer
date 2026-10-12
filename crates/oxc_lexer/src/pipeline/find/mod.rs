crate::backend! {
    avx2 {
        mod avx2;
        use avx2::define_find_function;
        pub(super) use avx2::{bracket_bits, find1, find2, find3, find4};
    }
    neon {
        mod neon;
        use neon::define_find_function;
        pub(super) use neon::{bracket_bits, find1, find2, find3, find4};
    }
    generic {
        mod generic;
        use generic::define_find_function;
        pub(super) use generic::{bracket_bits, find1, find2, find3, find4};
    }
}

mod common;
pub(super) use common::{
    find_jsx_tag, find_jsx_text, find_line_terminator, find_opener, find_opener_jsx5,
    find_opener_jsx7, find_opener6, find_regex, find_tmpl, unicode_ws_len,
};
