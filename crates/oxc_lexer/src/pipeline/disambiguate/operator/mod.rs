use crate::{
    opmap::OP_KIND_BASE,
    tables::{Tables, is_digit},
};

use super::super::{
    BCOM, BIGINT, HASHBANG, IDENT, IDENT_ESC, JEND, JSX_LT, LCOM, NUM, PRIV_IDENT, PRIV_IDENT_ESC,
    REGEX, STR, TMPL_HEAD, TMPL_MIDDLE, TMPL_NOSUB, TMPL_TAIL, WS,
    bitmap::{bm_next0, bm_next1, bm_prev1},
};
use super::{
    bm_prev_sig, ident_is, kind_at, lt_in_range, match_delim_back, walk, walk::After as WalkAfter,
};

#[cfg(test)]
mod tests;

/// Check if `p` is a position where an operator cannot go.
///
/// Operator position is directly after a complete value, e.g. after `a`, `f(x)`, or `a[0]`.
/// Anywhere else, an operator can't go, and something must start instead:
///
/// - An expression e.g. `x = /re/`, `x = <Foo />`.
/// - A statement e.g. `if (c) /re/.test(s)`, `if (c) <Foo />`.
/// - In TypeScript, a type e.g. `let f: <T>(x: T) => T`.
///
/// Returns `true` if `p` is not in operator position, `false` if it is.
///
/// Callers use this to decide:
///
/// - `/` starts a regex if `true`, or is a division operator (`/` or `/=`) if `false`.
/// - `<` may start a JSX element if `true`, or is a less-than operator if `false`.
///   In TS, a `<` in operator position can also open type arguments e.g. `f<T>()`.
///
/// The previous significant token decides most cases; `}`, `yield` / `await`, `of`, a TS `>` and
/// the TS line-break cases depend on unbounded left context and ask [`walk`](super::walk).
pub(crate) unsafe fn not_operator_position(
    t: &Tables,
    src: *const u8,
    st: *const u64,
    opch: *const u64,
    kind: *const u8,
    word: *const u64,
    n: usize,
    p: usize,
    ts: bool,
    module: bool,
) -> bool {
    let cx = walk::Cx { t, src, st, opch, kind, n, ts, kw_final: 0 };
    let mut q = bm_prev1(st, p);
    while q >= 0 {
        let qi = q as usize;
        let k = *kind.add(qi);
        if k == WS || k == LCOM || k == BCOM || k == HASHBANG {
            q = bm_prev1(st, qi);
            continue;
        }
        if k == STR {
            let e = bm_next1(st, qi + 1, n);
            if !lt_in_range(src, e, p) {
                return false;
            }
            return module_specifier_asi(src, st, kind, qi)
                || (ts && walk::after(cx, module, qi) == WalkAfter::EndsDecl);
        }
        if k == TMPL_NOSUB || k == TMPL_TAIL {
            if !(ts && lt_in_range(src, bm_next1(st, qi + 1, n), p)) {
                return false;
            }
            let from = if k == TMPL_TAIL { template_head(st, kind, qi) } else { qi };
            return walk::after_from(cx, module, qi, from) == WalkAfter::EndsDecl;
        }
        if k == REGEX || k == PRIV_IDENT || k == PRIV_IDENT_ESC || k == JEND || k == JSX_LT {
            return false;
        }
        if k == TMPL_HEAD || k == TMPL_MIDDLE {
            return true;
        }
        if k == NUM {
            // A word run that starts with a digit is a numeric literal, and a numeric literal ends
            // a value: division, unless TS ASI applies.
            let we = bm_next0(word, qi, n);
            return ts
                && lt_in_range(src, we, p)
                && walk::after(cx, module, qi) == WalkAfter::EndsDecl;
        }
        if k == IDENT || k == IDENT_ESC {
            let e = bm_next1(st, qi + 1, n);
            let newline = lt_in_range(src, e, p);
            if prop_name(src, st, kind, qi) {
                return ts && newline && walk::after(cx, module, qi) == WalkAfter::EndsDecl;
            }
            if k == IDENT && t.is_regex_keyword(src.add(qi), e - qi) {
                if ts && e - qi == 4 && ident_is(src, qi, b"void") {
                    // `x as void / 2` is division; `void /re/` is not.
                    return walk::after(cx, module, qi) != WalkAfter::Value;
                }
                if !module
                    && e - qi == 5
                    && (ident_is(src, qi, b"yield") || ident_is(src, qi, b"await"))
                {
                    return walk::after_scoped(cx, module, qi) != WalkAfter::Value;
                }
                return true;
            }
            if k == IDENT && e - qi == 2 && *src.add(qi) == b'o' && *src.add(qi + 1) == b'f' {
                return walk::after(cx, module, qi) == WalkAfter::Operand;
            }
            if newline {
                return walk::after(cx, module, qi) == WalkAfter::EndsDecl;
            }
            return false;
        }
        if k >= OP_KIND_BASE {
            let ch = *src.add(qi);
            // TS postfix non-null `!`: `x! / 2` is division, so look through the `!` unless a line
            // terminator precedes it (ASI makes it a prefix `!/re/`).
            if ts && ch == b'!' {
                let q2 = bm_prev_sig(st, kind, qi);
                if q2 < 0 || lt_in_range(src, bm_next1(st, q2 as usize + 1, n), qi) {
                    return true;
                }
                q = q2;
                continue;
            }
            if ch == b'.' {
                // Only a trailing-dot numeric literal (`1./2`) makes `.` end a value; member
                // access, `...` and `?.` all precede an operand.
                let q2 = bm_prev1(st, qi);
                return !(q2 >= 0
                    && *kind.add(q2 as usize) == NUM
                    && bm_next1(st, q2 as usize + 1, n) == qi);
            }
            if ch == b'+' || ch == b'-' {
                // Tail of `++`/`--`: postfix ends a value, prefix does not; a run of three or more
                // always ends in a prefix pair or a lone sign.
                if qi > 0 && *src.add(qi - 1) == ch {
                    return !incdec_is_postfix(t, src, st, kind, n, qi - 1);
                }
                return true;
            }
            if ch == b'}' {
                let from = match_delim_back(src, st, kind, qi, b'{', b'}').unwrap_or(qi);
                return walk::after_from(cx, module, qi, from) != WalkAfter::Value;
            }
            if ch == b')' {
                if paren_close_is_regex(src, st, kind, qi) {
                    return true;
                }
                if !(ts && lt_in_range(src, qi + 1, p)) {
                    return false;
                }
                let from = match_delim_back(src, st, kind, qi, b'(', b')').unwrap_or(qi);
                return walk::after_from(cx, module, qi, from) == WalkAfter::EndsDecl;
            }
            if ts && ch == b'>' && !(qi > 0 && *src.add(qi - 1) == b'=') {
                return walk::after(cx, module, qi) != WalkAfter::Value;
            }
            if ch == b']' {
                if !(ts && lt_in_range(src, qi + 1, p)) {
                    return false;
                }
                let from = match_delim_back(src, st, kind, qi, b'[', b']').unwrap_or(qi);
                return walk::after_from(cx, module, qi, from) == WalkAfter::EndsDecl;
            }
            return true;
        }
        return true;
    }
    true
}

/// Is the word at `pos` a property name: does a lone `.` (member access or `?.`, not the last dot
/// of `...`) precede it as a token? Trivia in between does not matter: `x. return / 2` divides.
#[inline]
unsafe fn prop_name(src: *const u8, st: *const u64, kind: *const u8, pos: usize) -> bool {
    let q = bm_prev_sig(st, kind, pos);
    if q < 0 {
        return false;
    }
    let w = q as usize;
    if *kind.add(w) < OP_KIND_BASE {
        return false;
    }
    let c = *src.add(w);
    if c == b'?' {
        // A `?.` that `coalesce` has already fused starts at the `?`.
        return *src.add(w + 1) == b'.' && !is_digit(*src.add(w + 2));
    }
    // A lone `.`: not a `...`, whether the run is still three token starts (the previous token is
    // then its last dot) or already fused (its first).
    c == b'.'
        && !(*src.add(w + 1) == b'.' && *src.add(w + 2) == b'.')
        && !(w >= 2 && *src.add(w - 1) == b'.' && *src.add(w - 2) == b'.')
}

/// The TemplateHead that opened the template whose TemplateTail is at `tail` (nested templates are
/// crossed); `tail` itself if unbalanced.
unsafe fn template_head(st: *const u64, kind: *const u8, tail: usize) -> usize {
    let mut depth = 0u32;
    let mut q = bm_prev_sig(st, kind, tail);
    while q >= 0 {
        let w = q as usize;
        match *kind.add(w) {
            TMPL_TAIL => depth += 1,
            TMPL_HEAD => {
                if depth == 0 {
                    return w;
                }
                depth -= 1;
            }
            _ => {}
        }
        q = bm_prev_sig(st, kind, w);
    }
    tail
}

#[inline]
unsafe fn module_specifier_asi(
    src: *const u8,
    st: *const u64,
    kind: *const u8,
    spec: usize,
) -> bool {
    let q = bm_prev_sig(st, kind, spec);
    if q < 0 {
        return false;
    }
    let w = q as usize;
    *kind.add(w) == IDENT
        && !prop_name(src, st, kind, w)
        && (ident_is(src, w, b"from") || ident_is(src, w, b"import"))
}

/// Is the `++`/`--` whose first byte is at `first` postfix? It is when a value ends right before it
/// on the same line; `tail_before` is the shared definition of "ends a value".
unsafe fn incdec_is_postfix(
    t: &Tables,
    src: *const u8,
    st: *const u64,
    kind: *const u8,
    n: usize,
    first: usize,
) -> bool {
    let q = bm_prev_sig(st, kind, first);
    q >= 0
        && tail_before(t, src, st, kind, n, first)
        && !lt_in_range(src, bm_next1(st, q as usize + 1, n), first)
}

unsafe fn tail_before(
    t: &Tables,
    src: *const u8,
    st: *const u64,
    kind: *const u8,
    n: usize,
    pos: usize,
) -> bool {
    let mut s = bm_prev_sig(st, kind, pos);
    loop {
        if s < 0 {
            return false;
        }
        let w = s as usize;
        if kind_at(kind, w) >= OP_KIND_BASE && *src.add(w) == b'!' && *src.add(w + 1) != b'=' {
            s = bm_prev_sig(st, kind, w);
            continue;
        }
        break;
    }
    if s < 0 {
        return false;
    }
    let sp = s as usize;
    let sk = kind_at(kind, sp);
    if sk >= OP_KIND_BASE {
        return matches!(*src.add(sp), b')' | b']');
    }
    if sk == IDENT {
        let e = bm_next1(st, sp + 1, n);
        return prop_name(src, st, kind, sp) || !t.is_regex_keyword(src.add(sp), e - sp);
    }
    matches!(sk, NUM | BIGINT | STR | TMPL_NOSUB | TMPL_TAIL | REGEX | PRIV_IDENT)
}

unsafe fn paren_close_is_regex(src: *const u8, st: *const u64, kind: *const u8, qi: usize) -> bool {
    let Some(lp) = match_delim_back(src, st, kind, qi, b'(', b')') else {
        return false;
    };
    let q = bm_prev_sig(st, kind, lp);
    if q < 0 {
        return false;
    }
    let mut w = q as usize;
    if *kind.add(w) != IDENT {
        return false;
    }
    if ident_is(src, w, b"await") {
        let q2 = bm_prev_sig(st, kind, w);
        if q2 < 0 || *kind.add(q2 as usize) != IDENT {
            return false;
        }
        w = q2 as usize;
        return !prop_name(src, st, kind, w) && ident_is(src, w, b"for");
    }
    !prop_name(src, st, kind, w)
        && (ident_is(src, w, b"if")
            || ident_is(src, w, b"while")
            || ident_is(src, w, b"for")
            || ident_is(src, w, b"with"))
}
