use core::cell::{Cell, RefCell};

use crate::{
    opmap::OP_KIND_BASE,
    tables::{Tables, is_digit, is_id_start, is_word, is_ws},
    token::{KW_BASE, KW_MAX, TokenKind},
};

use super::{
    BCOM, BIGINT, HASHBANG, IDENT, IDENT_ESC, LCOM, NUM, PRIV_IDENT_ESC, STR, TMPL_HEAD,
    TMPL_MIDDLE, TMPL_NOSUB, TMPL_TAIL, WS,
    bitmap::{bm_get, bm_next1, bm_prev1},
    find::{find_line_terminator, unicode_ws_len},
    scan::scan_block_comment,
};

mod operator;
mod walk;
pub(super) use operator::not_operator_position;

#[cfg(test)]
mod tests;

#[inline(always)]
unsafe fn kind_at(kind: *const u8, w: usize) -> u8 {
    let k = *kind.add(w);
    if k >= KW_BASE && k <= KW_MAX {
        return IDENT;
    }
    if k == IDENT_ESC || k == PRIV_IDENT_ESC {
        return k & !(IDENT_ESC ^ IDENT);
    }
    k
}

#[inline]
unsafe fn trivia_at(src: *const u8, i: usize) -> Option<(bool, usize)> {
    let b = *src.add(i);
    if b < 0x80 {
        return None;
    }
    let b1 = *src.add(i + 1);
    let b2 = *src.add(i + 2);
    match b {
        0xc2 if b1 == 0xa0 || b1 == 0x85 => Some((false, 2)),
        0xe1 if b1 == 0x9a && b2 == 0x80 => Some((false, 3)),
        0xe2 if b1 == 0x80 && ((0x80..=0x8b).contains(&b2) || b2 == 0xaf) => Some((false, 3)),
        0xe2 if b1 == 0x80 && (b2 == 0xa8 || b2 == 0xa9) => Some((true, 3)),
        0xe2 if b1 == 0x81 && b2 == 0x9f => Some((false, 3)),
        0xe3 if b1 == 0x80 && b2 == 0x80 => Some((false, 3)),
        0xef if b1 == 0xbb && b2 == 0xbf => Some((false, 3)),
        _ => None,
    }
}

/// Distance cap (in token starts) for the backward delimiter matches below;
/// past it we fall back to the safe legacy "`}` means regex" answer. Only
/// pathological input gets near it.
const BRACE_MATCH_CAP: u32 = 1024;

/// Previous significant token start before `pos` (skipping trivia), or -1 at
/// start of input.
#[inline]
pub(super) unsafe fn bm_prev_sig(st: *const u64, kind: *const u8, pos: usize) -> i64 {
    let mut q = bm_prev1(st, pos);
    while q >= 0 {
        let k = *kind.add(q as usize);
        if k == WS || k == LCOM || k == BCOM || k == HASHBANG {
            q = bm_prev1(st, q as usize);
            continue;
        }
        break;
    }
    q
}

unsafe fn lt_in_range(src: *const u8, a: usize, b: usize) -> bool {
    let mut i = a;
    while i < b {
        let c = *src.add(i);
        if c == b'\n' || c == b'\r' {
            return true;
        }
        if c == 0xe2
            && *src.add(i + 1) == 0x80
            && (*src.add(i + 2) == 0xa8 || *src.add(i + 2) == 0xa9)
        {
            return true;
        }
        i += 1;
    }
    false
}

/// Does the identifier at `pos` equal exactly `kw`? The following-byte check
/// rejects longer identifiers (the source pad makes it safe at EOF).
#[inline]
unsafe fn ident_is(src: *const u8, pos: usize, kw: &[u8]) -> bool {
    let mut i = 0;
    while i < kw.len() {
        if *src.add(pos + i) != kw[i] {
            return false;
        }
        i += 1;
    }
    let after = pos + kw.len();
    !is_word(*src.add(after)) || trivia_at(src, after).is_some()
}

struct DelimMemo {
    generation: u64,
    upto: usize,
    pairs: Vec<(u32, u32)>,
    open: [Vec<u32>; 3],
}

thread_local! {
    static MEMO_GEN: Cell<u64> = const { Cell::new(0) };
    static DELIM_MEMO: RefCell<DelimMemo> = const {
        RefCell::new(DelimMemo {
            generation: 0,
            upto: 0,
            pairs: Vec::new(),
            open: [Vec::new(), Vec::new(), Vec::new()],
        })
    };
}

pub(super) fn memo_new_lex() {
    MEMO_GEN.with(|g| g.set(g.get().wrapping_add(1)));
    walk::new_lex();
}

/// `coalesce` is about to ask about `>` runs: its token state differs from carve's, so the context
/// walk restarts.
pub(super) fn walk_new_pass() {
    walk::new_pass();
}

fn delim_slot(c: u8) -> usize {
    match c {
        b'(' | b')' => 0,
        b'[' | b']' => 1,
        _ => 2,
    }
}

#[inline(never)]
unsafe fn delim_memo_opener(
    src: *const u8,
    st: *const u64,
    kind: *const u8,
    from: usize,
) -> Option<usize> {
    let generation = MEMO_GEN.with(Cell::get);
    DELIM_MEMO.with(|cell| {
        let mut m = cell.borrow_mut();
        if m.generation != generation {
            m.generation = generation;
            m.upto = 0;
            m.pairs.clear();
            for stack in &mut m.open {
                stack.clear();
            }
        }
        let mut w = bm_next1(st, m.upto, from + 1);
        while w <= from {
            if *kind.add(w) >= OP_KIND_BASE {
                let c = *src.add(w);
                match c {
                    b'(' | b'[' | b'{' => m.open[delim_slot(c)].push(w as u32),
                    b')' | b']' | b'}' => {
                        if let Some(o) = m.open[delim_slot(c)].pop() {
                            m.pairs.push((w as u32, o));
                        }
                    }
                    _ => {}
                }
            }
            w = bm_next1(st, w + 1, from + 1);
        }
        m.upto = from + 1;
        match m.pairs.binary_search_by_key(&(from as u32), |pr| pr.0) {
            Ok(i) => Some(m.pairs[i].1 as usize),
            Err(_) => None,
        }
    })
}

/// Match the close punctuator at `from` back to its opener, counting only
/// punctuator delimiters — template-closing `}`s and cleared literal
/// interiors are invisible. Past the cap the answer comes from a per-file
/// closer-to-opener table built once, so it is exact; None if unbalanced.
#[inline]
unsafe fn match_delim_back(
    src: *const u8,
    st: *const u64,
    kind: *const u8,
    from: usize,
    open: u8,
    close: u8,
) -> Option<usize> {
    let mut depth: i32 = 1;
    let mut steps: u32 = 0;
    let mut q = bm_prev1(st, from);
    while q >= 0 {
        steps += 1;
        if steps > BRACE_MATCH_CAP {
            return delim_memo_opener(src, st, kind, from);
        }
        let pos = q as usize;
        if *kind.add(pos) >= OP_KIND_BASE {
            let c = *src.add(pos);
            if c == close {
                depth += 1;
            } else if c == open {
                depth -= 1;
                if depth == 0 {
                    return Some(pos);
                }
            }
        }
        q = bm_prev1(st, pos);
    }
    None
}

/// Cap for scanning `>` runs: matching `<` is within 100 bytes
/// Hitting the cap returns `None` (fuse), so it can only widen the residual, never split a shift.
const GT_SCAN_CAP: usize = 1 << 16;

/// Bytes `gt_run_closes_type_args` reacts to. Everything else is skipped without touching a bitmap.
static GT_SCAN_DELIM: [bool; 256] = {
    let mut t = [false; 256];
    t[b'<' as usize] = true;
    t[b'>' as usize] = true;
    t[b'(' as usize] = true;
    t[b')' as usize] = true;
    t[b'[' as usize] = true;
    t[b']' as usize] = true;
    t[b'{' as usize] = true;
    t[b'}' as usize] = true;
    t[b';' as usize] = true;
    t
};

#[inline(always)]
unsafe fn word_len(src: *const u8, w: usize) -> usize {
    let mut e = w + 1;
    while is_word(*src.add(e)) {
        e += 1;
    }
    e - w
}

unsafe fn word_is_any(src: *const u8, w: usize, words: &[&[u8]]) -> bool {
    let len = word_len(src, w);
    let first = *src.add(w);
    let mut i = 0;
    while i < words.len() {
        let kw = words[i];
        if kw.len() == len && kw[0] == first && ident_is(src, w, kw) {
            return true;
        }
        i += 1;
    }
    false
}

const fn kw(k: TokenKind) -> u8 {
    k as u8
}

const KW_EXTENDS: u8 = kw(TokenKind::KwExtends);
const KW_IS: u8 = kw(TokenKind::KwIs);
const KW_IN: u8 = kw(TokenKind::KwIn);
const KW_KEYOF: u8 = kw(TokenKind::KwKeyof);
const KW_TYPEOF: u8 = kw(TokenKind::KwTypeof);
const KW_READONLY: u8 = kw(TokenKind::KwReadonly);
const KW_UNIQUE: u8 = kw(TokenKind::KwUnique);
const KW_INFER: u8 = kw(TokenKind::KwInfer);
const KW_ABSTRACT: u8 = kw(TokenKind::KwAbstract);
const KW_NEW: u8 = kw(TokenKind::KwNew);
const KW_ASSERTS: u8 = kw(TokenKind::KwAsserts);
const KW_IMPORT: u8 = kw(TokenKind::KwImport);
const KW_AS: u8 = kw(TokenKind::KwAs);
const KW_AWAIT: u8 = kw(TokenKind::KwAwait);
const KW_YIELD: u8 = kw(TokenKind::KwYield);
const KW_DELETE: u8 = kw(TokenKind::KwDelete);
const KW_FUNCTION: u8 = kw(TokenKind::KwFunction);
const KW_CLASS: u8 = kw(TokenKind::KwClass);
const KW_INSTANCEOF: u8 = kw(TokenKind::KwInstanceof);
const KW_SUPER: u8 = kw(TokenKind::KwSuper);
const KW_THIS: u8 = kw(TokenKind::KwThis);
const KW_SWITCH: u8 = kw(TokenKind::KwSwitch);
const KW_CASE: u8 = kw(TokenKind::KwCase);
const KW_RETURN: u8 = kw(TokenKind::KwReturn);
const KW_THROW: u8 = kw(TokenKind::KwThrow);
const KW_VAR: u8 = kw(TokenKind::KwVar);
const KW_LET: u8 = kw(TokenKind::KwLet);
const KW_CONST: u8 = kw(TokenKind::KwConst);
const KW_IF: u8 = kw(TokenKind::KwIf);
const KW_ELSE: u8 = kw(TokenKind::KwElse);
const KW_FOR: u8 = kw(TokenKind::KwFor);
const KW_WHILE: u8 = kw(TokenKind::KwWhile);
const KW_DO: u8 = kw(TokenKind::KwDo);
const KW_BREAK: u8 = kw(TokenKind::KwBreak);
const KW_CONTINUE: u8 = kw(TokenKind::KwContinue);
const KW_WITH: u8 = kw(TokenKind::KwWith);
const KW_TRY: u8 = kw(TokenKind::KwTry);
const KW_CATCH: u8 = kw(TokenKind::KwCatch);
const KW_FINALLY: u8 = kw(TokenKind::KwFinally);
const KW_DEBUGGER: u8 = kw(TokenKind::KwDebugger);
const KW_DEFAULT: u8 = kw(TokenKind::KwDefault);
const KW_EXPORT: u8 = kw(TokenKind::KwExport);
const KW_ENUM: u8 = kw(TokenKind::KwEnum);

#[inline(always)]
fn type_illegal_kind(k: u8) -> bool {
    matches!(
        k,
        KW_AWAIT
            | KW_YIELD
            | KW_DELETE
            | KW_FUNCTION
            | KW_CLASS
            | KW_INSTANCEOF
            | KW_SUPER
            | KW_SWITCH
            | KW_CASE
            | KW_RETURN
            | KW_THROW
            | KW_VAR
            | KW_LET
            | KW_CONST
            | KW_IF
            | KW_ELSE
            | KW_FOR
            | KW_WHILE
            | KW_DO
            | KW_BREAK
            | KW_CONTINUE
            | KW_WITH
            | KW_TRY
            | KW_CATCH
            | KW_FINALLY
            | KW_DEBUGGER
            | KW_DEFAULT
            | KW_EXPORT
            | KW_ENUM
    )
}

#[inline(always)]
fn type_prefix_kind(k: u8) -> bool {
    matches!(
        k,
        KW_KEYOF
            | KW_TYPEOF
            | KW_READONLY
            | KW_UNIQUE
            | KW_INFER
            | KW_ABSTRACT
            | KW_NEW
            | KW_ASSERTS
            | KW_IMPORT
            | KW_EXTENDS
            | KW_IS
            | KW_IN
            | KW_AS
    )
}

unsafe fn type_list_legal(
    t: &Tables,
    src: *const u8,
    st: *const u64,
    kind: *const u8,
    lo: usize,
    hi: usize,
) -> bool {
    let mut start = true;
    let mut brc: i32 = 0;
    let mut brk: i32 = 0;
    let mut angle_bits: u64 = 0;
    let mut angle_depth: u32 = 0;
    let mut paren_ok = false;
    let mut cond_ok = false;
    let mut par: i32 = 0;
    let mut this_head = false;
    let mut skip = usize::MAX;
    let mut w = bm_next1(st, lo, hi);
    while w < hi {
        let mut k = kind_at(kind, w);
        if w == skip || k == WS || k == LCOM || k == BCOM {
            w = bm_next1(st, w + 1, hi);
            continue;
        }
        // Text carve has not reached yet: a raw comment is trivia, a raw string or template is a
        // literal type.
        let j = skip_raw_literal(src, kind, hi, w);
        if j != w {
            let c0 = *src.add(w);
            if c0 == b'/' {
                w = bm_next1(st, j, hi);
                continue;
            }
            k = if c0 == b'`' { TMPL_NOSUB } else { STR };
            if !start && brc == 0 {
                return false;
            }
            start = false;
            paren_ok = false;
            w = bm_next1(st, j, hi);
            continue;
        }
        let c = *src.add(w);
        let mut ok_paren = false;
        let was_this = this_head;
        this_head = false;
        if k == IDENT || k == IDENT_ESC {
            let kk = t.kwts.lookup(src.add(w), word_len(src, w)) as u8;
            this_head = kk == KW_THIS;
            if !start && brc == 0 && !matches!(kk, KW_EXTENDS | KW_IS | KW_IN) {
                return false;
            }
            if type_illegal_kind(kk) {
                return false;
            }
            if kk == KW_EXTENDS {
                cond_ok = true;
            }
            start = type_prefix_kind(kk);
        } else if k == NUM || k == BIGINT || k == STR || k == TMPL_NOSUB || k == TMPL_TAIL {
            if !start && brc == 0 && k != TMPL_TAIL {
                return false;
            }
            start = false;
        } else if k == TMPL_HEAD || k == TMPL_MIDDLE {
            if k == TMPL_HEAD && !start && brc == 0 {
                return false;
            }
            start = true;
        } else if k >= OP_KIND_BASE {
            match c {
                b'(' => {
                    if !start && brc == 0 && !paren_ok {
                        return false;
                    }
                    par += 1;
                    start = true;
                }
                b')' => {
                    par -= 1;
                    start = false;
                }
                b']' => {
                    brk -= 1;
                    start = false;
                }
                b'[' => {
                    brk += 1;
                    start = true;
                }
                b'{' => {
                    if !start && brc == 0 {
                        return false;
                    }
                    brc += 1;
                    start = true;
                }
                b'}' => {
                    brc -= 1;
                    start = false;
                }
                b'<' => {
                    let nx = *src.add(w + 1);
                    if was_this || nx == b'=' || (nx == b'<' && !bm_get(st, w + 1)) {
                        return false;
                    }
                    angle_bits = (angle_bits << 1) | u64::from(start);
                    angle_depth += 1;
                    start = true;
                }
                b'>' => {
                    let nx = *src.add(w + 1);
                    if (nx == b'=' || nx == b'>') && !bm_get(st, w + 1) {
                        return false;
                    }
                    if angle_depth > 0 {
                        ok_paren = angle_bits & 1 != 0;
                        angle_bits >>= 1;
                        angle_depth -= 1;
                    }
                    start = false;
                }
                b'=' => {
                    // `=>` of a function type, or a type-parameter default (the walk also reads
                    // member type-parameter lists here).
                    if *src.add(w + 1) == b'>' && bm_get(st, w + 1) {
                        skip = w + 1;
                    }
                    start = true;
                }
                b':' | b'.' => start = true,
                b',' => {
                    if angle_depth == 0 && brc == 0 && brk == 0 && par == 0 {
                        cond_ok = false;
                    }
                    start = true;
                }
                b'|' | b'&' => {
                    if *src.add(w + 1) == c {
                        return false;
                    }
                    start = true;
                }
                b'?' => {
                    let nx = *src.add(w + 1);
                    if nx == b'.' || nx == b'?' {
                        return false;
                    }
                    if brc == 0 && brk == 0 {
                        let optional = par > 0
                            && matches!(*src.add(skip_ws_fwd(src, w + 1, hi)), b':' | b',' | b')');
                        if !optional {
                            if !cond_ok {
                                return false;
                            }
                            cond_ok = false;
                        }
                    }
                    start = true;
                }
                b';' => {
                    if brc == 0 {
                        return false;
                    }
                    start = true;
                }
                b'-' => {
                    if !start && brc == 0 {
                        return false;
                    }
                    start = true;
                }
                b'+' => {
                    if brc == 0 {
                        return false;
                    }
                    start = true;
                }
                _ => return false,
            }
        } else {
            return false;
        }
        paren_ok = ok_paren;
        w = bm_next1(st, w + 1, hi);
    }
    true
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Follow {
    Split,
    Fuse,
    Ctx,
}

const FOLLOW_SPLIT_WORDS: &[&[u8]] =
    &[b"in", b"instanceof", b"as", b"satisfies", b"extends", b"implements"];

unsafe fn gt_follower(src: *const u8, n: usize, mut i: usize) -> Follow {
    let mut broke = false;
    loop {
        if i >= n {
            return Follow::Split;
        }
        let c = *src.add(i);
        match c {
            b' ' | b'\t' | 0x0b | 0x0c => {
                i += 1;
                continue;
            }
            b'\n' | b'\r' => {
                broke = true;
                i += 1;
                continue;
            }
            b'/' => {
                let d = *src.add(i + 1);
                if d == b'/' {
                    broke = true;
                    i = find_line_terminator(src, n, i + 2);
                    continue;
                }
                if d != b'*' {
                    return Follow::Split;
                }
                let e = scan_block_comment(src, n, i + 2).0;
                if e >= n {
                    return Follow::Split;
                }
                if lt_in_range(src, i + 2, e) {
                    broke = true;
                }
                i = e + 1;
                continue;
            }
            _ => {}
        }
        if c >= 0x80 {
            if c == 0xe2
                && *src.add(i + 1) == 0x80
                && (*src.add(i + 2) == 0xa8 || *src.add(i + 2) == 0xa9)
            {
                broke = true;
                i += 3;
                continue;
            }
            let wl = unicode_ws_len(src, i);
            if wl != 0 {
                i += wl;
                continue;
            }
            return if broke { Follow::Split } else { Follow::Fuse };
        }
        let nx = *src.add(i + 1);
        if broke {
            return match c {
                b'<' if nx != b'<' && nx != b'=' => Follow::Ctx,
                b'+' | b'-' if nx != b'=' && nx != c => Follow::Ctx,
                b'>' => Follow::Ctx,
                _ => Follow::Split,
            };
        }
        return match c {
            b'(' | b'`' | b'=' | b')' | b']' | b'}' | b',' | b';' | b':' | b'?' | b'|' | b'&'
            | b'*' | b'%' | b'^' => Follow::Split,
            b'!' => {
                if nx == b'=' {
                    Follow::Split
                } else {
                    Follow::Fuse
                }
            }
            b'.' => {
                if is_digit(nx) {
                    Follow::Fuse
                } else {
                    Follow::Split
                }
            }
            b'{' | b'[' | b'>' => Follow::Ctx,
            b'+' | b'-' => {
                if nx == b'=' {
                    Follow::Split
                } else {
                    Follow::Fuse
                }
            }
            b'<' => {
                if nx == b'<' || nx == b'=' {
                    Follow::Split
                } else {
                    Follow::Fuse
                }
            }
            b'~' | b'@' | b'#' | b'"' | b'\'' => Follow::Fuse,
            _ => {
                if is_digit(c) {
                    Follow::Fuse
                } else if is_id_start(c) {
                    if word_is_any(src, i, FOLLOW_SPLIT_WORDS) {
                        Follow::Split
                    } else {
                        Follow::Fuse
                    }
                } else {
                    Follow::Split
                }
            }
        };
    }
}

/// Forward angle match: from `i` at `depth`, the `>` that brings it to 0, or
/// `None` on an unmatched closer, a `;` outside every bracket, or the cap.
/// Same gating as [`gt_run_closes_type_args`] — `opch & st` for angles, `st`
/// for the bracket counters — and the same balance requirement at the close.
unsafe fn angle_close_fwd(
    src: *const u8,
    st: *const u64,
    opch: *const u64,
    kind: *const u8,
    i: usize,
    lim: usize,
    depth: i32,
) -> Option<usize> {
    angle_close_fwd_capped(src, st, opch, kind, i, lim, depth).0
}

/// If a comment, string or template carve has not reached yet starts at the token start `i`, the
/// position just past it; else `i`. Forward scans from before the carve cursor cross raw text;
/// carved literals have cleared interiors and are never re-entered.
#[inline]
unsafe fn skip_raw_literal(src: *const u8, kind: *const u8, n: usize, i: usize) -> usize {
    if *kind.add(i) < OP_KIND_BASE {
        return i;
    }
    let c = *src.add(i);
    match c {
        b'/' => match *src.add(i + 1) {
            b'/' => find_line_terminator(src, n, i + 2),
            b'*' => {
                let e = scan_block_comment(src, n, i + 2).0;
                if e < n { e + 1 } else { n }
            }
            _ => i,
        },
        b'"' | b'\'' => {
            let mut j = i + 1;
            while j < n {
                let d = *src.add(j);
                if d == b'\\' {
                    j += 2;
                    continue;
                }
                if d == c || d == b'\n' || d == b'\r' {
                    return j + 1;
                }
                j += 1;
            }
            n
        }
        b'`' => {
            let mut j = i + 1;
            while j < n {
                let d = *src.add(j);
                if d == b'\\' {
                    j += 2;
                    continue;
                }
                if d == b'`' {
                    return j + 1;
                }
                j += 1;
            }
            n
        }
        _ => i,
    }
}

unsafe fn angle_close_fwd_capped(
    src: *const u8,
    st: *const u64,
    opch: *const u64,
    kind: *const u8,
    mut i: usize,
    lim: usize,
    mut depth: i32,
) -> (Option<usize>, bool) {
    let mut par: i32 = 0;
    let mut brk: i32 = 0;
    let mut brc: i32 = 0;
    while i < lim {
        if bm_get(st, i) {
            let j = skip_raw_literal(src, kind, lim, i);
            if j != i {
                i = j;
                continue;
            }
        }
        let c = *src.add(i);
        if GT_SCAN_DELIM[c as usize] && bm_get(st, i) {
            let op = bm_get(opch, i);
            match c {
                b'<' => {
                    if op && *src.add(i + 1) != b'=' {
                        depth += 1;
                    }
                }
                b'>' => {
                    if op && !(i > 0 && *src.add(i - 1) == b'=') {
                        depth -= 1;
                        if depth == 0 {
                            return ((par == 0 && brk == 0 && brc == 0).then_some(i), false);
                        }
                    }
                }
                b'(' => par += 1,
                b')' => {
                    par -= 1;
                    if par < 0 {
                        return (None, false);
                    }
                }
                b'[' => brk += 1,
                b']' => {
                    brk -= 1;
                    if brk < 0 {
                        return (None, false);
                    }
                }
                b'{' => brc += 1,
                b'}' => {
                    let kk = kind_at(kind, i);
                    if kk != TMPL_MIDDLE && kk != TMPL_TAIL {
                        brc -= 1;
                        if brc < 0 {
                            return (None, false);
                        }
                    }
                }
                _ => {
                    if par == 0 && brk == 0 && brc == 0 {
                        return (None, false); // `;`
                    }
                }
            }
        }
        i += 1;
    }
    (None, true)
}

/// The `)` matching the `(` at `i`, or `None` past `lim`.
unsafe fn paren_close_fwd(
    src: *const u8,
    st: *const u64,
    mut i: usize,
    lim: usize,
) -> Option<usize> {
    let mut d: i32 = 0;
    while i < lim {
        if bm_get(st, i) {
            match *src.add(i) {
                b'(' => d += 1,
                b')' => {
                    d -= 1;
                    if d == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    None
}

#[inline]
unsafe fn skip_ws_fwd(src: *const u8, mut i: usize, lim: usize) -> usize {
    while i < lim && is_ws(*src.add(i)) {
        i += 1;
    }
    i
}

/// True when the `<<` at `lt` is two type-argument/type-parameter openers
/// rather than shift-left.
///
/// Only one TypeScript production puts two `<` next to each other: a
/// type-argument list whose first argument is a function type. The second
/// `<` therefore has to open a type-parameter list belonging to one —
///
/// ```text
/// Name < < TypeParams > ( Params ) => Type >
/// ```
///
/// — so the whole shape is checked, not a prefix of it. That is what
/// separates `Array<<T>(x: T) => T>` from `a << b >> c` (no `(` after the
/// first `>`) and from `a << b > (c)` (no `=>` after the parameters). Every
/// reject path returns false, i.e. today's fused `<<`, so a wrong answer can
/// never split a real shift.
unsafe fn lt_run_opens_type_args(
    src: *const u8,
    st: *const u64,
    opch: *const u64,
    kind: *const u8,
    n: usize,
    lt: usize,
) -> bool {
    let lim = (lt + GT_SCAN_CAP).min(n);
    // A TypeParameter starts with an identifier (or the `const` modifier),
    // which is what makes `<<=` cost two bytes to reject.
    let head = skip_ws_fwd(src, lt + 2, lim);
    if head >= lim {
        return false;
    }
    let hc = *src.add(head);
    if !(is_id_start(hc) || (hc == b'\\' && *src.add(head + 1) == b'u')) {
        return false;
    }
    let Some(gt) = angle_close_fwd(src, st, opch, kind, lt + 2, lim, 1) else {
        return false;
    };
    let lp = skip_ws_fwd(src, gt + 1, lim);
    if lp >= lim || *src.add(lp) != b'(' {
        return false;
    }
    let Some(rp) = paren_close_fwd(src, st, lp, lim) else {
        return false;
    };
    let ar = skip_ws_fwd(src, rp + 1, lim);
    if ar + 1 >= lim || *src.add(ar) != b'=' || *src.add(ar + 1) != b'>' {
        return false;
    }
    // The outer list must close too, or this was a comparison against a
    // generic arrow function and the parser would have backtracked as well.
    angle_close_fwd(src, st, opch, kind, ar + 2, lim, 1).is_some()
}

/// `coalesce` entry for a `<<` run: true when the two `<` must stay separate
/// tokens. Cold — `<<` is shift-left everywhere except this one shape.
#[inline(never)]
pub(super) unsafe fn lt_run_split(
    src: *const u8,
    st: *const u64,
    opch: *const u64,
    kind: *const u8,
    n: usize,
    p: usize,
) -> bool {
    lt_run_opens_type_args(src, st, opch, kind, n, p)
}

/// `.tsx`: at an operand-position `<IDENT>(` that may be a generic arrow or a JSX element, `(is
/// JSX, report an unterminated element)`. `generic` says whether `(...) =>` follows the `>`.
pub(super) unsafe fn jsx_ambiguous_verdict(
    t: &Tables,
    src: *const u8,
    st: *const u64,
    opch: *const u64,
    kind: *const u8,
    n: usize,
    lt: usize,
    generic: bool,
    module: bool,
) -> (bool, bool) {
    let cx = walk::Cx { t, src, st, opch, kind, n, ts: true, kw_final: 0 };
    let site = walk::before(cx, module, lt);
    if site.in_type || site.type_params {
        (false, false)
    } else if generic {
        (false, site.operand)
    } else {
        (true, false)
    }
}

/// `coalesce` entry for a `>` run (`>>`, `>>>`, and a `>` glued to `=`): how many leading `>` bytes
/// to leave unfused (each closes an open `<` list), or 0 to fuse. Keyword kinds below `kw_final`
/// are already written.
pub(super) unsafe fn gt_run_split(
    t: &Tables,
    src: *const u8,
    st: *const u64,
    opch: *const u64,
    kind: *const u8,
    n: usize,
    p: usize,
    run: usize,
    module: bool,
    kw_final: usize,
) -> usize {
    let cx = walk::Cx { t, src, st, opch, kind, n, ts: true, kw_final };
    let angles = walk::angles_before(cx, module, p);
    let mut g = 0usize;
    while g < run && *src.add(p + g) == b'>' {
        g += 1;
    }
    angles.min(g)
}
