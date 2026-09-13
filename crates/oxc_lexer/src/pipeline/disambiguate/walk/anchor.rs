use core::cell::Cell;

use crate::opmap::OP_KIND_BASE;

use super::super::super::{
    BIGINT, IDENT, JSX_LT, NUM, PRIV_IDENT, REGEX, STR, TMPL_HEAD, TMPL_MIDDLE, TMPL_NOSUB,
    TMPL_TAIL, bitmap::bm_next1,
};
use super::super::{bm_prev_sig, match_delim_back};
use super::*;

// Bounded walks.
//
// A walk starts at an anchor: a token whose context is certain from its neighbours alone. Those
// are statement keywords (not property names, members or JSX attributes), the `(` whose group holds
// the query, and the start of the source. The scan back to the anchor records the balanced groups
// it crosses and the last separator inside each open bracket, so the walk skips group interiors
// and, once a bracket's frame kind is known, resumes at the separator. Nothing is guessed: what the
// walk did not see was either inside a skipped group, whose closer depends only on how it opened,
// or before a separator, which resets the frame.

/// Step cap for the scan back to an anchor (tokens plus group jumps); past it the full walk
/// answers.
const SCAN_CAP: u32 = 2048;

/// Token cap for locating the JSX tag around a keyword.
const TAG_SCAN_CAP: u32 = 256;

/// Where a bounded walk starts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Anchor {
    /// A statement starts at the token here.
    Stmt(usize),
    /// An operand starts at the token here: an expression-position `function` / `class`, or the
    /// `(` whose group holds the query.
    Expr(usize),
    /// The bounded walk already covers this point and continues from where it stopped.
    Continue,
}

struct Scan {
    anchor: Anchor,
    /// Unmatched `<` on the query's own level: the type lists a `>` run there could close.
    angles: u32,
}

/// Step cap for the short scan deciding a `>` run without context.
const RUN_SCAN_CAP: u32 = 256;

/// The lists a `>` run of `run` bytes at `pos` closes, when its context cannot matter: the run
/// closes no list on its level, or TypeScript's expression speculation accepts the list that would
/// take the whole run. Speculation only happens after a plain name, and in a type a `<` after a
/// name always opens a list, so such a list is one in every context. None when the context
/// matters (the walk decides).
unsafe fn run_shortcut(cx: Cx, pos: usize, run: usize) -> Option<usize> {
    let mut pending = 0i32;
    let mut found = 0usize;
    let mut lt = 0usize;
    let mut bounded = false;
    let mut steps = 0u32;
    let mut q = bm_prev_sig(cx.st, cx.kind, pos);
    while q >= 0 {
        steps += 1;
        if steps > RUN_SCAN_CAP {
            return None;
        }
        let p = q as usize;
        let k = base_kind(cx, p);
        if k >= OP_KIND_BASE {
            let c = *cx.src.add(p);
            match c {
                b')' | b']' | b'}' => {
                    let open = match c {
                        b')' => b'(',
                        b']' => b'[',
                        _ => b'{',
                    };
                    let o = match_delim_back(cx.src, cx.st, cx.kind, cx.n, p, open, c)?;
                    q = bm_prev_sig(cx.st, cx.kind, o);
                    continue;
                }
                b'(' | b'[' | b'{' | b';' => {
                    bounded = true;
                    break;
                }
                b'>' => {
                    if !(p > 0 && *cx.src.add(p - 1) == b'=') {
                        pending += run_len(cx, p, b'>');
                    }
                }
                b'<' => {
                    if *cx.src.add(p + 1) != b'=' {
                        let n = run_len(cx, p, b'<');
                        if n != 1 {
                            return None;
                        }
                        if pending > 0 {
                            pending -= 1;
                        } else {
                            found += 1;
                            lt = p;
                            if found == run {
                                break;
                            }
                        }
                    }
                }
                _ => {}
            }
        } else if !matches!(k, IDENT | STR | NUM | BIGINT | PRIV_IDENT | REGEX | TMPL_NOSUB) {
            return None;
        }
        q = bm_prev_sig(cx.st, cx.kind, p);
    }
    if found == 0 {
        return if bounded || q < 0 { Some(0) } else { None };
    }
    if !matches!(prev_token(cx, lt), Prev::Word(_, 0)) {
        return None;
    }
    if type_args_at(cx, lt) { Some(found) } else { None }
}

/// Keyword code of the word at `w`.
unsafe fn ident_kw(cx: Cx, w: usize) -> u8 {
    let e = bm_next1(cx.st, w + 1, cx.n);
    word_kw(cx, w, e - w)
}

/// Number of `c` bytes the token at `p` starts with (a fused `>>>` counts three, a lone `>` one).
unsafe fn run_len(cx: Cx, p: usize, c: u8) -> i32 {
    let e = bm_next1(cx.st, p + 1, cx.n);
    let mut i = p;
    while i < e && *cx.src.add(i) == c {
        i += 1;
    }
    (i - p) as i32
}

/// The significant token before `p`.
#[derive(Clone, Copy)]
enum Prev {
    None,
    /// An operator: its start and first byte.
    Op(usize, u8),
    /// A word: its start and keyword code (0 for a plain name).
    Word(usize, u8),
    /// A literal, template part or JSX token.
    Other(usize),
}

unsafe fn prev_token(cx: Cx, p: usize) -> Prev {
    let q = bm_prev_sig(cx.st, cx.kind, p);
    if q < 0 {
        return Prev::None;
    }
    let q = q as usize;
    let k = base_kind(cx, q);
    if k >= OP_KIND_BASE {
        Prev::Op(q, *cx.src.add(q))
    } else if k == IDENT {
        Prev::Word(q, ident_kw(cx, q))
    } else {
        Prev::Other(q)
    }
}

/// Is the word at `p` an attribute inside a JSX opening tag? Scans back over attribute names,
/// values and `=` to the tag's `<`.
unsafe fn in_jsx_tag(cx: Cx, p: usize) -> bool {
    let mut q = bm_prev_sig(cx.st, cx.kind, p);
    let mut steps = 0u32;
    while q >= 0 {
        steps += 1;
        if steps > TAG_SCAN_CAP {
            return false;
        }
        let w = q as usize;
        let k = base_kind(cx, w);
        if k == JSX_LT {
            return true;
        }
        if k >= OP_KIND_BASE {
            match *cx.src.add(w) {
                b'=' => {}
                b'}' | b')' | b']' => {
                    let c = *cx.src.add(w);
                    let open = match c {
                        b')' => b'(',
                        b']' => b'[',
                        _ => b'{',
                    };
                    let Some(o) = match_delim_back(cx.src, cx.st, cx.kind, cx.n, w, open, c) else {
                        return false;
                    };
                    q = bm_prev_sig(cx.st, cx.kind, o);
                    continue;
                }
                _ => return false,
            }
        } else if !matches!(k, IDENT | STR | NUM | BIGINT | TMPL_NOSUB) {
            return false;
        }
        q = bm_prev_sig(cx.st, cx.kind, w);
    }
    false
}

/// Can the token at `f` follow a statement keyword but not a property, member or attribute name?
unsafe fn keyword_follower(cx: Cx, f: usize) -> bool {
    let k = base_kind(cx, f);
    if k >= OP_KIND_BASE {
        return matches!(*cx.src.add(f), b'{' | b'-' | b'+' | b'~' | b'*' | b'@');
    }
    matches!(k, IDENT | STR | NUM | BIGINT | TMPL_NOSUB | TMPL_HEAD | REGEX | PRIV_IDENT | JSX_LT)
}

/// Does a statement start at `p` as far as the token before it can tell? `}` needs the JSX check
/// (an attribute after a `{...}` value); a line break allows a statement after anything else.
unsafe fn stmt_boundary(cx: Cx, p: usize, prev: Prev) -> bool {
    match prev {
        Prev::None => true,
        Prev::Op(_, b';' | b'{' | b')' | b':') => true,
        Prev::Op(_, b'}') => !in_jsx_tag(cx, p),
        Prev::Word(_, K_ELSE | K_DO | K_EXPORT | K_DEFAULT | K_DECLARE | K_ABSTRACT) => true,
        Prev::Op(q, _) | Prev::Word(q, _) | Prev::Other(q) => {
            let e = bm_next1(cx.st, q + 1, cx.n);
            lt_in_range(cx.src, e, p) && !in_jsx_tag(cx, p)
        }
    }
}

/// `function` / `class` at `at` (`async` for `async function`): a declaration or an expression,
/// read off the token before. `named` says a name follows (a label's `:` then precedes a
/// declaration, a property's an expression; only a name allows ASI to start a declaration).
unsafe fn fn_class_anchor(
    cx: Cx,
    at: usize,
    prev: Prev,
    class: bool,
    named: bool,
) -> Option<Anchor> {
    let broken = |q: usize| {
        let e = bm_next1(cx.st, q + 1, cx.n);
        lt_in_range(cx.src, e, at)
    };
    let stmt = Some(Anchor::Stmt(at));
    let expr = Some(Anchor::Expr(at));
    match prev {
        Prev::None => stmt,
        Prev::Op(q, c) => match c {
            b';' | b'{' => stmt,
            b'}' => {
                if in_jsx_tag(cx, at) {
                    None
                } else {
                    stmt
                }
            }
            // `if (x) function f() {}`; a decorator's `)` before a class expression.
            b')' => {
                if !class || (named && broken(q)) {
                    stmt
                } else {
                    None
                }
            }
            b']' => {
                if named && broken(q) {
                    stmt
                } else {
                    None
                }
            }
            b'>' if q > 0 && *cx.src.add(q - 1) == b'=' => expr,
            // A value or type ended on the previous line.
            b'>' => {
                if named && broken(q) {
                    stmt
                } else {
                    expr
                }
            }
            b'+' | b'-' if *cx.src.add(q + 1) == c || (q > 0 && *cx.src.add(q - 1) == c) => {
                if named && broken(q) { stmt } else { None }
            }
            b':' => {
                if named {
                    None
                } else {
                    expr
                }
            }
            _ => expr,
        },
        Prev::Word(_, K_ELSE | K_DO | K_EXPORT | K_DEFAULT | K_DECLARE | K_ABSTRACT) => stmt,
        // Restricted productions: a line break ends the statement.
        Prev::Word(q, K_RETURN | K_YIELD) => {
            if named && broken(q) {
                stmt
            } else {
                expr
            }
        }
        Prev::Word(
            _,
            K_TYPEOF | K_THROW | K_AWAIT | K_VOID | K_DELETE | K_NEW | K_IN | K_OF | K_INSTANCEOF
            | K_CASE,
        ) => expr,
        // A heritage expression: what follows it is the enclosing class's body.
        Prev::Word(_, K_EXTENDS) => None,
        Prev::Word(q, _) | Prev::Other(q) => {
            if named && broken(q) {
                stmt
            } else {
                None
            }
        }
    }
}

/// `function` at `at` (or the `async` before it), with the token after `function` at `f`: a named
/// function is a declaration or an expression by its context; an anonymous one is an expression,
/// or a method named `function`, which is no anchor.
unsafe fn function_anchor(cx: Cx, at: usize, prev: Prev, f: usize) -> Option<Anchor> {
    if f >= cx.n {
        return None;
    }
    let fk = base_kind(cx, f);
    let fc = *cx.src.add(f);
    if fk == IDENT || (fk >= OP_KIND_BASE && fc == b'*') {
        fn_class_anchor(cx, at, prev, false, true)
    } else if fk >= OP_KIND_BASE && fc == b'(' {
        match fn_class_anchor(cx, at, prev, false, false) {
            Some(Anchor::Expr(a)) => Some(Anchor::Expr(a)),
            _ => None,
        }
    } else {
        None
    }
}

/// Is the word at `p` an anchor? Also its keyword code (0: a plain name).
unsafe fn anchor_at(cx: Cx, p: usize) -> (u8, Option<Anchor>) {
    let e = bm_next1(cx.st, p + 1, cx.n);
    // Keywords are lowercase words of up to ten letters: skip the hash for the rest.
    if !cx.src.add(p).read().is_ascii_lowercase() || e - p > 10 {
        return (0, None);
    }
    let kwc = word_kw(cx, p, e - p);
    let anchor = anchor_of(cx, p, e, kwc);
    (kwc, anchor)
}

unsafe fn anchor_of(cx: Cx, p: usize, e: usize, kwc: u8) -> Option<Anchor> {
    if kwc == 0 {
        return None;
    }
    let prev = prev_token(cx, p);
    // A property name.
    if let Prev::Op(q, c) = prev {
        if c == b'.' || (c == b'?' && *cx.src.add(q + 1) == b'.') {
            return None;
        }
    }
    let f = next_sig(cx, e);
    if f >= cx.n {
        return None;
    }
    let fk = base_kind(cx, f);
    let fc = *cx.src.add(f);
    let f_kw = if fk == IDENT { ident_kw(cx, f) } else { 0 };
    let same_line = !lt_in_range(cx.src, e, f);
    match kwc {
        K_FUNCTION => function_anchor(cx, p, prev, f),
        K_CLASS => {
            if fk == IDENT {
                fn_class_anchor(cx, p, prev, true, true)
            } else if fk >= OP_KIND_BASE && matches!(fc, b'{' | b'<') {
                fn_class_anchor(cx, p, prev, true, false)
            } else {
                None
            }
        }
        K_ASYNC => {
            if f_kw == K_FUNCTION && same_line {
                let e2 = bm_next1(cx.st, f + 1, cx.n);
                function_anchor(cx, p, prev, next_sig(cx, e2))
            } else {
                None
            }
        }
        K_VAR | K_CONST | K_RETURN | K_THROW | K_CASE | K_EXPORT | K_IMPORT | K_ENUM | K_ELSE
        | K_DO | K_TRY | K_FINALLY | K_BREAK | K_CONTINUE | K_DEBUGGER | K_IF | K_FOR | K_WHILE
        | K_SWITCH | K_WITH | K_CATCH => {
            if keyword_follower(cx, f) && stmt_boundary(cx, p, prev) {
                Some(Anchor::Stmt(p))
            } else {
                None
            }
        }
        K_LET | K_TYPE | K_INTERFACE | K_NAMESPACE | K_MODULE | K_DECLARE | K_ABSTRACT => {
            if !same_line {
                return None;
            }
            let ok = match kwc {
                K_LET => {
                    fk == IDENT && f_kw == 0 && second_follower(cx, f, &[b'=', b';', b':', b','])
                }
                K_TYPE => fk == IDENT && f_kw == 0 && second_follower(cx, f, &[b'=', b'<']),
                K_INTERFACE => {
                    fk == IDENT
                        && f_kw == 0
                        && (second_follower(cx, f, &[b'{', b'<'])
                            || second_word(cx, f) == K_EXTENDS)
                }
                K_NAMESPACE => fk == IDENT && f_kw == 0 && second_follower(cx, f, &[b'{', b'.']),
                K_MODULE => {
                    ((fk == IDENT && f_kw == 0) || fk == STR) && second_follower(cx, f, &[b'{'])
                }
                K_DECLARE => matches!(
                    f_kw,
                    K_CONST
                        | K_LET
                        | K_VAR
                        | K_FUNCTION
                        | K_CLASS
                        | K_MODULE
                        | K_NAMESPACE
                        | K_GLOBAL
                        | K_ENUM
                        | K_INTERFACE
                        | K_TYPE
                        | K_ABSTRACT
                        | K_ASYNC
                ),
                _ => f_kw == K_CLASS,
            };
            if !ok {
                return None;
            }
            let boundary = match prev {
                Prev::None => true,
                Prev::Op(_, b';') => true,
                Prev::Op(_, b'{') => kwc == K_LET,
                Prev::Op(_, b'}') => !in_jsx_tag(cx, p),
                Prev::Word(_, K_EXPORT | K_DECLARE) => true,
                _ => false,
            };
            if boundary { Some(Anchor::Stmt(p)) } else { None }
        }
        _ => None,
    }
}

/// Is the significant token after the word at `f` one of `ops`?
unsafe fn second_follower(cx: Cx, f: usize, ops: &[u8]) -> bool {
    let e = bm_next1(cx.st, f + 1, cx.n);
    let s = next_sig(cx, e);
    s < cx.n && base_kind(cx, s) >= OP_KIND_BASE && ops.contains(&*cx.src.add(s))
}

/// Keyword code of the significant word after the word at `f` (0 if none).
unsafe fn second_word(cx: Cx, f: usize) -> u8 {
    let e = bm_next1(cx.st, f + 1, cx.n);
    let s = next_sig(cx, e);
    if s < cx.n && base_kind(cx, s) == IDENT { ident_kw(cx, s) } else { 0 }
}

/// Record the separators and brace boundary found on a level for the opener (or anchor) at `at`.
fn push_level(
    w: &mut Walk,
    at: usize,
    semi: Option<usize>,
    comma: Option<usize>,
    brace: Option<usize>,
) {
    if semi.is_some() || comma.is_some() || brace.is_some() {
        w.plan.push(Jump::Sep {
            at: at as u32,
            semi: semi.map_or(0, |s| s as u32),
            comma: comma.map_or(0, |c| c as u32),
            brace: brace.map_or(0, |b| b as u32),
        });
    }
}

/// The walk continues inside the level scanned last: record where it may resume.
fn push_resume(w: &mut Walk, semi: Option<usize>, brace: Option<usize>) {
    if semi.is_some() || brace.is_some() {
        w.plan.push(Jump::Resume {
            semi: semi.map_or(0, |s| s as u32),
            brace: brace.map_or(0, |b| b as u32),
        });
    }
}

/// After the `}` at `c`: the position of a token that must start a statement or member there
/// (a name, string, number, private name or decorator; not `as` / `satisfies` / `in` /
/// `instanceof`, which continue a value), or None.
unsafe fn brace_boundary(cx: Cx, c: usize) -> Option<usize> {
    let f = next_sig(cx, c + 1);
    if f >= cx.n {
        return None;
    }
    let k = base_kind(cx, f);
    let ok = match k {
        IDENT => !matches!(
            ident_kw(cx, f),
            K_AS | K_SATISFIES | K_IN | K_INSTANCEOF | K_OF | K_IMPLEMENTS | K_EXTENDS | K_FROM
        ),
        PRIV_IDENT | STR | NUM | BIGINT => true,
        _ => k >= OP_KIND_BASE && *cx.src.add(f) == b'@',
    };
    if ok { Some(f) } else { None }
}

/// Scan back from `from` (exclusive) to the anchor of a query, recording the walk's jumps in
/// `w.plan` (nearest first). `cont` is where the current bounded walk stopped: reaching it, or a
/// group holding it, means the walk continues from there. None past the scan cap.
unsafe fn scan(cx: Cx, w: &mut Walk, from: usize, cont: Option<usize>) -> Option<Scan> {
    w.plan.clear();
    let mut steps = 0u32;
    // Per level: the last separators seen (nearest to the query), the `>` minus `<` balance going
    // back, and the unmatched `<` found.
    let mut level = 0u32;
    let mut semi: Option<usize> = None;
    let mut comma: Option<usize> = None;
    let mut brace: Option<usize> = None;
    // `>` closers seen on this level going back, waiting for their `<` (`u32::MAX`: one of a run,
    // which the walk cannot jump to).
    let mut gts: Vec<u32> = Vec::with_capacity(8);
    let mut angles0 = 0u32;
    let mut q = bm_prev_sig(cx.st, cx.kind, from);
    loop {
        if q < 0 {
            push_level(w, 0, semi, comma, brace);
            return Some(Scan { anchor: Anchor::Stmt(0), angles: angles0 });
        }
        let p = q as usize;
        if cont.is_some_and(|c| p < c) {
            push_resume(w, semi, brace);
            return Some(Scan { anchor: Anchor::Continue, angles: angles0 });
        }
        steps += 1;
        if steps > SCAN_CAP {
            return None;
        }
        let k = base_kind(cx, p);
        if k >= OP_KIND_BASE {
            let c = *cx.src.add(p);
            match c {
                b')' | b']' | b'}' => {
                    let open = match c {
                        b')' => b'(',
                        b']' => b'[',
                        _ => b'{',
                    };
                    let o = match_delim_back(cx.src, cx.st, cx.kind, cx.n, p, open, c)?;
                    if cont.is_some_and(|c| o < c) {
                        push_resume(w, semi, brace);
                        return Some(Scan { anchor: Anchor::Continue, angles: angles0 });
                    }
                    if c == b'}' && brace.is_none() && gts.is_empty() {
                        brace = brace_boundary(cx, p);
                    }
                    w.plan.push(Jump::Skip { at: o as u32, to: p as u32 });
                    q = bm_prev_sig(cx.st, cx.kind, o);
                    continue;
                }
                b'(' => {
                    push_level(w, p, semi, comma, brace);
                    if let Some(anchor) = paren_anchor(cx, p) {
                        return Some(Scan { anchor, angles: angles0 });
                    }
                    // A paren that may sit in a type: the walk reaches it from further back.
                    level += 1;
                    semi = None;
                    comma = None;
                    brace = None;
                    gts.clear();
                }
                b'[' | b'{' => {
                    push_level(w, p, semi, comma, brace);
                    level += 1;
                    semi = None;
                    comma = None;
                    brace = None;
                    gts.clear();
                }
                // A separator inside a balanced `<...>` before the query (`bal > 0`) is not the
                // level's.
                b';' => {
                    if semi.is_none() && gts.is_empty() {
                        semi = Some(p);
                    }
                }
                b',' => {
                    if comma.is_none() && gts.is_empty() {
                        comma = Some(p);
                    }
                }
                b'>' => {
                    if !(p > 0 && *cx.src.add(p - 1) == b'=') {
                        let n = run_len(cx, p, b'>');
                        if n == 1 {
                            gts.push(p as u32);
                        } else {
                            gts.extend(core::iter::repeat_n(u32::MAX, n as usize));
                        }
                    }
                }
                b'<' => {
                    if *cx.src.add(p + 1) != b'=' {
                        let n = run_len(cx, p, b'<');
                        for _ in 0..n {
                            match gts.pop() {
                                Some(g) => {
                                    if n == 1 && g != u32::MAX {
                                        w.plan.push(Jump::Angle { at: p as u32, to: g });
                                    }
                                }
                                None => {
                                    // An open list around everything seen on this level so far:
                                    // the separators inside it are not the level's.
                                    if level == 0 {
                                        angles0 += 1;
                                    }
                                    semi = None;
                                    comma = None;
                                    brace = None;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        } else if k == IDENT {
            let (kwc, anchor) = anchor_at(cx, p);
            if let Some(anchor) = anchor {
                let at = match anchor {
                    Anchor::Stmt(a) | Anchor::Expr(a) => a,
                    Anchor::Continue => p,
                };
                push_level(w, at, semi, comma, brace);
                return Some(Scan { anchor, angles: angles0 });
            }
            // A `,` after a class or interface keyword may sit in its heritage list, which belongs
            // to the head, not to the frame around it.
            if matches!(kwc, K_CLASS | K_INTERFACE) && !property_name(cx, p) {
                comma = None;
                brace = None;
            }
        } else if k == TMPL_TAIL || k == TMPL_MIDDLE {
            // A template: skip back to its head. A middle also opens the substitution the query is
            // in, so the level changes there.
            let mut depth = 1i32;
            let mut t = bm_prev_sig(cx.st, cx.kind, p);
            let head = loop {
                if t < 0 {
                    return None;
                }
                steps += 1;
                if steps > SCAN_CAP {
                    return None;
                }
                let tp = t as usize;
                match base_kind(cx, tp) {
                    TMPL_TAIL => depth += 1,
                    TMPL_HEAD => {
                        depth -= 1;
                        if depth == 0 {
                            break tp;
                        }
                    }
                    _ => {}
                }
                t = bm_prev_sig(cx.st, cx.kind, tp);
            };
            if cont.is_some_and(|c| head < c) {
                push_resume(w, semi, brace);
                return Some(Scan { anchor: Anchor::Continue, angles: angles0 });
            }
            if k == TMPL_MIDDLE {
                push_level(w, p, semi, comma, brace);
                level += 1;
                semi = None;
                comma = None;
                brace = None;
                gts.clear();
            }
            w.plan.push(Jump::Skip { at: head as u32, to: p as u32 });
            q = bm_prev_sig(cx.st, cx.kind, head);
            continue;
        } else if k == TMPL_HEAD {
            // The substitution the query is in.
            push_level(w, p, semi, comma, brace);
            level += 1;
            semi = None;
            comma = None;
            brace = None;
            gts.clear();
        }
        q = bm_prev_sig(cx.st, cx.kind, p);
    }
}

/// The `(` at `p` holds the query: where the walk starts when the token before makes the paren an
/// expression's (a call, a grouping, a statement head). None when it may be a type's, such as after
/// `:`, `=`, `,`, `<` or `=>`: the walk then reaches it from an anchor further back.
unsafe fn paren_anchor(cx: Cx, p: usize) -> Option<Anchor> {
    match prev_token(cx, p) {
        Prev::None => Some(Anchor::Expr(p)),
        Prev::Word(kp, K_IF | K_WHILE | K_FOR | K_WITH | K_SWITCH | K_CATCH)
            if !property_name(cx, kp) =>
        {
            Some(Anchor::Stmt(kp))
        }
        Prev::Word(kp, K_AWAIT) => match prev_token(cx, kp) {
            Prev::Word(fp, K_FOR) if !property_name(cx, fp) => Some(Anchor::Stmt(fp)),
            _ => Some(Anchor::Expr(p)),
        },
        Prev::Word(
            _,
            0 | K_THIS | K_SUPER | K_RETURN | K_THROW | K_TYPEOF | K_YIELD | K_VOID | K_DELETE
            | K_IN | K_OF | K_INSTANCEOF | K_CASE | K_ELSE | K_DO | K_ASYNC | K_NULL | K_TRUE
            | K_FALSE,
        ) => Some(Anchor::Expr(p)),
        Prev::Word(..) => None,
        Prev::Op(q, c) => match c {
            b')' | b']' | b'}' | b'!' | b'+' | b'-' | b'*' | b'/' | b'%' | b'^' | b'~' => {
                Some(Anchor::Expr(p))
            }
            b'>' if !(q > 0 && *cx.src.add(q - 1) == b'=') => Some(Anchor::Expr(p)),
            _ => None,
        },
        Prev::Other(q) => match base_kind(cx, q) {
            STR | NUM | BIGINT | TMPL_NOSUB | TMPL_TAIL | REGEX | PRIV_IDENT => {
                Some(Anchor::Expr(p))
            }
            _ => None,
        },
    }
}

/// Is the word at `w` a property name (`x.if`)?
unsafe fn property_name(cx: Cx, w: usize) -> bool {
    matches!(prev_token(cx, w), Prev::Op(q, c) if c == b'.' || (c == b'?' && *cx.src.add(q + 1) == b'.'))
}

/// Run `f` on a bounded walk for the token at `pos`, scanning back from `from` (the matching
/// opener of a closer at `pos`, else `pos`). None when no anchor lies within the scan cap or the
/// walk lost its footing.
unsafe fn local_walk<R>(
    cx: Cx,
    module: bool,
    pos: usize,
    from: usize,
    f: impl FnOnce(&mut Walk, Cx) -> R,
) -> Option<R> {
    let generation = GEN.with(Cell::get);
    LOCAL.with(|cell| {
        let mut w = cell.borrow_mut();
        let cont =
            (w.generation == generation && w.seed_depth != 0 && !w.seed_lost && w.upto <= from)
                .then_some(w.upto);
        let s = scan(cx, &mut w, from, cont);
        let Some(s) = s else {
            return None;
        };
        w.start(cx, module, generation, s.anchor, pos, from);
        w.advance(cx, pos);
        if w.seed_lost {
            w.generation = u64::MAX;
            return None;
        }
        Some(f(&mut w, cx))
    })
}

impl Walk {
    /// Seed the walk at `anchor` for a query at `pos` whose closer group opens at `from`, with the
    /// scan's jumps in `plan` (nearest first).
    unsafe fn start(
        &mut self,
        cx: Cx,
        module: bool,
        generation: u64,
        anchor: Anchor,
        pos: usize,
        from: usize,
    ) {
        let _ = cx;
        self.plan.reverse();
        if from < pos {
            self.plan.push(Jump::Skip { at: from as u32, to: pos as u32 });
        }
        self.plan_next = 0;
        match anchor {
            Anchor::Continue => {
                if let Some(Jump::Resume { semi, brace }) = self.plan.first().copied() {
                    self.plan.remove(0);
                    self.resume(semi as usize, brace as usize, pos);
                }
            }
            Anchor::Stmt(at) | Anchor::Expr(at) => {
                self.reset(module);
                self.generation = generation;
                self.upto = at;
                self.prev_end = at;
                if matches!(anchor, Anchor::Expr(_)) {
                    self.stmt_start = false;
                }
                self.seed_depth = self.frames.len();
                self.seed_lost = false;
            }
        }
    }
}

/// Context at the token starting at `pos` (not through it).
pub(crate) unsafe fn before(cx: Cx, module: bool, pos: usize) -> Site {
    if let Some(s) = local_walk(cx, module, pos, pos, |w, _| w.site()) {
        return s;
    }
    let generation = GEN.with(Cell::get);
    WALK.with(|cell| {
        let mut w = cell.borrow_mut();
        if w.generation != generation || w.upto > pos {
            w.generation = generation;
            w.reset(module);
        }
        w.advance(cx, pos);
        w.site()
    })
}

/// Open `<` lists a `>` run of `run` bytes at `pos` would close. Two answers need no walk: the
/// scan back to the anchor meets no open `<` on the run's level (nothing to close), or the list
/// that would take the whole run is one TypeScript's expression speculation accepts, which opens a
/// list in any context.
pub(crate) unsafe fn angles_before(cx: Cx, module: bool, pos: usize, run: usize) -> usize {
    if let Some(k) = run_shortcut(cx, pos, run) {
        return k;
    }
    let generation = GEN.with(Cell::get);
    let local = LOCAL.with(|cell| {
        let mut w = cell.borrow_mut();
        let cont =
            (w.generation == generation && w.seed_depth != 0 && !w.seed_lost && w.upto <= pos)
                .then_some(w.upto);
        let s = scan(cx, &mut w, pos, cont);
        let Some(s) = s else {
            return None;
        };
        if s.angles == 0 && s.anchor != Anchor::Continue {
            return Some(0);
        }
        w.start(cx, module, generation, s.anchor, pos, pos);
        w.advance(cx, pos);
        if w.seed_lost {
            w.generation = u64::MAX;
            return None;
        }
        Some(w.open_angles())
    });
    local.unwrap_or_else(|| before(cx, module, pos).angles)
}

/// What the significant token at `pos` leaves behind.
pub(crate) unsafe fn after(cx: Cx, module: bool, pos: usize) -> After {
    after_from(cx, module, pos, pos)
}

/// Like [`after`], with the bounded walk scanning back from `from` (the matching opener of a
/// closer at `pos`): the walk reaches the opener, skips the group and steps the closer.
pub(crate) unsafe fn after_from(cx: Cx, module: bool, pos: usize, from: usize) -> After {
    if let Some(a) = local_walk(cx, module, pos, from, |w, cx| {
        if w.upto > pos { w.classify_after() } else { w.after_token(cx, pos) }
    }) {
        return a;
    }
    after_scoped(cx, module, pos)
}
