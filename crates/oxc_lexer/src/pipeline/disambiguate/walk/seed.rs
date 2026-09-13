use core::cell::{Cell, RefCell};

use crate::opmap::OP_KIND_BASE;

use super::super::super::{
    BIGINT, IDENT, JEND, JSX_LT, JTEXT, NUM, PRIV_IDENT, REGEX, STR, TMPL_HEAD, TMPL_MIDDLE,
    TMPL_NOSUB, TMPL_TAIL,
    bitmap::{bm_get, bm_next1},
};
use super::super::{
    Follow, angle_close_fwd_capped, bm_prev_sig, gt_follower, ident_is, lt_in_range,
    lt_run_opens_type_args, match_delim_back, type_list_legal,
};
use super::*;

// Bounded walks.
//
// Most questions depend only on the current statement or member, so the walk starts near the query:
// at the nearest `;` on its nesting level, or right after the innermost unmatched opener. What that
// boundary sits in is read off the tokens before it by rules that answer only when sure; otherwise
// the full memoized walk answers.

/// A bounded walk keeps going for a later query this close to where it stopped, instead of
/// re-seeding.
const SEED_CONTINUE: usize = 512;

/// Recursion cap for the seed rules.
const SEED_RULE_DEPTH: u32 = 24;

/// Token budget for the seed rules of one query: past it they give up and the full walk answers.
const SEED_RULE_BUDGET: u32 = 32768;

/// Where a bounded walk starts.
#[derive(Clone, Copy)]
enum Seed {
    /// A statement starts here (top level or inside a block).
    Stmt(usize),
    /// A member starts at `.1` inside the class body / type literal of kind `.2` opened at `.0`
    /// (`usize::MAX` when the opener lies beyond the scan and the kind came from the member).
    Member(usize, usize, Fk),
    /// Right after the `{` here, which opens a frame of kind `.1`.
    Brace(usize, Fk),
    /// Right after the `(` / `[` here.
    Paren(usize),
    Bracket(usize),
    /// A template substitution opened by the head / middle token here.
    Sub(usize),
    /// Right after the `<` here, the outermost of the type lists holding the query.
    Angle(usize, Lt),
}

/// What a `<` opens.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lt {
    /// Type arguments inside a type (`let x: Foo<`); the flag says whether that type is a
    /// declaration's.
    Type(bool),
    /// Type parameters of a declaration head or member (`class C<`).
    Params,
    /// Type arguments on an expression (`f<T>(`).
    Args,
    /// An assertion or generic arrow (`= <T>`).
    Assert,
}

/// Token cap for the backward scan to a boundary or opener.
const SEED_SCAN_CAP: u32 = 512;

/// Token cap for the search for a ternary's `?` before a `:`.
const SEED_TERNARY_SCAN: u32 = 96;

/// Past a boundary, how much farther the scan looks for the frame around it before reading the kind
/// off the tokens after the boundary.
const SEED_CONTAINER_CAP: u32 = 64;

const MEMO_SLOTS: usize = 32;

thread_local! {
    static BUDGET: Cell<u32> = const { Cell::new(0) };
    /// Answers a query asks repeatedly (nested lists re-ask about every enclosing `<` and `:`),
    /// cleared with the budget.
    static MEMO: RefCell<[(u64, u64); MEMO_SLOTS]> = const { RefCell::new([(u64::MAX, 0); MEMO_SLOTS]) };
    static MEMO_NEXT: Cell<usize> = const { Cell::new(0) };
}

/// Start a query: a fresh budget and memo.
fn start_query() {
    BUDGET.with(|b| b.set(SEED_RULE_BUDGET));
    MEMO.with(|m| *m.borrow_mut() = [(u64::MAX, 0); MEMO_SLOTS]);
    MEMO_NEXT.with(|n| n.set(0));
}

const MEMO_ANGLE: u64 = 1 << 56;
const MEMO_COLON: u64 = 2 << 56;
const MEMO_TYPE: u64 = 3 << 56;

fn memo_get(key: u64) -> Option<u64> {
    MEMO.with(|m| {
        // Slots fill in order, so the first empty one ends the search.
        for (k, v) in m.borrow().iter() {
            if *k == key {
                return Some(*v);
            }
            if *k == u64::MAX {
                return None;
            }
        }
        None
    })
}

fn memo_put(key: u64, val: u64) {
    let i = MEMO_NEXT.with(|n| {
        let i = n.get();
        n.set((i + 1) % MEMO_SLOTS);
        i
    });
    MEMO.with(|m| m.borrow_mut()[i] = (key, val));
}

/// Spend one step of the seed rules' budget; false once it is used up.
#[inline]
fn tick() -> bool {
    BUDGET.with(|b| {
        let v = b.get();
        if v == 0 {
            return false;
        }
        b.set(v - 1);
        true
    })
}

/// What ends at a boundary found by the scan.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Boundary {
    /// A `;`: the frame around it says what starts next.
    Semi,
    /// A block or declaration closed: a statement starts next.
    Stmt,
    /// A method body or static block closed: a member starts next.
    Member,
}

/// Where a bounded walk for a query at `pos` starts: the outermost type list holding it, else the
/// nearest `;` or block end on its level, else the innermost unmatched opener. None when nothing
/// usable lies within the scan cap.
unsafe fn find_seed(cx: Cx, pos: usize) -> Option<Seed> {
    let (mut par, mut brk, mut brc, mut tdep, mut ang) = (0i32, 0i32, 0i32, 0i32, 0i32);
    let mut steps = 0u32;
    // The boundary, and how far past it the scan has gone.
    let mut at = usize::MAX;
    let mut at_kind = Boundary::Semi;
    let mut past = 0u32;
    // A `}` on the query's level whose `{` the scan has not reached.
    let mut pending = usize::MAX;
    // The outermost type list found so far.
    let mut angle: Option<(usize, Lt)> = None;
    // The enclosing opener, once known (`usize::MAX`: the top level).
    let mut enc = usize::MAX;
    let mut known = false;
    // The operator char of the previous (later) token, 0 for other tokens.
    let mut after: u8 = 0;
    let mut q = bm_prev_sig(cx.st, cx.kind, pos);
    loop {
        if q < 0 {
            known = true;
            break;
        }
        if at != usize::MAX {
            past += 1;
            if past > SEED_CONTAINER_CAP {
                break;
            }
        }
        steps += 1;
        if steps > SEED_SCAN_CAP || !tick() {
            break;
        }
        let w = q as usize;
        let closed = par == 0 && brk == 0 && brc == 0 && tdep == 0;
        let k = base_kind(cx, w);
        if k >= OP_KIND_BASE {
            let c = *cx.src.add(w);
            match c {
                b')' => par += 1,
                b']' => brk += 1,
                b'}' => {
                    if closed && at == usize::MAX {
                        // `{...}` followed by a list token sits inside a list; any other `}` closed
                        // a block, which no type list spans.
                        if angle.is_some()
                            && !matches!(
                                after,
                                b'>' | b','
                                    | b'&'
                                    | b'|'
                                    | b')'
                                    | b']'
                                    | b'?'
                                    | b':'
                                    | b'['
                                    | b'.'
                                    | b'='
                            )
                        {
                            break;
                        }
                        pending = w;
                    }
                    brc += 1;
                }
                b'(' => {
                    if par == 0 {
                        enc = w;
                        known = true;
                        break;
                    }
                    par -= 1;
                }
                b'[' => {
                    if brk == 0 {
                        enc = w;
                        known = true;
                        break;
                    }
                    brk -= 1;
                }
                b'{' => {
                    if brc == 0 {
                        enc = w;
                        known = true;
                        break;
                    }
                    brc -= 1;
                    if brc == 0 && par == 0 && brk == 0 && tdep == 0 && pending != usize::MAX {
                        // The group on the query's level just crossed: a boundary if it was a
                        // block, body or declaration.
                        if at == usize::MAX {
                            if let Some(b) = block_end_boundary(cx, w, pending) {
                                at = pending + 1;
                                at_kind = b;
                            }
                        }
                        pending = usize::MAX;
                    }
                }
                b';' => {
                    if closed && at == usize::MAX {
                        if angle.is_some() {
                            break;
                        }
                        at = w + 1;
                        at_kind = Boundary::Semi;
                    }
                }
                b'>' if closed && at == usize::MAX => {
                    if !(w > 0 && *cx.src.add(w - 1) == b'=') {
                        ang += run_len(cx, w, b'>');
                    }
                }
                b'<' if closed && at == usize::MAX => {
                    // A `<<` token may hold one matched and one unmatched opener (`a<<T>(x: T) =>
                    // T>`).
                    ang -= run_len(cx, w, b'<');
                    if ang < 0 {
                        ang = 0;
                        match angle_kind(cx, w, 0) {
                            LtKind::List(l) => {
                                angle = Some((w, l));
                                // Only a type name's list can sit inside another list: the others
                                // are outermost.
                                if !matches!(l, Lt::Type(_)) {
                                    break;
                                }
                            }
                            // A comparison, or nothing the rules can tell: the lists found so far
                            // are the query's.
                            LtKind::Cmp | LtKind::Unknown => {
                                if angle.is_some() {
                                    break;
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        } else if k == TMPL_TAIL {
            tdep += 1;
        } else if k == TMPL_MIDDLE || k == TMPL_HEAD {
            if tdep == 0 {
                if !closed {
                    return None;
                }
                enc = w;
                known = true;
                break;
            }
            if k == TMPL_HEAD {
                tdep -= 1;
            }
        } else if k == JSX_LT && closed {
            // Leaving a JSX element backwards: whatever holds it is not worth the walk from there.
            break;
        } else if k == IDENT && closed && angle.is_some() {
            // A statement keyword: no type list spans it. `import("m")` inside a type is not one.
            let kwc = ident_kw(cx, w);
            let import_type = kwc == K_IMPORT && {
                let nx = next_sig(cx, bm_next1(cx.st, w + 1, cx.n));
                nx < cx.n && base_kind(cx, nx) >= OP_KIND_BASE && *cx.src.add(nx) == b'('
            };
            if is_stmt_keyword(kwc, true, cx.ts) && !import_type {
                break;
            }
        }
        after = if k >= OP_KIND_BASE { *cx.src.add(w) } else { 0 };
        q = bm_prev_sig(cx.st, cx.kind, w);
    }
    if let Some((lt, kind)) = angle {
        return Some(Seed::Angle(lt, kind));
    }
    if known && enc != usize::MAX && *cx.src.add(enc) != b'{' {
        return Some(match *cx.src.add(enc) {
            b'(' => Seed::Paren(enc),
            b'[' => Seed::Bracket(enc),
            _ => Seed::Sub(enc),
        });
    }
    if at != usize::MAX {
        if known {
            if enc == usize::MAX {
                return Some(Seed::Stmt(at));
            }
            let kind = brace_seed_kind(cx, enc, 0)?;
            return Some(if matches!(kind, Fk::ClassBody | Fk::TypeLit) {
                Seed::Member(enc, at, kind)
            } else {
                Seed::Stmt(at)
            });
        }
        let member = match at_kind {
            Boundary::Stmt => false,
            Boundary::Member => true,
            Boundary::Semi => looks_like_member(cx, at),
        };
        return Some(if member {
            Seed::Member(usize::MAX, at, Fk::ClassBody)
        } else {
            Seed::Stmt(at)
        });
    }
    if known {
        if enc == usize::MAX {
            return Some(Seed::Stmt(0));
        }
        let kind = brace_seed_kind(cx, enc, 0)?;
        return Some(Seed::Brace(enc, kind));
    }
    None
}

/// The frame around a name or `[` that follows a `;`, `,` or `}`: what its opener opens (`Call` for
/// a parenthesis, `TypeBracket` / `Array` for a bracket).
unsafe fn container_kind(cx: Cx, at: usize, depth: u32) -> Option<Fk> {
    let o = enclosing_opener(cx, at)?;
    if o == usize::MAX {
        return Some(Fk::Block);
    }
    match *cx.src.add(o) {
        b'{' => brace_seed_kind(cx, o, depth),
        b'(' => Some(Fk::Call),
        b'[' => match type_start(cx, o, depth) {
            TypeStart::No => Some(Fk::Array),
            TypeStart::Unknown => None,
            _ => Some(Fk::TypeBracket),
        },
        _ => Some(Fk::Sub),
    }
}

/// Token cap for the scan to the opener around a member.
const SEED_CONTAINER_SCAN: u32 = 48;

/// The innermost unmatched opener before `pos` (`usize::MAX`: none, the top level), or None when it
/// lies beyond the scan.
unsafe fn enclosing_opener(cx: Cx, pos: usize) -> Option<usize> {
    let (mut par, mut brk, mut brc, mut tdep) = (0i32, 0i32, 0i32, 0i32);
    let mut q = bm_prev_sig(cx.st, cx.kind, pos);
    for _ in 0..SEED_CONTAINER_SCAN {
        if q < 0 {
            return Some(usize::MAX);
        }
        if !tick() {
            return None;
        }
        let w = q as usize;
        let k = base_kind(cx, w);
        if k >= OP_KIND_BASE {
            match *cx.src.add(w) {
                b')' => par += 1,
                b']' => brk += 1,
                b'}' => brc += 1,
                b'(' => {
                    if par == 0 {
                        return Some(w);
                    }
                    par -= 1;
                }
                b'[' => {
                    if brk == 0 {
                        return Some(w);
                    }
                    brk -= 1;
                }
                b'{' => {
                    if brc == 0 {
                        return Some(w);
                    }
                    brc -= 1;
                }
                _ => {}
            }
        } else if k == TMPL_TAIL {
            tdep += 1;
        } else if k == TMPL_MIDDLE || k == TMPL_HEAD {
            if tdep == 0 {
                return if par == 0 && brk == 0 && brc == 0 { Some(w) } else { None };
            }
            if k == TMPL_HEAD {
                tdep -= 1;
            }
        }
        q = bm_prev_sig(cx.st, cx.kind, w);
    }
    None
}

/// What a `function` body's head is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FnKind {
    /// A function declaration: a statement follows its body.
    Decl,
    /// A function expression: its body ends a value.
    Expr,
    /// A class or object method: a member (or `,`) follows its body.
    Method,
    Unknown,
}

/// Classify the function whose body opens with the `{` at `lb`.
unsafe fn fn_body_head(cx: Cx, lb: usize) -> FnKind {
    let mut rp = bm_prev_sig(cx.st, cx.kind, lb);
    if rp >= 0 && !(base_kind(cx, rp as usize) >= OP_KIND_BASE && *cx.src.add(rp as usize) == b')')
    {
        // A return type: its `:` follows the parameter list.
        let (ts, at) = type_start_at(cx, lb, 1);
        if ts != TypeStart::Decl || base_kind(cx, at) < OP_KIND_BASE || *cx.src.add(at) != b':' {
            return FnKind::Unknown;
        }
        rp = bm_prev_sig(cx.st, cx.kind, at);
    }
    if rp < 0 || base_kind(cx, rp as usize) < OP_KIND_BASE || *cx.src.add(rp as usize) != b')' {
        return FnKind::Unknown;
    }
    let Some(lp) = match_delim_back(cx.src, cx.st, cx.kind, rp as usize, b'(', b')') else {
        return FnKind::Unknown;
    };
    let mut h = bm_prev_sig(cx.st, cx.kind, lp);
    if h >= 0 && base_kind(cx, h as usize) >= OP_KIND_BASE && *cx.src.add(h as usize) == b'>' {
        // Type parameters.
        let Some(lt) = angle_back(cx, h as usize) else { return FnKind::Unknown };
        h = bm_prev_sig(cx.st, cx.kind, lt);
    }
    if h < 0 {
        return FnKind::Unknown;
    }
    let hw = h as usize;
    let hk = base_kind(cx, hw);
    let mut kw = if hk == IDENT { ident_kw(cx, hw) } else { 0 };
    let mut before = hw;
    if hk == IDENT && kw != K_FUNCTION {
        // A named function or a method: look before the name.
        let p = bm_prev_sig(cx.st, cx.kind, hw);
        if p < 0 {
            return FnKind::Unknown;
        }
        before = p as usize;
        kw = ident_kw(cx, before);
    } else if hk >= OP_KIND_BASE && *cx.src.add(hw) == b'*' {
        // `function* (`
        let p = bm_prev_sig(cx.st, cx.kind, hw);
        if p < 0 {
            return FnKind::Unknown;
        }
        before = p as usize;
        kw = ident_kw(cx, before);
    } else if hk != IDENT && !matches!(hk, STR | NUM | BIGINT | PRIV_IDENT) {
        if hk >= OP_KIND_BASE && *cx.src.add(hw) == b']' {
            return FnKind::Method;
        }
        return FnKind::Unknown;
    } else if hk != IDENT {
        return FnKind::Method;
    }
    if kw == K_FUNCTION {
        return function_keyword_head(cx, before);
    }
    if base_kind(cx, before) >= OP_KIND_BASE && *cx.src.add(before) == b'*' {
        let p = bm_prev_sig(cx.st, cx.kind, before);
        if p >= 0 && ident_kw(cx, p as usize) == K_FUNCTION {
            return function_keyword_head(cx, p as usize);
        }
        return FnKind::Method;
    }
    // A method name preceded by a modifier, accessor keyword, decorator or member separator.
    let bk = base_kind(cx, before);
    if bk == IDENT {
        let e = bm_next1(cx.st, before + 1, cx.n);
        if is_modifier(kw) || (kw == 0 && is_accessor_word(cx, before, e)) {
            return FnKind::Method;
        }
        if kw == 0 && before_decorator(cx, before).is_some() {
            return FnKind::Method;
        }
        return FnKind::Unknown;
    }
    if bk >= OP_KIND_BASE {
        return match *cx.src.add(before) {
            b'{' | b';' | b'}' | b',' => FnKind::Method,
            b')' if before_decorator(cx, before).is_some() => FnKind::Method,
            _ => FnKind::Unknown,
        };
    }
    FnKind::Unknown
}

/// A `function` keyword at `w`: a declaration or an expression, from the token before it.
unsafe fn function_keyword_head(cx: Cx, w: usize) -> FnKind {
    let mut kw = w;
    let mut p = bm_prev_sig(cx.st, cx.kind, w);
    if p >= 0 && ident_kw(cx, p as usize) == K_ASYNC {
        kw = p as usize;
        p = bm_prev_sig(cx.st, cx.kind, kw);
    }
    match stmt_position(cx, p, kw) {
        Some(true) => FnKind::Decl,
        Some(false) => FnKind::Expr,
        None => FnKind::Unknown,
    }
}

/// Is the `class` keyword at `w` a declaration (a statement follows its body) rather than an
/// expression?
unsafe fn class_is_declaration(cx: Cx, w: usize) -> Option<bool> {
    stmt_position(cx, bm_prev_sig(cx.st, cx.kind, w), w)
}

/// Does the declaration keyword at `kw` start a statement, given the token `p` before it?
/// Decorators are skipped; a line break after a value or restricted keyword ends the statement.
unsafe fn stmt_position(cx: Cx, p: i64, kw: usize) -> Option<bool> {
    if p < 0 {
        return Some(true);
    }
    if !tick() {
        return None;
    }
    let pw = p as usize;
    let pk = base_kind(cx, pw);
    let newline = lt_in_range(cx.src, bm_next1(cx.st, pw + 1, cx.n), kw);
    if pk == IDENT {
        return match ident_kw(cx, pw) {
            K_EXPORT | K_DEFAULT | K_DECLARE | K_ABSTRACT | K_ELSE | K_DO => Some(true),
            K_RETURN | K_THROW | K_YIELD | K_BREAK | K_CONTINUE | K_DEBUGGER => Some(newline),
            K_THIS | K_SUPER | K_TRUE | K_FALSE | K_NULL => Some(newline),
            0 => match before_decorator(cx, pw) {
                Some(d) => stmt_position(cx, d, kw),
                None => Some(newline),
            },
            _ => Some(false),
        };
    }
    if pk >= OP_KIND_BASE {
        let c = *cx.src.add(pw);
        return match c {
            b';' | b'{' | b'}' => Some(true),
            b')' => match before_decorator(cx, pw) {
                Some(d) => stmt_position(cx, d, kw),
                None => Some(newline),
            },
            b']' => Some(newline),
            b'>' if newline => {
                // `let x: Foo<T>` then a declaration on the next line.
                let lt = angle_back(cx, pw)?;
                match angle_kind(cx, lt, 1) {
                    LtKind::List(Lt::Type(_)) => Some(true),
                    // An assertion `<T>` wants its operand.
                    LtKind::List(_) | LtKind::Cmp => Some(false),
                    LtKind::Unknown => None,
                }
            }
            b'+' | b'-' if *cx.src.add(pw + 1) == c => Some(newline),
            b'!' if cx.ts => Some(newline),
            // A label or a case clause, or an object value.
            b':' => None,
            _ => Some(false),
        };
    }
    if matches!(pk, STR | NUM | BIGINT | TMPL_NOSUB | TMPL_TAIL | REGEX | PRIV_IDENT) {
        return Some(newline);
    }
    Some(false)
}

/// The `class` keyword of the declaration head ending before the `{` at `lb`, if `decl_head_kind`
/// found one.
unsafe fn class_keyword_before(cx: Cx, lb: usize, outermost: bool) -> Option<usize> {
    let mut q = bm_prev_sig(cx.st, cx.kind, lb);
    let (mut ang, mut par, mut brc) = (0i32, 0i32, 0i32);
    for _ in 0..128 {
        if q < 0 || !tick() {
            return None;
        }
        let w = q as usize;
        let k = base_kind(cx, w);
        if k >= OP_KIND_BASE {
            match *cx.src.add(w) {
                b'>' => ang += run_len(cx, w, b'>'),
                b'<' => ang -= run_len(cx, w, b'<'),
                b')' => par += 1,
                b'(' => par -= 1,
                b'}' => brc += 1,
                b'{' => brc -= 1,
                _ => {}
            }
        } else if k == IDENT && ang == 0 && par == 0 && brc == 0 && ident_kw(cx, w) == K_CLASS {
            // `class extends class {} {`: the inner class is heritage.
            let p = bm_prev_sig(cx.st, cx.kind, w);
            if outermost && p >= 0 && ident_kw(cx, p as usize) == K_EXTENDS {
                q = p;
                continue;
            }
            return Some(w);
        }
        q = bm_prev_sig(cx.st, cx.kind, w);
    }
    None
}

/// The `{` at `lb` closed a group on the query's level: what starts after its `}`, when that does
/// not depend on anything before the group.
unsafe fn block_end_boundary(cx: Cx, lb: usize, rb: usize) -> Option<Boundary> {
    match brace_seed_kind(cx, lb, 0)? {
        Fk::Block | Fk::EnumBody => Some(Boundary::Stmt),
        Fk::StaticBlock => Some(Boundary::Member),
        Fk::ClassBody => {
            let kw = class_keyword_before(cx, lb, true)?;
            if class_is_declaration(cx, kw)? { Some(Boundary::Stmt) } else { None }
        }
        Fk::TypeLit => {
            // An interface body; a type literal continues its type.
            let p = bm_prev_sig(cx.st, cx.kind, lb);
            if p >= 0 && decl_head_kind(cx, p as usize) == Some(Fk::TypeLit) {
                Some(Boundary::Stmt)
            } else {
                None
            }
        }
        Fk::FnBody => match fn_body_head(cx, lb) {
            FnKind::Decl => Some(Boundary::Stmt),
            FnKind::Method => {
                // An object literal's method is followed by `,` or `}`.
                let nx = next_sig(cx, rb + 1);
                if nx < cx.n && base_kind(cx, nx) >= OP_KIND_BASE && *cx.src.add(nx) == b',' {
                    None
                } else {
                    Some(Boundary::Member)
                }
            }
            FnKind::Expr | FnKind::Unknown => None,
        },
        _ => None,
    }
}

/// What starts after the `}` at `rb` that closes the `{` at `lb`, when the group alone settles it.
unsafe fn brace_close_after(cx: Cx, lb: usize) -> Option<After> {
    match brace_seed_kind(cx, lb, 0)? {
        Fk::Block | Fk::EnumBody | Fk::StaticBlock => Some(After::Operand),
        Fk::ClassBody => {
            let kw = class_keyword_before(cx, lb, true)?;
            Some(if class_is_declaration(cx, kw)? { After::Operand } else { After::Value })
        }
        Fk::TypeLit => {
            let p = bm_prev_sig(cx.st, cx.kind, lb);
            if p >= 0 && decl_head_kind(cx, p as usize) == Some(Fk::TypeLit) {
                Some(After::Operand)
            } else {
                None
            }
        }
        Fk::FnBody => match fn_body_head(cx, lb) {
            FnKind::Decl | FnKind::Method => Some(After::Operand),
            FnKind::Expr => Some(After::Value),
            FnKind::Unknown => None,
        },
        // A `/` cannot continue an arrow function: a new statement starts.
        Fk::ArrowBody => Some(After::Operand),
        Fk::Object | Fk::Pattern => Some(After::Value),
        _ => None,
    }
}

/// Do the tokens from `at` read as a class or interface member rather than a statement? Used when
/// the frame around a boundary lies beyond the scan.
unsafe fn looks_like_member(cx: Cx, at: usize) -> bool {
    let mut p = next_sig(cx, at);
    // Decorators.
    let mut decorated = false;
    for _ in 0..8 {
        if p >= cx.n || base_kind(cx, p) < OP_KIND_BASE || *cx.src.add(p) != b'@' {
            break;
        }
        decorated = true;
        p = next_sig(cx, p + 1);
        // `a.b.c`
        loop {
            if p >= cx.n || base_kind(cx, p) != IDENT {
                return false;
            }
            p = next_sig(cx, p + 1);
            if p < cx.n && base_kind(cx, p) >= OP_KIND_BASE && *cx.src.add(p) == b'.' {
                p = next_sig(cx, p + 1);
                continue;
            }
            break;
        }
        if p < cx.n && base_kind(cx, p) >= OP_KIND_BASE && *cx.src.add(p) == b'(' {
            let Some(rp) = skip_group_fwd(cx, p, b'(', b')') else { return false };
            p = next_sig(cx, rp + 1);
        }
    }
    if p >= cx.n {
        return false;
    }
    let k = base_kind(cx, p);
    if k == IDENT {
        let e = bm_next1(cx.st, p + 1, cx.n);
        let kwc = word_kw(cx, p, e - p);
        if decorated && matches!(kwc, K_CLASS | K_EXPORT | K_ABSTRACT | K_DECLARE) {
            return false;
        }
        let nx = next_sig(cx, e);
        let nx_is = |c: u8| nx < cx.n && base_kind(cx, nx) >= OP_KIND_BASE && *cx.src.add(nx) == c;
        let nx_name =
            nx < cx.n && matches!(base_kind(cx, nx), IDENT | STR | NUM | BIGINT | PRIV_IDENT);
        if is_modifier(kwc) || (kwc == 0 && is_accessor_word(cx, p, e)) {
            // A modifier before a member name, or `static {`.
            if nx_name || nx_is(b'[') || nx_is(b'*') || nx_is(b'{') {
                return true;
            }
            // Used as a name itself.
            return name_looks_like_member(cx, p, e);
        }
        return match kwc {
            K_DECLARE | K_TYPE | K_INTERFACE | K_NAMESPACE | K_MODULE | K_ENUM | K_LET
            | K_CONST | K_VAR | K_USING => {
                // `type X =` / `let x =` statements vs members so named.
                name_looks_like_member(cx, p, e)
            }
            0 => name_looks_like_member(cx, p, e),
            // `default:` / `case x:` label clauses.
            K_DEFAULT | K_CASE => false,
            // Any other keyword before `:` names a member (`finally: T`).
            _ => nx_is(b':'),
        };
    }
    if matches!(k, STR | NUM | BIGINT | PRIV_IDENT) {
        let e = bm_next1(cx.st, p + 1, cx.n);
        return name_looks_like_member(cx, p, e);
    }
    if k >= OP_KIND_BASE {
        return match *cx.src.add(p) {
            b'*' => true,
            b'[' => {
                let Some(rb) = skip_group_fwd(cx, p, b'[', b']') else { return false };
                let nx = next_sig(cx, rb + 1);
                nx < cx.n
                    && base_kind(cx, nx) >= OP_KIND_BASE
                    && matches!(*cx.src.add(nx), b':' | b'(' | b'?' | b'<' | b'!')
            }
            _ => false,
        };
    }
    false
}

/// After a member-like name at `p..e`: does what follows make it a member?
unsafe fn name_looks_like_member(cx: Cx, p: usize, e: usize) -> bool {
    let _ = p;
    let nx = next_sig(cx, e);
    if nx >= cx.n || base_kind(cx, nx) < OP_KIND_BASE {
        return false;
    }
    let c = *cx.src.add(nx);
    let c1 = *cx.src.add(nx + 1);
    match c {
        b':' => {
            // A label names a statement; an annotation starts a type.
            let t = next_sig(cx, nx + 1);
            if t < cx.n && base_kind(cx, t) == IDENT {
                !matches!(
                    ident_kw(cx, t),
                    K_FOR
                        | K_WHILE
                        | K_DO
                        | K_IF
                        | K_SWITCH
                        | K_TRY
                        | K_THROW
                        | K_RETURN
                        | K_BREAK
                        | K_CONTINUE
                        | K_VAR
                        | K_LET
                        | K_CONST
                        | K_FUNCTION
                        | K_CLASS
                        | K_WITH
                        | K_DEBUGGER
                        | K_IMPORT
                        | K_EXPORT
                )
            } else {
                t < cx.n && base_kind(cx, t) != IDENT
            }
        }
        b'?' => c1 != b'.' && c1 != b'?',
        b'!' => {
            let t = next_sig(cx, nx + 1);
            t < cx.n && base_kind(cx, t) >= OP_KIND_BASE && *cx.src.add(t) == b':'
        }
        b'(' => {
            let Some(rp) = skip_group_fwd(cx, nx, b'(', b')') else { return false };
            let t = next_sig(cx, rp + 1);
            t < cx.n && base_kind(cx, t) >= OP_KIND_BASE && matches!(*cx.src.add(t), b'{' | b':')
        }
        b'<' if c1 != b'<' && c1 != b'=' => {
            let lim = (nx + 4096).min(cx.n);
            let (close, _) =
                angle_close_fwd_capped(cx.src, cx.st, cx.opch, cx.kind, nx + 1, lim, 1);
            let Some(gt) = close else { return false };
            let t = next_sig(cx, gt + 1);
            if !(t < cx.n && base_kind(cx, t) >= OP_KIND_BASE && *cx.src.add(t) == b'(') {
                return false;
            }
            let Some(rp) = skip_group_fwd(cx, t, b'(', b')') else { return false };
            let u = next_sig(cx, rp + 1);
            u < cx.n && base_kind(cx, u) >= OP_KIND_BASE && matches!(*cx.src.add(u), b'{' | b':')
        }
        _ => false,
    }
}

/// The closer matching the opener at `o` (raw scan over token starts, no nesting of other bracket
/// kinds).
unsafe fn skip_group_fwd(cx: Cx, o: usize, open: u8, close: u8) -> Option<usize> {
    let mut depth = 0i32;
    let mut i = o;
    for _ in 0..4096 {
        if i >= cx.n || !tick() {
            return None;
        }
        if base_kind(cx, i) >= OP_KIND_BASE {
            let c = *cx.src.add(i);
            if c == open {
                depth += 1;
            } else if c == close {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
        }
        i = bm_next1(cx.st, i + 1, cx.n);
    }
    None
}

/// What a name sits in, for the `:` after it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    /// Class body / type literal member: a type annotation follows.
    Member,
    /// Object literal key: a value follows.
    ObjectKey,
    /// Parameter, declarator, named tuple member: a type follows.
    Param,
    /// `function f`: a signature.
    FnName,
    /// Statement start: a label.
    Label,
    /// `case x`.
    Case,
    /// An operand of an expression.
    Value,
}

fn is_modifier(kwc: u8) -> bool {
    matches!(
        kwc,
        K_PUBLIC
            | K_PRIVATE
            | K_PROTECTED
            | K_STATIC
            | K_READONLY
            | K_DECLARE
            | K_ABSTRACT
            | K_OVERRIDE
            | K_ACCESSOR
            | K_ASYNC
    )
}

/// `get` / `set` spelled at `w..e`.
unsafe fn is_accessor_word(cx: Cx, w: usize, e: usize) -> bool {
    e - w == 3 && (ident_is(cx.src, w, b"get") || ident_is(cx.src, w, b"set"))
}

/// The token before a decorator that ends with the token at `w` (its name, or the `)` of its call),
/// if it is one.
unsafe fn before_decorator(cx: Cx, w: usize) -> Option<i64> {
    let mut t = w;
    if base_kind(cx, t) >= OP_KIND_BASE {
        if *cx.src.add(t) != b')' {
            return None;
        }
        let lp = match_delim_back(cx.src, cx.st, cx.kind, t, b'(', b')')?;
        let h = bm_prev_sig(cx.st, cx.kind, lp);
        if h < 0 {
            return None;
        }
        t = h as usize;
    }
    for _ in 0..16 {
        if base_kind(cx, t) != IDENT {
            return None;
        }
        let u = bm_prev_sig(cx.st, cx.kind, t);
        if u < 0 || base_kind(cx, u as usize) < OP_KIND_BASE {
            return None;
        }
        match *cx.src.add(u as usize) {
            b'@' => return Some(bm_prev_sig(cx.st, cx.kind, u as usize)),
            b'.' => {
                let v = bm_prev_sig(cx.st, cx.kind, u as usize);
                if v < 0 {
                    return None;
                }
                t = v as usize;
            }
            _ => return None,
        }
    }
    None
}

/// The role of the name token at `w`, from the tokens before it.
unsafe fn name_role(cx: Cx, w: usize, depth: u32) -> Option<Role> {
    if depth > SEED_RULE_DEPTH {
        return None;
    }
    let mut p = bm_prev_sig(cx.st, cx.kind, w);
    for _ in 0..16 {
        if !tick() {
            return None;
        }
        if p < 0 {
            return Some(Role::Label);
        }
        let pw = p as usize;
        let pk = base_kind(cx, pw);
        if pk == IDENT {
            let e = bm_next1(cx.st, pw + 1, cx.n);
            let kwc = word_kw(cx, pw, e - pw);
            if is_modifier(kwc) || (kwc == 0 && is_accessor_word(cx, pw, e)) {
                // Only members carry modifiers.
                return Some(Role::Member);
            }
            if kwc == 0 {
                if before_decorator(cx, pw).is_some() {
                    return Some(Role::Member);
                }
                // A value or type on the previous line: a member or a statement starts here.
                return if lt_in_range(cx.src, e, w) {
                    Some(after_break_role(cx, w))
                } else {
                    None
                };
            }
            return match kwc {
                K_CASE => Some(Role::Case),
                K_LET | K_CONST | K_VAR | K_USING => Some(Role::Param),
                K_FUNCTION => Some(Role::FnName),
                K_RETURN | K_THROW | K_YIELD | K_AWAIT | K_DELETE | K_VOID | K_TYPEOF | K_NEW
                | K_IN | K_OF | K_INSTANCEOF => Some(Role::Value),
                K_ELSE | K_DO => Some(Role::Label),
                // `null`, `this`, `string`, ...: a value or type ended on the previous line.
                _ => {
                    if lt_in_range(cx.src, e, w) {
                        Some(after_break_role(cx, w))
                    } else {
                        None
                    }
                }
            };
        }
        if pk >= OP_KIND_BASE {
            let c = *cx.src.add(pw);
            return match c {
                b'{' => match brace_seed_kind(cx, pw, depth + 1)? {
                    Fk::Object | Fk::Pattern => Some(Role::ObjectKey),
                    Fk::TypeLit | Fk::ClassBody => Some(Role::Member),
                    Fk::Block | Fk::FnBody | Fk::ArrowBody | Fk::StaticBlock => Some(Role::Label),
                    _ => None,
                },
                b';' | b'}' | b',' => match container_kind(cx, w, depth + 1) {
                    Some(Fk::Object | Fk::Pattern) => Some(Role::ObjectKey),
                    Some(Fk::TypeLit | Fk::ClassBody) => Some(Role::Member),
                    Some(Fk::Block | Fk::FnBody | Fk::ArrowBody | Fk::StaticBlock) if c != b',' => {
                        Some(Role::Label)
                    }
                    Some(Fk::Call | Fk::TypeBracket) if c == b',' => Some(Role::Param),
                    Some(_) => None,
                    // Beyond the scan a `,` list is almost always an object literal; after `;` or
                    // `}` the tokens after the name decide.
                    None if c == b',' => Some(Role::ObjectKey),
                    None => {
                        if looks_like_member(cx, w) {
                            Some(Role::Member)
                        } else {
                            Some(Role::Label)
                        }
                    }
                },
                b'(' => Some(Role::Param),
                b'[' => match type_start(cx, pw, depth + 1) {
                    // A named tuple member.
                    TypeStart::Decl | TypeStart::Expr => Some(Role::Param),
                    _ => None,
                },
                b'?' => Some(Role::Value),
                b'.' => {
                    // `...name: T`: a rest parameter or a named rest element of a tuple type; a
                    // spread's operand otherwise.
                    if *cx.src.add(pw + 1) == b'.' {
                        let e = bm_next1(cx.st, w + 1, cx.n);
                        let nx = next_sig(cx, e);
                        let annotated = nx < cx.n
                            && base_kind(cx, nx) >= OP_KIND_BASE
                            && *cx.src.add(nx) == b':';
                        if annotated { Some(Role::Param) } else { Some(Role::Value) }
                    } else {
                        Some(Role::Value)
                    }
                }
                b':' => Some(Role::Label),
                b']' | b'>' => {
                    let e = bm_next1(cx.st, pw + 1, cx.n);
                    if lt_in_range(cx.src, e, w) { Some(after_break_role(cx, w)) } else { None }
                }
                b')' => {
                    if before_decorator(cx, pw).is_some() {
                        Some(Role::Member)
                    } else if lt_in_range(cx.src, pw + 1, w) {
                        Some(after_break_role(cx, w))
                    } else {
                        Some(Role::Label)
                    }
                }
                b'*' => {
                    // A generator marker at member position, else a product: what precedes it
                    // decides.
                    p = bm_prev_sig(cx.st, cx.kind, pw);
                    continue;
                }
                _ => None,
            };
        }
        if matches!(pk, STR | NUM | BIGINT | TMPL_NOSUB | TMPL_TAIL | REGEX) {
            let e = bm_next1(cx.st, pw + 1, cx.n);
            return if lt_in_range(cx.src, e, w) { Some(after_break_role(cx, w)) } else { None };
        }
        return None;
    }
    None
}

/// A name starting a member or statement after a line break: a member when what follows reads as an
/// annotation or signature (`baz: Qux`, `m(): T`), a label otherwise.
unsafe fn after_break_role(cx: Cx, w: usize) -> Role {
    let e = bm_next1(cx.st, w + 1, cx.n);
    if name_looks_like_member(cx, w, e) { Role::Member } else { Role::Label }
}

/// What the `{` after the `:` at `colon` opens: a type literal (annotation), an object literal
/// (value, ternary branch) or a block (label, `case`).
unsafe fn colon_kind(cx: Cx, colon: usize, depth: u32) -> Option<Fk> {
    let key = MEMO_COLON | colon as u64;
    if let Some(v) = memo_get(key) {
        return if v == 0 { None } else { Some(FK_LIST[v as usize - 1]) };
    }
    let r = colon_kind_uncached(cx, colon, depth);
    memo_put(key, r.and_then(|k| FK_LIST.iter().position(|f| *f == k)).map_or(0, |i| i as u64 + 1));
    r
}

/// `Fk` by discriminant, for the memo.
const FK_LIST: [Fk; 5] = [Fk::TypeLit, Fk::Object, Fk::Block, Fk::Container, Fk::ClassBody];

unsafe fn colon_kind_uncached(cx: Cx, colon: usize, depth: u32) -> Option<Fk> {
    if depth > SEED_RULE_DEPTH {
        return None;
    }
    // `function (...): {`: a return type, whatever surrounds it.
    let q = bm_prev_sig(cx.st, cx.kind, colon);
    if q >= 0 && base_kind(cx, q as usize) >= OP_KIND_BASE && *cx.src.add(q as usize) == b')' {
        if let Some(lp) = match_delim_back(cx.src, cx.st, cx.kind, q as usize, b'(', b')') {
            if paren_head_is_function(cx, lp) {
                return Some(Fk::TypeLit);
            }
        }
    }
    // A pending `?` on this level makes it a ternary or conditional type, `case` a case clause.
    // Type lists count as groups so a `,` inside one does not end the search.
    let (mut par, mut brk, mut brc, mut ang) = (0i32, 0i32, 0i32, 0i32);
    // Earlier `:` on the level, each owed a `?` of its own.
    let mut owed = 0u32;
    let mut q = bm_prev_sig(cx.st, cx.kind, colon);
    let mut first = true;
    let mut steps = 0u32;
    while q >= 0 {
        steps += 1;
        if steps > SEED_TERNARY_SCAN || !tick() {
            break;
        }
        let w = q as usize;
        let k = base_kind(cx, w);
        if k >= OP_KIND_BASE {
            let c = *cx.src.add(w);
            let arrow = c == b'>' && w > 0 && *cx.src.add(w - 1) == b'=';
            if par > 0 || brk > 0 || brc > 0 {
                match c {
                    b')' => par += 1,
                    b'(' => par -= 1,
                    b']' => brk += 1,
                    b'[' => brk -= 1,
                    b'}' => brc += 1,
                    b'{' => brc -= 1,
                    _ => {}
                }
            } else if ang > 0 {
                match c {
                    b'>' if !arrow => ang += run_len(cx, w, b'>'),
                    b'<' => ang = (ang - run_len(cx, w, b'<')).max(0),
                    b')' => par += 1,
                    b']' => brk += 1,
                    b'}' => brc += 1,
                    b'=' if *cx.src.add(w + 1) == b'>' => {}
                    b'(' | b'[' | b'{' | b';' | b'=' => break,
                    _ => {}
                }
            } else {
                match c {
                    b')' => par += 1,
                    b']' => brk += 1,
                    // `{...} : {` continues a consequent object; any other `}` closed a block
                    // before the statement.
                    b'}' if first => brc += 1,
                    b'>' if !arrow => ang += run_len(cx, w, b'>'),
                    // An arrow inside a branch (`c ? () => x : {`).
                    b'=' if *cx.src.add(w + 1) == b'>' => {}
                    b'(' | b'[' | b'{' | b'}' | b';' | b',' | b'=' => break,
                    b':' => owed += 1,
                    b'?' => {
                        let c1 = *cx.src.add(w + 1);
                        if first {
                            // `x?: {`
                            return Some(Fk::TypeLit);
                        }
                        if c1 != b'.' && c1 != b'?' {
                            if owed > 0 {
                                owed -= 1;
                            } else {
                                return question_kind(cx, w, depth);
                            }
                        }
                    }
                    _ => {}
                }
            }
        } else if k == IDENT && !first && par == 0 && brk == 0 && brc == 0 {
            let e = bm_next1(cx.st, w + 1, cx.n);
            if word_kw(cx, w, e - w) == K_CASE {
                return Some(Fk::Block);
            }
        }
        first = false;
        q = bm_prev_sig(cx.st, cx.kind, w);
    }
    // The token before the `:` decides.
    let q = bm_prev_sig(cx.st, cx.kind, colon);
    if q < 0 {
        return None;
    }
    let w = q as usize;
    let k = base_kind(cx, w);
    if k >= OP_KIND_BASE {
        return match *cx.src.add(w) {
            b')' => {
                let lp = match_delim_back(cx.src, cx.st, cx.kind, w, b'(', b')')?;
                signature_kind(cx, lp, depth)
            }
            b']' => {
                let lb = match_delim_back(cx.src, cx.st, cx.kind, w, b'[', b']')?;
                bracket_member(cx, lb, depth)
            }
            b'?' => Some(Fk::TypeLit),
            b'}' => {
                // `const {a}: {` / `({a}: {`: a pattern's annotation.
                let lb = match_delim_back(cx.src, cx.st, cx.kind, w, b'{', b'}')?;
                let p = bm_prev_sig(cx.st, cx.kind, lb);
                if p < 0 {
                    return None;
                }
                let pw = p as usize;
                let pattern = matches!(ident_kw(cx, pw), K_LET | K_CONST | K_VAR | K_USING)
                    || (base_kind(cx, pw) >= OP_KIND_BASE
                        && matches!(*cx.src.add(pw), b'(' | b','));
                if pattern { Some(Fk::TypeLit) } else { None }
            }
            b'!' => {
                // `x!: {`
                let p = bm_prev_sig(cx.st, cx.kind, w);
                if p < 0 || !matches!(base_kind(cx, p as usize), IDENT | PRIV_IDENT) {
                    return None;
                }
                match name_role(cx, p as usize, depth + 1)? {
                    Role::Member | Role::Param => Some(Fk::TypeLit),
                    _ => None,
                }
            }
            _ => None,
        };
    }
    if matches!(k, IDENT | STR | NUM | BIGINT | PRIV_IDENT) {
        return match name_role(cx, w, depth + 1)? {
            Role::Member | Role::Param => Some(Fk::TypeLit),
            Role::ObjectKey => Some(Fk::Object),
            Role::Label | Role::Case => Some(Fk::Block),
            Role::FnName | Role::Value => None,
        };
    }
    None
}

/// The `?` at `w` on the level of a later `:`: a conditional type (type literal follows), an
/// optional member marker `x?(` / `x?<` (return type follows) or a ternary (object follows).
unsafe fn question_kind(cx: Cx, w: usize, depth: u32) -> Option<Fk> {
    match type_start(cx, w, depth + 1) {
        TypeStart::Decl | TypeStart::Expr => return Some(Fk::TypeLit),
        TypeStart::Unknown => return None,
        TypeStart::No => {}
    }
    let nx = next_sig(cx, w + 1);
    if nx < cx.n && base_kind(cx, nx) >= OP_KIND_BASE && matches!(*cx.src.add(nx), b'(' | b'<') {
        let p = bm_prev_sig(cx.st, cx.kind, w);
        if p >= 0 && matches!(base_kind(cx, p as usize), IDENT | STR | NUM | BIGINT | PRIV_IDENT) {
            if name_role(cx, p as usize, depth + 1) == Some(Role::Member) {
                return Some(Fk::TypeLit);
            }
        }
    }
    Some(Fk::Object)
}

/// The `[` at `lb` starts a member key (`[k]: {`, `[k](): {`, `readonly [k: string]: {`): what
/// container it is in.
unsafe fn bracket_member(cx: Cx, lb: usize, depth: u32) -> Option<Fk> {
    let p = bm_prev_sig(cx.st, cx.kind, lb);
    if p < 0 {
        return None;
    }
    let pw = p as usize;
    let pk = base_kind(cx, pw);
    if pk == IDENT {
        let e = bm_next1(cx.st, pw + 1, cx.n);
        let kwc = word_kw(cx, pw, e - pw);
        if is_modifier(kwc) || (kwc == 0 && is_accessor_word(cx, pw, e)) {
            return Some(Fk::TypeLit);
        }
        // `const [a]: T`: an array pattern's annotation.
        if matches!(kwc, K_LET | K_CONST | K_VAR | K_USING) {
            return Some(Fk::TypeLit);
        }
        // `@dec [k]`: a decorated member.
        if kwc == 0 && before_decorator(cx, pw).is_some() {
            return Some(Fk::TypeLit);
        }
        return None;
    }
    if pk >= OP_KIND_BASE {
        let kind = match *cx.src.add(pw) {
            b'{' => brace_seed_kind(cx, pw, depth + 1)?,
            b';' | b'}' => container_kind(cx, lb, depth + 1).unwrap_or(Fk::TypeLit),
            b',' => container_kind(cx, lb, depth + 1).unwrap_or(Fk::Object),
            // `([a]: T)`: a parameter pattern's annotation.
            b'(' => Fk::Call,
            b'*' => Fk::ClassBody,
            _ => return None,
        };
        return match kind {
            Fk::TypeLit | Fk::ClassBody | Fk::Call => Some(Fk::TypeLit),
            Fk::Object => Some(Fk::Object),
            _ => None,
        };
    }
    None
}

/// The `<` matching the `>` at `gt` (counting glued `>>` / `<<` runs).
unsafe fn angle_back(cx: Cx, gt: usize) -> Option<usize> {
    let mut depth = run_len(cx, gt, b'>');
    let (mut par, mut brk, mut brc) = (0i32, 0i32, 0i32);
    let mut q = bm_prev_sig(cx.st, cx.kind, gt);
    for _ in 0..128 {
        if q < 0 || !tick() {
            return None;
        }
        let w = q as usize;
        if base_kind(cx, w) >= OP_KIND_BASE {
            let inside = par > 0 || brk > 0 || brc > 0;
            match *cx.src.add(w) {
                b'>' if !inside => {
                    if !(w > 0 && *cx.src.add(w - 1) == b'=') {
                        depth += run_len(cx, w, b'>');
                    }
                }
                b'<' if !inside => {
                    depth -= run_len(cx, w, b'<');
                    if depth <= 0 {
                        return Some(w);
                    }
                }
                // Inside a group (a type literal constraint) anything goes.
                b';' if inside => {}
                b')' => par += 1,
                b'(' => {
                    if par == 0 {
                        return None;
                    }
                    par -= 1;
                }
                b']' => brk += 1,
                b'[' => {
                    if brk == 0 {
                        return None;
                    }
                    brk -= 1;
                }
                b'}' => brc += 1,
                b'{' => {
                    if brc == 0 {
                        return None;
                    }
                    brc -= 1;
                }
                b';' => return None,
                _ => {}
            }
        }
        q = bm_prev_sig(cx.st, cx.kind, w);
    }
    None
}

/// Length of the token at `w` if it is made only of `c` bytes, else 0.
unsafe fn run_len(cx: Cx, w: usize, c: u8) -> i32 {
    let e = bm_next1(cx.st, w + 1, cx.n);
    let mut i = w;
    while i < e && *cx.src.add(i) == c {
        i += 1;
    }
    if i == e { (e - w) as i32 } else { 0 }
}

/// What the `{` after `(...):` opens, for the `(` at `lp`: a return type (signature, arrow
/// function) or a block (`case f(x): {`).
unsafe fn signature_kind(cx: Cx, lp: usize, depth: u32) -> Option<Fk> {
    let h = bm_prev_sig(cx.st, cx.kind, lp);
    if h < 0 {
        return Some(Fk::TypeLit);
    }
    let hw = h as usize;
    let hk = base_kind(cx, hw);
    if hk == IDENT {
        let e = bm_next1(cx.st, hw + 1, cx.n);
        return match word_kw(cx, hw, e - hw) {
            K_FUNCTION | K_ASYNC => Some(Fk::TypeLit),
            K_CASE => Some(Fk::Block),
            // Any other keyword before `(...):` names a method (`return(): T`).
            _ => named_signature(cx, hw),
        };
    }
    match hk {
        STR | NUM | BIGINT | PRIV_IDENT => named_signature(cx, hw),
        _ if hk >= OP_KIND_BASE => match *cx.src.add(hw) {
            b']' => {
                // A computed method name.
                let lb = match_delim_back(cx.src, cx.st, cx.kind, hw, b'[', b']')?;
                match bracket_member(cx, lb, depth)? {
                    Fk::TypeLit | Fk::Object => Some(Fk::TypeLit),
                    _ => None,
                }
            }
            b'>' if hw > 0 && *cx.src.add(hw - 1) == b'=' => Some(Fk::TypeLit),
            b'>' => {
                // Type parameters: `m<T>(): {`, `<T>(a): {`.
                let lt = angle_back(cx, hw)?;
                let g = bm_prev_sig(cx.st, cx.kind, lt);
                if g < 0 {
                    return Some(Fk::TypeLit);
                }
                let gw = g as usize;
                let gk = base_kind(cx, gw);
                if gk == IDENT {
                    let e = bm_next1(cx.st, gw + 1, cx.n);
                    return match word_kw(cx, gw, e - gw) {
                        K_CASE => Some(Fk::Block),
                        K_FUNCTION | K_ASYNC | K_RETURN | K_YIELD | K_AWAIT | K_TYPEOF | K_VOID
                        | K_DELETE | K_THROW | K_ELSE | K_DO | K_IN | K_OF | K_INSTANCEOF
                        | K_DEFAULT => Some(Fk::TypeLit),
                        K_NEW | K_THIS | K_SUPER | K_IMPORT | K_EXTENDS | K_IF | K_WHILE
                        | K_FOR | K_SWITCH | K_CATCH | K_WITH => None,
                        _ => named_signature(cx, gw),
                    };
                }
                if gk >= OP_KIND_BASE {
                    let gc = *cx.src.add(gw);
                    // After a value on the same line `<` compares; after a member that ended on the
                    // previous line (`): T` then `<U>(...)`) a call signature starts.
                    let value_end = matches!(gc, b')' | b']')
                        || (gc == b'>' && !(gw > 0 && *cx.src.add(gw - 1) == b'='));
                    let e = bm_next1(cx.st, gw + 1, cx.n);
                    return match gc {
                        b'.' => None,
                        _ if value_end && !lt_in_range(cx.src, e, lt) => None,
                        _ => Some(Fk::TypeLit),
                    };
                }
                None
            }
            b'*' | b'=' | b'(' | b',' | b':' | b'[' | b'{' | b'}' | b';' | b'!' | b'&' | b'|'
            | b'+' | b'-' | b'~' | b'?' | b'^' | b'%' | b'/' | b'<' => Some(Fk::TypeLit),
            _ => None,
        },
        _ => None,
    }
}

/// Is the `(` at `lp` the parameter list of a `function` keyword (`function (`, `function* f(`,
/// `function f<T>(`)?
unsafe fn paren_head_is_function(cx: Cx, lp: usize) -> bool {
    let mut h = bm_prev_sig(cx.st, cx.kind, lp);
    if h >= 0 && base_kind(cx, h as usize) >= OP_KIND_BASE && *cx.src.add(h as usize) == b'>' {
        match angle_back(cx, h as usize) {
            Some(lt) => h = bm_prev_sig(cx.st, cx.kind, lt),
            None => return false,
        }
    }
    if h < 0 {
        return false;
    }
    let hw = h as usize;
    let hk = base_kind(cx, hw);
    if hk == IDENT {
        if ident_kw(cx, hw) == K_FUNCTION {
            return true;
        }
        let p = bm_prev_sig(cx.st, cx.kind, hw);
        if p < 0 {
            return false;
        }
        let pw = p as usize;
        if ident_kw(cx, pw) == K_FUNCTION {
            return true;
        }
        if base_kind(cx, pw) >= OP_KIND_BASE && *cx.src.add(pw) == b'*' {
            let pp = bm_prev_sig(cx.st, cx.kind, pw);
            return pp >= 0 && ident_kw(cx, pp as usize) == K_FUNCTION;
        }
        return false;
    }
    if hk >= OP_KIND_BASE && *cx.src.add(hw) == b'*' {
        let p = bm_prev_sig(cx.st, cx.kind, hw);
        return p >= 0 && ident_kw(cx, p as usize) == K_FUNCTION;
    }
    false
}

/// `name(...): {` with the name at `w`: a method or function signature's return type, unless a
/// `case` clause tests a call.
unsafe fn named_signature(cx: Cx, w: usize) -> Option<Fk> {
    let p = bm_prev_sig(cx.st, cx.kind, w);
    if p < 0 {
        return Some(Fk::TypeLit);
    }
    let pw = p as usize;
    if base_kind(cx, pw) == IDENT {
        let e = bm_next1(cx.st, pw + 1, cx.n);
        let broken = lt_in_range(cx.src, e, w);
        return match ident_kw(cx, pw) {
            K_CASE => Some(Fk::Block),
            K_RETURN | K_THROW | K_YIELD | K_AWAIT | K_DELETE | K_TYPEOF | K_NEW | K_IN | K_OF
            | K_INSTANCEOF | K_ELSE | K_DO => None,
            // `): void` then a signature on the next line.
            K_VOID if !broken => None,
            _ => Some(Fk::TypeLit),
        };
    }
    if base_kind(cx, pw) >= OP_KIND_BASE && matches!(*cx.src.add(pw), b'.' | b'?') {
        return None;
    }
    Some(Fk::TypeLit)
}

/// What the type (if any) that the opener at `o` belongs to started from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TypeStart {
    /// Not in a type.
    No,
    /// A declaration type (`let x: `, `(a: `, `type X = `, `extends`, ...): the region ends the
    /// statement.
    Decl,
    /// `x as T` / `x satisfies T`: the region ends inside an expression.
    Expr,
    /// The rules cannot tell.
    Unknown,
}

unsafe fn type_start(cx: Cx, o: usize, depth: u32) -> TypeStart {
    type_start_at(cx, o, depth).0
}

/// Scan back from `o` over a type's tail (names, dots, balanced groups) to the token that decided
/// whether a type is being read: what it decided and where. An unmatched opener holding `o` is
/// decided by its own context, a `<` list by its head, or by TypeScript's speculation when the head
/// is an expression.
unsafe fn type_start_at(cx: Cx, o: usize, depth: u32) -> (TypeStart, usize) {
    let key = MEMO_TYPE | o as u64;
    if let Some(v) = memo_get(key) {
        let kind = match v >> 56 {
            0 => TypeStart::No,
            1 => TypeStart::Decl,
            2 => TypeStart::Expr,
            _ => TypeStart::Unknown,
        };
        return (kind, (v & ((1 << 56) - 1)) as usize);
    }
    let r = type_start_at_uncached(cx, o, depth);
    let tag: u64 = match r.0 {
        TypeStart::No => 0,
        TypeStart::Decl => 1,
        TypeStart::Expr => 2,
        TypeStart::Unknown => 3,
    };
    memo_put(key, (tag << 56) | r.1 as u64);
    r
}

unsafe fn type_start_at_uncached(cx: Cx, o: usize, depth: u32) -> (TypeStart, usize) {
    if !cx.ts {
        return (TypeStart::No, o);
    }
    if depth > SEED_RULE_DEPTH {
        return (TypeStart::Unknown, o);
    }
    let (mut ang, mut brk, mut par, mut brc) = (0i32, 0i32, 0i32, 0i32);
    // After `,` / `=`: only the list's opener decides.
    let mut skip = false;
    let mut q = bm_prev_sig(cx.st, cx.kind, o);
    let mut first = true;
    while q >= 0 {
        if !tick() {
            return (TypeStart::Unknown, o);
        }
        let w = q as usize;
        let at_start = first;
        first = false;
        let k = base_kind(cx, w);
        let c = *cx.src.add(w);
        if k >= OP_KIND_BASE {
            let c1 = *cx.src.add(w + 1);
            let arrow = c == b'>' && w > 0 && *cx.src.add(w - 1) == b'=';
            let fused_arrow = c == b'=' && c1 == b'>';
            let lvl = ang > 0 || brk > 0 || par > 0 || brc > 0;
            match c {
                b'>' if !arrow => ang += run_len(cx, w, b'>'),
                b']' => brk += 1,
                b')' => par += 1,
                b'}' => brc += 1,
                b'<' if ang > 0 => {
                    ang -= run_len(cx, w, b'<');
                    if ang < 0 {
                        // The first `<` of a `<<` holds `o`.
                        return angle_list_start(cx, w, o, depth);
                    }
                }
                b'[' if brk > 0 => brk -= 1,
                b'(' if par > 0 => par -= 1,
                b'{' if brc > 0 => {
                    brc -= 1;
                    if brc == 0 && ang == 0 && brk == 0 && par == 0 {
                        // Only a type literal can sit in a type's tail.
                        match brace_seed_kind(cx, w, depth + 1) {
                            Some(Fk::TypeLit) => {}
                            Some(_) => return (TypeStart::No, o),
                            None => return (TypeStart::Unknown, o),
                        }
                    }
                }
                b';' => return (TypeStart::No, o),
                _ if lvl => {
                    if matches!(c, b'(' | b'[' | b'{') {
                        // The group holding `o` closed inside another one.
                        return (TypeStart::No, o);
                    }
                }
                b'(' | b'[' => return type_start_at(cx, w, depth + 1),
                b'{' => {
                    return match brace_seed_kind(cx, w, depth + 1) {
                        Some(Fk::TypeLit) => {
                            let r = type_start_at(cx, w, depth + 1);
                            if r.0 == TypeStart::No { (TypeStart::Decl, w) } else { r }
                        }
                        Some(_) => (TypeStart::No, o),
                        None => (TypeStart::Unknown, o),
                    };
                }
                b'<' => return angle_list_start(cx, w, o, depth),
                _ if skip => {}
                b'>' | b'=' if arrow || fused_arrow => {
                    // `=>`: a function type continues after its parameter list; an arrow function's
                    // body is an expression.
                    let eq = if fused_arrow { w } else { w - 1 };
                    let p = bm_prev_sig(cx.st, cx.kind, eq);
                    if p < 0 || base_kind(cx, p as usize) < OP_KIND_BASE {
                        return (TypeStart::No, o);
                    }
                    if *cx.src.add(p as usize) != b')' {
                        return (TypeStart::No, o);
                    }
                    let Some(lp) = match_delim_back(cx.src, cx.st, cx.kind, p as usize, b'(', b')')
                    else {
                        return (TypeStart::Unknown, o);
                    };
                    return type_start_at(cx, lp, depth + 1);
                }
                b'=' => {
                    if c1 == b'=' {
                        return (TypeStart::No, o);
                    }
                    if type_alias_eq(cx, w) {
                        return (TypeStart::Decl, w);
                    }
                    skip = true;
                }
                b',' => skip = true,
                b':' => {
                    return match colon_kind(cx, w, depth + 1) {
                        Some(Fk::TypeLit) => (TypeStart::Decl, w),
                        Some(_) => (TypeStart::No, o),
                        None => (TypeStart::Unknown, o),
                    };
                }
                b'?' => {
                    if c1 == b'.' || c1 == b'?' {
                        return (TypeStart::No, o);
                    }
                    // A conditional type's `?`, or an optional `a?, `.
                    let r = type_start_at(cx, w, depth + 1);
                    return if r.0 == TypeStart::No { (TypeStart::No, o) } else { r };
                }
                b'|' | b'&' if c1 != c => {}
                b'.' if c1 != b'.' || *cx.src.add(w + 2) == b'.' => {}
                b'-' | b'+' if c1 != c && c1 != b'=' => {
                    // A signed literal type (`-1`) or a sum: what precedes decides either way.
                    let nx = next_sig(cx, w + 1);
                    if !(nx < cx.n && matches!(base_kind(cx, nx), NUM | BIGINT)) {
                        return (TypeStart::No, o);
                    }
                }
                _ => return (TypeStart::No, o),
            }
        } else if k == IDENT {
            if ang == 0 && brk == 0 && par == 0 && brc == 0 {
                let e = bm_next1(cx.st, w + 1, cx.n);
                match word_kw(cx, w, e - w) {
                    // In a list (`f(x as T, {`) an `as` belongs to an earlier item.
                    K_AS | K_SATISFIES if skip => {}
                    K_AS | K_SATISFIES => return (TypeStart::Expr, w),
                    K_EXTENDS => {
                        // `class C extends (`: a heritage expression right after the keyword;
                        // `class C extends A<` a type.
                        let p = bm_prev_sig(cx.st, cx.kind, w);
                        let mut after_class = false;
                        if at_start && p >= 0 && base_kind(cx, p as usize) == IDENT {
                            let pw = p as usize;
                            let pe = bm_next1(cx.st, pw + 1, cx.n);
                            after_class = match word_kw(cx, pw, pe - pw) {
                                K_CLASS => true,
                                0 => {
                                    let pp = bm_prev_sig(cx.st, cx.kind, pw);
                                    pp >= 0 && ident_kw(cx, pp as usize) == K_CLASS
                                }
                                _ => false,
                            };
                        }
                        return if after_class { (TypeStart::No, o) } else { (TypeStart::Decl, w) };
                    }
                    K_IMPLEMENTS => return (TypeStart::Decl, w),
                    K_IN => {
                        // `[K in T]` (a mapped type) vs `x in y`.
                        let p = bm_prev_sig(cx.st, cx.kind, w);
                        if p >= 0 && base_kind(cx, p as usize) == IDENT {
                            let pp = bm_prev_sig(cx.st, cx.kind, p as usize);
                            if pp >= 0
                                && base_kind(cx, pp as usize) >= OP_KIND_BASE
                                && *cx.src.add(pp as usize) == b'['
                            {
                                return (TypeStart::Decl, w);
                            }
                        }
                        return (TypeStart::No, o);
                    }
                    K_IMPORT => {
                        // `import('m')` in a type (or a dynamic import) vs an import statement.
                        let nx = next_sig(cx, e);
                        if !(nx < cx.n
                            && base_kind(cx, nx) >= OP_KIND_BASE
                            && *cx.src.add(nx) == b'(')
                        {
                            return (TypeStart::No, o);
                        }
                    }
                    K_RETURN | K_THROW | K_YIELD | K_AWAIT | K_DELETE | K_CASE | K_OF
                    | K_INSTANCEOF | K_ELSE | K_DO | K_DEFAULT | K_IF | K_WHILE | K_FOR
                    | K_SWITCH | K_WITH | K_LET | K_CONST | K_VAR | K_USING | K_FUNCTION
                    | K_CLASS | K_EXPORT | K_SUPER | K_NAMESPACE | K_MODULE | K_TYPE
                    | K_INTERFACE | K_ENUM | K_BREAK | K_CONTINUE | K_DEBUGGER | K_TRY
                    | K_CATCH | K_FINALLY | K_FROM | K_GLOBAL => {
                        return (TypeStart::No, o);
                    }
                    // Names, modifiers and `typeof` / `keyof` / `readonly` / `infer` / `unique` /
                    // `asserts` / `is` / `new` / `abstract`: the type or expression continues.
                    _ => {}
                }
            }
        } else if !matches!(
            k,
            STR | NUM | BIGINT | TMPL_NOSUB | TMPL_HEAD | TMPL_MIDDLE | TMPL_TAIL | PRIV_IDENT
        ) {
            return (TypeStart::No, o);
        }
        q = bm_prev_sig(cx.st, cx.kind, w);
    }
    (TypeStart::No, o)
}

/// Keyword code of the identifier token at `w` (0 for a plain name or a non-identifier).
unsafe fn ident_kw(cx: Cx, w: usize) -> u8 {
    if base_kind(cx, w) != IDENT {
        return 0;
    }
    let e = bm_next1(cx.st, w + 1, cx.n);
    word_kw(cx, w, e - w)
}

/// What a `<` is, for the seed rules.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LtKind {
    List(Lt),
    /// A comparison (or `<<`, `<=`).
    Cmp,
    Unknown,
}

/// Classify the `<` at `lt` by its head: type parameters of a declaration or member, type arguments
/// inside a type, an assertion or generic arrow at operand position, or TypeScript's speculation
/// after a value.
unsafe fn angle_kind(cx: Cx, lt: usize, depth: u32) -> LtKind {
    let key = MEMO_ANGLE | lt as u64;
    if let Some(v) = memo_get(key) {
        return LtKind::decode(v);
    }
    let r = angle_kind_uncached(cx, lt, depth);
    memo_put(key, r.encode());
    r
}

impl LtKind {
    fn encode(self) -> u64 {
        match self {
            LtKind::List(Lt::Type(false)) => 0,
            LtKind::List(Lt::Type(true)) => 1,
            LtKind::List(Lt::Params) => 2,
            LtKind::List(Lt::Args) => 3,
            LtKind::List(Lt::Assert) => 4,
            LtKind::Cmp => 5,
            LtKind::Unknown => 6,
        }
    }

    fn decode(v: u64) -> LtKind {
        match v {
            0 => LtKind::List(Lt::Type(false)),
            1 => LtKind::List(Lt::Type(true)),
            2 => LtKind::List(Lt::Params),
            3 => LtKind::List(Lt::Args),
            4 => LtKind::List(Lt::Assert),
            5 => LtKind::Cmp,
            _ => LtKind::Unknown,
        }
    }
}

unsafe fn angle_kind_uncached(cx: Cx, lt: usize, depth: u32) -> LtKind {
    if !cx.ts {
        return LtKind::Cmp;
    }
    let c1 = *cx.src.add(lt + 1);
    if c1 == b'=' {
        return LtKind::Cmp;
    }
    // `<<` is a shift unless it opens type arguments holding a generic arrow's type parameters.
    if c1 == b'<' && !lt_run_opens_type_args(cx.src, cx.st, cx.opch, cx.kind, cx.n, lt) {
        return LtKind::Cmp;
    }
    let h = bm_prev_sig(cx.st, cx.kind, lt);
    if h < 0 {
        return LtKind::List(Lt::Assert);
    }
    let hw = h as usize;
    let hk = base_kind(cx, hw);
    if hk == IDENT {
        let e = bm_next1(cx.st, hw + 1, cx.n);
        let broken = lt_in_range(cx.src, e, lt);
        match word_kw(cx, hw, e - hw) {
            K_TRUE | K_FALSE | K_NULL => {
                if broken {
                    // `let x: true` then `<T>y` on the next line.
                    return match type_start(cx, hw + 1, depth + 1) {
                        TypeStart::Decl => LtKind::List(Lt::Assert),
                        TypeStart::Unknown => LtKind::Unknown,
                        TypeStart::Expr | TypeStart::No => LtKind::Cmp,
                    };
                }
                return LtKind::Cmp;
            }
            // `this<T>()` in an expression, `extends this<T>` in a heritage clause; inside any
            // other type `this` ends it.
            K_THIS | K_SUPER => {
                let b = bm_prev_sig(cx.st, cx.kind, hw);
                if b >= 0 && matches!(ident_kw(cx, b as usize), K_EXTENDS | K_IMPLEMENTS) {
                    return LtKind::List(Lt::Type(true));
                }
                return match type_start(cx, hw + 1, depth + 1) {
                    TypeStart::Decl | TypeStart::Expr => LtKind::Cmp,
                    TypeStart::Unknown => LtKind::Unknown,
                    TypeStart::No => {
                        if lt_speculates_type_args(cx, lt) {
                            LtKind::List(Lt::Args)
                        } else {
                            LtKind::Cmp
                        }
                    }
                };
            }
            K_FUNCTION => return LtKind::List(Lt::Params),
            // Statements these end cannot be continued by a comparison.
            K_DEBUGGER | K_BREAK | K_CONTINUE => return LtKind::List(Lt::Assert),
            // A keyword that takes an operand: `<T>` starts an assertion or a generic arrow.
            K_RETURN | K_THROW | K_YIELD | K_AWAIT | K_DELETE | K_VOID | K_TYPEOF | K_CASE
            | K_IN | K_OF | K_INSTANCEOF | K_ELSE | K_DO | K_DEFAULT | K_NEW | K_EXTENDS
            | K_IMPLEMENTS | K_ASYNC | K_KEYOF => return LtKind::List(Lt::Assert),
            _ => {}
        }
        // A declaration head's name (`class C<`, `function f<`, `type X<`), a type name after
        // `extends` / `implements`, or `let x` ended by a line break (`<` starts a statement).
        let b = bm_prev_sig(cx.st, cx.kind, hw);
        if b >= 0 {
            let bw = b as usize;
            match ident_kw(cx, bw) {
                K_CLASS | K_INTERFACE | K_TYPE | K_FUNCTION => return LtKind::List(Lt::Params),
                K_EXTENDS | K_IMPLEMENTS => return LtKind::List(Lt::Type(true)),
                K_LET | K_CONST | K_VAR | K_USING if broken => return LtKind::List(Lt::Assert),
                _ => {}
            }
            if base_kind(cx, bw) >= OP_KIND_BASE && *cx.src.add(bw) == b'*' {
                let bb = bm_prev_sig(cx.st, cx.kind, bw);
                if bb >= 0 && ident_kw(cx, bb as usize) == K_FUNCTION {
                    return LtKind::List(Lt::Params);
                }
            }
        }
        // Type arguments on a type name: the type before the head decides, unless a line break
        // ended that type before the `<`.
        match type_start(cx, hw + 1, depth + 1) {
            // A declaration's type ends at the line break and a statement (an assertion) starts; an
            // expression's type continues with a comparison.
            TypeStart::Decl if broken => return LtKind::List(Lt::Assert),
            TypeStart::Expr if broken => return LtKind::Cmp,
            TypeStart::Decl => return LtKind::List(Lt::Type(true)),
            TypeStart::Expr => return LtKind::List(Lt::Type(false)),
            TypeStart::Unknown => return LtKind::Unknown,
            TypeStart::No => {}
        }
        // A member's type parameters: `m<T>(`.
        if matches!(
            name_role(cx, hw, depth + 1),
            Some(Role::Member | Role::ObjectKey | Role::FnName)
        ) {
            return LtKind::List(Lt::Params);
        }
        return if lt_speculates_type_args(cx, lt) { LtKind::List(Lt::Args) } else { LtKind::Cmp };
    }
    if hk == PRIV_IDENT {
        // A private member declaration takes type parameters (`#p<T>() {}`); an access (`o.#p<`)
        // compares or instantiates.
        let b = bm_prev_sig(cx.st, cx.kind, hw);
        let access = b >= 0
            && base_kind(cx, b as usize) >= OP_KIND_BASE
            && matches!(*cx.src.add(b as usize), b'.' | b'?');
        if !access {
            return LtKind::List(Lt::Params);
        }
        return if lt_speculates_type_args(cx, lt) { LtKind::List(Lt::Args) } else { LtKind::Cmp };
    }
    if hk >= OP_KIND_BASE {
        // After a value a `<` compares; after an operator or opener it starts an assertion or a
        // generic arrow.
        let hc = *cx.src.add(hw);
        let arrow = hc == b'>' && hw > 0 && *cx.src.add(hw - 1) == b'=';
        if matches!(hc, b')' | b']' | b'}' | b'>') && !arrow {
            // A declaration's type ending at a line break (`let x: Foo[]` then `<T>y`): a statement
            // starts.
            let e = bm_next1(cx.st, hw + 1, cx.n);
            if lt_in_range(cx.src, e, lt) {
                match type_start(cx, hw + 1, depth + 1) {
                    TypeStart::Decl => return LtKind::List(Lt::Assert),
                    TypeStart::Unknown => return LtKind::Unknown,
                    TypeStart::Expr | TypeStart::No => {}
                }
            }
        }
        match hc {
            b'}' => {
                // A block or declaration body: a statement (an assertion) starts; an expression's
                // `}` is a value.
                let Some(lb) = match_delim_back(cx.src, cx.st, cx.kind, hw, b'{', b'}') else {
                    return LtKind::Unknown;
                };
                return match brace_close_after(cx, lb) {
                    Some(After::Operand | After::EndsDecl) => LtKind::List(Lt::Assert),
                    Some(After::Value) => LtKind::Cmp,
                    None => LtKind::Unknown,
                };
            }
            b']' => {
                // A computed member key takes type parameters.
                let Some(lb) = match_delim_back(cx.src, cx.st, cx.kind, hw, b'[', b']') else {
                    return LtKind::Unknown;
                };
                return if bracket_member(cx, lb, depth + 1).is_some() {
                    LtKind::List(Lt::Params)
                } else {
                    LtKind::Cmp
                };
            }
            b')' => {
                // `if (x) <T>y`: a statement head, then a statement.
                let Some(lp) = match_delim_back(cx.src, cx.st, cx.kind, hw, b'(', b')') else {
                    return LtKind::Unknown;
                };
                let p = bm_prev_sig(cx.st, cx.kind, lp);
                if p >= 0 && matches!(ident_kw(cx, p as usize), K_IF | K_WHILE | K_FOR | K_WITH) {
                    return LtKind::List(Lt::Assert);
                }
            }
            b'>' if !(hw > 0 && *cx.src.add(hw - 1) == b'=') => {
                // After an assertion `<T>` another may start; after type arguments or a comparison
                // a `<` compares.
                let Some(lt2) = angle_back(cx, hw) else { return LtKind::Unknown };
                return match angle_kind(cx, lt2, depth + 1) {
                    LtKind::List(Lt::Assert) => LtKind::List(Lt::Assert),
                    LtKind::Unknown => LtKind::Unknown,
                    _ => LtKind::Cmp,
                };
            }
            _ => {}
        }
        let value_end = matches!(hc, b')')
            || ((hc == b'+' || hc == b'-') && *cx.src.add(hw + 1) == hc)
            || (hc == b'!' && *cx.src.add(hw + 1) != b'=' && {
                // Postfix non-null assertion after a value.
                let p = bm_prev_sig(cx.st, cx.kind, hw);
                p >= 0 && {
                    let pk = base_kind(cx, p as usize);
                    matches!(pk, IDENT | STR | NUM | BIGINT | PRIV_IDENT)
                        || (pk >= OP_KIND_BASE
                            && matches!(*cx.src.add(p as usize), b')' | b']' | b'}'))
                }
            });
        return if value_end { LtKind::Cmp } else { LtKind::List(Lt::Assert) };
    }
    if matches!(hk, STR | NUM | BIGINT | TMPL_NOSUB | TMPL_TAIL) {
        let e = bm_next1(cx.st, hw + 1, cx.n);
        if lt_in_range(cx.src, e, lt) {
            // A literal type ending a declaration at a line break.
            return match type_start(cx, hw + 1, depth + 1) {
                TypeStart::Decl => LtKind::List(Lt::Assert),
                TypeStart::Unknown => LtKind::Unknown,
                TypeStart::Expr | TypeStart::No => LtKind::Cmp,
            };
        }
        if matches!(hk, STR | NUM | BIGINT)
            && matches!(name_role(cx, hw, depth + 1), Some(Role::Member | Role::ObjectKey))
        {
            // `"k"<T>() {}`: a member's type parameters.
            return LtKind::List(Lt::Params);
        }
    }
    LtKind::Cmp
}

/// The unmatched `<` at `lt` holds `o`: is its list a type?
unsafe fn angle_list_start(cx: Cx, lt: usize, o: usize, depth: u32) -> (TypeStart, usize) {
    match angle_kind(cx, lt, depth) {
        LtKind::List(Lt::Type(false)) => (TypeStart::Expr, lt),
        LtKind::List(_) => (TypeStart::Decl, lt),
        LtKind::Cmp => (TypeStart::No, o),
        LtKind::Unknown => (TypeStart::Unknown, o),
    }
}

/// TypeScript's type-argument speculation at an expression-position `<`.
unsafe fn lt_speculates_type_args(cx: Cx, lt: usize) -> bool {
    let lim = (lt + 4096).min(cx.n);
    let (close, _) = angle_close_fwd_capped(cx.src, cx.st, cx.opch, cx.kind, lt + 1, lim, 1);
    let Some(gt) = close else { return false };
    if matches!(*cx.src.add(gt + 1), b'=' | b'>') {
        return false;
    }
    match gt_follower(cx.src, cx.n, gt + 1) {
        Follow::Split => type_list_legal(cx.t, cx.src, cx.st, cx.kind, lt + 1, gt),
        Follow::Fuse | Follow::Ctx => false,
    }
}

/// A type literal when the `{` at `o` is inside a type, else `other`.
unsafe fn in_type_or(cx: Cx, o: usize, depth: u32, other: Fk) -> Option<Fk> {
    match type_start(cx, o, depth) {
        TypeStart::No => Some(other),
        TypeStart::Unknown => None,
        _ => Some(Fk::TypeLit),
    }
}

/// The `{` at `o` follows a completed type: a body after a return type (`m(): T {`) or a heritage
/// clause; `other` when it is not in a type.
unsafe fn completed_type_body(cx: Cx, o: usize, depth: u32, other: Option<Fk>) -> Option<Fk> {
    let (ts, at) = type_start_at(cx, o, depth);
    match ts {
        TypeStart::Unknown => None,
        TypeStart::No | TypeStart::Expr => other,
        TypeStart::Decl => {
            let ak = base_kind(cx, at);
            if ak >= OP_KIND_BASE && *cx.src.add(at) == b':' {
                let p = bm_prev_sig(cx.st, cx.kind, at);
                if p >= 0
                    && base_kind(cx, p as usize) >= OP_KIND_BASE
                    && *cx.src.add(p as usize) == b')'
                {
                    return Some(Fk::FnBody);
                }
                return other;
            }
            if ak == IDENT && matches!(ident_kw(cx, at), K_EXTENDS | K_IMPLEMENTS) {
                return decl_head_kind(cx, at);
            }
            other
        }
    }
}

/// What the `{` at `o` opens, from the tokens before it. None when they do not settle it.
unsafe fn brace_seed_kind(cx: Cx, o: usize, depth: u32) -> Option<Fk> {
    if depth > SEED_RULE_DEPTH || !tick() {
        return None;
    }
    let q = bm_prev_sig(cx.st, cx.kind, o);
    if q < 0 {
        return Some(Fk::Block);
    }
    let w = q as usize;
    let k = base_kind(cx, w);
    let c = *cx.src.add(w);
    if k >= OP_KIND_BASE {
        return match c {
            b')' => paren_body_kind(cx, w),
            b';' => Some(Fk::Block),
            b'{' => match brace_seed_kind(cx, w, depth + 1) {
                // An expression inside a JSX container; a malformed nested object.
                Some(Fk::Container | Fk::Object) => Some(Fk::Object),
                Some(_) => Some(Fk::Block),
                None => None,
            },
            b'}' => {
                let o2 = match_delim_back(cx.src, cx.st, cx.kind, w, b'{', b'}')?;
                match brace_seed_kind(cx, o2, depth + 1)? {
                    // Another JSX expression container just closed: children again.
                    Fk::Container => Some(Fk::Container),
                    // A return type literal before a body, or a type alias before a block.
                    Fk::TypeLit => completed_type_body(cx, o, depth, Some(Fk::Block)),
                    // Heritage: `class extends class {} {`, `extends {} {`.
                    Fk::ClassBody | Fk::Object => {
                        let h = if brace_seed_kind(cx, o2, depth + 1) == Some(Fk::Object) {
                            bm_prev_sig(cx.st, cx.kind, o2)
                        } else {
                            class_keyword_before(cx, o2, false)
                                .map_or(-1, |k| bm_prev_sig(cx.st, cx.kind, k))
                        };
                        if h >= 0 && ident_kw(cx, h as usize) == K_EXTENDS {
                            Some(Fk::ClassBody)
                        } else {
                            Some(Fk::Block)
                        }
                    }
                    _ => Some(Fk::Block),
                }
            }
            b'>' => {
                if w > 0 && *cx.src.add(w - 1) == b'=' {
                    return arrow_body_kind(cx, w - 1, depth);
                }
                if !bm_get(cx.opch, w) {
                    // A JSX tag's own `>`: children follow.
                    return Some(Fk::Container);
                }
                // `class C<T> {`, `f(): A<T> {`, `<T>{ ... }`, `a > {`.
                if let Some(k) = decl_head_kind(cx, w) {
                    return Some(k);
                }
                completed_type_body(cx, o, depth, Some(Fk::Object))
            }
            b'=' => {
                if *cx.src.add(w + 1) == b'>' {
                    // A fused `=>` (coalesce has run).
                    return arrow_body_kind(cx, w, depth);
                }
                if type_alias_eq(cx, w) {
                    return Some(Fk::TypeLit);
                }
                match jsx_attr_eq(cx, w) {
                    Some(true) => Some(Fk::Container),
                    // `x = {`, or a type-parameter default `<T = {`.
                    Some(false) => in_type_or(cx, o, depth, Fk::Object),
                    None => None,
                }
            }
            b':' => colon_kind(cx, w, depth),
            b',' | b'|' | b'&' | b'<' | b'(' | b'[' | b'?' => in_type_or(cx, o, depth, Fk::Object),
            b'!' | b'+' | b'-' => {
                // Postfix `x++` / `x!` then a line break: a block starts.
                let postfix = if c == b'!' {
                    cx.ts && {
                        let p = bm_prev_sig(cx.st, cx.kind, w);
                        p >= 0 && {
                            let pk = base_kind(cx, p as usize);
                            matches!(pk, IDENT | STR | NUM | BIGINT | PRIV_IDENT)
                                || (pk >= OP_KIND_BASE
                                    && matches!(*cx.src.add(p as usize), b')' | b']' | b'}'))
                        }
                    }
                } else {
                    *cx.src.add(w + 1) == c || (w > 0 && *cx.src.add(w - 1) == c)
                };
                if postfix && lt_in_range(cx.src, w + 1, o) {
                    Some(Fk::Block)
                } else {
                    Some(Fk::Object)
                }
            }
            b'~' | b'*' | b'/' | b'%' | b'^' | b'.' => Some(Fk::Object),
            _ => None,
        };
    }
    if k == IDENT {
        let e = bm_next1(cx.st, w + 1, cx.n);
        let newline = lt_in_range(cx.src, e, o);
        // A property name (`x.in`) is a value: a `{` on the next line is a block, on the same line
        // malformed.
        let p = bm_prev_sig(cx.st, cx.kind, w);
        if p >= 0
            && base_kind(cx, p as usize) >= OP_KIND_BASE
            && *cx.src.add(p as usize) == b'.'
            && *cx.src.add(p as usize + 1) != b'.'
        {
            return if newline { Some(Fk::Block) } else { None };
        }
        return match word_kw(cx, w, e - w) {
            K_ELSE | K_DO | K_TRY | K_FINALLY | K_CATCH | K_GLOBAL => Some(Fk::Block),
            // Restricted productions: nothing follows them on a new line.
            K_RETURN | K_THROW | K_YIELD if newline => Some(Fk::Block),
            K_RETURN | K_CASE | K_IN | K_OF | K_TYPEOF | K_AWAIT | K_YIELD | K_DELETE | K_NEW
            | K_DEFAULT | K_THROW | K_INSTANCEOF => Some(Fk::Object),
            // `f(): void {` / `f(): this {` vs `void {`.
            K_VOID | K_NULL | K_TRUE | K_FALSE => {
                completed_type_body(cx, o, depth, Some(Fk::Object))
            }
            K_THIS => completed_type_body(cx, o, depth, None),
            K_EXTENDS => {
                // `class extends {` (an object as heritage) vs a conditional type's `T extends {`.
                let p = bm_prev_sig(cx.st, cx.kind, w);
                let mut after_class = false;
                if p >= 0 {
                    let pw = p as usize;
                    after_class = match ident_kw(cx, pw) {
                        K_CLASS => true,
                        0 if base_kind(cx, pw) == IDENT => {
                            let pp = bm_prev_sig(cx.st, cx.kind, pw);
                            pp >= 0 && ident_kw(cx, pp as usize) == K_CLASS
                        }
                        _ => false,
                    };
                }
                Some(if after_class { Fk::Object } else { Fk::TypeLit })
            }
            K_KEYOF | K_READONLY | K_INFER | K_AS | K_SATISFIES | K_IS | K_IMPLEMENTS
            | K_UNIQUE | K_ASSERTS => Some(Fk::TypeLit),
            K_WITH | K_IMPORT | K_EXPORT => Some(Fk::ModuleSpec),
            K_STATIC => Some(Fk::StaticBlock),
            K_LET | K_CONST | K_VAR | K_USING => Some(Fk::Pattern),
            0 | K_CLASS | K_INTERFACE | K_ENUM | K_NAMESPACE | K_MODULE => {
                if let Some(k) = decl_head_kind(cx, w) {
                    return Some(k);
                }
                // A value then a block on the next line (`x\n{}`).
                completed_type_body(cx, o, depth, if newline { Some(Fk::Block) } else { None })
            }
            // Keyword-named types (`): number {`) and anything else: a body after a completed type,
            // or a block after a line break.
            _ => completed_type_body(cx, o, depth, if newline { Some(Fk::Block) } else { None }),
        };
    }
    match k {
        // `declare module 'm' {`
        STR => Some(Fk::Block),
        JTEXT | JEND => Some(Fk::Container),
        _ => None,
    }
}

/// What the `{` after `=>` opens, for the `=` at `eq`: an arrow body, or the return type literal of
/// a function type.
unsafe fn arrow_body_kind(cx: Cx, eq: usize, depth: u32) -> Option<Fk> {
    let p = bm_prev_sig(cx.st, cx.kind, eq);
    if p < 0 {
        return None;
    }
    let pw = p as usize;
    if !(base_kind(cx, pw) >= OP_KIND_BASE && *cx.src.add(pw) == b')') {
        // `x => {`, or `(a): T => {` where the return type just ended.
        return Some(Fk::ArrowBody);
    }
    let lp = match_delim_back(cx.src, cx.st, cx.kind, pw, b'(', b')')?;
    match type_start(cx, lp, depth + 1) {
        TypeStart::No => Some(Fk::ArrowBody),
        TypeStart::Unknown => None,
        _ => Some(Fk::TypeLit),
    }
}

/// What the `{` after the `)` at `rp` opens: a statement body, a function body or a class body
/// after a heritage call.
unsafe fn paren_body_kind(cx: Cx, rp: usize) -> Option<Fk> {
    let lp = match_delim_back(cx.src, cx.st, cx.kind, rp, b'(', b')')?;
    let h = bm_prev_sig(cx.st, cx.kind, lp);
    if h < 0 {
        return None;
    }
    let hw = h as usize;
    let hk = base_kind(cx, hw);
    if hk == IDENT {
        let e = bm_next1(cx.st, hw + 1, cx.n);
        return match word_kw(cx, hw, e - hw) {
            K_IF | K_WHILE | K_FOR | K_WITH | K_SWITCH | K_CATCH => Some(Fk::Block),
            K_AWAIT => {
                // `for await (`
                let f = bm_prev_sig(cx.st, cx.kind, hw);
                if f >= 0 && ident_kw(cx, f as usize) == K_FOR { Some(Fk::Block) } else { None }
            }
            K_FUNCTION => Some(Fk::FnBody),
            K_EXTENDS => Some(Fk::ClassBody),
            K_RETURN | K_TYPEOF | K_VOID | K_DELETE | K_YIELD | K_NEW | K_THROW | K_IN | K_OF
            | K_INSTANCEOF | K_CASE | K_ELSE | K_DO | K_ASYNC | K_THIS | K_SUPER | K_IMPORT
            | K_DEFAULT | K_TRUE | K_FALSE | K_NULL => None,
            _ => decl_head_kind(cx, hw).or(Some(Fk::FnBody)),
        };
    }
    match hk {
        STR | NUM | BIGINT | PRIV_IDENT => Some(Fk::FnBody),
        _ if hk >= OP_KIND_BASE => match *cx.src.add(hw) {
            b']' | b'*' => Some(Fk::FnBody),
            b'>' if hw > 0 && *cx.src.add(hw - 1) == b'=' => None,
            b'>' => decl_head_kind(cx, hw).or(Some(Fk::FnBody)),
            _ => None,
        },
        _ => None,
    }
}

/// The body kind of a declaration whose head ends at `w`: walk back over names, dots, commas and
/// balanced `<...>` / `(...)` to its keyword.
unsafe fn decl_head_kind(cx: Cx, mut w: usize) -> Option<Fk> {
    let (mut ang, mut par) = (0i32, 0i32);
    for _ in 0..64 {
        if !tick() {
            return None;
        }
        let k = base_kind(cx, w);
        if k >= OP_KIND_BASE {
            match *cx.src.add(w) {
                b'>' => ang += run_len(cx, w, b'>'),
                b'<' => {
                    ang -= run_len(cx, w, b'<');
                    if ang < 0 {
                        return None;
                    }
                }
                b')' => par += 1,
                b'(' => {
                    if par == 0 {
                        return None;
                    }
                    par -= 1;
                }
                b'.' | b',' => {}
                _ if ang > 0 || par > 0 => {}
                _ => return None,
            }
        } else if k == IDENT && ang == 0 && par == 0 {
            let e = bm_next1(cx.st, w + 1, cx.n);
            match word_kw(cx, w, e - w) {
                K_CLASS => return Some(Fk::ClassBody),
                K_INTERFACE => return Some(Fk::TypeLit),
                K_ENUM => return Some(Fk::EnumBody),
                K_NAMESPACE | K_MODULE => return Some(Fk::Block),
                0 | K_EXTENDS | K_IMPLEMENTS | K_ABSTRACT | K_DECLARE | K_EXPORT | K_CONST => {}
                _ => return None,
            }
        } else if ang == 0 && par == 0 && !matches!(k, IDENT | STR | NUM | BIGINT | TMPL_NOSUB) {
            return None;
        }
        let q = bm_prev_sig(cx.st, cx.kind, w);
        if q < 0 {
            return None;
        }
        w = q as usize;
    }
    None
}

/// Is the `=` at `w` the one of `type X = ` / `type X<T> = `?
unsafe fn type_alias_eq(cx: Cx, w: usize) -> bool {
    let mut q = bm_prev_sig(cx.st, cx.kind, w);
    if q < 0 {
        return false;
    }
    if base_kind(cx, q as usize) >= OP_KIND_BASE && *cx.src.add(q as usize) == b'>' {
        // Skip the type-parameter list.
        let mut ang = 0i32;
        loop {
            if !tick() {
                return false;
            }
            let p = q as usize;
            if base_kind(cx, p) >= OP_KIND_BASE {
                match *cx.src.add(p) {
                    b'>' => ang += run_len(cx, p, b'>'),
                    b'<' => {
                        ang -= run_len(cx, p, b'<');
                        if ang <= 0 {
                            q = bm_prev_sig(cx.st, cx.kind, p);
                            break;
                        }
                    }
                    b';' | b'{' | b'}' => return false,
                    _ => {}
                }
            }
            q = bm_prev_sig(cx.st, cx.kind, p);
            if q < 0 {
                return false;
            }
        }
    }
    if q < 0 || base_kind(cx, q as usize) != IDENT {
        return false;
    }
    let p = bm_prev_sig(cx.st, cx.kind, q as usize);
    p >= 0 && ident_kw(cx, p as usize) == K_TYPE
}

/// Is the `=` at `w` a JSX attribute's `=` (`<Tag attr={`)? None when the tag is too long to tell.
unsafe fn jsx_attr_eq(cx: Cx, w: usize) -> Option<bool> {
    let q = bm_prev_sig(cx.st, cx.kind, w);
    if q < 0 || base_kind(cx, q as usize) != IDENT {
        return Some(false);
    }
    let mut q = bm_prev_sig(cx.st, cx.kind, q as usize);
    let mut brc = 0i32;
    for _ in 0..256 {
        if q < 0 || !tick() {
            return Some(false);
        }
        let p = q as usize;
        let k = base_kind(cx, p);
        if k == JSX_LT {
            return Some(brc == 0);
        }
        if k >= OP_KIND_BASE {
            match *cx.src.add(p) {
                b'}' => brc += 1,
                b'{' => {
                    if brc == 0 {
                        return Some(false);
                    }
                    brc -= 1;
                }
                b'.' | b':' | b'-' | b'=' => {}
                _ if brc > 0 => {}
                _ => return Some(false),
            }
        } else if brc == 0 && !matches!(k, IDENT | STR) {
            return Some(false);
        }
        q = bm_prev_sig(cx.st, cx.kind, p);
    }
    None
}

impl Walk {
    /// Start a bounded walk at `seed`. False when the opener there cannot be classified locally.
    unsafe fn seed(&mut self, cx: Cx, module: bool, seed: Seed) -> bool {
        self.reset(module);
        match seed {
            Seed::Stmt(at) => {
                self.upto = at;
                self.prev_end = at;
            }
            Seed::Member(o, at, kind) => {
                if !self.seed_brace(cx, o, kind) {
                    return false;
                }
                self.upto = at;
                self.prev_end = at;
            }
            Seed::Brace(o, kind) => {
                if !self.seed_brace(cx, o, kind) {
                    return false;
                }
                self.upto = o + 1;
                self.prev_end = o + 1;
            }
            Seed::Angle(lt, kind) => {
                let (region, decl, s) = match kind {
                    Lt::Type(decl) => (Some(if decl { R_INLINE } else { R_EXPR }), decl, 3),
                    Lt::Params => (None, true, 1),
                    Lt::Args => (None, false, 2),
                    Lt::Assert => (Some(R_ASSERT), false, 4),
                };
                if let Some(rule) = region {
                    let r = self.push(Fk::TypeRegion);
                    r.decl = decl;
                    r.s = rule;
                }
                let f = self.push(Fk::Angle);
                f.decl = decl;
                f.s = s;
                self.set_operand();
                self.upto = lt + 1;
                self.prev_end = lt + 1;
                self.seed_once = true;
            }
            Seed::Paren(o) => {
                let ts_start = type_start(cx, o, 0);
                if ts_start == TypeStart::Unknown {
                    return false;
                }
                if ts_start != TypeStart::No {
                    let expr = ts_start == TypeStart::Expr;
                    let r = self.push(Fk::TypeRegion);
                    r.decl = !expr;
                    r.s = if expr { R_EXPR } else { R_INLINE };
                    let f = self.push(Fk::TypeParen);
                    f.decl = true;
                    self.set_operand();
                    self.upto = o + 1;
                    self.prev_end = o + 1;
                    self.seed_depth = self.frames.len();
                    self.seed_lost = false;
                    return true;
                }
                let q = bm_prev_sig(cx.st, cx.kind, o);
                let mut kind = Fk::Group;
                let mut head = 0u8;
                if q >= 0 {
                    let p = q as usize;
                    let k = base_kind(cx, p);
                    if k == IDENT {
                        let e = bm_next1(cx.st, p + 1, cx.n);
                        let mut kwc = word_kw(cx, p, e - p);
                        if kwc == K_AWAIT {
                            let f = bm_prev_sig(cx.st, cx.kind, p);
                            if f >= 0 && ident_kw(cx, f as usize) == K_FOR {
                                kwc = K_FOR;
                            }
                        }
                        head = match kwc {
                            K_IF => H_IF,
                            K_WHILE => H_WHILE,
                            K_FOR => H_FOR,
                            K_WITH => H_WITH,
                            K_SWITCH => H_SWITCH,
                            K_CATCH => H_CATCH,
                            _ => 0,
                        };
                        if head != 0 {
                            kind = Fk::Head;
                        } else if matches!(kwc, 0 | K_THIS | K_SUPER | K_IMPORT) {
                            kind = Fk::Call;
                        }
                    } else if k >= OP_KIND_BASE && matches!(*cx.src.add(p), b')' | b']') {
                        kind = Fk::Call;
                    }
                }
                let f = self.push(kind);
                f.head = head;
                f.s = F_START;
                self.set_operand();
                self.upto = o + 1;
                self.prev_end = o + 1;
            }
            Seed::Bracket(o) => {
                let ts_start = type_start(cx, o, 0);
                if ts_start == TypeStart::Unknown {
                    return false;
                }
                if ts_start != TypeStart::No {
                    let expr = ts_start == TypeStart::Expr;
                    let r = self.push(Fk::TypeRegion);
                    r.decl = !expr;
                    r.s = if expr { R_EXPR } else { R_INLINE };
                    let f = self.push(Fk::TypeBracket);
                    f.decl = true;
                } else {
                    self.push(Fk::Array);
                }
                self.set_operand();
                self.upto = o + 1;
                self.prev_end = o + 1;
            }
            Seed::Sub(h) => {
                let e = bm_next1(cx.st, h + 1, cx.n);
                self.push(Fk::Sub);
                self.set_operand();
                self.upto = e;
                self.prev_end = e;
            }
        }
        self.seed_depth = self.frames.len();
        self.seed_lost = false;
        true
    }

    /// Push the frame the `{` at `o` opens (`usize::MAX`: a class body whose `{` lies beyond the
    /// scan), with the state right after it.
    unsafe fn seed_brace(&mut self, cx: Cx, o: usize, kind: Fk) -> bool {
        if kind == Fk::TypeLit {
            let ts = if o == usize::MAX { TypeStart::Decl } else { type_start(cx, o, 0) };
            if ts == TypeStart::Unknown {
                return false;
            }
            let expr = ts == TypeStart::Expr;
            let r = self.push(Fk::TypeRegion);
            r.decl = !expr;
            r.s = if expr { R_EXPR } else { R_INLINE };
            let f = self.push(Fk::TypeLit);
            f.decl = true;
            f.value = expr;
        } else {
            let f = self.push(kind);
            match kind {
                Fk::ClassBody => {
                    f.strict = true;
                    f.s = M_KEY_POS;
                }
                Fk::Object => {
                    f.value = true;
                    f.s = M_KEY_POS;
                }
                Fk::StaticBlock => {
                    f.reserved = true;
                    f.strict = true;
                }
                Fk::FnBody | Fk::ArrowBody => {
                    f.prologue = 1;
                }
                _ => {}
            }
        }
        self.expr_allowed = true;
        self.stmt_start = matches!(kind, Fk::Block | Fk::FnBody | Fk::ArrowBody | Fk::StaticBlock);
        true
    }
}

/// Run `f` on a bounded walk whose seed lies before `from`, advanced to `pos`; None when no seed
/// can be found or classified.
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
        let cont = w.generation == generation
            && w.seed_depth != 0
            && !w.seed_lost
            && !w.seed_once
            && from >= w.upto
            && from - w.upto <= SEED_CONTINUE;
        if cont {
            w.advance(cx, pos);
            if !w.seed_lost {
                return Some(f(&mut w, cx));
            }
            // The walk left its seed frame on the way: start over nearer.
        }
        start_query();
        let seed = find_seed(cx, from)?;
        w.generation = generation;
        if !w.seed(cx, module, seed) {
            w.generation = u64::MAX;
            return None;
        }
        w.advance(cx, pos);
        if w.seed_lost {
            return None;
        }
        Some(f(&mut w, cx))
    })
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

/// Open `<` lists a `>` run at `pos` would close. Cheaper than [`before`]: only a walk seeded
/// inside a type list can count any, so every other seed answers zero without walking.
pub(crate) unsafe fn angles_before(cx: Cx, module: bool, pos: usize) -> usize {
    let generation = GEN.with(Cell::get);
    let local = LOCAL.with(|cell| {
        let mut w = cell.borrow_mut();
        let cont = w.generation == generation
            && w.seed_depth != 0
            && !w.seed_lost
            && !w.seed_once
            && pos >= w.upto
            && pos - w.upto <= SEED_CONTINUE;
        if cont {
            w.advance(cx, pos);
            if !w.seed_lost {
                return Some(w.open_angles());
            }
        }
        start_query();
        let seed = find_seed(cx, pos)?;
        if !matches!(seed, Seed::Angle(..)) {
            return Some(0);
        }
        w.generation = generation;
        if !w.seed(cx, module, seed) {
            w.generation = u64::MAX;
            return None;
        }
        w.advance(cx, pos);
        if w.seed_lost {
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

/// Like [`after`], with the bounded walk seeded before `from` (the matching opener of a closer at
/// `pos`).
pub(crate) unsafe fn after_from(cx: Cx, module: bool, pos: usize, from: usize) -> After {
    if from < pos && base_kind(cx, pos) >= OP_KIND_BASE && *cx.src.add(pos) == b'}' {
        // A closed brace group usually settles it by itself.
        start_query();
        if let Some(a) = brace_close_after(cx, from) {
            return a;
        }
    }
    if let Some(a) = local_walk(cx, module, pos, from, |w, cx| {
        if w.upto > pos { w.classify_after() } else { w.after_token(cx, pos) }
    }) {
        return a;
    }
    after_scoped(cx, module, pos)
}
