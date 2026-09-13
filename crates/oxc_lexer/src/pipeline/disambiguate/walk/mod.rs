//! Forward context walk.
//!
//! The hard regex-vs-division questions ("did this `}` close a value?", "is this `yield` a
//! keyword?", "does a type annotation end here?") are questions about the parser's state, which is
//! a pushdown state: a stack of bracket frames tagged with what they opened, a few scope flags, and
//! whether an operand may start next. That state is a function of the token prefix, so it is
//! computed forward and memoized instead of reconstructed backward at every site.
//!
//! The walk is lazy and reads the carved bitmaps as carve left them: keywords are still
//! identifiers, multi-byte operators are still one token start per byte, and `misc_pre` has already
//! split Unicode whitespace. Sites ask in source order, so the walk never rewinds in practice; a
//! query behind the current position rebuilds from the start.

use core::cell::{Cell, RefCell};

use crate::{
    opmap::{OP_KIND_BASE, OP_QDOT},
    tables::{Tables, is_digit, is_op_char},
    token::{KW_BASE, TokenKind},
};

use super::super::{
    BCOM, BIGINT, HASHBANG, IDENT, IDENT_ESC, JEND, JSX_LT, JTEXT, LCOM, NUM, PRIV_IDENT,
    PRIV_IDENT_ESC, REGEX, STR, TMPL_HEAD, TMPL_MIDDLE, TMPL_NOSUB, TMPL_TAIL, WS,
    bitmap::bm_next1,
};
use super::{
    Follow, angle_close_fwd_capped, gt_follower, ident_is, lt_in_range, lt_run_opens_type_args,
    type_list_legal,
};

mod seed;
pub(super) use seed::{after, after_from, angles_before, before};

#[cfg(test)]
mod tests;

/// Everything the walk needs to read a lex in progress. Copied into every call; the pointers are
/// stable for the duration of one `lex_raw`.
#[derive(Clone, Copy)]
pub(super) struct Cx<'a> {
    pub t: &'a Tables,
    pub src: *const u8,
    pub st: *const u64,
    pub opch: *const u64,
    pub kind: *const u8,
    pub n: usize,
    pub ts: bool,
    /// Token starts below this position carry their final keyword kind (`coalesce` has flushed
    /// them): an `IDENT` there is a plain name.
    pub kw_final: usize,
}

/// What the token at a site leaves behind, for the `/` right after it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum After {
    /// The token ends a value: `/` is division.
    Value,
    /// An operand may start: `/` is a regex.
    Operand,
    /// The token ends a declaration nothing can continue (an annotation, a declarator without
    /// initializer, a bodiless signature, `break label`, a module specifier): `/` is a regex.
    EndsDecl,
}

/// Keyword codes: the token kinds `coalesce` writes, used here on the spellings `classify` left as
/// identifiers.
macro_rules! kw_codes {
    ($($name:ident = $kind:ident),* $(,)?) => {
        $(const $name: u8 = TokenKind::$kind as u8;)*
    };
}
kw_codes! {
    K_BREAK = KwBreak, K_CASE = KwCase, K_CATCH = KwCatch,
    K_CLASS = KwClass, K_CONST = KwConst, K_CONTINUE = KwContinue,
    K_DEBUGGER = KwDebugger, K_DEFAULT = KwDefault, K_DELETE = KwDelete,
    K_DO = KwDo, K_ELSE = KwElse, K_ENUM = KwEnum, K_EXPORT = KwExport,
    K_EXTENDS = KwExtends, K_FALSE = KwFalse, K_FINALLY = KwFinally,
    K_FOR = KwFor, K_FUNCTION = KwFunction, K_IF = KwIf,
    K_IMPORT = KwImport, K_IN = KwIn, K_INSTANCEOF = KwInstanceof,
    K_NEW = KwNew, K_NULL = KwNull, K_RETURN = KwReturn, K_SUPER = KwSuper,
    K_SWITCH = KwSwitch, K_THIS = KwThis, K_THROW = KwThrow,
    K_TRUE = KwTrue, K_TRY = KwTry, K_TYPEOF = KwTypeof, K_VAR = KwVar,
    K_VOID = KwVoid, K_WHILE = KwWhile, K_WITH = KwWith, K_YIELD = KwYield,
    K_LET = KwLet, K_STATIC = KwStatic, K_ASYNC = KwAsync,
    K_AWAIT = KwAwait, K_OF = KwOf, K_FROM = KwFrom, K_AS = KwAs,
    K_ABSTRACT = KwAbstract, K_ACCESSOR = KwAccessor,
    K_ASSERTS = KwAsserts, K_DECLARE = KwDeclare, K_GLOBAL = KwGlobal,
    K_IMPLEMENTS = KwImplements, K_INFER = KwInfer,
    K_INTERFACE = KwInterface, K_IS = KwIs, K_KEYOF = KwKeyof,
    K_MODULE = KwModule, K_NAMESPACE = KwNamespace,
    K_OVERRIDE = KwOverride, K_PRIVATE = KwPrivate,
    K_PROTECTED = KwProtected, K_PUBLIC = KwPublic,
    K_READONLY = KwReadonly, K_SATISFIES = KwSatisfies, K_TYPE = KwType,
    K_UNIQUE = KwUnique, K_USING = KwUsing,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fk {
    Root,
    // Braces.
    Block,
    FnBody,
    ArrowBody,
    ClassBody,
    StaticBlock,
    Object,
    Pattern,
    TypeLit,
    EnumBody,
    ModuleSpec,
    Container,
    // A template substitution (`${` .. `}`), opened by a TemplateHead or TemplateMiddle and closed
    // by the next Middle/Tail.
    Sub,
    // Parens.
    Head,
    Params,
    Call,
    Group,
    TypeParen,
    // Brackets.
    Index,
    Array,
    ComputedKey,
    TypeBracket,
    ArrayPattern,
    // JSX.
    JsxTag,
    JsxElem,
    // Virtual frames (no bracket of their own).
    Concise,
    TypeRegion,
    Angle,
    FnHead,
    ClassHead,
}

// Head kinds.
const H_IF: u8 = 1;
const H_WHILE: u8 = 2;
const H_FOR: u8 = 3;
const H_WITH: u8 = 4;
const H_SWITCH: u8 = 5;
const H_CATCH: u8 = 6;

// Declarator state on statement frames and for-heads (`s`).
const D_NONE: u8 = 0;
const D_BINDING: u8 = 1;
const D_BOUND: u8 = 2;
const D_INIT: u8 = 3;

// Statement register on statement frames (`m`).
const S_NONE: u8 = 0;
const S_CASE: u8 = 1;
const S_LABEL: u8 = 2;
const S_IMPORT: u8 = 3;
const S_EXPORT: u8 = 4;
const S_TYPE: u8 = 5;
const S_TYPE_NAME: u8 = 6;
const S_BREAK: u8 = 7;
const S_NAMESPACE: u8 = 8;
const S_ENUM: u8 = 9;
const S_EXPORT_AS: u8 = 11;
const S_EXPORT_AS_NS: u8 = 12;
const S_IMPORT_NAME: u8 = 13;
const S_DECLARE_MODULE: u8 = 14;

// For-head state (`s`).
const F_START: u8 = 0;
const F_BOUND: u8 = 1;
const F_EXPR: u8 = 2;
const F_ITER: u8 = 3;

// Member state on Object / ClassBody / TypeLit (`s`).
const M_KEY_POS: u8 = 0;
const M_KEY_SEEN: u8 = 1;
const M_VALUE: u8 = 2;

// Member modifier bits (`m`) on Object / ClassBody.
const MOD_ASYNC: u8 = 1;
const MOD_GEN: u8 = 2;
const MOD_STATIC: u8 = 4;

// TypeRegion end rule (`s`).
const R_ASSERT: u8 = 1; // `<T>x`: ends at its closing `>`
const R_ARROW_RET: u8 = 2; // `(a): T =>`: ends at `=>`
const R_INLINE: u8 = 3; // declarator/param/member annotation: ends at `=`/`,`/closer/`{`
const R_STMT: u8 = 4; // alias / import-equals / bodiless module: ends at `;`/ASI
const R_EXPR: u8 = 5; // `as T` / `satisfies T`: ends at any expression token
const R_INTERFACE: u8 = 6; // `interface X ... { }`: ends after its body

#[derive(Clone, Copy)]
struct Frame {
    kind: Fk,
    is_gen: bool,
    asyn: bool,
    strict: bool,
    reserved: bool,
    /// Braces: closing this frame ends a value.
    value: bool,
    /// Type frames: part of a declaration type (vs embedded in an expression).
    decl: bool,
    /// TypeRegion: the last token completed a type.
    atom: bool,
    /// TypeRegion: the last token closed a `(` opened inside the region.
    inner: bool,
    s: u8,
    m: u8,
    /// Member modifier bits (Object / ClassBody).
    mods: u8,
    /// For-head: the binding came from a declaration (`for (let x of`).
    fdecl: bool,
    head: u8,
    qdebt: u16,
    prologue: u8,
}

impl Frame {
    fn child(&self, kind: Fk) -> Frame {
        Frame {
            kind,
            is_gen: self.is_gen,
            asyn: self.asyn,
            strict: self.strict,
            reserved: self.reserved,
            value: false,
            decl: false,
            atom: false,
            inner: false,
            s: 0,
            m: 0,
            mods: 0,
            fdecl: false,
            head: 0,
            qdebt: 0,
            prologue: 0,
        }
    }
}

pub(super) struct Walk {
    generation: u64,
    /// Frame count right after seeding a bounded walk (0: unseeded); the walk stays valid while
    /// that frame is on the stack.
    seed_depth: usize,
    /// A bounded walk popped its seed frame (or an unbalanced closer): its state is a guess from
    /// here on.
    seed_lost: bool,
    /// The seed answers one query only (its frames are a guess past it).
    seed_once: bool,
    frames: Vec<Frame>,
    /// Next unprocessed byte position: every token start below it has been walked.
    upto: usize,
    expr_allowed: bool,
    stmt_start: bool,
    prev_end: usize,
    /// The previous significant token was a numeric literal ending exactly at `prev_end` (so a `.`
    /// there continues the number).
    prev_num: bool,
    /// Previous token was `.` / `?.`: the next word is a property name.
    after_dot: bool,
    /// Keyword code of the previous significant token, 0 if none.
    prev_kw: u8,
    /// Previous token was `=>`, with the async-ness of the arrow.
    prev_arrow: bool,
    arrow_async: bool,
    /// Previous token closed a Group (`)`), and whether `async` preceded it.
    closed_group: bool,
    closed_group_async: bool,
    /// Previous token closed a Params frame.
    closed_params: bool,
    /// The previous significant token was `async` (same line as this one).
    prev_async: bool,
    /// `export default` was just seen.
    export_default: bool,
    /// Decorator at statement level / operand level (0 none).
    deco: u8,
    /// `await` immediately after `for`.
    for_await: bool,
    /// A closing JSX tag is being skipped until its JEND.
    jsx_closing: bool,
    /// Set by the last processed token when the statement it completed cannot be continued by
    /// anything (`break label`, module specifier).
    stmt_done: bool,
    /// Last `after_token` query, so a site asking twice gets one answer.
    last_query: (usize, After),
    /// Start of the last processed token (a query inside it, e.g. at the last `>` of a fused `>>>`,
    /// reports the state after it).
    last_start: usize,
    /// The previous token ended an `as` type or closed a type-argument list; TypeScript never tries
    /// type arguments after those, so a `<` here compares.
    no_type_args: bool,
}

thread_local! {
    static GEN: Cell<u64> = const { Cell::new(0) };
    /// The full walk from the start of the source (memoized).
    static WALK: RefCell<Walk> = RefCell::new(Walk::new());
    /// A bounded walk seeded at a boundary near the query.
    static LOCAL: RefCell<Walk> = RefCell::new(Walk::new());
}

/// Called once per `lex_raw`: invalidates the memoized walk.
pub(super) fn new_lex() {
    GEN.with(|g| g.set(g.get().wrapping_add(1)));
}

/// Called when a later pipeline stage (coalesce) starts asking: the token state it sees differs
/// from carve's, so the walk restarts.
pub(super) fn new_pass() {
    new_lex();
}

/// Context at the (not yet processed) token starting at a position.
#[derive(Clone, Copy, Debug)]
pub(super) struct Site {
    /// Inside a type: annotation, alias, type argument list, type literal.
    pub in_type: bool,
    /// An operand may start here, in expression context.
    pub operand: bool,
    /// A `<` here opens the type-parameter list of a declaration or member.
    pub type_params: bool,
    /// Open `<` lists a `>` run here would close.
    pub angles: usize,
}

impl Walk {
    fn new() -> Walk {
        Walk {
            generation: u64::MAX,
            seed_depth: 0,
            seed_lost: false,
            seed_once: false,
            frames: Vec::with_capacity(64),
            upto: 0,
            expr_allowed: true,
            stmt_start: true,
            prev_end: 0,
            prev_num: false,
            after_dot: false,
            prev_kw: 0,
            prev_arrow: false,
            arrow_async: false,
            closed_group: false,
            closed_group_async: false,
            closed_params: false,
            prev_async: false,
            export_default: false,
            deco: 0,
            for_await: false,
            jsx_closing: false,
            stmt_done: false,
            last_query: (usize::MAX, After::Operand),
            last_start: 0,
            no_type_args: false,
        }
    }

    fn reset(&mut self, module: bool) {
        self.frames.clear();
        self.frames.push(Frame {
            kind: Fk::Root,
            is_gen: false,
            asyn: module,
            strict: module,
            reserved: false,
            value: false,
            decl: false,
            atom: false,
            inner: false,
            s: 0,
            m: 0,
            mods: 0,
            fdecl: false,
            head: 0,
            qdebt: 0,
            prologue: 1,
        });
        self.upto = 0;
        self.expr_allowed = true;
        self.stmt_start = true;
        self.prev_end = 0;
        self.prev_num = false;
        self.after_dot = false;
        self.prev_kw = 0;
        self.prev_arrow = false;
        self.arrow_async = false;
        self.closed_group = false;
        self.closed_group_async = false;
        self.closed_params = false;
        self.prev_async = false;
        self.export_default = false;
        self.deco = 0;
        self.for_await = false;
        self.jsx_closing = false;
        self.stmt_done = false;
        self.last_query = (usize::MAX, After::Operand);
        self.last_start = 0;
        self.no_type_args = false;
        self.seed_depth = 0;
        self.seed_lost = false;
        self.seed_once = false;
    }

    #[inline]
    fn top(&self) -> &Frame {
        self.frames.last().unwrap()
    }

    #[inline]
    fn top_mut(&mut self) -> &mut Frame {
        self.frames.last_mut().unwrap()
    }

    #[inline]
    fn top_kind(&self) -> Fk {
        self.top().kind
    }

    fn push(&mut self, kind: Fk) -> &mut Frame {
        let f = self.top().child(kind);
        self.frames.push(f);
        self.frames.last_mut().unwrap()
    }

    fn pop(&mut self) -> Frame {
        if self.frames.len() > 1 {
            let f = self.frames.pop().unwrap();
            if self.frames.len() < self.seed_depth {
                self.seed_lost = true;
            }
            f
        } else {
            *self.top()
        }
    }

    /// A closer with no frame to close: past here a bounded walk guesses.
    fn unbalanced(&mut self) {
        if self.seed_depth != 0 {
            self.seed_lost = true;
        }
    }

    /// Index of the innermost frame that owns statements / declarations.
    fn stmt_frame(&self) -> usize {
        let mut i = self.frames.len() - 1;
        loop {
            match self.frames[i].kind {
                Fk::Concise | Fk::TypeRegion | Fk::Angle | Fk::FnHead | Fk::ClassHead => {}
                _ => return i,
            }
            if i == 0 {
                return 0;
            }
            i -= 1;
        }
    }

    /// The frame whose declarator state (`let x`, `for (let x`) governs the next token, if the
    /// innermost real frame can hold one.
    fn decl_frame(&self) -> Option<usize> {
        let i = self.stmt_frame();
        if is_stmt_holder(self.frames[i].kind) { Some(i) } else { None }
    }

    /// Statement register of the innermost statement holder (S_NONE inside expression frames).
    fn stmt_reg(&self) -> u8 {
        let i = self.stmt_frame();
        if is_stmt_holder(self.frames[i].kind) { self.frames[i].m } else { S_NONE }
    }

    fn set_stmt_reg(&mut self, v: u8) {
        let i = self.stmt_frame();
        if is_stmt_holder(self.frames[i].kind) {
            self.frames[i].m = v;
        }
    }

    /// Innermost function-like scope: where `yield` / `await` look up their keyword-ness.
    fn scope(&self) -> &Frame {
        let mut i = self.frames.len() - 1;
        loop {
            match self.frames[i].kind {
                Fk::Root
                | Fk::FnBody
                | Fk::ArrowBody
                | Fk::Concise
                | Fk::ClassBody
                | Fk::StaticBlock
                | Fk::Params => return &self.frames[i],
                _ => {}
            }
            if i == 0 {
                return &self.frames[0];
            }
            i -= 1;
        }
    }

    /// Is the top of the stack (ignoring nothing) a type context?
    fn in_type(&self) -> bool {
        matches!(
            self.top_kind(),
            Fk::TypeRegion | Fk::Angle | Fk::TypeParen | Fk::TypeBracket | Fk::TypeLit
        ) || (self.top_kind() == Fk::Sub && self.top().decl)
    }

    /// The nearest TypeRegion above the nearest bracket frame, if any.
    fn region_index(&self) -> Option<usize> {
        let mut i = self.frames.len() - 1;
        loop {
            match self.frames[i].kind {
                Fk::TypeRegion => return Some(i),
                Fk::Concise | Fk::Angle | Fk::FnHead | Fk::ClassHead => {}
                _ => return None,
            }
            if i == 0 {
                return None;
            }
            i -= 1;
        }
    }

    /// Pop concise arrow bodies sitting on top of the stack.
    fn pop_concise(&mut self) {
        while self.top_kind() == Fk::Concise {
            self.pop();
        }
    }

    /// End every virtual frame above the nearest bracket frame (used by closers and separators).
    fn pop_virtual(&mut self) {
        while matches!(
            self.top_kind(),
            Fk::Concise | Fk::TypeRegion | Fk::Angle | Fk::FnHead | Fk::ClassHead
        ) {
            self.pop();
        }
    }

    /// Pop through the nearest bracket frame if its kind is in `kinds`; virtual frames are crossed,
    /// any other bracket frame is a barrier and nothing is popped.
    fn pop_to(&mut self, kinds: &[Fk]) -> Option<Frame> {
        let mut i = self.frames.len();
        while i > 1 {
            i -= 1;
            let k = self.frames[i].kind;
            if kinds.contains(&k) {
                let f = self.frames[i];
                self.frames.truncate(i);
                if i < self.seed_depth {
                    self.seed_lost = true;
                }
                return Some(f);
            }
            if !matches!(k, Fk::Concise | Fk::TypeRegion | Fk::Angle | Fk::FnHead | Fk::ClassHead) {
                return None;
            }
        }
        None
    }

    fn after_statement(&mut self) {
        self.expr_allowed = true;
        self.stmt_start = true;
        let i = self.stmt_frame();
        let f = &mut self.frames[i];
        f.s = D_NONE;
        f.m = S_NONE;
        f.head = 0;
        f.qdebt = 0;
        self.export_default = false;
        self.deco = 0;
    }

    /// Statement boundary reached (`;`, ASI, block end).
    fn end_statement(&mut self) {
        // Bodiless signatures and open type regions end with the statement.
        self.pop_virtual();
        self.after_statement();
    }

    fn set_value(&mut self) {
        self.expr_allowed = false;
        self.stmt_start = false;
    }

    fn set_operand(&mut self) {
        self.expr_allowed = true;
        self.stmt_start = false;
    }

    /// An operator or keyword after which an operand may start.
    fn operand_done(&mut self) {
        self.set_operand();
        self.clear_prev();
    }

    /// A keyword that expects an operand, remembered for the next token.
    fn keyword(&mut self, kwc: u8) {
        self.operand_done();
        self.prev_kw = kwc;
    }
}

/// Frames that hold statements (and so declarator state and statement registers).
fn is_stmt_holder(k: Fk) -> bool {
    matches!(k, Fk::Root | Fk::Block | Fk::FnBody | Fk::ArrowBody | Fk::StaticBlock)
}

#[inline]
unsafe fn raw_kind(cx: Cx, pos: usize) -> u8 {
    *cx.kind.add(pos)
}

/// Base kind: keywords back to IDENT, escaped names to their plain kind.
#[inline]
unsafe fn base_kind(cx: Cx, pos: usize) -> u8 {
    let k = *cx.kind.add(pos);
    if k >= KW_BASE {
        return IDENT;
    }
    if k == IDENT_ESC {
        return IDENT;
    }
    if k == PRIV_IDENT_ESC {
        return PRIV_IDENT;
    }
    k
}

/// Keyword code of the word at `pos` (0 for a plain identifier).
#[inline]
unsafe fn word_kw(cx: Cx, pos: usize, len: usize) -> u8 {
    let raw = *cx.kind.add(pos);
    if raw >= KW_BASE {
        return raw;
    }
    if raw != IDENT || pos < cx.kw_final {
        return 0;
    }
    let set = if cx.ts { &cx.t.kwts } else { &cx.t.kwjs };
    let k = set.lookup(cx.src.add(pos), len) as u8;
    if k >= KW_BASE { k } else { 0 }
}

/// Position of the next significant token start at or after `i` (skipping trivia), or `n`.
#[inline]
unsafe fn next_sig(cx: Cx, mut i: usize) -> usize {
    loop {
        i = bm_next1(cx.st, i, cx.n);
        if i >= cx.n {
            return cx.n;
        }
        let k = *cx.kind.add(i);
        if k == WS || k == LCOM || k == BCOM || k == HASHBANG {
            i += 1;
            continue;
        }
        return i;
    }
}

/// Fused operator at `pos`: `(len, kind)`; `kind` is the punct kind for a single byte (0 for
/// unknown).
#[inline]
unsafe fn munch(cx: Cx, pos: usize) -> (usize, u32) {
    let b0 = *cx.src.add(pos);
    let b1 = *cx.src.add(pos + 1);
    let b2 = *cx.src.add(pos + 2);
    let b3 = *cx.src.add(pos + 3);
    let mut l = 4u32;
    while l >= 2 {
        if pos + l as usize <= cx.n {
            let k = cx.t.op.opmap_lookup(b0, b1, b2, b3, l);
            if k != 0 && !(k == OP_QDOT as u32 && is_digit(b2)) {
                return (l as usize, k);
            }
        }
        l -= 1;
    }
    (1, 0)
}

/// Can `tok` (the token starting at `pos`) continue an expression that a value token ended on the
/// previous line? Used for ASI.
unsafe fn continues_expression(cx: Cx, pos: usize) -> bool {
    let k = base_kind(cx, pos);
    if k >= OP_KIND_BASE {
        let c = *cx.src.add(pos);
        let c1 = *cx.src.add(pos + 1);
        if (c == b'+' || c == b'-') && c1 == c {
            return false;
        }
        return matches!(
            c,
            b'+' | b'-'
                | b'*'
                | b'/'
                | b'%'
                | b'&'
                | b'|'
                | b'^'
                | b'<'
                | b'>'
                | b'='
                | b'?'
                | b'.'
                | b','
                | b'('
                | b'['
                | b':'
                | b')'
                | b']'
                | b'}'
        );
    }
    if k == IDENT {
        let e = bm_next1(cx.st, pos + 1, cx.n);
        let kwc = word_kw(cx, pos, e - pos);
        return kwc == K_IN
            || kwc == K_INSTANCEOF
            || (cx.ts && (kwc == K_AS || kwc == K_SATISFIES));
    }
    matches!(k, TMPL_HEAD | TMPL_NOSUB)
}

/// Can `pos` continue a type after a completed type atom on the previous line? `.`, `|`, `&` may
/// follow a line break; `[`, `<`, `extends` may not.
unsafe fn continues_type_after_break(cx: Cx, pos: usize) -> bool {
    let k = base_kind(cx, pos);
    if k >= OP_KIND_BASE {
        let c = *cx.src.add(pos);
        let c1 = *cx.src.add(pos + 1);
        return (c == b'.' && c1 != b'.')
            || (c == b'|' && c1 != b'|')
            || (c == b'&' && c1 != b'&')
            || c == b'?'
            || c == b':'
            || c == b','
            || c == b')'
            || c == b']'
            || c == b'}'
            || c == b'>'
            || (c == b'=' && c1 == b'>');
    }
    false
}

/// Does the word start a statement no expression can continue? Reserved statement keywords always
/// do, contextual ones only after a line break.
fn is_stmt_keyword(kwc: u8, newline: bool, ts: bool) -> bool {
    match kwc {
        K_IF | K_FOR | K_WHILE | K_RETURN | K_VAR | K_CONST | K_SWITCH | K_TRY | K_THROW | K_DO
        | K_WITH | K_BREAK | K_CONTINUE | K_DEBUGGER | K_FUNCTION | K_CLASS | K_IMPORT
        | K_EXPORT | K_ENUM => true,
        K_LET | K_ASYNC | K_TYPE | K_INTERFACE | K_DECLARE | K_NAMESPACE | K_MODULE
        | K_ABSTRACT | K_USING => newline && (ts || matches!(kwc, K_LET | K_ASYNC | K_USING)),
        _ => false,
    }
}

/// [`after`] on the full walk from the start of the source: needed when the answer depends on the
/// enclosing functions (`yield` / `await`).
pub(super) unsafe fn after_scoped(cx: Cx, module: bool, pos: usize) -> After {
    let generation = GEN.with(Cell::get);
    WALK.with(|cell| {
        let mut w = cell.borrow_mut();
        if w.generation == generation && w.last_query.0 == pos {
            return w.last_query.1;
        }
        let inside_last = w.generation == generation && pos < w.upto && pos >= w.last_start;
        if w.generation != generation || (w.upto > pos && !inside_last) {
            w.generation = generation;
            w.reset(module);
        }
        w.advance(cx, pos);
        // A query inside the token just processed (the tail of a fused operator run such as `>>>`)
        // is answered by the state after it.
        let a = if w.upto > pos { w.classify_after() } else { w.after_token(cx, pos) };
        w.last_query = (pos, a);
        a
    })
}

impl Walk {
    fn advance(&mut self, cx: Cx, limit: usize) {
        let mut pos = self.upto;
        loop {
            pos = unsafe { bm_next1(cx.st, pos, cx.n) };
            if pos >= limit || pos >= cx.n {
                break;
            }
            pos = unsafe { self.step(cx, pos) };
        }
        self.upto = pos.max(self.upto);
    }

    /// Process the single token at `pos` (which must be the next unprocessed token) and report what
    /// it leaves behind.
    pub(super) unsafe fn after_token(&mut self, cx: Cx, pos: usize) -> After {
        debug_assert!(pos >= self.upto);
        let end = self.step(cx, pos);
        self.upto = end.max(self.upto);
        self.classify_after()
    }

    /// The state left behind by the last processed token.
    fn classify_after(&self) -> After {
        if self.stmt_done {
            return After::EndsDecl;
        }
        // A declaration-type region whose last token completed a type.
        if let Some(i) = self.region_index() {
            let r = &self.frames[i];
            if r.decl && r.atom && matches!(r.s, R_INLINE | R_STMT | R_INTERFACE) {
                // Parameter annotations (`(a: T` then `/`) are never followed by a regex; only
                // statement-level regions matter.
                let below = if i > 0 { self.frames[i - 1].kind } else { Fk::Root };
                if !matches!(below, Fk::Params | Fk::Group | Fk::Call | Fk::Angle | Fk::TypeParen) {
                    return After::EndsDecl;
                }
            }
        }
        // A bodiless function signature: `function f(a)` then a line break.
        if self.top_kind() == Fk::FnHead && self.closed_params {
            return After::EndsDecl;
        }
        // `let x` with nothing after the binding.
        if let Some(si) = self.decl_frame() {
            let sf = &self.frames[si];
            if sf.s == D_BOUND && si == self.frames.len() - 1 && sf.kind != Fk::Head {
                return After::EndsDecl;
            }
        }
        if self.expr_allowed { After::Operand } else { After::Value }
    }

    /// Is the identifier `yield` at the (unprocessed) token `pos` a keyword?
    fn yield_is_keyword(&self) -> bool {
        let s = self.scope();
        s.is_gen || s.strict || s.reserved
    }

    fn await_is_keyword(&self) -> bool {
        let s = self.scope();
        s.asyn || s.reserved
    }

    fn site(&self) -> Site {
        let in_type = self.in_type();
        Site {
            in_type,
            operand: self.expr_allowed && !in_type && !self.stmt_done,
            type_params: self.type_params_expected(),
            angles: self.open_angles(),
        }
    }

    /// Would a `<` at the next token open a type-parameter list of a declaration head or member?
    fn type_params_expected(&self) -> bool {
        match self.top_kind() {
            Fk::FnHead | Fk::ClassHead => true,
            Fk::Object | Fk::ClassBody => self.top().s == M_KEY_SEEN,
            _ => self.stmt_reg() == S_TYPE_NAME,
        }
    }

    /// Consecutive open `<` lists at the top of the stack.
    fn open_angles(&self) -> usize {
        let mut i = self.frames.len();
        let mut k = 0;
        while i > 1 {
            i -= 1;
            if self.frames[i].kind == Fk::Angle {
                k += 1;
            } else {
                break;
            }
        }
        k
    }

    unsafe fn step(&mut self, cx: Cx, pos: usize) -> usize {
        self.last_start = pos;
        let raw = raw_kind(cx, pos);
        let k = base_kind(cx, pos);
        if k == WS || k == LCOM || k == BCOM || k == HASHBANG {
            return pos + 1;
        }
        let newline = lt_in_range(cx.src, self.prev_end, pos);
        self.stmt_done = false;

        // Skipping the inside of a closing JSX tag.
        if self.jsx_closing {
            if k == JEND {
                self.jsx_closing = false;
                self.jsx_element_done();
            }
            return pos + 1;
        }
        // Inside an opening tag / children: only structure matters.
        if matches!(self.top_kind(), Fk::JsxTag | Fk::JsxElem) {
            return self.step_jsx(cx, pos, k);
        }

        // ASI: a value ended the previous line and this token cannot continue it.
        if newline && !self.expr_allowed && !continues_expression(cx, pos) {
            self.asi(cx, pos);
        }
        // `let x` then a line break: only `=`, `,`, `;`, `:` and `!` can continue the declarator,
        // anything else starts a new statement.
        if newline && !self.expr_allowed {
            if let Some(di) = self.decl_frame() {
                if self.frames[di].s == D_BOUND && di == self.frames.len() - 1 {
                    let c = *cx.src.add(pos);
                    let k = base_kind(cx, pos);
                    if !(k >= OP_KIND_BASE && matches!(c, b'=' | b',' | b';' | b':' | b'!')) {
                        self.end_statement();
                    }
                }
            }
        }
        // A type ended on the previous line and this token cannot continue it: the annotation, and
        // any statement it belongs to, is over.
        if newline && self.top_kind() == Fk::TypeRegion && self.top().atom {
            if !continues_type_after_break(cx, pos) {
                self.end_region_by_break(cx, pos);
            }
        }
        // Restricted productions: `return` / `throw` / `yield` / `break` / `continue` followed by a
        // line break end their statement.
        if newline
            && matches!(
                self.prev_kw,
                K_RETURN | K_THROW | K_BREAK | K_CONTINUE | K_DEBUGGER | K_YIELD
            )
            && self.top_kind() != Fk::TypeRegion
        {
            self.end_statement();
        }
        // The directive prologue ends at the first statement that is not a string literal.
        if self.stmt_start {
            let si = self.stmt_frame();
            if self.frames[si].prologue == 1 && k != STR {
                self.frames[si].prologue = 0;
            }
        }

        let end = match k {
            IDENT => self.step_word(cx, pos, raw, newline),
            NUM | BIGINT | STR | REGEX | TMPL_NOSUB | PRIV_IDENT => {
                let e = bm_next1(cx.st, pos + 1, cx.n);
                self.literal(cx, pos, k, e, newline);
                e
            }
            TMPL_HEAD => {
                let e = bm_next1(cx.st, pos + 1, cx.n);
                let in_type = self.in_type();
                let f = self.push(Fk::Sub);
                f.decl = in_type;
                self.set_operand();
                e
            }
            TMPL_MIDDLE => {
                let e = bm_next1(cx.st, pos + 1, cx.n);
                if self.pop_to(&[Fk::Sub]).is_none() {
                    self.unbalanced();
                }
                let in_type = self.in_type();
                let f = self.push(Fk::Sub);
                f.decl = in_type;
                self.set_operand();
                e
            }
            TMPL_TAIL => {
                let e = bm_next1(cx.st, pos + 1, cx.n);
                if self.pop_to(&[Fk::Sub]).is_none() {
                    self.unbalanced();
                }
                if self.in_type() {
                    self.type_atom();
                } else {
                    self.value_done();
                }
                e
            }
            JSX_LT => {
                let tpos = next_sig(cx, pos + 1);
                if *cx.src.add(tpos) == b'/' {
                    // Closing tag: the element it closes is the nearest JsxElem frame.
                    if self.pop_to(&[Fk::JsxElem]).is_none() {
                        self.unbalanced();
                    }
                    self.jsx_closing = true;
                } else {
                    self.push(Fk::JsxTag);
                }
                pos + 1
            }
            JEND | JTEXT => pos + 1,
            _ => self.step_op(cx, pos, newline),
        };
        self.prev_end = end;
        end
    }

    fn value_done(&mut self) {
        self.set_value();
        self.clear_prev();
        // The first value in a `for (` head is its binding.
        if self.top_kind() == Fk::Head && self.top().s == F_START {
            self.top_mut().s = F_BOUND;
        }
    }

    fn clear_prev(&mut self) {
        self.no_type_args = false;
        self.after_dot = false;
        self.prev_kw = 0;
        self.prev_arrow = false;
        self.arrow_async = false;
        self.closed_group = false;
        self.closed_params = false;
        self.prev_async = false;
        self.prev_num = false;
    }

    unsafe fn asi(&mut self, cx: Cx, pos: usize) {
        let _ = cx;
        let _ = pos;
        // Concise arrow bodies end with the statement.
        self.pop_concise();
        // An expression-embedded type region ends too.
        while self.top_kind() == Fk::TypeRegion && self.top().s == R_EXPR {
            self.pop();
        }
        if self.top_kind() == Fk::TypeRegion {
            // A declaration type: handled by the caller's type check.
            return;
        }
        // A head continues onto the next line when its body (or more head) follows; otherwise the
        // break ends a bodiless signature.
        if matches!(self.top_kind(), Fk::FnHead | Fk::ClassHead) {
            let c = *cx.src.add(pos);
            let k = base_kind(cx, pos);
            if k >= OP_KIND_BASE && (c == b'{' || c == b'<' || c == b'(') {
                return;
            }
            if k == IDENT {
                let e = bm_next1(cx.st, pos + 1, cx.n);
                if matches!(word_kw(cx, pos, e - pos), K_EXTENDS | K_IMPLEMENTS) {
                    return;
                }
            }
            self.end_statement();
            return;
        }
        match self.top_kind() {
            Fk::ClassBody => {
                // Member initializer ended; a new member starts.
                let f = self.top_mut();
                f.s = M_KEY_POS;
                f.mods = 0;
                self.set_operand();
            }
            Fk::Object
            | Fk::Call
            | Fk::Group
            | Fk::Array
            | Fk::Index
            | Fk::Params
            | Fk::Sub
            | Fk::Container
            | Fk::Head
            | Fk::ComputedKey => {
                // No statements here; nothing to end.
            }
            _ => self.end_statement(),
        }
    }

    unsafe fn end_region_by_break(&mut self, cx: Cx, pos: usize) {
        let _ = (cx, pos);
        let r = self.pop();
        match r.s {
            R_STMT | R_INTERFACE => self.end_statement(),
            R_INLINE => {
                // A declarator / member annotation ended by a line break.
                match self.top_kind() {
                    Fk::ClassBody => {
                        let f = self.top_mut();
                        f.s = M_KEY_POS;
                        f.mods = 0;
                        self.set_operand();
                    }
                    Fk::FnHead => {
                        // Return type of a bodiless signature.
                        self.pop();
                        self.end_statement();
                    }
                    _ => self.end_statement(),
                }
            }
            R_EXPR => {
                // `x as T` then a new line: the value is complete.
                self.set_value();
                self.no_type_args = true;
                if !continues_expression(cx, pos) {
                    self.asi(cx, pos);
                }
            }
            _ => self.set_value(),
        }
    }

    fn jsx_element_done(&mut self) {
        // Back in whatever expression held the element (or in a parent's children, where it does
        // not matter).
        if !matches!(self.top_kind(), Fk::JsxTag | Fk::JsxElem) {
            self.value_done();
        }
    }

    unsafe fn step_jsx(&mut self, cx: Cx, pos: usize, k: u8) -> usize {
        let c = *cx.src.add(pos);
        match k {
            JSX_LT => {
                let tpos = next_sig(cx, pos + 1);
                if *cx.src.add(tpos) == b'/' {
                    if self.pop_to(&[Fk::JsxElem]).is_none() {
                        self.unbalanced();
                    }
                    self.jsx_closing = true;
                } else {
                    self.push(Fk::JsxTag);
                }
                pos + 1
            }
            JEND => {
                // Self-closing tag or closing tag end.
                self.pop_to(&[Fk::JsxTag, Fk::JsxElem]);
                self.jsx_element_done();
                pos + 1
            }
            JTEXT | STR | IDENT | NUM | BIGINT | TMPL_NOSUB => bm_next1(cx.st, pos + 1, cx.n),
            _ if k >= OP_KIND_BASE => {
                match c {
                    b'{' => {
                        self.push(Fk::Container);
                        self.operand_done();
                    }
                    b'<' => {
                        if self.top_kind() == Fk::JsxTag {
                            self.top_mut().s += 1;
                        }
                    }
                    b'>' => {
                        if self.top_kind() == Fk::JsxTag {
                            if self.top().s > 0 {
                                self.top_mut().s -= 1;
                            } else {
                                self.top_mut().kind = Fk::JsxElem;
                            }
                        }
                    }
                    _ => {}
                }
                pos + 1
            }
            _ => pos + 1,
        }
    }

    unsafe fn literal(&mut self, cx: Cx, pos: usize, k: u8, end: usize, newline: bool) {
        let _ = newline;
        if self.top_kind() == Fk::TypeRegion || self.in_type() {
            self.type_atom();
            return;
        }
        // Directive prologue.
        if k == STR {
            let si = self.stmt_frame();
            if self.frames[si].prologue != 0 && self.stmt_start {
                let j = next_sig(cx, end);
                let confirmed = j >= cx.n
                    || (base_kind(cx, j) >= OP_KIND_BASE
                        && (*cx.src.add(j) == b';' || *cx.src.add(j) == b'}'))
                    || (lt_in_range(cx.src, end, j) && !continues_expression(cx, j));
                if confirmed {
                    if end - pos == 12
                        && ident_is(cx.src, pos + 1, b"use strict")
                        && *cx.src.add(end - 1) == *cx.src.add(pos)
                    {
                        self.frames[si].strict = true;
                    }
                } else {
                    self.frames[si].prologue = 0;
                }
            }
            // Module specifier: `import "x"`, `... from "x"`.
            let reg = self.stmt_reg();
            if (matches!(reg, S_IMPORT | S_IMPORT_NAME)
                && matches!(self.prev_kw, K_IMPORT | K_FROM))
                || (reg == S_EXPORT && self.prev_kw == K_FROM)
            {
                self.value_done();
                let nx = next_sig(cx, end);
                let attrs = nx < cx.n
                    && base_kind(cx, nx) == IDENT
                    && !lt_in_range(cx.src, end, nx)
                    && (ident_is(cx.src, nx, b"with") || ident_is(cx.src, nx, b"assert"));
                if attrs {
                    self.set_stmt_reg(S_IMPORT);
                    return;
                }
                self.stmt_done = true;
                self.after_statement();
                return;
            }
            if reg == S_DECLARE_MODULE {
                // `declare module "x"`: bodiless unless `{` follows.
                self.value_done();
                self.stmt_done = true;
                return;
            }
        }
        // Member keys.
        match self.top_kind() {
            Fk::Object | Fk::ClassBody if self.top().s == M_KEY_POS => {
                self.top_mut().s = M_KEY_SEEN;
                self.value_done();
                return;
            }
            Fk::Head => {
                if self.top().s == F_START {
                    self.top_mut().s = F_EXPR;
                }
            }
            _ => {}
        }
        self.value_done();
        self.prev_num = k == NUM || k == BIGINT;
    }

    fn type_atom(&mut self) {
        if let Some(i) = self.region_index() {
            let r = &mut self.frames[i];
            r.atom = true;
            r.inner = false;
        }
        self.set_value();
        self.clear_prev();
    }

    fn type_operator(&mut self) {
        if let Some(i) = self.region_index() {
            let r = &mut self.frames[i];
            r.atom = false;
            r.inner = false;
        }
        self.operand_done();
    }

    unsafe fn step_word(&mut self, cx: Cx, pos: usize, raw: u8, newline: bool) -> usize {
        let end = bm_next1(cx.st, pos + 1, cx.n);
        let len = end - pos;
        let mut kwc = if self.after_dot { 0 } else { word_kw(cx, pos, len) };
        let _ = raw;

        // Inside a type: prefix / infix type keywords, else an atom.
        if self.in_type() {
            match kwc {
                K_KEYOF | K_TYPEOF | K_READONLY | K_UNIQUE | K_INFER | K_ABSTRACT | K_NEW
                | K_ASSERTS | K_IMPORT | K_EXTENDS | K_IS | K_IN | K_AS => {
                    self.type_operator();
                    self.prev_kw = kwc;
                    if kwc == K_EXTENDS {
                        // A conditional type: the `?` and `:` to come belong to the type, not to an
                        // enclosing expression.
                        if let Some(i) = self.region_index() {
                            self.frames[i].qdebt += 1;
                        }
                    }
                }
                _ => {
                    // A statement keyword right after a completed type is an error on the same line
                    // and was handled by the break rule on a new one; read it as an atom.
                    self.type_atom();
                    if kwc == K_THIS {
                        self.prev_kw = K_THIS;
                    }
                }
            }
            return end;
        }

        // Contextual keyword resolution.
        match kwc {
            K_YIELD => {
                if !self.yield_is_keyword() {
                    kwc = 0;
                }
            }
            K_AWAIT => {
                if self.prev_kw == K_FOR {
                    self.for_await = true;
                    self.prev_kw = K_FOR;
                    return end;
                }
                if !self.await_is_keyword() {
                    kwc = 0;
                }
            }
            K_OF => {
                if !(self.top_kind() == Fk::Head
                    && self.top().head == H_FOR
                    && self.top().s == F_BOUND)
                {
                    kwc = 0;
                }
            }
            K_LET => {
                let nx = next_sig(cx, end);
                let ok = nx < cx.n && {
                    let nk = base_kind(cx, nx);
                    let c = *cx.src.add(nx);
                    nk == IDENT || (nk >= OP_KIND_BASE && (c == b'[' || c == b'{'))
                };
                let at_stmt = self.stmt_start
                    || self.top_kind() == Fk::Head
                    || matches!(self.prev_kw, K_DECLARE | K_EXPORT);
                if !ok || !at_stmt {
                    kwc = 0;
                }
            }
            K_USING => {
                let nx = next_sig(cx, end);
                let ok = nx < cx.n
                    && base_kind(cx, nx) == IDENT
                    && !lt_in_range(cx.src, end, nx)
                    && (self.stmt_start || matches!(self.prev_kw, K_DECLARE | K_EXPORT));
                if !ok {
                    kwc = 0;
                }
            }
            K_ASYNC => {
                // `async` is a modifier only when the next token is on the same line and continues
                // a function / arrow head.
                let nx = next_sig(cx, end);
                let same_line = nx < cx.n && !lt_in_range(cx.src, end, nx);
                let nk = if nx < cx.n { base_kind(cx, nx) } else { 0 };
                let nc = if nx < cx.n { *cx.src.add(nx) } else { 0 };
                let follows = same_line
                    && (nk == IDENT
                        || (nk >= OP_KIND_BASE && (nc == b'(' || nc == b'*' || nc == b'['))
                        || nk == STR
                        || nk == NUM
                        || nk == PRIV_IDENT);
                if !follows {
                    kwc = 0;
                }
            }
            K_TYPE | K_INTERFACE | K_NAMESPACE | K_MODULE | K_DECLARE | K_ABSTRACT | K_GLOBAL => {
                // Statement-level TS declarations only.
                let nx = next_sig(cx, end);
                let nk = if nx < cx.n { base_kind(cx, nx) } else { 0 };
                let nc = if nx < cx.n { *cx.src.add(nx) } else { 0 };
                let same_line = nx < cx.n && !lt_in_range(cx.src, end, nx);
                let starts_decl = same_line
                    && (nk == IDENT
                        || nk == STR
                        || (kwc == K_GLOBAL && nk >= OP_KIND_BASE && nc == b'{'));
                let at_stmt = self.stmt_start
                    || matches!(self.prev_kw, K_EXPORT | K_DECLARE | K_DEFAULT | K_ABSTRACT)
                    || (kwc == K_NAMESPACE && self.stmt_reg() == S_EXPORT_AS);
                if !(cx.ts && starts_decl && at_stmt) {
                    kwc = 0;
                }
            }
            K_AS | K_SATISFIES => {
                // Only after a value in an expression, in TS; `export as` opens `export as
                // namespace X`.
                let export_as =
                    kwc == K_AS && self.stmt_reg() == S_EXPORT && self.prev_kw == K_EXPORT;
                let in_module_clause = self.top_kind() == Fk::ModuleSpec
                    || matches!(self.stmt_reg(), S_IMPORT | S_EXPORT);
                if !export_as && (!cx.ts || self.expr_allowed || in_module_clause) {
                    kwc = 0;
                }
            }
            K_STATIC => {
                if !(self.top_kind() == Fk::ClassBody && self.top().s == M_KEY_POS) {
                    kwc = 0;
                }
            }
            K_IMPLEMENTS => {
                if self.top_kind() != Fk::ClassHead {
                    kwc = 0;
                }
            }
            K_FROM => {
                if !matches!(self.stmt_reg(), S_IMPORT | S_EXPORT | S_IMPORT_NAME) {
                    kwc = 0;
                }
            }
            _ => {}
        }

        // Member keys in object literals / class bodies.
        if matches!(self.top_kind(), Fk::Object | Fk::ClassBody) && self.top().s == M_KEY_POS {
            return self.member_word(cx, pos, end, kwc, newline);
        }

        // Statement-level keywords that cannot continue an expression start a new statement even
        // without a separator.
        let import_attrs =
            kwc == K_WITH && matches!(self.stmt_reg(), S_IMPORT | S_IMPORT_NAME | S_EXPORT);
        if !self.expr_allowed
            && kwc != 0
            && self.deco == 0
            && !import_attrs
            && is_stmt_keyword(kwc, newline, cx.ts)
        {
            if matches!(
                self.top_kind(),
                Fk::Root | Fk::Block | Fk::FnBody | Fk::ArrowBody | Fk::StaticBlock | Fk::FnHead
            ) {
                self.end_statement();
            }
        }

        let si = self.stmt_frame();
        let stmt_reg = self.stmt_reg();
        let at_start = self.stmt_start;

        // Statement registers that consume a name.
        match stmt_reg {
            S_BREAK => {
                // `break label`: the statement is complete.
                self.set_stmt_reg(S_NONE);
                self.value_done();
                self.stmt_done = true;
                self.after_statement();
                return end;
            }
            S_TYPE => {
                self.set_stmt_reg(S_TYPE_NAME);
                self.value_done();
                return end;
            }
            S_NAMESPACE | S_ENUM if kwc == 0 || kwc == K_GLOBAL => {
                // The declared name (dotted for namespaces).
                self.value_done();
                return end;
            }
            S_IMPORT if kwc == 0 || kwc == K_TYPE => {
                // `import x` / `import type x` / `import x = ...`
                if kwc == K_TYPE && self.prev_kw == K_IMPORT {
                    let nx = next_sig(cx, end);
                    let nk = if nx < cx.n { base_kind(cx, nx) } else { 0 };
                    if nx < cx.n
                        && (nk == IDENT
                            || (nk >= OP_KIND_BASE && matches!(*cx.src.add(nx), b'{' | b'*')))
                    {
                        self.prev_kw = K_TYPE;
                        return end;
                    }
                }
                self.set_stmt_reg(S_IMPORT_NAME);
                self.value_done();
                return end;
            }
            S_EXPORT_AS if kwc == K_NAMESPACE => {
                self.set_stmt_reg(S_EXPORT_AS_NS);
                self.set_operand();
                self.prev_kw = K_NAMESPACE;
                return end;
            }
            S_EXPORT_AS_NS => {
                self.set_stmt_reg(S_NONE);
                self.value_done();
                self.stmt_done = true;
                self.after_statement();
                return end;
            }
            _ => {}
        }

        match kwc {
            0 => self.plain_word(cx, pos, end, at_start),
            K_THIS | K_SUPER | K_NULL | K_TRUE | K_FALSE => {
                self.plain_word(cx, pos, end, false);
            }
            K_FUNCTION => {
                let value = !self.stmt_start
                    && !self.export_default
                    && self.deco == 0
                    && self.expr_allowed
                    && !matches!(self.prev_kw, K_EXPORT | K_DECLARE);
                let asyn = self.prev_async;
                let f = self.push(Fk::FnHead);
                f.value = value;
                f.asyn = asyn;
                f.is_gen = false;
                self.set_value();
                self.clear_prev();
                self.prev_kw = K_FUNCTION;
                self.export_default = false;
            }
            K_CLASS => {
                let value = if self.deco != 0 {
                    self.deco == 2
                } else {
                    !self.stmt_start
                        && !self.export_default
                        && self.expr_allowed
                        && !matches!(self.prev_kw, K_EXPORT | K_DECLARE | K_ABSTRACT)
                };
                let f = self.push(Fk::ClassHead);
                f.value = value;
                self.set_value();
                self.clear_prev();
                self.prev_kw = K_CLASS;
                self.export_default = false;
                self.deco = 0;
            }
            K_EXTENDS => {
                // Class heritage expression.
                if self.top_kind() == Fk::ClassHead {
                    self.top_mut().s = 1;
                }
                self.keyword(K_EXTENDS);
            }
            K_IMPLEMENTS => {
                // Type references follow.
                self.top_mut().s = 2;
                self.keyword(K_IMPLEMENTS);
            }
            K_WITH if matches!(stmt_reg, S_IMPORT | S_IMPORT_NAME | S_EXPORT) => {
                // Import attributes: `from "x" with { type: "json" }`.
                self.keyword(K_WITH);
            }
            K_IF | K_WHILE | K_FOR | K_WITH | K_SWITCH | K_CATCH => {
                self.operand_done();
                if kwc == K_CATCH {
                    // `catch {` without a binding.
                    self.stmt_start = true;
                }
                self.prev_kw = kwc;
                let _ = si;
                let hi = self.stmt_frame();
                self.frames[hi].head = match kwc {
                    K_IF => H_IF,
                    K_WHILE => H_WHILE,
                    K_FOR => H_FOR,
                    K_WITH => H_WITH,
                    K_SWITCH => H_SWITCH,
                    _ => H_CATCH,
                };
                self.for_await = false;
            }
            K_ELSE | K_DO | K_TRY | K_FINALLY => {
                self.expr_allowed = true;
                self.stmt_start = true;
                self.clear_prev();
                self.prev_kw = kwc;
            }
            K_RETURN | K_THROW | K_YIELD | K_AWAIT | K_TYPEOF | K_VOID | K_DELETE | K_NEW
            | K_IN | K_INSTANCEOF | K_OF | K_DEBUGGER => {
                if self.top_kind() == Fk::Head && matches!(kwc, K_OF | K_IN) {
                    self.top_mut().s = F_ITER;
                }
                self.keyword(kwc);
            }
            K_CASE => {
                self.set_stmt_reg(S_CASE);
                self.keyword(K_CASE);
            }
            K_DEFAULT => {
                if self.prev_kw == K_EXPORT {
                    self.export_default = true;
                    self.set_stmt_reg(S_NONE);
                    self.set_operand();
                } else {
                    self.set_stmt_reg(S_CASE);
                    self.set_operand();
                }
                self.clear_prev();
                self.prev_kw = K_DEFAULT;
            }
            K_BREAK | K_CONTINUE => {
                self.set_stmt_reg(S_BREAK);
                self.keyword(kwc);
            }
            K_VAR | K_CONST | K_LET | K_USING => {
                if kwc == K_CONST {
                    // `const enum`
                    let nx = next_sig(cx, end);
                    if nx < cx.n && base_kind(cx, nx) == IDENT && ident_is(cx.src, nx, b"enum") {
                        self.keyword(K_CONST);
                        return end;
                    }
                }
                if let Some(di) = self.decl_frame() {
                    self.frames[di].s = D_BINDING;
                }
                self.keyword(kwc);
            }
            K_IMPORT => {
                let nx = next_sig(cx, end);
                let nc = if nx < cx.n { *cx.src.add(nx) } else { 0 };
                if nx < cx.n && base_kind(cx, nx) >= OP_KIND_BASE && (nc == b'(' || nc == b'.') {
                    // `import(...)` / `import.meta`: an expression.
                    self.set_value();
                    self.clear_prev();
                    self.prev_kw = K_IMPORT;
                } else {
                    self.set_stmt_reg(S_IMPORT);
                    self.keyword(K_IMPORT);
                }
            }
            K_EXPORT => {
                self.set_stmt_reg(S_EXPORT);
                self.keyword(K_EXPORT);
            }
            K_AS => {
                if stmt_reg == S_EXPORT && self.prev_kw == K_EXPORT {
                    self.set_stmt_reg(S_EXPORT_AS);
                    self.keyword(K_AS);
                } else {
                    self.open_region(R_EXPR, false);
                    self.prev_kw = K_AS;
                }
            }
            K_SATISFIES => {
                self.open_region(R_EXPR, false);
                self.prev_kw = K_SATISFIES;
            }
            K_ASYNC => {
                // A modifier: the function or arrow it modifies decides expression-ness, so it is
                // transparent.
                self.clear_prev();
                self.prev_async = true;
                self.prev_kw = K_ASYNC;
            }
            K_TYPE => {
                self.set_stmt_reg(S_TYPE);
                self.keyword(K_TYPE);
            }
            K_INTERFACE => {
                // Same head frame as a class: `extends`, `<` and the body may follow on later
                // lines.
                let f = self.push(Fk::ClassHead);
                f.value = false;
                f.m = 1;
                self.set_value();
                self.clear_prev();
                self.prev_kw = K_INTERFACE;
            }
            K_ENUM => {
                self.set_stmt_reg(S_ENUM);
                self.keyword(K_ENUM);
            }
            K_NAMESPACE | K_MODULE => {
                self.set_stmt_reg(S_NAMESPACE);
                let declare = self.prev_kw == K_DECLARE;
                self.keyword(kwc);
                if declare && kwc == K_MODULE {
                    // `declare module "x"` may have no body.
                    self.set_stmt_reg(S_DECLARE_MODULE);
                }
            }
            K_DECLARE | K_ABSTRACT | K_GLOBAL => {
                self.keyword(kwc);
                if kwc == K_GLOBAL {
                    self.stmt_start = true;
                }
            }
            K_STATIC => {
                self.top_mut().mods |= MOD_STATIC;
                self.keyword(K_STATIC);
            }
            K_FROM => {
                self.keyword(K_FROM);
            }
            _ => {
                // Any other keyword spelling used as a plain word.
                self.plain_word(cx, pos, end, at_start);
            }
        }
        end
    }

    /// A plain identifier (or keyword used as a name) in expression / statement position.
    unsafe fn plain_word(&mut self, cx: Cx, pos: usize, end: usize, at_start: bool) {
        // Declarator binding.
        if let Some(si) = self.decl_frame() {
            if self.frames[si].s == D_BINDING && si == self.frames.len() - 1 {
                if self.frames[si].kind == Fk::Head {
                    self.frames[si].fdecl = true;
                }
                self.frames[si].s = D_BOUND;
                self.value_done();
                return;
            }
        }
        // Label candidate: a lone identifier at statement start.
        if at_start && self.stmt_reg() == S_NONE && self.top_kind() != Fk::Head {
            let nx = next_sig(cx, end);
            if nx < cx.n
                && base_kind(cx, nx) >= OP_KIND_BASE
                && *cx.src.add(nx) == b':'
                && *cx.src.add(nx + 1) != b':'
            {
                self.set_stmt_reg(S_LABEL);
            }
        }
        // `async x => ...`: remember the modifier for the arrow.
        let asyn = self.prev_async;
        self.value_done();
        self.arrow_async = asyn;
        let _ = pos;
    }

    /// A word at member-key position of an object literal or class body.
    unsafe fn member_word(
        &mut self,
        cx: Cx,
        pos: usize,
        end: usize,
        kwc: u8,
        newline: bool,
    ) -> usize {
        let _ = newline;
        let is_class = self.top_kind() == Fk::ClassBody;
        // Modifiers apply when a key can follow on the same line.
        let nx = next_sig(cx, end);
        let nk = if nx < cx.n { base_kind(cx, nx) } else { 0 };
        let nc = if nx < cx.n { *cx.src.add(nx) } else { 0 };
        let key_follows = nx < cx.n
            && !lt_in_range(cx.src, end, nx)
            && (matches!(nk, IDENT | STR | NUM | BIGINT | PRIV_IDENT)
                || (nk >= OP_KIND_BASE && (nc == b'[' || nc == b'*' || nc == b'#')));
        let key_follows_any_line = nx < cx.n
            && (matches!(nk, IDENT | STR | NUM | BIGINT | PRIV_IDENT)
                || (nk >= OP_KIND_BASE && (nc == b'[' || nc == b'*' || nc == b'#')));
        if kwc == K_ASYNC && key_follows {
            self.top_mut().mods |= MOD_ASYNC;
            self.operand_done();
            return end;
        }
        if is_class && kwc == K_STATIC {
            if nx < cx.n && nk >= OP_KIND_BASE && nc == b'{' {
                self.top_mut().mods |= MOD_STATIC;
                self.keyword(K_STATIC);
                return end;
            }
            if key_follows_any_line {
                self.top_mut().mods |= MOD_STATIC;
                self.operand_done();
                return end;
            }
        }
        if is_class
            && matches!(
                kwc,
                K_PUBLIC
                    | K_PRIVATE
                    | K_PROTECTED
                    | K_READONLY
                    | K_ABSTRACT
                    | K_OVERRIDE
                    | K_DECLARE
                    | K_ACCESSOR
            )
            && key_follows_any_line
        {
            self.operand_done();
            return end;
        }
        if kwc == 0
            && (ident_is(cx.src, pos, b"get") || ident_is(cx.src, pos, b"set"))
            && key_follows_any_line
        {
            self.operand_done();
            return end;
        }
        // The key itself.
        self.top_mut().s = M_KEY_SEEN;
        self.value_done();
        end
    }

    unsafe fn open_region(&mut self, rule: u8, decl: bool) {
        let f = self.push(Fk::TypeRegion);
        f.decl = decl;
        f.s = rule;
        f.atom = false;
        f.inner = false;
        self.operand_done();
    }

    /// End the type region on top because `pos` is an expression token. Returns true if a region
    /// was ended.
    unsafe fn end_region_for(&mut self, cx: Cx, pos: usize) -> bool {
        if self.top_kind() != Fk::TypeRegion {
            return false;
        }
        let _ = (cx, pos);
        let r = self.pop();
        self.set_value();
        if r.s == R_EXPR {
            self.no_type_args = true;
        }
        true
    }

    unsafe fn step_op(&mut self, cx: Cx, pos: usize, newline: bool) -> usize {
        let c = *cx.src.add(pos);
        let (mut len, _) = if is_op_char(c) || c == b'/' { munch(cx, pos) } else { (1, 0) };
        let c1 = *cx.src.add(pos + 1);

        // In a `for (` head an operator other than member access makes the binding an expression
        // (`for (x = of / 2;;)`), so a later `of` is a plain identifier.
        if self.top_kind() == Fk::Head
            && self.top().s == F_BOUND
            && !matches!(c, b'.' | b'[' | b'(' | b')' | b']' | b'}' | b'{' | b',' | b';')
            && !(c == b'?' && c1 == b'.')
        {
            self.top_mut().s = F_EXPR;
        }

        // Type context: brackets and separators belong to the type.
        if self.in_type() {
            // `>` closes one Angle per byte; `<<` opens two.
            if c == b'>' && !(c1 == b'=' && self.top_kind() != Fk::Angle) {
                len = 1;
            }
            if c == b'<' && c1 == b'<' {
                len = 1;
            }
            if c == b'=' && c1 == b'>' {
                len = 2;
            }
            return self.type_op(cx, pos, c, len);
        }

        match c {
            b'{' => {
                self.open_brace(cx, pos, newline);
                pos + 1
            }
            b'}' => {
                self.close_brace(cx, pos);
                pos + 1
            }
            b'(' => {
                self.open_paren(cx, pos);
                pos + 1
            }
            b')' => {
                self.close_paren(cx, pos);
                pos + 1
            }
            b'[' => {
                self.open_bracket(cx, pos);
                pos + 1
            }
            b']' => {
                self.close_bracket(cx, pos);
                pos + 1
            }
            b';' => {
                self.semicolon(cx, pos);
                pos + 1
            }
            b',' => {
                self.comma(cx, pos);
                pos + 1
            }
            b':' => {
                self.colon(cx, pos);
                pos + 1
            }
            b'?' => {
                if len >= 2 {
                    // `?.` / `??` / `??=`
                    if c1 == b'.' {
                        self.operand_done();
                        self.after_dot = true;
                    } else {
                        self.operand_done();
                    }
                    return pos + len;
                }
                self.question(cx, pos);
                pos + 1
            }
            b'.' => {
                if len == 3 {
                    // spread: what follows is a value, not a member key
                    if self.top_kind() == Fk::Object {
                        self.top_mut().s = M_VALUE;
                    }
                    self.operand_done();
                    return pos + 3;
                }
                if self.prev_num && self.prev_end == pos {
                    // `1.` continues the numeric literal.
                    self.set_value();
                    self.prev_num = false;
                    return pos + 1;
                }
                self.operand_done();
                self.after_dot = true;
                pos + 1
            }
            b'=' => {
                if len == 2 && c1 == b'>' {
                    self.arrow(cx, pos);
                    return pos + 2;
                }
                if len == 1 {
                    self.assign(cx, pos);
                    return pos + 1;
                }
                // `==` / `===`
                self.operand_done();
                pos + len
            }
            b'!' => {
                if len == 1 && cx.ts && !self.expr_allowed && !newline {
                    // Postfix non-null / definite assignment.
                    self.set_value();
                    self.clear_prev();
                    return pos + 1;
                }
                self.operand_done();
                pos + len
            }
            b'+' | b'-' => {
                if len == 2 && c1 == c {
                    // `++` / `--`: postfix keeps the value.
                    if !self.expr_allowed && !newline {
                        self.set_value();
                        self.clear_prev();
                    } else {
                        self.operand_done();
                    }
                    return pos + 2;
                }
                self.operand_done();
                pos + len
            }
            b'*' => {
                if len == 1 {
                    if self.top_kind() == Fk::FnHead {
                        self.top_mut().is_gen = true;
                        self.set_value();
                        self.clear_prev();
                        self.prev_kw = K_FUNCTION;
                        return pos + 1;
                    }
                    if matches!(self.top_kind(), Fk::Object | Fk::ClassBody)
                        && self.top().s == M_KEY_POS
                    {
                        self.top_mut().mods |= MOD_GEN;
                        self.operand_done();
                        return pos + 1;
                    }
                    if self.prev_kw == K_YIELD {
                        // `yield*`
                        self.set_operand();
                        return pos + 1;
                    }
                }
                self.operand_done();
                pos + len
            }
            b'<' => {
                if len >= 2 {
                    if c1 == b'<'
                        && len == 2
                        && cx.ts
                        && !self.expr_allowed
                        && !self.no_type_args
                        && lt_run_opens_type_args(cx.src, cx.st, cx.opch, cx.kind, cx.n, pos)
                    {
                        // Two openers, not a shift.
                        self.less_than(cx, pos);
                        return pos + 1;
                    }
                    self.operand_done();
                    return pos + len;
                }
                self.less_than(cx, pos);
                pos + 1
            }
            b'>' => {
                if self.top_kind() == Fk::Angle {
                    self.close_angle(cx, pos);
                    return pos + 1;
                }
                self.operand_done();
                pos + len
            }
            b'@' => {
                if self.deco == 0 {
                    self.deco = if self.stmt_start || !self.expr_allowed { 1 } else { 2 };
                }
                self.operand_done();
                pos + 1
            }
            b'#' => {
                // Stray `#` (private names are PRIV_IDENT tokens).
                self.operand_done();
                pos + 1
            }
            _ => {
                // Every other operator expects an operand.
                self.operand_done();
                if self.top_kind() == Fk::Head && self.top().s != F_ITER {
                    self.top_mut().s = F_EXPR;
                }
                pos + len
            }
        }
    }

    unsafe fn type_op(&mut self, cx: Cx, pos: usize, c: u8, len: usize) -> usize {
        let top = self.top_kind();
        match c {
            b'(' => {
                if top == Fk::TypeRegion && self.top().atom && self.top().s == R_EXPR {
                    // `x as T (` cannot continue the type.
                    self.end_region_for(cx, pos);
                    self.open_paren(cx, pos);
                    return pos + 1;
                }
                let f = self.push(Fk::TypeParen);
                f.decl = true;
                self.operand_done();
                pos + 1
            }
            b')' => {
                if top == Fk::TypeParen {
                    self.pop();
                    if let Some(i) = self.region_index() {
                        let r = &mut self.frames[i];
                        r.atom = true;
                        r.inner = true;
                    }
                    self.set_value();
                    self.clear_prev();
                    return pos + 1;
                }
                // Closes something outside the type.
                self.pop_virtual();
                self.close_paren(cx, pos);
                pos + 1
            }
            b'[' => {
                if top == Fk::TypeRegion
                    && self.top().atom
                    && self.top().s == R_EXPR
                    && lt_in_range(cx.src, self.prev_end, pos)
                {
                    self.end_region_for(cx, pos);
                    self.open_bracket(cx, pos);
                    return pos + 1;
                }
                let f = self.push(Fk::TypeBracket);
                f.decl = true;
                self.operand_done();
                pos + 1
            }
            b']' => {
                if top == Fk::TypeBracket {
                    self.pop();
                    if let Some(i) = self.region_index() {
                        let r = &mut self.frames[i];
                        r.atom = true;
                        r.inner = false;
                    }
                    self.set_value();
                    self.clear_prev();
                    return pos + 1;
                }
                self.pop_virtual();
                self.close_bracket(cx, pos);
                pos + 1
            }
            b'{' => {
                if top == Fk::TypeRegion && self.top().atom {
                    // A body follows a completed type (`): T {`).
                    let r = self.pop();
                    if r.s == R_INTERFACE {
                        // `interface X extends Y {`: the body.
                        let f = self.push(Fk::TypeLit);
                        f.decl = true;
                        f.value = false;
                        f.s = 1; // ends the interface statement when closed
                        self.operand_done();
                        return pos + 1;
                    }
                    self.set_value();
                    self.open_brace(cx, pos, false);
                    return pos + 1;
                }
                let expr = self.region_index().is_some_and(|i| !self.frames[i].decl);
                let f = self.push(Fk::TypeLit);
                f.decl = !expr;
                f.value = expr;
                self.operand_done();
                pos + 1
            }
            b'}' => {
                if top == Fk::TypeLit {
                    let f = self.pop();
                    if f.s == 1 {
                        // Interface body done: statement over.
                        if self.top_kind() == Fk::TypeRegion {
                            self.pop();
                        }
                        self.end_statement();
                        return pos + 1;
                    }
                    if let Some(i) = self.region_index() {
                        let r = &mut self.frames[i];
                        r.atom = true;
                        r.inner = false;
                    }
                    self.set_value();
                    self.clear_prev();
                    return pos + 1;
                }
                self.pop_virtual();
                self.close_brace(cx, pos);
                pos + 1
            }
            b'<' => {
                if self.prev_kw == K_THIS {
                    // `this` takes no type arguments: the type is over and this `<` is a
                    // comparison.
                    self.end_region_for(cx, pos);
                    self.operand_done();
                    return pos + 1;
                }
                let decl = self.region_index().is_none_or(|i| self.frames[i].decl);
                let f = self.push(Fk::Angle);
                f.decl = decl;
                f.s = 3;
                self.operand_done();
                pos + 1
            }
            b'>' => {
                if top == Fk::Angle {
                    self.close_angle(cx, pos);
                    return pos + 1;
                }
                // Relational `>` after `x as T`: the type is over.
                self.end_region_for(cx, pos);
                self.operand_done();
                pos + len
            }
            b',' => {
                match top {
                    Fk::Angle | Fk::TypeParen | Fk::TypeBracket | Fk::TypeLit => {
                        self.type_operator();
                    }
                    _ => {
                        // Ends the region: next declarator / parameter / argument.
                        self.pop();
                        self.comma(cx, pos);
                    }
                }
                pos + 1
            }
            b';' => {
                if top == Fk::TypeLit {
                    self.type_operator();
                    return pos + 1;
                }
                self.pop_virtual();
                self.semicolon(cx, pos);
                pos + 1
            }
            b'=' => {
                if len == 2 {
                    // `=>` continues a function type only right after its parameter list.
                    if top == Fk::TypeRegion && self.top().inner {
                        self.type_operator();
                        return pos + 2;
                    }
                    if top == Fk::TypeRegion && self.top().s == R_ARROW_RET {
                        let r = self.pop();
                        self.closed_group = true;
                        self.closed_group_async = r.asyn;
                        self.arrow(cx, pos);
                        return pos + 2;
                    }
                    if matches!(top, Fk::Angle | Fk::TypeParen | Fk::TypeBracket | Fk::TypeLit) {
                        self.type_operator();
                        return pos + 2;
                    }
                    self.end_region_for(cx, pos);
                    self.arrow(cx, pos);
                    return pos + 2;
                }
                if top == Fk::Angle {
                    // Type-parameter default.
                    self.type_operator();
                    return pos + 1;
                }
                if matches!(top, Fk::TypeLit | Fk::TypeParen | Fk::TypeBracket) {
                    self.type_operator();
                    return pos + 1;
                }
                // Initializer / default value: the region ends.
                self.pop();
                self.assign(cx, pos);
                pos + 1
            }
            b'?' | b':' | b'|' | b'&' | b'.' | b'-' | b'+' | b'*' => {
                if len >= 2 && matches!(c, b'|' | b'&' | b'?') && *cx.src.add(pos + 1) == c {
                    // `||` / `&&` / `??`: expression operators.
                    self.end_region_for(cx, pos);
                    self.operand_done();
                    return pos + len;
                }
                if c == b'?' && *cx.src.add(pos + 1) == b'.' {
                    self.end_region_for(cx, pos);
                    self.operand_done();
                    self.after_dot = true;
                    return pos + 2;
                }
                if c == b'.' && len == 3 {
                    self.type_operator();
                    return pos + 3;
                }
                if c == b'?'
                    && top == Fk::TypeRegion
                    && self.top().atom
                    && self.top().s == R_EXPR
                    && self.top().qdebt == 0
                {
                    // `x as T ? a : b`: a conditional expression.
                    self.end_region_for(cx, pos);
                    self.question(cx, pos);
                    return pos + 1;
                }
                if c == b':' && top == Fk::TypeRegion && self.top().qdebt > 0 {
                    // The `:` of a conditional type pays its `?`.
                    self.top_mut().qdebt -= 1;
                }
                self.type_operator();
                pos + len
            }
            b'!' => {
                // `x as T!`: not a type token.
                self.end_region_for(cx, pos);
                self.set_value();
                self.clear_prev();
                pos + 1
            }
            _ => {
                // Any other operator ends an expression-embedded type; in a declaration type it is
                // an error and we treat it the same.
                self.end_region_for(cx, pos);
                self.operand_done();
                pos + len
            }
        }
    }

    unsafe fn close_angle(&mut self, cx: Cx, pos: usize) {
        let _ = (cx, pos);
        let a = self.pop();
        match a.s {
            4 => {
                // Type assertion `<T>`: an operand follows.
                if self.top_kind() == Fk::TypeRegion && self.top().s == R_ASSERT {
                    self.pop();
                }
                self.operand_done();
            }
            2 => {
                // Type arguments on an expression: the instantiation is a value, and no second list
                // may follow.
                self.set_value();
                self.clear_prev();
                self.no_type_args = true;
            }
            1 => {
                // Type parameters of a declaration head.
                self.set_value();
                self.clear_prev();
                match self.top_kind() {
                    Fk::FnHead => self.prev_kw = K_FUNCTION,
                    Fk::ClassHead => self.prev_kw = K_CLASS,
                    _ => {}
                }
            }
            _ => {
                if let Some(i) = self.region_index() {
                    let r = &mut self.frames[i];
                    r.atom = true;
                    r.inner = false;
                }
                self.set_value();
                self.clear_prev();
            }
        }
    }

    unsafe fn less_than(&mut self, cx: Cx, pos: usize) {
        // Type parameters of a declaration head or member.
        let head = match self.top_kind() {
            Fk::FnHead | Fk::ClassHead => true,
            Fk::Object | Fk::ClassBody => self.top().s == M_KEY_SEEN,
            _ => self.stmt_reg() == S_TYPE_NAME,
        };
        if cx.ts && head {
            let f = self.push(Fk::Angle);
            f.decl = true;
            f.s = 1;
            self.operand_done();
            return;
        }
        if cx.ts && self.expr_allowed {
            // `<T>x` assertion / `<T,>() =>` generic arrow: a type list.
            self.open_region(R_ASSERT, false);
            let f = self.push(Fk::Angle);
            f.decl = false;
            f.s = 4;
            self.operand_done();
            return;
        }
        if cx.ts && !self.expr_allowed && !self.no_type_args {
            // After a value: type arguments (`f<T>(x)`) or less-than.
            if self.expr_type_args(cx, pos) {
                let f = self.push(Fk::Angle);
                f.decl = false;
                f.s = 2;
                self.operand_done();
                return;
            }
        }
        self.operand_done();
        if self.top_kind() == Fk::Head && self.top().s != F_ITER {
            self.top_mut().s = F_EXPR;
        }
    }

    /// TypeScript's speculative parse of a type-argument list in expression position, on the
    /// forward scans `coalesce` already uses.
    unsafe fn expr_type_args(&mut self, cx: Cx, lt: usize) -> bool {
        let lim = (lt + 4096).min(cx.n);
        let (close, _capped) =
            angle_close_fwd_capped(cx.src, cx.st, cx.opch, cx.kind, lt + 1, lim, 1);
        let Some(gt) = close else { return false };
        if matches!(*cx.src.add(gt + 1), b'=' | b'>') {
            return false;
        }
        match gt_follower(cx.src, cx.n, gt + 1) {
            Follow::Split => type_list_legal(cx.t, cx.src, cx.st, cx.kind, lt + 1, gt),
            Follow::Fuse | Follow::Ctx => false,
        }
    }

    unsafe fn arrow(&mut self, cx: Cx, pos: usize) {
        let asyn = if self.closed_group { self.closed_group_async } else { self.arrow_async };
        self.prev_arrow = true;
        // Concise body unless `{` follows.
        let nx = next_sig(cx, pos + 2);
        let block = nx < cx.n && base_kind(cx, nx) >= OP_KIND_BASE && *cx.src.add(nx) == b'{';
        if !block {
            let f = self.push(Fk::Concise);
            f.is_gen = false;
            f.asyn = asyn;
            f.reserved = false;
        }
        self.operand_done();
        self.prev_arrow = true;
        self.arrow_async = asyn;
    }

    unsafe fn assign(&mut self, cx: Cx, pos: usize) {
        let _ = (cx, pos);
        let si = self.stmt_frame();
        let reg = self.stmt_reg();
        if reg == S_TYPE_NAME && si == self.frames.len() - 1 {
            // `type X =`: the alias type.
            self.set_stmt_reg(S_NONE);
            self.open_region(R_STMT, true);
            return;
        }
        if reg == S_IMPORT_NAME && si == self.frames.len() - 1 {
            // `import X = ...`: a module reference.
            self.set_stmt_reg(S_NONE);
            self.open_region(R_STMT, true);
            return;
        }
        if let Some(di) = self.decl_frame() {
            if self.frames[di].s == D_BOUND && di == self.frames.len() - 1 {
                self.frames[di].s = D_INIT;
            }
        }
        if self.top_kind() == Fk::ClassBody {
            self.top_mut().s = M_VALUE;
        }
        self.operand_done();
    }

    unsafe fn question(&mut self, cx: Cx, pos: usize) {
        // Optional marker (`a?: T`, `a?,`, `a?)`) vs conditional.
        let nx = next_sig(cx, pos + 1);
        let nc = if nx < cx.n { *cx.src.add(nx) } else { 0 };
        let member = self.top_kind() == Fk::ClassBody;
        let optional = nx < cx.n
            && base_kind(cx, nx) >= OP_KIND_BASE
            && (matches!(nc, b':' | b',' | b')' | b']' | b'>')
                || (member && matches!(nc, b'(' | b'<')));
        if optional && !self.expr_allowed {
            // Keep the member / parameter state.
            self.clear_prev();
            return;
        }
        self.top_mut().qdebt += 1;
        self.operand_done();
    }

    unsafe fn colon(&mut self, cx: Cx, pos: usize) {
        let _ = pos;
        // A concise arrow body without a pending `?` of its own ends at a `:` (the `?` belongs to
        // the frame below).
        while self.top_kind() == Fk::Concise && self.top().qdebt == 0 {
            self.pop();
        }
        let si = self.stmt_frame();
        let top = self.top_kind();
        // Ternary with a pending `?` on this frame.
        if self.top().qdebt > 0 {
            self.top_mut().qdebt -= 1;
            self.operand_done();
            return;
        }
        match top {
            Fk::Object => {
                self.top_mut().s = M_VALUE;
                self.operand_done();
                return;
            }
            Fk::ClassBody => {
                if cx.ts {
                    self.open_region(R_INLINE, true);
                } else {
                    self.operand_done();
                }
                return;
            }
            Fk::Params => {
                if cx.ts {
                    self.open_region(R_INLINE, true);
                } else {
                    self.operand_done();
                }
                return;
            }
            Fk::Group | Fk::Call => {
                if cx.ts && !self.expr_allowed {
                    // Arrow parameter annotation.
                    self.open_region(R_INLINE, true);
                } else {
                    self.operand_done();
                }
                return;
            }
            Fk::FnHead => {
                // Return type.
                self.open_region(R_INLINE, true);
                return;
            }
            Fk::ComputedKey => {
                // Index signature `[k: string]`.
                if cx.ts {
                    self.open_region(R_INLINE, true);
                } else {
                    self.operand_done();
                }
                return;
            }
            _ => {}
        }
        let _ = si;
        match self.stmt_reg() {
            S_CASE | S_LABEL => {
                self.set_stmt_reg(S_NONE);
                self.after_statement();
                self.clear_prev();
                return;
            }
            _ => {}
        }
        if let Some(di) = self.decl_frame() {
            if cx.ts && self.frames[di].s == D_BOUND && di == self.frames.len() - 1 {
                // Declarator type annotation.
                self.open_region(R_INLINE, true);
                return;
            }
        }
        if cx.ts && self.closed_group && top != Fk::Head {
            // `(a): T =>` arrow return type; remember the group's `async`.
            let asyn = self.closed_group_async;
            self.open_region(R_ARROW_RET, true);
            self.top_mut().asyn = asyn;
            return;
        }
        self.operand_done();
    }

    unsafe fn semicolon(&mut self, cx: Cx, pos: usize) {
        let _ = (cx, pos);
        self.pop_concise();
        match self.top_kind() {
            Fk::Head => {
                let f = self.top_mut();
                f.s = F_ITER;
                f.qdebt = 0;
                f.fdecl = false;
                let si = self.stmt_frame();
                self.frames[si].s = D_NONE;
                self.operand_done();
            }
            Fk::ClassBody => {
                let f = self.top_mut();
                f.s = M_KEY_POS;
                f.mods = 0;
                self.operand_done();
            }
            _ => {
                self.end_statement();
                self.clear_prev();
            }
        }
    }

    unsafe fn comma(&mut self, cx: Cx, pos: usize) {
        let _ = (cx, pos);
        self.pop_concise();
        while self.top_kind() == Fk::TypeRegion {
            self.pop();
        }
        let top = self.top_kind();
        match top {
            Fk::Object | Fk::ClassBody => {
                let f = self.top_mut();
                f.s = M_KEY_POS;
                f.mods = 0;
                f.qdebt = 0;
            }
            Fk::Head => {
                let f = self.top_mut();
                f.s = F_START;
                f.qdebt = 0;
            }
            _ => {
                if let Some(di) = self.decl_frame() {
                    if di == self.frames.len() - 1 && self.frames[di].s != D_NONE {
                        self.frames[di].s = D_BINDING;
                    }
                }
                self.top_mut().qdebt = 0;
            }
        }
        self.operand_done();
    }

    unsafe fn open_brace(&mut self, cx: Cx, pos: usize, newline: bool) {
        let _ = (pos, newline);
        let top = self.top_kind();
        let kind;
        let mut value = false;
        let mut genr = false;
        let mut asyn = false;
        let mut strict = self.top().strict;
        let mut reserved = false;
        let mut prologue = 0u8;
        if matches!(top, Fk::JsxTag | Fk::JsxElem) {
            kind = Fk::Container;
        } else if top == Fk::ClassHead {
            if self.top().m == 1 {
                // Interface body: a type literal that ends the statement.
                self.pop();
                let f = self.push(Fk::TypeLit);
                f.decl = true;
                f.value = false;
                f.s = 1;
                self.operand_done();
                self.deco = 0;
                return;
            }
            if self.top().s == 1 && self.prev_kw == K_EXTENDS {
                // Object literal as heritage.
                kind = Fk::Object;
                value = true;
            } else {
                let h = self.pop();
                kind = Fk::ClassBody;
                value = h.value;
                strict = true;
            }
        } else if top == Fk::FnHead {
            let h = self.pop();
            kind = Fk::FnBody;
            value = h.value;
            genr = h.is_gen;
            asyn = h.asyn;
            prologue = 1;
        } else if self.prev_arrow {
            kind = Fk::ArrowBody;
            asyn = self.arrow_async;
        } else if top == Fk::ClassBody
            && self.top().mods & MOD_STATIC != 0
            && self.top().s == M_KEY_POS
        {
            kind = Fk::StaticBlock;
            reserved = true;
            strict = true;
        } else {
            let reg = self.stmt_reg();
            let binding = self
                .decl_frame()
                .is_some_and(|di| self.frames[di].s == D_BINDING && di == self.frames.len() - 1);
            if binding {
                kind = Fk::Pattern;
            } else if matches!(reg, S_IMPORT | S_EXPORT | S_IMPORT_NAME) {
                kind = Fk::ModuleSpec;
            } else if reg == S_ENUM {
                kind = Fk::EnumBody;
            } else if reg == S_NAMESPACE || reg == S_DECLARE_MODULE || self.prev_kw == K_GLOBAL {
                kind = Fk::Block;
            } else if matches!(top, Fk::Object) && self.top().s != M_VALUE {
                // `{` at key position of an object literal: malformed; treat as a nested object.
                kind = Fk::Object;
                value = true;
            } else if self.stmt_start {
                kind = Fk::Block;
            } else if self.expr_allowed {
                kind = Fk::Object;
                value = true;
            } else {
                kind = Fk::Block;
            }
        }
        // Statement frames reset their registers when a block opens.
        if kind == Fk::EnumBody || kind == Fk::Block || kind == Fk::TypeLit {
            self.set_stmt_reg(S_NONE);
        }
        let f = self.push(kind);
        f.value = value;
        f.strict = strict;
        f.reserved = reserved || (f.reserved && kind != Fk::FnBody && kind != Fk::ArrowBody);
        f.prologue = prologue;
        if matches!(kind, Fk::FnBody | Fk::ArrowBody) {
            f.is_gen = genr;
            f.asyn = asyn;
            f.reserved = false;
            f.prologue = 1;
        }
        if kind == Fk::StaticBlock {
            f.is_gen = false;
            f.asyn = false;
        }
        if kind == Fk::TypeLit {
            f.decl = true;
            f.s = 1;
        }
        self.expr_allowed = true;
        self.stmt_start = matches!(kind, Fk::Block | Fk::FnBody | Fk::ArrowBody | Fk::StaticBlock);
        self.clear_prev();
        self.deco = 0;
        let _ = cx;
    }

    unsafe fn close_brace(&mut self, cx: Cx, pos: usize) {
        let _ = (cx, pos);
        // Virtual frames above the brace end with it.
        let Some(f) = self.pop_to(&[
            Fk::Block,
            Fk::FnBody,
            Fk::ArrowBody,
            Fk::ClassBody,
            Fk::StaticBlock,
            Fk::Object,
            Fk::Pattern,
            Fk::TypeLit,
            Fk::EnumBody,
            Fk::ModuleSpec,
            Fk::Container,
        ]) else {
            // Unbalanced: treat as a block end.
            self.unbalanced();
            self.after_statement();
            self.clear_prev();
            return;
        };
        match f.kind {
            Fk::Object | Fk::TypeLit if f.value => {
                self.value_done();
                self.member_done();
            }
            Fk::FnBody | Fk::ClassBody if f.value => {
                self.value_done();
                self.member_done();
            }
            Fk::Container => {
                self.clear_prev();
            }
            Fk::Pattern => {
                if let Some(di) = self.decl_frame() {
                    self.frames[di].s = D_BOUND;
                    if self.frames[di].kind == Fk::Head {
                        self.frames[di].fdecl = true;
                    }
                }
                self.value_done();
            }
            Fk::ModuleSpec => {
                self.operand_done();
            }
            Fk::TypeLit => {
                if f.s == 1 {
                    if self.top_kind() == Fk::TypeRegion {
                        self.pop();
                    }
                    self.end_statement();
                } else if let Some(i) = self.region_index() {
                    let r = &mut self.frames[i];
                    r.atom = true;
                    r.inner = false;
                    self.set_value();
                    self.clear_prev();
                } else {
                    self.after_statement();
                    self.clear_prev();
                }
            }
            Fk::FnBody | Fk::ClassBody | Fk::StaticBlock | Fk::Block | Fk::EnumBody => {
                // A statement-level body: a new statement may start; inside a class body a new
                // member may start.
                if matches!(self.top_kind(), Fk::ClassBody | Fk::Object) {
                    self.member_done();
                    self.operand_done();
                } else if f.kind == Fk::FnBody && matches!(self.top_kind(), Fk::TypeLit) {
                    self.operand_done();
                } else {
                    self.after_statement();
                    self.clear_prev();
                }
            }
            Fk::ArrowBody => {
                // The arrow function is complete: it cannot be continued.
                self.pop_concise();
                self.expr_allowed = true;
                self.stmt_start = true;
                self.clear_prev();
            }
            _ => {
                self.after_statement();
                self.clear_prev();
            }
        }
    }

    /// After a method / accessor body or nested object closed inside a member container, the next
    /// member key may follow.
    fn member_done(&mut self) {
        if matches!(self.top_kind(), Fk::ClassBody | Fk::Object) {
            let f = self.top_mut();
            if f.s == M_KEY_SEEN {
                f.s = M_KEY_POS;
                f.mods = 0;
            }
        }
    }

    unsafe fn open_paren(&mut self, cx: Cx, pos: usize) {
        let _ = (cx, pos);
        let top = self.top_kind();
        let si = self.stmt_frame();
        let head = if is_stmt_holder(self.frames[si].kind) { self.frames[si].head } else { 0 };
        let kind;
        let mut genr = false;
        let mut asyn = false;
        let mut hk = 0u8;
        if top == Fk::FnHead {
            kind = Fk::Params;
            genr = self.top().is_gen;
            asyn = self.top().asyn;
        } else if matches!(top, Fk::Object | Fk::ClassBody) && self.top().s == M_KEY_SEEN {
            // Method: give it a head so the body picks up its kind.
            let m = self.top().mods;
            let f = self.push(Fk::FnHead);
            f.value = false;
            f.is_gen = m & MOD_GEN != 0;
            f.asyn = m & MOD_ASYNC != 0;
            kind = Fk::Params;
            genr = m & MOD_GEN != 0;
            asyn = m & MOD_ASYNC != 0;
        } else if head != 0
            && matches!(self.prev_kw, K_IF | K_WHILE | K_FOR | K_WITH | K_SWITCH | K_CATCH)
        {
            kind = Fk::Head;
            hk = head;
            self.frames[si].head = 0;
        } else if self.expr_allowed || self.prev_kw == K_NEW {
            kind = Fk::Group;
            asyn = self.prev_async;
        } else {
            kind = Fk::Call;
        }
        let f = self.push(kind);
        f.head = hk;
        f.qdebt = 0;
        if kind == Fk::Params {
            f.is_gen = genr;
            f.asyn = asyn;
            f.reserved = false;
        }
        if kind == Fk::Group {
            f.asyn = asyn;
        }
        if kind == Fk::Head {
            f.s = F_START;
            f.fdecl = false;
        }
        self.operand_done();
    }

    unsafe fn close_paren(&mut self, cx: Cx, pos: usize) {
        let _ = (cx, pos);
        let Some(f) = self.pop_to(&[Fk::Head, Fk::Params, Fk::Call, Fk::Group, Fk::TypeParen])
        else {
            self.unbalanced();
            self.after_statement();
            self.clear_prev();
            return;
        };
        match f.kind {
            Fk::Head => {
                // A statement (or `{`) follows.
                self.expr_allowed = true;
                self.stmt_start = true;
                self.clear_prev();
                let si = self.stmt_frame();
                self.frames[si].head = 0;
            }
            Fk::Params => {
                self.set_value();
                self.clear_prev();
                self.closed_params = true;
            }
            Fk::Group => {
                self.value_done();
                self.closed_group = true;
                self.closed_group_async = f.asyn;
            }
            _ => {
                self.value_done();
            }
        }
    }

    unsafe fn open_bracket(&mut self, cx: Cx, pos: usize) {
        let _ = (cx, pos);
        let top = self.top_kind();
        let binding = self
            .decl_frame()
            .is_some_and(|di| self.frames[di].s == D_BINDING && di == self.frames.len() - 1);
        let kind = if matches!(top, Fk::Object | Fk::ClassBody) && self.top().s == M_KEY_POS {
            Fk::ComputedKey
        } else if binding {
            Fk::ArrayPattern
        } else if self.expr_allowed {
            Fk::Array
        } else {
            Fk::Index
        };
        let f = self.push(kind);
        f.qdebt = 0;
        self.operand_done();
    }

    unsafe fn close_bracket(&mut self, cx: Cx, pos: usize) {
        let _ = (cx, pos);
        let Some(f) = self.pop_to(&[
            Fk::Index,
            Fk::Array,
            Fk::ComputedKey,
            Fk::TypeBracket,
            Fk::ArrayPattern,
        ]) else {
            self.unbalanced();
            self.after_statement();
            self.clear_prev();
            return;
        };
        match f.kind {
            Fk::ComputedKey => {
                self.top_mut().s = M_KEY_SEEN;
                self.value_done();
            }
            Fk::ArrayPattern => {
                if let Some(di) = self.decl_frame() {
                    self.frames[di].s = D_BOUND;
                    if self.frames[di].kind == Fk::Head {
                        self.frames[di].fdecl = true;
                    }
                }
                self.value_done();
            }
            _ => {
                self.value_done();
            }
        }
    }
}
