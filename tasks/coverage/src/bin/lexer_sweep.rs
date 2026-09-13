//! Corpus sweep for the incubating `oxc_lexer`: lex every JS/TS file under the given directories
//! and compare the token stream with `oxc_parser`'s. Files the parser rejects are skipped.
//!
//! ```sh
//! cargo build -p oxc_coverage --bin lexer_sweep --features lexer --profile coverage
//! target/coverage/lexer_sweep <dir>...
//! ```
//!
//! `SWEEP_SKIP_TO=<path fragment>` resumes after that file (for a driver that restarts after a
//! parser stack overflow); `SWEEP_TRACE=1` prints each path to stderr.
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::cast_possible_truncation,
    clippy::disallowed_methods
)]

#[cfg(not(feature = "lexer"))]
fn main() {
    eprintln!("lexer_sweep: build with `--features lexer`");
}

#[cfg(feature = "lexer")]
fn main() {
    sweep::main();
}

#[cfg(feature = "lexer")]
mod sweep {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use oxc::{
        allocator::Allocator,
        parser::{Parser, config::TokensParserConfig},
        span::SourceType,
    };
    use oxc_lexer::{LexOptions, PAD, TokenKind};

    static ORACLE_DROPS: AtomicUsize = AtomicUsize::new(0);

    /// One file: the first token-stream difference, or None when the streams agree or the parser
    /// rejects the file.
    fn stream_diff(code: &str, st: SourceType) -> Option<String> {
        let run = || {
            let alloc = Allocator::default();
            let ret = Parser::new(&alloc, code, st).with_config(TokensParserConfig).parse();
            if ret.fatal_error || !ret.diagnostics.is_empty() {
                return None;
            }
            let oracle: Vec<(u32, u32)> = ret.tokens.iter().map(|t| (t.start(), t.end())).collect();

            let n = code.len();
            let mut buf = Vec::with_capacity(n + PAD);
            buf.extend_from_slice(code.as_bytes());
            buf.resize(n + PAD, 0);
            let options = LexOptions {
                source_type_module: ret.program.source_type.is_module(),
                jsx: st.is_jsx(),
                ts: st.is_typescript(),
                ..Default::default()
            };
            let (result, arena) = oxc_lexer::lex_utf8(&buf, n as u32, options);
            let kinds = result.tok_kinds(&arena);
            let spans = result.tok_spans(&arena);
            let mine: Vec<(u32, u32)> = (0..kinds.len())
                .filter(|&i| kinds[i] != TokenKind::Eof && !kinds[i].is_trivia())
                .map(|i| (spans[i].start, spans[i].end))
                .collect();

            let bytes = code.as_bytes();
            let (mut mi, mut oi) = (0usize, 0usize);
            let mut drops = 0usize;
            while mi < mine.len() && oi < oracle.len() {
                let (ours, theirs) = (mine[mi], oracle[oi]);
                if ours == theirs {
                    mi += 1;
                    oi += 1;
                    continue;
                }
                // The parser leaves the first `<` of a split `<<` out of its own token list; the
                // lexer emits both.
                let split_lt = ours.1 == ours.0 + 1 && bytes.get(ours.0 as usize) == Some(&b'<');
                let next_lt = theirs.0 == ours.1
                    && theirs.1 == theirs.0 + 1
                    && bytes.get(theirs.0 as usize) == Some(&b'<');
                if split_lt && next_lt {
                    drops += 1;
                    mi += 1;
                    continue;
                }
                let at = code.get(theirs.0 as usize..theirs.1 as usize).unwrap_or("");
                let ctx_start = (theirs.0 as usize).saturating_sub(40);
                let ctx = code.get(ctx_start..(theirs.1 as usize + 20).min(n)).unwrap_or("");
                return Some(format!(
                    "token {mi}: lexer {}..{} / parser {}..{} {at:?} | {:?}",
                    ours.0,
                    ours.1,
                    theirs.0,
                    theirs.1,
                    ctx.replace(['\n', '\r'], " ")
                ));
            }
            if mi != mine.len() || oi != oracle.len() {
                return Some(format!("token count: lexer {}, parser {}", mine.len(), oracle.len()));
            }
            ORACLE_DROPS.fetch_add(drops, Ordering::Relaxed);
            None
        };
        catch_unwind(AssertUnwindSafe(run)).ok().flatten()
    }

    fn run_files(dir: &str) {
        let d = dir.to_string();
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || run_files_inner(&d))
            .unwrap()
            .join()
            .unwrap();
    }

    fn run_files_inner(dir: &str) {
        let mut checked = 0usize;
        let mut bad = 0usize;
        let mut skip_to = std::env::var("SWEEP_SKIP_TO").ok();
        let mut files: Vec<PathBuf> = Vec::new();
        let mut stack = vec![PathBuf::from(dir)];
        while let Some(d) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    if p.file_name().is_some_and(|f| f == ".git") {
                        continue;
                    }
                    stack.push(p);
                    continue;
                }
                let Some(ext) = p.extension().and_then(|s| s.to_str()) else { continue };
                if matches!(ext, "js" | "mjs" | "cjs" | "ts" | "mts" | "cts" | "jsx" | "tsx") {
                    files.push(p);
                }
            }
        }
        files.sort();
        for p in files {
            if let Some(marker) = &skip_to {
                if !p.to_string_lossy().contains(marker.as_str()) {
                    continue;
                }
                skip_to = None;
                continue; // resume after the offending file
            }
            let Ok(code) = std::fs::read_to_string(&p) else { continue };
            let st = SourceType::from_path(&p).unwrap_or_default();
            checked += 1;
            if std::env::var("SWEEP_TRACE").is_ok() {
                eprintln!("{}", p.display());
            }
            if let Some(why) = stream_diff(&code, st) {
                bad += 1;
                println!("  {}: {why}", p.display());
            }
        }
        println!(
            "files checked {checked}, mismatched {bad}, oracle drops {}",
            ORACLE_DROPS.load(Ordering::Relaxed)
        );
    }

    pub fn main() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let dirs: Vec<&String> = args.iter().filter(|a| *a != "--files").collect();
        if dirs.is_empty() {
            eprintln!("usage: lexer_sweep <dir>...");
            std::process::exit(2);
        }
        std::panic::set_hook(Box::new(|_| {}));
        for dir in dirs {
            println!("=== {dir} ===");
            run_files(dir);
        }
    }
}
