#!/usr/bin/env -S just --justfile

set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]
set shell := ["bash", "-cu"]

_default:
  @just --list -u

# ==================== ALIASES ====================
alias r := ready
alias c := conformance
alias f := fix

# ==================== SETUP & INITIALIZATION ====================

# Initialize the project by installing all necessary tools
init:
  # Rust related init
  cargo binstall watchexec-cli cargo-insta typos-cli cargo-shear@1.13.1 -y
  # Node.js related init
  pnpm install

# Clone or update submodules
submodules:
  node .github/scripts/clone-parallel.mjs
  just update-transformer-fixtures

# Install git pre-commit hook to format files
install-hook:
  echo -e "#!/bin/sh\njust fmt" > .git/hooks/pre-commit
  chmod +x .git/hooks/pre-commit

# ==================== CORE DEVELOPMENT ====================

# When ready, run the same CI commands
ready:
  git diff --exit-code --quiet
  pnpm install
  typos
  cargo lintgen
  just fmt
  just check
  just test
  just lint
  just doc
  just ast
  git status

# Run cargo check
check *args:
  cargo ck {{args}}

# Run all the tests
test *args:
  cargo test --all-features {{args}}

# Lint the whole project
[unix]
lint *args:
  CARGO_BUILD_WARNINGS=deny cargo lint {{args}}

[windows]
lint *args:
  $Env:CARGO_BUILD_WARNINGS='deny'; cargo lint {{args}}

# Format all files
fmt:
  -cargo shear --fix --check-test-targets # remove all unused dependencies
  cargo fmt
  node --run fmt

[unix]
doc *args:
  RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --document-private-items --all-features {{args}}

[windows]
doc *args:
  $Env:RUSTDOCFLAGS='-D warnings'; cargo doc --no-deps --document-private-items --all-features {{args}}

# Fix all auto-fixable format and lint issues
fix:
  cargo clippy --fix --allow-staged --no-deps
  just fmt
  typos -w
  git status

# ==================== DEVELOPMENT TOOLS ====================

watch *args='':
  watchexec --no-vcs-ignore {{args}}

watch-check:
  just watch "'cargo check; cargo clippy'"

watch-example *args='':
  just watch 'just example {{args}}'

# Run examples in parser, formatter, linter
example tool *args='':
  cargo run -p oxc_{{tool}} --example {{tool}} -- {{args}}

# Run the benchmarks
benchmark:
  cargo benchmark

# Run benchmarks for a single component
benchmark-one *args:
  cargo benchmark --bench {{args}} --no-default-features --features {{ if args == "linter" { "linter" } else { "compiler" } }}

# ==================== TESTING & CONFORMANCE ====================

# Run all conformance tests
coverage:
  cargo coverage
  cargo run -p oxc_transform_conformance -- --exec

# Run Test262, Babel and TypeScript conformance suite
conformance *args='':
  cargo coverage -- {{args}}

# Test ESTree
test-estree *args='':
  cargo run -p oxc_coverage --profile coverage -- estree {{args}}

test-estree-tokens *args='':
  cargo run -p oxc_coverage --profile coverage -- estree_tokens {{args}}

# Get code coverage
codecov:
  cargo codecov --html

# ==================== AST & CODEGEN ====================

# Generate AST related boilerplate code.
# If fails first time, run with JS generators disabled first, and then again with JS generators enabled.
# This is necessary because JS generators use `oxc_*` crates (e.g. `oxc_minifier`), and those crates may not compile
# unless Rust code is generated first.
# See: https://github.com/oxc-project/oxc/issues/15564
[unix]
ast:
  cargo run -p oxc_ast_tools || { cargo run -p oxc_ast_tools --no-default-features && cargo run -p oxc_ast_tools; }
[windows]
ast:
  try { cargo run -p oxc_ast_tools } catch { cargo run -p oxc_ast_tools --no-default-features; cargo run -p oxc_ast_tools }

# ==================== PARSER ====================

# Parser-specific commands will be added here as needed

# ==================== LEXER ====================

# `oxc_lexer` compiles to one of three backends, chosen at build time:
#
# * `avx2`, on x86_64 with `avx2` + `bmi2` + `popcnt` enabled
# * `neon`, on aarch64
# * `scalar` fallback, everywhere else
#
# All three need testing, and they should produce identical results. The recipes without a suffix build
# whichever one this host selects, `-simd` forces `avx2`, and `-generic` forces `scalar`.
# `oxc_lexer::BACKEND` names the one a build actually got.
#
# None of `avx2`, `bmi2` and `popcnt` is in the x86_64 baseline on any platform, so reaching the AVX2 core
# always means asking for them explicitly - even on an x86_64 host. The flags live in `.cargo/lexer-simd.toml`,
# passed with `cargo --config`, which avoids shell quoting entirely (this justfile runs PowerShell on Windows).
# NEON needs no flags, because it is in the aarch64 baseline: on an ARM host it is simply what `cargo test` builds.
#
# `--target` is always passed to the forcing recipes, so that the flags land on the triple the config file names,
# and so that the builds get separate target dirs instead of invalidating each other.
#
# On an ARM host the x86_64 builds are cross-compiled and run under emulation - macOS provides that via Rosetta 2.
# On Mac, ensure x86_64 target is installed: `rustup target add x86_64-apple-darwin`.
# If it's not, you'll get a "can't find crate for std" error.
#
# On ARM Linux or ARM Windows they will fail, loudly, when linking or running.
_lexer-simd-target := if os() == "macos" { "x86_64-apple-darwin" } else if os() == "windows" { "x86_64-pc-windows-msvc" } else { "x86_64-unknown-linux-gnu" }
_lexer-simd := "--target " + _lexer-simd-target + " --config .cargo/lexer-simd.toml"

# Reaching the scalar fallback means a target whose baseline has no SIMD core - which is the x86_64 triple
# with no flags added. On an x86_64 host that is the native build already, so no `--target` is passed and the
# `-generic` recipes reuse the plain ones' target dir. On an ARM host they cross-compile and run under emulation,
# which is the only way to reach the scalar fallback there at all, since `neon` is unconditional on aarch64.
_lexer-generic := if arch() == "x86_64" { "" } else { "--target " + _lexer-simd-target }

# Run `oxc_lexer`'s tests against this host's backend (`neon` on ARM, `scalar` on x86_64)
test-lexer *args='':
  cargo test -p oxc_lexer {{args}}

# Run `oxc_lexer`'s tests against the AVX2 core
test-lexer-simd *args='':
  cargo test -p oxc_lexer {{_lexer-simd}} {{args}}

# Run `oxc_lexer`'s tests against the scalar fallback
test-lexer-generic *args='':
  cargo test -p oxc_lexer {{_lexer-generic}} {{args}}

# Run lexer conformance against this host's backend
conformance-lexer *args='':
  cargo run -p oxc_coverage --profile coverage --features lexer -- lexer {{args}}

# Run lexer conformance against the AVX2 core
conformance-lexer-simd *args='':
  cargo run -p oxc_coverage --profile coverage {{_lexer-simd}} --features lexer -- lexer {{args}}

# Run lexer conformance against the scalar fallback
conformance-lexer-generic *args='':
  cargo run -p oxc_coverage --profile coverage {{_lexer-generic}} --features lexer -- lexer {{args}}

# Lint `oxc_lexer` and the conformance harness against this host's backend
[unix]
lint-lexer *args='':
  CARGO_BUILD_WARNINGS=deny cargo clippy -p oxc_lexer -p oxc_coverage --all-targets --all-features {{args}}

[windows]
lint-lexer *args='':
  $Env:CARGO_BUILD_WARNINGS='deny'; cargo clippy -p oxc_lexer -p oxc_coverage --all-targets --all-features {{args}}

# Lint `oxc_lexer` and the conformance harness against the AVX2 core
lint-lexer-simd *args='':
  just lint-lexer {{_lexer-simd}} {{args}}

# Lint `oxc_lexer` and the conformance harness against the scalar fallback
lint-lexer-generic *args='':
  just lint-lexer {{_lexer-generic}} {{args}}

# The lexer benchmarks measure wallclock time. Unlike the other benchmarks, they don't run on CodSpeed.
#
# `oxc_parser`'s lexer is always built without the SIMD flags, so it's benchmarked in a separate build
# from `oxc_lexer`. Its results are saved as criterion baseline `lexer_old`, and then `oxc_lexer`
# is compared against that baseline.
# The "change" which criterion reports for `oxc_lexer` is its time relative to `oxc_parser`'s.
# The "change" reported for `oxc_parser` (if any) is relative to its previous run.
#
# Both benchmarks are built before either is run, so the 2 runs happen back-to-back,
# without a build between them.
#
# `args` are passed to criterion, e.g. `just bench-lexer App.tsx` to run only that file.
# The recipes set the baseline options themselves, so don't pass those.
#
# On an ARM host, `bench-lexer-simd` compares `oxc_parser` running natively against `oxc_lexer`
# running under emulation, so the comparison is meaningless. Use plain `bench-lexer` there, which
# benchmarks the NEON core natively against `oxc_parser`, also native.
_bench-lexer := "cargo bench -p oxc_benchmark --no-default-features --features lexer_wallclock --bench"

# Benchmark `oxc_lexer`'s native backend (`neon` on ARM, `scalar` on x86_64) against `oxc_parser`'s lexer
bench-lexer *args='':
  {{_bench-lexer}} lexer_wallclock_old --bench lexer_wallclock_new --no-run
  {{_bench-lexer}} lexer_wallclock_old -- --save-baseline lexer_old {{args}}
  {{_bench-lexer}} lexer_wallclock_new -- --baseline lexer_old {{args}}

# Benchmark `oxc_lexer`'s AVX2 core against `oxc_parser`'s lexer
bench-lexer-simd *args='':
  {{_bench-lexer}} lexer_wallclock_old --no-run
  {{_bench-lexer}} lexer_wallclock_new {{_lexer-simd}} --no-run
  {{_bench-lexer}} lexer_wallclock_old -- --save-baseline lexer_old {{args}}
  {{_bench-lexer}} lexer_wallclock_new {{_lexer-simd}} -- --baseline lexer_old {{args}}

# `ready-lexer` fails if any step fails, or if conformance changes the lexer snapshots.
# Only the lexer snapshots are checked for changes, so it can be run with other uncommitted changes.
# Snapshots are checked after each conformance run, because each run overwrites the one before's snapshots.
_lexer-snapshots := "tasks/coverage/snapshots/lexer_*"

# Lint, test, and run conformance for `oxc_lexer` against all three backends, and check snapshots are unchanged.
# On an x86_64 host the plain and `-generic` legs are the same build, so the repeats are cache hits.
ready-lexer:
  just lint-lexer
  just lint-lexer-simd
  just lint-lexer-generic
  just test-lexer
  just test-lexer-simd
  just test-lexer-generic
  just conformance-lexer
  git diff --exit-code HEAD -- '{{_lexer-snapshots}}'
  just conformance-lexer-simd
  git diff --exit-code HEAD -- '{{_lexer-snapshots}}'
  just conformance-lexer-generic
  git diff --exit-code HEAD -- '{{_lexer-snapshots}}'
  just doc -p oxc_lexer

# ==================== LINTER ====================

# oxlint release build
oxlint:
  cargo build -p oxlint --release --features allocator

# watch oxlint, e.g. `just watch-oxlint test.js`
watch-oxlint *args='':
  just watch 'cargo run -p oxlint -- --disable-nested-config {{args}}'

# oxlint release build for node.js
# After building, you can run oxlint with `node <oxc-root>/apps/oxlint/dist/cli.js`
oxlint-node:
  pnpm -C apps/oxlint run build

# oxlint dev build, for testing with Node.js locally.
# This uses a non-release Rust build without the `allocator` feature (no mimalloc) and sets DEBUG options for the JS bundle,
# which mainly affects build time, performance, and debug assertions rather than available linting functionality.
# After building, you can run oxlint with `node <oxc-root>/apps/oxlint/dist/cli.js`
oxlint-node-dev:
  pnpm -C apps/oxlint run build-dev

watch-oxlint-node *args='':
  just watch 'pnpm run -C apps/oxlint build-dev && node apps/oxlint/dist/cli.js --disable-nested-config {{args}}'

# Create a new lint rule for any plugin
new-rule name plugin='eslint':
  cargo run -p rulegen {{name}} {{plugin}}
  just linter-schema-json
  just linter-config-ts
  just fmt

# Update test cases for an existing lint rule from upstream
update-rule-tests name plugin='eslint':
  cargo run -p rulegen {{name}} {{plugin}} --update-tests
  just fmt

# Legacy aliases for backward compatibility
new-eslint-rule name: (new-rule name "eslint")
new-jest-rule name: (new-rule name "jest")
new-ts-rule name: (new-rule name "typescript")
new-unicorn-rule name: (new-rule name "unicorn")
new-import-rule name: (new-rule name "import")
new-react-rule name: (new-rule name "react")
new-jsx-a11y-rule name: (new-rule name "jsx-a11y")
new-oxc-rule name: (new-rule name "oxc")
new-nextjs-rule name: (new-rule name "nextjs")
new-jsdoc-rule name: (new-rule name "jsdoc")
new-react-perf-rule name: (new-rule name "react-perf")
new-n-rule name: (new-rule name "n")
new-promise-rule name: (new-rule name "promise")
new-vitest-rule name: (new-rule name "vitest")
new-vue-rule name: (new-rule name "vue")

# Alias for backward compatibility
alias new-typescript-rule := new-ts-rule

# ==================== FORMATTER ====================

# oxfmt release build
oxfmt:
  cargo build -p oxfmt --release --features allocator

# watch oxfmt, e.g. `just watch-oxfmt test.js`
watch-oxfmt *args='':
  just watch 'cargo run -p oxfmt -- {{args}}'

# Build oxfmt in release build
# After building, you can run oxfmt with `node <oxc-root>/apps/oxfmt/dist/cli.js`
oxfmt-node:
  pnpm -C apps/oxfmt run build

# oxfmt dev build, for testing with Node.js locally.
# This builds faster than the release build and may differ in performance or behavior.
# After building, you can run oxfmt with `node <oxc-root>/apps/oxfmt/dist/cli.js`
oxfmt-node-dev:
  pnpm -C apps/oxfmt run build-dev

watch-oxfmt-node *args='':
  just watch 'pnpm run -C apps/oxfmt build-dev && node apps/oxfmt/dist/cli.js {{args}}'

# ==================== TRANSFORMER ====================

# Test Transform
test-transform *args='':
  cargo run -p oxc_transform_conformance -- --exec {{args}}

# Update transformer conformance test fixtures
update-transformer-fixtures:
  cd tasks/coverage/babel; git reset --hard HEAD; git clean -f -q
  node tasks/transform_conformance/update_fixtures.ts

# ==================== MINIFIER ====================

# Update minifier size snapshots
minsize:
  cargo minsize
  just allocs

# Update memory allocation snapshots
allocs:
  cargo allocs

# Update linter timing snapshots
lint-timings:
  cargo lint-timings

# Generate minifier size comparison
minifier-diff:
  #!/usr/bin/env bash
  cargo minsize --compress-only pr
  git checkout main
  cargo minsize --compress-only main
  for file in antd bundle.min d3 echarts jquery lodash moment react.development three typescript victory vue
  do
      echo $file.js >> diff
      diff target/minifier/main/$file.js target/minifier/pr/$file.js >> diff
  done
  git checkout -

# ==================== PLAYGROUND ====================

# Install wasm32-wasip1-threads for playground
install-wasm:
  rustup target add wasm32-wasip1-threads

build-playground:
  pnpm --filter oxc-playground build

watch-playground:
  just watch 'pnpm --filter oxc-playground build-dev'

# ==================== UTILITIES & ADVANCED ====================

# Generate website documentation, intended for updating the oxc.rs website.
# Path should be the path to your clone of https://github.com/oxc-project/website
# When testing changes to the website documentation, you may also want to run `pnpm run fmt`
# in the website directory.
website path:
  cargo run -p website_linter rules --rules-json {{path}}/.vitepress/data/rules.json --rule-docs {{path}}/src/docs/guide/usage/linter/rules --git-ref $(git rev-parse HEAD) --rule-count {{path}}/src/docs/guide/usage
  cargo run -p website_linter cli > {{path}}/src/docs/guide/usage/linter/generated-cli.md
  cargo run -p website_linter schema-markdown > {{path}}/src/docs/guide/usage/linter/generated-config.md
  cargo run -p website_linter schema-markdown-lsp > {{path}}/src/docs/guide/usage/linter/generated-lsp-config.md
  cargo run -p website_formatter cli > {{path}}/src/docs/guide/usage/formatter/generated-cli.md
  cargo run -p website_formatter schema-markdown > {{path}}/src/docs/guide/usage/formatter/generated-config.md

# Generate linter schema json for `npm/oxlint/configuration_schema.json`
linter-schema-json:
  cargo run -p website_linter schema-json > npm/oxlint/configuration_schema.json

# Generate linter config TypeScript types for `apps/oxlint/src-js/package/config.generated.ts`
linter-config-ts:
  pnpm --filter oxlint-app generate-config-types

# Generate formatter schema json for `npm/oxfmt/configuration_schema.json`
formatter-schema-json:
  cargo run -p website_formatter schema-json > npm/oxfmt/configuration_schema.json

# Generate formatter config TypeScript types for `apps/oxfmt/src-js/config.generated.ts`
formatter-config-ts:
  pnpm --filter oxfmt-app generate-config-types

# Automatically DRY up Cargo.toml manifests in a workspace
autoinherit:
  cargo binstall cargo-autoinherit
  cargo autoinherit

# ==================== PLATFORM HELPERS ====================

[unix]
clone-submodule dir url sha:
  cd {{dir}} || git init {{dir}}
  cd {{dir}} && git remote add origin {{url}} || true
  cd {{dir}} && git fetch --depth=1 origin {{sha}} && git reset --hard {{sha}} && git clean -f -q

[windows]
clone-submodule dir url sha:
  if (-not (Test-Path {{dir}}/.git)) { git init {{dir}} }
  cd {{dir}} ; if ((git remote) -notcontains 'origin') { git remote add origin {{url}} } else { git remote set-url origin {{url}} }
  cd {{dir}} ; git fetch --depth=1 origin {{sha}} ; git reset --hard {{sha}} ; git clean -f -q
