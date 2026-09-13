#!/bin/bash
# usage: tasks/coverage/lexer_verify/sweep_driver.sh <corpus-dir> <log-file>
#
# Runs `lexer_sweep --files <corpus-dir>` and resumes past every file on which
# oxc_parser dies (stack overflow, exit 127 - not a panic, so the sweep cannot
# catch it). Mismatch lines and CRASH markers are appended to <log-file>; the
# last line is "DONE crashes=N" when the whole corpus has been walked.
set -u
DIR="$1"
LOG="$2"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
EXE="$ROOT/target/coverage/lexer_sweep.exe"
[ -x "$EXE" ] || EXE="$ROOT/target/coverage/lexer_sweep"
: > "$LOG"
skip=""
crashes=0
while :; do
  if [ -n "$skip" ]; then
    export SWEEP_SKIP_TO="$skip"
  else
    unset SWEEP_SKIP_TO
  fi
  SWEEP_TRACE=1 timeout 1800 "$EXE" --files "$DIR" > "$LOG.part" 2> "$LOG.trace"
  code=$?
  cat "$LOG.part" >> "$LOG"
  if grep -q "files checked" "$LOG.part"; then
    echo "DONE crashes=$crashes" >> "$LOG"
    break
  fi
  last=$(tail -1 "$LOG.trace")
  if [ -z "$last" ] || [ "$last" == "$skip" ]; then
    echo "ABORT code=$code last=$last" >> "$LOG"
    break
  fi
  crashes=$((crashes+1))
  echo "CRASH code=$code at $last" >> "$LOG"
  skip="$last"
done
rm -f "$LOG.part" "$LOG.trace"
grep -E "^(  |CRASH|DONE|ABORT|files checked)" "$LOG" | tail -n 40
