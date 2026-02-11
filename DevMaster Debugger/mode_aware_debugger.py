#!/usr/bin/env python3
"""
MODE-AWARE UNIVERSAL DEBUGGER

Inspired by Flask/Django development vs production modes.

INSIGHT: Auto-fixing everything is wrong for learning. Different contexts
need different error handling behaviors.

Modes:
  development - Show errors and solutions, don't fix (LEARNING MODE)
  review      - Show fixes, ask confirmation (SAFE MODE)
  production  - Auto-fix everything, log only (DEPLOYMENT MODE)

Usage:
  # Learning mode (see errors, understand fixes)
  DEBUG_MODE=development python mode_aware_debugger.py script.py

  # Safe mode (review each fix before applying)
  DEBUG_MODE=review python mode_aware_debugger.py script.py

  # Deployment mode (auto-fix everything)
  DEBUG_MODE=production python mode_aware_debugger.py script.py
"""

import sys
import os
import re
import subprocess
import shutil
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

# Import from existing universal_debugger
from universal_debugger import (
    ERROR_DATABASE,
    get_indent,
    wrap_in_try_except,
    get_indented_block,
    wrap_block_in_try_except,
    run_and_capture_error,
    parse_error
)

# Configuration
MODE = os.environ.get("DEBUG_MODE", "production")
LOG_FILE = "debugger_fixes.log"
UNKNOWN_ERRORS_FILE = "unknown_errors.json"

# Setup logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class FixProposal:
    """Represents a proposed fix for an error."""

    def __init__(self, error_type: str, file_path: str, line_number: int,
                 original_line: str, fixed_line: str, explanation: str,
                 pattern_name: str, confidence: float):
        self.error_type = error_type
        self.file_path = file_path
        self.line_number = line_number
        self.original_line = original_line.strip()
        self.fixed_line = fixed_line.strip()
        self.explanation = explanation
        self.pattern_name = pattern_name
        self.confidence = confidence


class ModeAwareDebugger:
    """
    Universal debugger with context-aware behavior.

    Modes:
    - development: Show errors, suggest fixes, don't apply (learning mode)
    - review: Show fixes, ask confirmation (safe mode)
    - production: Auto-fix everything, log only (deployment mode)
    """

    def __init__(self, mode: str = None):
        self.mode = mode or MODE
        self.fixes_applied: List[Dict] = []
        self.fixes_suggested: List[Dict] = []
        self.unknown_errors: List[Dict] = []

        if self.mode not in ['development', 'review', 'production']:
            raise ValueError(f"Unknown mode: {self.mode}. Use: development, review, or production")

    def handle_error(self, error_type: str, file_path: str, line_number: int,
                    error_message: str) -> bool:
        """
        Handle error based on current mode.

        Returns True if fix was applied, False otherwise.
        """
        # Try to generate fix proposal
        fix_proposal = self._generate_fix_proposal(
            error_type, file_path, line_number, error_message
        )

        if not fix_proposal:
            return self._handle_unknown_error(
                error_type, file_path, line_number, error_message
            )

        # Mode-specific behavior
        if self.mode == "development":
            return self._development_mode(fix_proposal)

        elif self.mode == "review":
            return self._review_mode(fix_proposal)

        elif self.mode == "production":
            return self._production_mode(fix_proposal)

        return False

    def _generate_fix_proposal(self, error_type: str, file_path: str,
                               line_number: int, error_message: str) -> Optional[FixProposal]:
        """Generate a fix proposal from ERROR_DATABASE."""

        # Handle module-qualified error names (e.g., json.decoder.JSONDecodeError)
        if '.' in error_type:
            error_type = error_type.split('.')[-1]

        if error_type not in ERROR_DATABASE:
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception:
            return None

        if line_number > len(lines) or line_number < 1:
            return None

        target_line = lines[line_number - 1]
        indent = get_indent(target_line)

        # Try each pattern for this error type
        error_info = ERROR_DATABASE[error_type]
        for pattern_idx, pattern in enumerate(error_info['patterns']):
            if re.search(pattern['detect'], target_line):
                try:
                    fixed = pattern['fix'](target_line, indent, error_message)

                    if fixed and fixed != target_line:
                        return FixProposal(
                            error_type=error_type,
                            file_path=file_path,
                            line_number=line_number,
                            original_line=target_line,
                            fixed_line=fixed,
                            explanation=error_info['description'],
                            pattern_name=f"pattern_{pattern_idx + 1}",
                            confidence=pattern.get('confidence', 0.8)
                        )
                except Exception:
                    continue

        return None

    def _development_mode(self, fix: FixProposal) -> bool:
        """
        Development mode: Show errors and solutions, but don't fix.

        Goal: Help developer learn what's wrong and how to fix it.
        """
        print(f"\n{'='*70}")
        print(f"🎓 DEVELOPMENT MODE - Learning")
        print(f"{'='*70}")
        print(f"📍 Error: {fix.error_type}")
        print(f"📂 Location: {fix.file_path}:{fix.line_number}")
        print(f"\n❓ What's wrong:")
        print(f"   {fix.explanation}")
        print(f"\n💡 How to fix:")
        print(f"   Confidence: {fix.confidence*100:.0f}%")
        print(f"\n   The fix would change:")
        print(f"   - Original: {fix.original_line}")
        print(f"   + Fixed:    {fix.fixed_line}")

        # Show what the fix does
        if "try:" in fix.fixed_line and "except" in fix.fixed_line:
            print(f"\n   → Wraps code in try/except for safety")
        elif ".get(" in fix.fixed_line:
            print(f"\n   → Uses .get() for safe dictionary access")
        elif "if" in fix.fixed_line and "!= 0" in fix.fixed_line:
            print(f"\n   → Adds zero-division check")
        elif "len(" in fix.fixed_line:
            print(f"\n   → Adds bounds checking")

        print(f"\n💻 To apply fixes:")
        print(f"   DEBUG_MODE=review python mode_aware_debugger.py {Path(fix.file_path).name}")
        print(f"   (or DEBUG_MODE=production to auto-apply all)")
        print(f"{'='*70}\n")

        self.fixes_suggested.append({
            'error_type': fix.error_type,
            'file': fix.file_path,
            'line': fix.line_number,
            'timestamp': datetime.now().isoformat()
        })

        return False  # Don't apply in development mode

    def _review_mode(self, fix: FixProposal) -> bool:
        """
        Review mode: Show fixes and ask for confirmation.

        Goal: Give developer control while making fixes easy.
        """
        print(f"\n{'='*70}")
        print(f"🔍 REVIEW MODE")
        print(f"{'='*70}")
        print(f"📍 {fix.error_type} at {fix.file_path}:{fix.line_number}")
        print(f"📝 {fix.explanation}")
        print(f"🎯 Confidence: {fix.confidence*100:.0f}%")
        print(f"\n   - {fix.original_line}")
        print(f"   + {fix.fixed_line}")
        print(f"{'='*70}")

        response = input("\n❓ Apply this fix? [y/n/a=all/s=skip all]: ").lower().strip()

        if response == 'a':
            print("✅ Switching to production mode (auto-apply all)")
            self.mode = "production"
            return self._apply_fix(fix)

        elif response == 's':
            print("⏭️  Switching to development mode (skip all)")
            self.mode = "development"
            return False

        elif response == 'y':
            return self._apply_fix(fix)

        else:
            print("⏭️  Skipped")
            return False

    def _production_mode(self, fix: FixProposal) -> bool:
        """
        Production mode: Auto-fix everything, log for review.

        Goal: Keep application running, never crash.
        """
        success = self._apply_fix(fix)

        if success:
            # Log fix for later review
            logging.info(
                f"Auto-fixed {fix.error_type} at {fix.file_path}:{fix.line_number}"
            )

            # Minimal output
            print(f"[FIX] {fix.error_type} at line {fix.line_number}")

        return success

    def _apply_fix(self, fix: FixProposal) -> bool:
        """Actually apply the fix to the file."""
        try:
            with open(fix.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            line_idx = fix.line_number - 1
            target_line = lines[line_idx]

            # Check if this needs multi-line block wrapping
            needs_block_wrap = (
                'try:' in fix.fixed_line and
                (re.search(r'\b(with|for|while)\b', target_line) or
                 target_line.strip().endswith(':'))
            )

            if needs_block_wrap:
                # Get the entire indented block
                block_lines, base_indent = get_indented_block(lines, line_idx)

                if block_lines and fix.error_type in ['FileNotFoundError', 'JSONDecodeError', 'PermissionError']:
                    fixed = wrap_block_in_try_except(block_lines, base_indent, fix.error_type)
                    lines_to_replace = len(block_lines)
                    fixed_lines = fixed.split('\n')
                    new_lines = [line + '\n' for line in fixed_lines if line]
                    lines[line_idx:line_idx + lines_to_replace] = new_lines
                else:
                    lines[line_idx] = fix.fixed_line
                    if not lines[line_idx].endswith('\n'):
                        lines[line_idx] += '\n'
            else:
                lines[line_idx] = fix.fixed_line
                if not lines[line_idx].endswith('\n'):
                    lines[line_idx] += '\n'

            # Write back
            with open(fix.file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            self.fixes_applied.append({
                'error_type': fix.error_type,
                'file': fix.file_path,
                'line': fix.line_number,
                'timestamp': datetime.now().isoformat(),
                'original': fix.original_line,
                'fixed': fix.fixed_line
            })

            return True

        except Exception as e:
            logging.error(f"Failed to apply fix: {e}")
            return False

    def _handle_unknown_error(self, error_type: str, file_path: str,
                              line_number: int, error_msg: str) -> bool:
        """
        Handle errors not in ERROR_DATABASE.

        This is CRITICAL for empirical database growth!
        """
        # Read the problematic line
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            problem_line = lines[line_number - 1].strip()
        except:
            problem_line = "<could not read line>"

        if self.mode == "production":
            # In production, log and continue
            logging.warning(
                f"Unknown error type: {error_type} at {file_path}:{line_number}"
            )
            print(f"[SKIP] No fix available for {error_type}")
            return False

        else:
            # In development: CAPTURE FOR DATABASE EXPANSION
            print(f"\n{'='*70}")
            print(f"❌ UNKNOWN ERROR - Not in database yet")
            print(f"{'='*70}")
            print(f"Error Type: {error_type}")
            print(f"Location: {file_path}:{line_number}")
            print(f"Message: {error_msg[:200]}")
            print(f"Problematic line: {problem_line}")

            # LOG FOR EMPIRICAL DISCOVERY
            unknown_error = {
                'error_type': error_type,
                'message': error_msg,
                'line': problem_line,
                'file': file_path,
                'line_number': line_number,
                'timestamp': datetime.now().isoformat()
            }

            self._log_unknown_error(unknown_error)

            print(f"\n✅ Pattern logged to {UNKNOWN_ERRORS_FILE}")
            print(f"   This helps improve the database!")

            # SHOW TEMPLATE FOR CONTRIBUTION
            print(f"\n📝 To add this fix to ERROR_DATABASE:")
            print(f"   '{error_type}': {{")
            print(f"       'description': '<what this error means>',")
            print(f"       'patterns': [{{")
            print(f"           'detect': r'<regex to match>',")
            print(f"           'fix': lambda line, indent, error_msg: '<your fix>',")
            print(f"           'multiline': False,")
            print(f"           'confidence': 0.85")
            print(f"       }}]")
            print(f"   }}")
            print(f"{'='*70}\n")

            self.unknown_errors.append(unknown_error)
            return False

    def _log_unknown_error(self, error_data: Dict):
        """Log unknown error for database expansion."""
        try:
            # Read existing logs
            if os.path.exists(UNKNOWN_ERRORS_FILE):
                with open(UNKNOWN_ERRORS_FILE, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []

            # Append new error
            logs.append(error_data)

            # Write back
            with open(UNKNOWN_ERRORS_FILE, 'w') as f:
                json.dump(logs, f, indent=2)

        except Exception as e:
            logging.error(f"Failed to log unknown error: {e}")

    def generate_report(self):
        """Generate summary of debugging session."""
        print(f"\n{'='*70}")

        if self.mode == "development":
            print(f"📚 DEVELOPMENT SESSION SUMMARY")
            print(f"{'='*70}")
            print(f"Errors encountered: {len(self.fixes_suggested) + len(self.unknown_errors)}")
            print(f"Known errors: {len(self.fixes_suggested)}")
            print(f"Unknown errors: {len(self.unknown_errors)}")
            print(f"Fixes applied: 0 (development mode)")

            if self.fixes_suggested:
                print(f"\n💡 To apply suggested fixes:")
                print(f"   DEBUG_MODE=review python mode_aware_debugger.py <script>")

            if self.unknown_errors:
                print(f"\n❌ Unknown errors found:")
                for err in self.unknown_errors:
                    print(f"   - {err['error_type']} at line {err['line_number']}")
                print(f"\n   Check {UNKNOWN_ERRORS_FILE} for details")

        elif self.mode == "production":
            print(f"🚀 PRODUCTION SESSION SUMMARY")
            print(f"{'='*70}")
            print(f"Fixes applied: {len(self.fixes_applied)}")

            if self.fixes_applied:
                print(f"\n✅ Auto-fixed errors:")
                error_counts = {}
                for fix in self.fixes_applied:
                    error_type = fix['error_type']
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1

                for error_type, count in error_counts.items():
                    print(f"   {error_type}: {count}x")

            print(f"\n📋 See {LOG_FILE} for details")

            # Save JSON report
            with open('debugger_report.json', 'w') as f:
                json.dump({
                    'mode': 'production',
                    'fixes_applied': self.fixes_applied,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)

        elif self.mode == "review":
            print(f"🔍 REVIEW SESSION SUMMARY")
            print(f"{'='*70}")
            print(f"Fixes applied: {len(self.fixes_applied)}")
            print(f"Fixes suggested: {len(self.fixes_suggested)}")

        print(f"{'='*70}\n")


def main():
    if len(sys.argv) < 2:
        print("MODE-AWARE UNIVERSAL DEBUGGER")
        print("=" * 70)
        print("\nUsage: python mode_aware_debugger.py <script.py>")
        print("\nModes (set with DEBUG_MODE environment variable):")
        print("  development - Show errors and learn (don't fix)")
        print("  review      - Review each fix before applying")
        print("  production  - Auto-fix everything (default)")
        print("\nExamples:")
        print("  DEBUG_MODE=development python mode_aware_debugger.py script.py")
        print("  DEBUG_MODE=review python mode_aware_debugger.py script.py")
        print("  DEBUG_MODE=production python mode_aware_debugger.py script.py")
        sys.exit(1)

    script_path = sys.argv[1]

    if not os.path.exists(script_path):
        print(f"[ERROR] File not found: {script_path}")
        sys.exit(1)

    # Create backup
    backup_path = script_path + '.backup'
    shutil.copy2(script_path, backup_path)

    mode = os.environ.get("DEBUG_MODE", "production")
    debugger = ModeAwareDebugger(mode=mode)

    print(f"{'='*70}")
    print(f"MODE-AWARE UNIVERSAL DEBUGGER")
    print(f"{'='*70}")
    print(f"Mode: {mode.upper()}")
    print(f"Target: {os.path.abspath(script_path)}")
    print(f"Database: {len(ERROR_DATABASE)} error types")
    print(f"Backup: {backup_path}")
    print(f"{'='*70}\n")

    max_iterations = 50
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"[ITERATION {iteration}] Running script...")

        stderr = run_and_capture_error(script_path)

        if not stderr:
            print(f"[SUCCESS] No errors detected!")
            break

        error_type, error_file, error_line, full_error = parse_error(stderr, script_path)

        if not error_type or not error_file or not error_line:
            print(f"[ERROR] Could not parse error:")
            print(stderr[:500])
            break

        # Only fix errors in the target script
        if os.path.abspath(error_file) != os.path.abspath(script_path):
            print(f"[SKIP] Error is in external file: {error_file}")
            break

        print(f"[DETECTED] {error_type} at line {error_line}")

        # Handle error based on mode
        was_fixed = debugger.handle_error(error_type, error_file, error_line, full_error)

        # In development mode, we never fix, so break after first error
        if mode == "development" and not was_fixed:
            print("\n💡 Development mode: Stopping after first error for learning")
            break

        # In review mode, if user skipped, we might have switched modes
        if not was_fixed and debugger.mode == "development":
            print("\n⏭️  Skipping remaining errors (switched to development mode)")
            break

        # If no fix was applied and we're not in dev mode, something went wrong
        if not was_fixed and mode != "development":
            break

    if iteration >= max_iterations:
        print(f"[TIMEOUT] Max iterations reached")

    # Generate report
    debugger.generate_report()

    print(f"\n💾 Backup saved at: {backup_path}")
    if len(debugger.fixes_applied) == 0:
        print(f"   No changes were made to the original file")


if __name__ == "__main__":
    main()
