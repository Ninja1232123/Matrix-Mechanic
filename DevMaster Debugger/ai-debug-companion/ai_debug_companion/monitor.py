"""
Error monitoring system that watches various sources for errors.
"""

import asyncio
import subprocess
from pathlib import Path
from typing import Callable, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .models import ParsedError
from .parsers import UniversalErrorParser


class ErrorCallback:
    """Callback handler for when errors are detected."""

    def __init__(self, callback: Callable[[ParsedError], None]):
        self.callback = callback

    def __call__(self, error: ParsedError):
        self.callback(error)


class LogFileHandler(FileSystemEventHandler):
    """Watches log files for errors."""

    def __init__(self, error_parser: UniversalErrorParser, on_error: ErrorCallback):
        self.parser = error_parser
        self.on_error = on_error
        self.file_positions: dict[str, int] = {}

    def on_modified(self, event: FileSystemEvent):
        if event.is_directory:
            return

        file_path = event.src_path
        if not self._is_log_file(file_path):
            return

        self._check_file_for_errors(file_path)

    def _is_log_file(self, path: str) -> bool:
        """Check if file is a log file we should monitor."""
        log_extensions = {'.log', '.out', '.err'}
        return Path(path).suffix in log_extensions

    def _check_file_for_errors(self, file_path: str):
        """Read new content from file and check for errors."""
        try:
            # Get last read position
            last_pos = self.file_positions.get(file_path, 0)

            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(last_pos)
                new_content = f.read()
                self.file_positions[file_path] = f.tell()

            if new_content.strip():
                errors = self.parser.parse_multiple(new_content)
                for error in errors:
                    self.on_error(error)

        except Exception as e:
            # Silently ignore errors in error monitoring
            pass


class ProcessMonitor:
    """Monitors a subprocess for errors in its output."""

    def __init__(self, error_parser: UniversalErrorParser, on_error: ErrorCallback):
        self.parser = error_parser
        self.on_error = on_error
        self.process: Optional[subprocess.Popen] = None
        self._buffer = ""

    async def monitor_command(self, command: list[str], cwd: Optional[str] = None):
        """Run a command and monitor its output for errors."""
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
            bufsize=1
        )

        # Read output line by line
        if self.process.stdout:
            for line in self.process.stdout:
                await self._process_line(line)

        return_code = self.process.wait()
        return return_code

    async def _process_line(self, line: str):
        """Process a single line of output."""
        self._buffer += line

        # Try to parse errors from buffer
        error = self.parser.parse(self._buffer)
        if error:
            self.on_error(error)
            self._buffer = ""  # Reset buffer after finding error

        # Keep buffer from growing indefinitely
        if len(self._buffer) > 10000:
            self._buffer = self._buffer[-5000:]  # Keep last 5000 chars


class ErrorMonitor:
    """Main error monitoring coordinator."""

    def __init__(self, watch_path: str = "."):
        self.watch_path = Path(watch_path).resolve()
        self.parser = UniversalErrorParser()
        self.observer: Optional[Observer] = None
        self.process_monitor: Optional[ProcessMonitor] = None
        self.error_callbacks: list[ErrorCallback] = []

    def add_error_callback(self, callback: Callable[[ParsedError], None]):
        """Register a callback to be called when errors are detected."""
        self.error_callbacks.append(ErrorCallback(callback))

    def _on_error_detected(self, error: ParsedError):
        """Internal handler when an error is detected."""
        for callback in self.error_callbacks:
            callback(error)

    def start_watching_files(self):
        """Start watching files in the directory for errors."""
        event_handler = LogFileHandler(
            self.parser,
            ErrorCallback(self._on_error_detected)
        )

        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.watch_path), recursive=True)
        self.observer.start()

    def stop_watching_files(self):
        """Stop watching files."""
        if self.observer:
            self.observer.stop()
            self.observer.join()

    async def monitor_command(self, command: list[str]) -> int:
        """Monitor a specific command for errors."""
        self.process_monitor = ProcessMonitor(
            self.parser,
            ErrorCallback(self._on_error_detected)
        )
        return await self.process_monitor.monitor_command(
            command,
            cwd=str(self.watch_path)
        )

    def analyze_text(self, text: str) -> list[ParsedError]:
        """Analyze text for errors without monitoring."""
        return self.parser.parse_multiple(text)
