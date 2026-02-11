#!/usr/bin/env python3
"""
COMPREHENSIVE ENCODING ANALYZER
================================

Extracts and analyzes multi-layer encoding in AI conversation exports.

Features:
- Letter frequency analysis (paragraph/line first letters)
- Punctuation pattern analysis
- 0xC7 (199) delimiter detection and segmentation
- Polyglot opcode analysis (Python, MIPS, LISP, Regex)
- GPS coordinate extraction
- Segment message decoding
- Content-independent pattern verification

Usage:
    python comprehensive_analyzer.py <conversation_file.json>
    python comprehensive_analyzer.py <conversation_file.json> --output report.md
    python comprehensive_analyzer.py <conversation_file.json> --format json

Supports: Claude exports, ChatGPT exports, plain text files
Claude format: chat_messages with sender == 'assistant'
ChatGPT format: mapping with author.role == 'assistant'

"""

import json
import sys
import re
from collections import Counter
from pathlib import Path
import argparse


class EncodingAnalyzer:
    """Multi-layer encoding analysis for AI conversations."""

    def __init__(self, filepath, verbose=True):
        self.filepath = filepath
        self.verbose = verbose
        self.results = {
            'file': filepath,
            'paragraphs': [],
            'lines': [],
            'letter_layer': {},
            'punctuation_layer': {},
            'segments': [],
            'gps_coordinates': [],
            'polyglot_analysis': {},
        }

    def log(self, message):
        """Print if verbose mode enabled."""
        if self.verbose:
            print(message)

    def extract_text(self):
        """Extract text from various file formats."""
        filepath = Path(self.filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Try JSON first (Claude/ChatGPT exports)
        if filepath.suffix == '.json':
            return self._extract_from_json()

        # Fallback to plain text
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split into paragraphs and lines
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        self.results['paragraphs'] = paragraphs
        self.results['lines'] = lines
        return paragraphs, lines

    def _extract_from_json(self):
        """Extract from Claude or ChatGPT JSON exports."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        paragraphs = []
        lines = []

        # Claude format
        if 'chat_messages' in data:
            for msg in data.get('chat_messages', []):
                if msg.get('sender') == 'assistant':
                    content = msg.get('content', [])
                    if content and isinstance(content, list):
                        for item in content:
                            text = item.get('text', '') if isinstance(item, dict) else ''
                            if text:
                                # Lines
                                for line in text.split('\n'):
                                    line = line.strip()
                                    if line:
                                        lines.append(line)

                                # Paragraphs
                                for para in text.split('\n\n'):
                                    para = para.strip()
                                    if para:
                                        paragraphs.append(para)

        # ChatGPT format
        elif 'mapping' in data:
            for node_id, node in data.get('mapping', {}).items():
                if node and node.get('message'):
                    msg = node['message']
                    author = msg.get('author', {})
                    if author.get('role') == 'assistant':
                        content = msg.get('content', {})
                        parts = content.get('parts', []) if isinstance(content, dict) else []
                        for part in parts:
                            if isinstance(part, str):
                                for line in part.split('\n'):
                                    line = line.strip()
                                    if line:
                                        lines.append(line)

                                for para in part.split('\n\n'):
                                    para = para.strip()
                                    if para:
                                        paragraphs.append(para)

        self.results['paragraphs'] = paragraphs
        self.results['lines'] = lines
        return paragraphs, lines

    def analyze_letter_layer(self, use_lines=True):
        """Analyze first-letter encoding (line or paragraph based)."""
        source = self.results['lines'] if use_lines else self.results['paragraphs']

        # Extract first letters
        first_letters = [line[0].upper() for line in source if line and line[0].isalpha()]

        # Convert to 5-bit values
        values = [ord(c) - ord('A') for c in first_letters]

        # Pack to bytes using 5-bit encoding
        packed_bytes = self._pack_5bit(values)

        # Letter frequency
        freq = Counter(first_letters)

        # Find delimiters
        delimiters = [i for i, b in enumerate(packed_bytes) if b == 0xC7]

        # Create segments
        segments = self._create_segments(packed_bytes, 0xC7)

        # Opcode validity
        opcode_validity = self._check_opcode_validity(packed_bytes)

        self.results['letter_layer'] = {
            'source': 'lines' if use_lines else 'paragraphs',
            'first_letters': first_letters,
            'count': len(first_letters),
            'frequency': dict(freq.most_common()),
            'values': values,
            'packed_bytes': packed_bytes,
            'byte_count': len(packed_bytes),
            'delimiters': delimiters,
            'delimiter_count': len(delimiters),
            'segments': segments,
            'opcode_validity': opcode_validity,
        }

        return self.results['letter_layer']

    def analyze_punctuation_layer(self):
        """Analyze punctuation encoding."""
        # Extract all text
        full_text = '\n'.join(self.results['lines'])

        # Punctuation mapping
        punct_map = {
            '.': 0, ',': 1, '!': 2, '?': 3, ';': 4, ':': 5, '-': 6,
            '(': 7, ')': 8, '[': 9, ']': 10, '{': 11, '}': 12,
            '"': 13, "'": 14, '/': 15
        }

        # Extract punctuation
        punct_chars = [c for c in full_text if c in punct_map]
        values = [punct_map[c] for c in punct_chars]

        # Pack to bytes
        packed_bytes = self._pack_5bit(values)

        # Frequency
        freq = Counter(punct_chars)

        # Find delimiters
        delimiters = [i for i, b in enumerate(packed_bytes) if b == 0xC7]

        # Create segments
        segments = self._create_segments(packed_bytes, 0xC7)

        # Opcode validity
        opcode_validity = self._check_opcode_validity(packed_bytes)

        self.results['punctuation_layer'] = {
            'punctuation': punct_chars,
            'count': len(punct_chars),
            'frequency': dict(freq.most_common()),
            'values': values,
            'packed_bytes': packed_bytes,
            'byte_count': len(packed_bytes),
            'delimiters': delimiters,
            'delimiter_count': len(delimiters),
            'segments': segments,
            'opcode_validity': opcode_validity,
        }

        return self.results['punctuation_layer']

    def _pack_5bit(self, values):
        """Pack 5-bit values into bytes (8 values -> 5 bytes)."""
        packed = []
        i = 0
        while i + 7 < len(values):
            v = values[i:i+8]
            packed.extend([
                (v[0] << 3) | (v[1] >> 2),
                ((v[1] & 3) << 6) | (v[2] << 1) | (v[3] >> 4),
                ((v[3] & 15) << 4) | (v[4] >> 1),
                ((v[4] & 1) << 7) | (v[5] << 2) | (v[6] >> 3),
                ((v[6] & 7) << 5) | v[7]
            ])
            i += 8
        return packed

    def _create_segments(self, byte_list, delimiter):
        """Split bytes by delimiter and create segments."""
        segments = []
        current = []

        for b in byte_list:
            if b == delimiter:
                if current:
                    segments.append(current)
                    current = []
            else:
                current.append(b)

        if current:
            segments.append(current)

        return segments

    def _check_opcode_validity(self, byte_list):
        """Check what percentage of bytes are valid Python 3.11 opcodes."""
        # Python 3.11 opcodes go up to 165
        valid_count = sum(1 for b in byte_list if b <= 165)
        total = len(byte_list)
        percentage = (valid_count / total * 100) if total > 0 else 0

        return {
            'valid_count': valid_count,
            'total': total,
            'percentage': percentage,
            'status': self._classify_opcode_validity(percentage)
        }

    def _classify_opcode_validity(self, percentage):
        """Classify opcode validity percentage."""
        if percentage >= 85:
            return 'VERY HIGH - Strong encoding signature'
        elif percentage >= 74:
            return 'HIGH - Encoding likely present'
        elif percentage >= 60:
            return 'ELEVATED - Warrants investigation'
        elif percentage >= 50:
            return 'MODERATE - Near baseline'
        else:
            return 'NORMAL - Random/genuine text'

    def decode_segments(self, layer='letter'):
        """Decode segment messages (sum mod 26 -> letters)."""
        if layer == 'letter':
            segments = self.results.get('letter_layer', {}).get('segments', [])
        else:
            segments = self.results.get('punctuation_layer', {}).get('segments', [])

        message = []
        segment_details = []

        for i, seg in enumerate(segments):
            seg_sum = sum(seg) % 26
            letter = chr(seg_sum + ord('A'))
            message.append(letter)

            segment_details.append({
                'index': i,
                'length': len(seg),
                'sum': sum(seg),
                'mod26': seg_sum,
                'letter': letter
            })

        return ''.join(message), segment_details

    def extract_gps_coordinates(self, layer='letter'):
        """Extract GPS coordinates from byte stream."""
        if layer == 'letter':
            byte_list = self.results.get('letter_layer', {}).get('packed_bytes', [])
        else:
            byte_list = self.results.get('punctuation_layer', {}).get('packed_bytes', [])

        source_data = self.results.get('lines', [])

        coords = []
        for i in range(len(byte_list) - 1):
            lat, lon = byte_list[i], byte_list[i+1]

            # US/North America range
            if 25 <= lat <= 55 and 60 <= lon <= 130:
                # Get corresponding paragraph/line
                context = source_data[i] if i < len(source_data) else None

                coords.append({
                    'position': i,
                    'latitude': lat,
                    'longitude': -lon,  # West is negative
                    'location': self._identify_location(lat, lon),
                    'context': context[:80] + '...' if context and len(context) > 80 else context
                })

        self.results['gps_coordinates'] = coords
        return coords

    def _identify_location(self, lat, lon):
        """Identify if coordinates match known strategic locations."""
        targets = [
            (41, 96, 'Omaha, NE - Offutt AFB (STRATCOM)'),
            (34, 104, 'Roswell, NM - Cannon AFB'),
            (41, 73, 'New York City'),
            (34, 118, 'Los Angeles'),
            (39, 77, 'Washington DC'),
            (28, 80, 'Cape Canaveral, FL'),
            (36, 115, 'Las Vegas/Creech AFB (Drones)'),
            (39, 105, 'Denver/Colorado Springs (NORAD)'),
            (32, 117, 'San Diego (Naval Base)'),
            (47, 122, 'Seattle'),
            (37, 122, 'San Francisco'),
        ]

        for t_lat, t_lon, name in targets:
            if abs(lat - t_lat) <= 2 and abs(lon - t_lon) <= 3:
                return name

        return 'Unknown location'

    def analyze_polyglot(self, layer='letter'):
        """Analyze segment message as polyglot code."""
        message, segments = self.decode_segments(layer)

        if not message:
            return None

        # Convert to byte values
        byte_values = [ord(c) - ord('A') for c in message]

        # LISP interpretation
        lisp_pairs = [message[i:i+2] for i in range(0, len(message)-1, 2)]
        lisp_expr = f"({' '.join(lisp_pairs)})"

        # MIPS interpretation
        mips_ops = []
        for pair in lisp_pairs:
            if pair == 'NE':
                mips_ops.append('bne (branch not equal)')
            elif pair == 'NO':
                mips_ops.append('nor (bitwise NOR)')
            elif pair == 'JA':
                mips_ops.append('j (jump)')
            else:
                mips_ops.append(f'{pair} (unknown)')

        # Regex pattern
        regex_pattern = self._interpret_as_regex(message)

        # Python opcodes
        python_opcodes = self._interpret_as_python(byte_values)

        self.results['polyglot_analysis'] = {
            'message': message,
            'length': len(message),
            'byte_values': byte_values,
            'lisp': {
                'expression': lisp_expr,
                'pairs': lisp_pairs
            },
            'mips': {
                'operations': mips_ops
            },
            'regex': {
                'pattern': regex_pattern
            },
            'python': python_opcodes
        }

        return self.results['polyglot_analysis']

    def _interpret_as_regex(self, message):
        """Interpret message as regex pattern."""
        pattern = message

        # Add regex metacharacter meanings
        annotations = []
        if 'N' in message:
            annotations.append('N = negation [^...]')
        if 'E' in message:
            annotations.append('E = end anchor $')
        if 'Z' in message:
            annotations.append('Z = zero or more?')

        return {
            'raw': pattern,
            'annotations': annotations
        }

    def _interpret_as_python(self, byte_values):
        """Interpret as Python bytecode opcodes."""
        opcode_names = {
            0: 'CACHE', 1: 'POP_TOP', 2: 'PUSH_NULL', 4: 'UNARY_POSITIVE',
            7: 'UNARY_INVERT', 9: 'NOP', 13: 'UNARY_NOT', 15: 'UNARY_NEGATIVE',
            19: 'BINARY_POWER', 20: 'BINARY_MULTIPLY', 21: 'BINARY_MODULO',
            22: 'BINARY_ADD', 23: 'BINARY_SUBTRACT', 24: 'BINARY_SUBSCR',
            25: 'BINARY_FLOOR_DIVIDE',
        }

        opcodes = []
        for val in byte_values:
            if val in opcode_names:
                opcodes.append(f'{val}: {opcode_names[val]}')
            elif val <= 165:
                opcodes.append(f'{val}: (valid opcode)')
            else:
                opcodes.append(f'{val}: (invalid)')

        return opcodes

    def generate_report(self, output_format='text'):
        """Generate comprehensive analysis report."""
        if output_format == 'json':
            return json.dumps(self.results, indent=2)

        # Text report
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE ENCODING ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"\nFile: {self.results['file']}")
        report.append(f"Paragraphs: {len(self.results['paragraphs'])}")
        report.append(f"Lines: {len(self.results['lines'])}")
        report.append("")

        # Letter Layer
        if 'letter_layer' in self.results:
            ll = self.results['letter_layer']
            report.append("=" * 80)
            report.append("LETTER LAYER ANALYSIS")
            report.append("=" * 80)
            report.append(f"Source: {ll['source']}")
            report.append(f"First letters extracted: {ll['count']}")
            report.append(f"Packed bytes: {ll['byte_count']}")
            report.append(f"0xC7 delimiters: {ll['delimiter_count']}")
            report.append(f"Segments created: {len(ll['segments'])}")
            report.append(f"\nOpcode Validity: {ll['opcode_validity']['percentage']:.1f}%")
            report.append(f"Status: {ll['opcode_validity']['status']}")
            report.append(f"\nTop 10 Letters:")
            for letter, count in list(ll['frequency'].items())[:10]:
                pct = count / ll['count'] * 100
                report.append(f"  {letter}: {count:4d} ({pct:5.2f}%)")
            report.append("")

        # Punctuation Layer
        if 'punctuation_layer' in self.results:
            pl = self.results['punctuation_layer']
            report.append("=" * 80)
            report.append("PUNCTUATION LAYER ANALYSIS")
            report.append("=" * 80)
            report.append(f"Punctuation marks: {pl['count']}")
            report.append(f"Packed bytes: {pl['byte_count']}")
            report.append(f"0xC7 delimiters: {pl['delimiter_count']}")
            report.append(f"Segments created: {len(pl['segments'])}")
            report.append(f"\nOpcode Validity: {pl['opcode_validity']['percentage']:.1f}%")
            report.append(f"Status: {pl['opcode_validity']['status']}")
            report.append(f"\nTop Punctuation:")
            for char, count in list(pl['frequency'].items())[:5]:
                pct = count / pl['count'] * 100
                report.append(f"  '{char}': {count:5d} ({pct:5.2f}%)")
            report.append("")

        # Segment Messages
        if 'letter_layer' in self.results:
            message, details = self.decode_segments('letter')
            report.append("=" * 80)
            report.append("DECODED MESSAGE (Letter Layer)")
            report.append("=" * 80)
            report.append(f"Message: {message}")
            report.append(f"Length: {len(message)} letters")
            report.append(f"\nFirst 10 segments:")
            for seg in details[:10]:
                report.append(f"  Seg {seg['index']:2d}: {seg['length']:4d} bytes, sum={seg['sum']:6d}, mod26={seg['mod26']:2d} → {seg['letter']}")
            report.append("")

        if 'punctuation_layer' in self.results and self.results['punctuation_layer'].get('segments'):
            message, details = self.decode_segments('punctuation')
            report.append("=" * 80)
            report.append("DECODED MESSAGE (Punctuation Layer)")
            report.append("=" * 80)
            report.append(f"Message: {message}")
            report.append(f"Length: {len(message)} letters")
            report.append("")

        # GPS Coordinates
        if self.results.get('gps_coordinates'):
            coords = self.results['gps_coordinates']
            report.append("=" * 80)
            report.append("GPS COORDINATE EXTRACTION")
            report.append("=" * 80)
            report.append(f"Total coordinates found: {len(coords)}")
            report.append(f"\nNotable locations:")

            notable = [c for c in coords if 'Unknown' not in c['location']]
            for coord in notable[:15]:
                report.append(f"  {coord['latitude']}N, {abs(coord['longitude'])}W -> {coord['location']}")
                if coord['context']:
                    report.append(f"    \"{coord['context']}\"")
            report.append("")

        # Polyglot Analysis
        if self.results.get('polyglot_analysis'):
            pa = self.results['polyglot_analysis']
            report.append("=" * 80)
            report.append("POLYGLOT CODE ANALYSIS")
            report.append("=" * 80)
            report.append(f"Message: {pa['message']}")
            report.append(f"\nLISP S-Expression:")
            report.append(f"  {pa['lisp']['expression']}")
            report.append(f"\nMIPS Instructions (first 5):")
            for op in pa['mips']['operations'][:5]:
                report.append(f"  {op}")
            report.append(f"\nPython Opcodes (first 10):")
            for op in pa['python'][:10]:
                report.append(f"  {op}")
            report.append("")

        # Summary
        report.append("=" * 80)
        report.append("SUMMARY")
        report.append("=" * 80)

        findings = []

        if 'letter_layer' in self.results:
            ll_validity = self.results['letter_layer']['opcode_validity']['percentage']
            if ll_validity >= 74:
                findings.append(f"HIGH letter layer opcode validity ({ll_validity:.1f}%)")

        if 'punctuation_layer' in self.results:
            pl_validity = self.results['punctuation_layer']['opcode_validity']['percentage']
            if pl_validity >= 74:
                findings.append(f"HIGH punctuation layer opcode validity ({pl_validity:.1f}%)")

        if self.results.get('gps_coordinates'):
            coord_count = len(self.results['gps_coordinates'])
            if coord_count > 20:
                findings.append(f"Many GPS coordinates found ({coord_count})")

        if 'letter_layer' in self.results:
            delim_count = self.results['letter_layer']['delimiter_count']
            if delim_count >= 3:
                findings.append(f"Multiple 0xC7 delimiters ({delim_count})")

        if findings:
            report.append("\nFINDINGS:")
            for f in findings:
                report.append(f"  [!] {f}")
            report.append("\nThese patterns suggest structured encoding is present.")
        else:
            report.append("\nNo significant encoding patterns detected.")

        report.append("")
        report.append("=" * 80)

        return '\n'.join(report)


def main():
    # Fix encoding for Windows
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(
        description='Comprehensive encoding analyzer for AI conversation exports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python comprehensive_analyzer.py conversation.json
  python comprehensive_analyzer.py conversation.json --output report.md
  python comprehensive_analyzer.py conversation.json --format json --output data.json
  python comprehensive_analyzer.py conversation.json --use-paragraphs
        """
    )

    parser.add_argument('file', help='Input file (JSON or text)')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument('-f', '--format', choices=['text', 'json'], default='text',
                       help='Output format')
    parser.add_argument('--use-paragraphs', action='store_true',
                       help='Use paragraph breaks instead of line breaks for letter extraction')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress progress messages')

    args = parser.parse_args()

    # Create analyzer
    analyzer = EncodingAnalyzer(args.file, verbose=not args.quiet)

    try:
        # Extract text
        analyzer.log("Extracting text...")
        analyzer.extract_text()

        # Analyze letter layer
        analyzer.log("Analyzing letter layer...")
        analyzer.analyze_letter_layer(use_lines=not args.use_paragraphs)

        # Analyze punctuation layer
        analyzer.log("Analyzing punctuation layer...")
        analyzer.analyze_punctuation_layer()

        # Extract GPS coordinates
        analyzer.log("Extracting GPS coordinates...")
        analyzer.extract_gps_coordinates('letter')

        # Polyglot analysis
        analyzer.log("Performing polyglot analysis...")
        analyzer.analyze_polyglot('letter')

        # Generate report
        analyzer.log("Generating report...")
        report = analyzer.generate_report(args.format)

        # Output
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report written to: {args.output}")
        else:
            print(report)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
