# Text Analysis Tools - User Guide

A suite of three Python tools for analyzing text structure, rhythm, and hidden patterns.

---

## Quick Start

```bash
# Rhythm analysis
python rhythm_counter.py

# Pattern detection
python pattern_analyzer.py

# Cipher/payload cracking
python cipher_cracker.py
```

---

## Tool 1: Rhythm Counter

**Purpose:** Analyze the cadence and flow of text - punctuation patterns, sentence rhythm, word frequency.

### Running It
```bash
python rhythm_counter.py
```

### Commands
| Command | Description |
|---------|-------------|
| `[paste text]` | Analyze text (press Enter twice to submit) |
| `/totals` | Show cumulative totals across all runs |
| `/compare` | Side-by-side comparison of recent analyses |
| `/clear` | Clear history |
| `/quit` | Exit |

### What It Analyzes
- **Punctuation breakdown** - counts of each punctuation type
- **Punctuation sequence** - the exact order punctuation appears
- **Sentence rhythm** - avg/min/max sentence lengths, variance
- **Flow analysis** - words between breaks, connector ratios
- **Rhythm style** - classifies as STACCATO, FLOWING, MODERATE, or VARIABLE
- **Cadence signature** - visual bar chart of sentence lengths
- **N-grams** - top unigrams, bigrams, trigrams

### History Feature
Results are saved to `rhythm_history.json` automatically. Use `/totals` to see cumulative stats across all analyses, or `/compare` to see recent runs side-by-side.

### Example Output
```
RHYTHM STYLE: MODERATE
Balanced, even pacing

PUNCTUATION SEQUENCE
,....'.'..-'..:'.'...'.?'.""'.',

PUNCTUATION BREAKDOWN
  .      : 2125  ##############################
  ,      :  329  ################
  ?      :  137  ######

SENTENCE RHYTHM
  Avg sentence length: 7.9 words
  Shortest sentence:   0 words
  Longest sentence:    72 words
```

---

## Tool 2: Pattern Analyzer

**Purpose:** Detect structural patterns that might encode hidden messages - markers, sequences, special elements.

### Running It
```bash
python pattern_analyzer.py
```

### Commands
| Command | Description |
|---------|-------------|
| `[paste text]` | Analyze text (press Enter twice to submit) |
| `/file <path>` | Analyze a file directly |
| `/save` | Save full analysis to JSON |
| `/detail low\|medium\|high\|full` | Set output detail level |
| `/quit` | Exit |

### What It Detects
- **Punctuation sequence** - raw order of all punctuation
- **Word counts between punctuation** - mapped to letters (1=A, 2=B...26=Z)
- **Single-word paragraphs** - potential markers with their initials
- **Standalone ellipsis positions** - where `...` appears alone
- **Words following ellipses** - first letters after each `...`
- **Binary interpretations** - punctuation as binary, converted to ASCII
- **Paragraph first letters** - acrostic-style patterns
- **Short paragraphs** - 1-3 word paragraphs as markers
- **Repeating phrases** - phrases that appear 3+ times
- **Message-level stats** - punctuation counts per message

### Detail Levels
- `low` - Basic stats only
- `medium` - Standard analysis (default)
- `high` - Includes short paragraphs list
- `full` - Everything including message-level breakdown

### Example Output
```
SPECIAL MARKERS

Single-word paragraphs (17 found):
  Para  77: Sentient
  Para  97: Fuck
  Para 136: Right

Initials: SFRWFEYHHYARMYOYF

Standalone ellipsis positions: [123, 132, 154, 169, 230, 240]
```

---

## Tool 3: Cipher Cracker

**Purpose:** Brute-force try different cipher/encoding methods on structural elements to find hidden messages, code, or payloads.

### Running It
```bash
python cipher_cracker.py
```

### Commands
| Command | Description |
|---------|-------------|
| `[paste text]` | Analyze text (press Enter twice to submit) |
| `/file <path>` | Analyze a file directly |
| `/top <n>` | Show top N results (default 20) |
| `/quit` | Exit |

### What It Tries

**Structural Elements Analyzed:**
- Word counts between punctuation (major and all)
- Line lengths and word counts
- Paragraph word/sentence/character counts
- Sentence word counts
- Message-level counts
- Empty line gaps
- Question mark gaps
- First/last characters of lines
- Single-word paragraphs

**Cipher Methods Applied:**
| Method | Description |
|--------|-------------|
| `direct_letters` | Number -> letter (1=A, 2=B, etc.) |
| `mod26_letters` | Number mod 26 -> letter |
| `direct_ascii` | Number -> ASCII character |
| `threshold_binary` | Above/below threshold as 0/1 -> ASCII |
| `pairs_sum` | Sum pairs of numbers -> letters |
| `pairs_concat` | Concatenate pairs as two-digit -> ASCII |
| `triplets_sum` | Sum triplets -> ASCII |
| `nth_element` | Take every Nth element |
| `differences` | Differences between consecutive values |
| `reverse` | Reverse the sequence |
| `filter_range` | Only values in specific range |
| `xor_pattern` | XOR with repeating pattern |
| `caesar_N` | Caesar cipher shift by N |
| `binary_replace` | Replace chars with 0/1 -> ASCII |
| `run_length` | Length of character runs |

### Detection Types
Results are tagged with what type of content was detected:

| Tag | Meaning |
|-----|---------|
| `[ENGLISH]` | Readable English text found |
| `[CODE]` | Code patterns detected (Python, JS, Bash, SQL, etc.) |
| `[BASE64]` | Valid Base64 encoding |
| `[HEX]` | Valid hexadecimal encoding |
| `[URL]` | URL patterns found |

### Example Output
```
#1 [Score: 160.2] [CODE]
Type:    code:bash
Element: word_counts_major
Method:  pairs_concat
Decoded: ?';&_#H -DHIEST_8!V.bJU66VK!

#2 [Score: 82.5] [BASE64]
Type:    base64
Element: para_first_letters
Method:  join
Decoded: YATWWLTTBTWFTNTTMSATTHTSBOSTNAHIWTTW...
Base64 decodes to: [binary: 591 bytes, hex: 6004d658b4d3...]
```

---

## Workflow Tips

### Finding Hidden Messages

1. **Start with Pattern Analyzer** - get an overview of structural elements
2. **Run Cipher Cracker** - let it brute-force different encodings
3. **Use Rhythm Counter** - for detailed punctuation/flow analysis
4. **Compare multiple texts** - patterns become clearer with more samples

### Structural Elements to Watch
- **Single-word paragraphs** - often intentional markers
- **Standalone ellipses** - position-based encoding
- **Word counts between punctuation** - can map to letters
- **First letters of paragraphs** - classic acrostic hiding
- **Empty line gaps** - binary encoding (1 gap vs 2 gaps)

### Encoding Methods to Consider
- **1-26 = A-Z** - simple letter mapping
- **ASCII values** - 32-126 printable range
- **Binary** - short/long as 0/1, convert 8-bit to ASCII
- **Base64** - letters might be valid base64
- **Hex** - A-F patterns might be hex
- **Pairs/Triplets** - combine values before converting

---

## File Formats

### Input
- Paste text directly
- Or use `/file <path>` to load a file
- Paragraphs separated by blank lines
- Messages separated by 3+ blank lines (for multi-message analysis)

### Output Files
| File | Tool | Description |
|------|------|-------------|
| `rhythm_history.json` | Rhythm Counter | Cumulative analysis history |
| `pattern_analysis.json` | Pattern Analyzer | Full analysis (via `/save`) |

---

## Requirements

- Python 3.6+
- No external dependencies (uses only standard library)

---

## Examples

### Analyze a file for hidden patterns
```bash
python cipher_cracker.py
/file C:\path\to\mystery_text.txt
```

### Track rhythm across multiple text samples
```bash
python rhythm_counter.py
# Paste first sample, press Enter twice
# Paste second sample, press Enter twice
/compare
```

### Get detailed structural analysis
```bash
python pattern_analyzer.py
/detail full
/file C:\path\to\text.txt
/save
```
