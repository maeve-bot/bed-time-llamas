# bed-time-llamas

Epub to audiobook pipeline using Qwen3-TTS

## Requirements

```bash
pip install ebooklib beautifulsoup4 lxml pydub soundfile tqdm qwen-tts
pip install num2words pymorphy3
```

## English Version

`epub_to_audiobook.py` - Convert English epub files to audiobooks.

```bash
# Basic usage with built-in voice
python3 epub_to_audiobook.py book.epub -o output/

# With specific speaker
python3 epub_to_audiobook.py book.epub -o output/ -s Ryan

# List available speakers
python3 epub_to_audiobook.py --list-speakers

# Skip synthesis (just extract text)
python3 epub_to_audiobook.py book.epub --skip-synthesize
```

## Russian Version

`epub_to_audiobook_ru.py` - Convert Russian epub files to audiobooks with voice cloning.

### Basic Usage

```bash
# With voice cloning (requires reference audio and text)
python3 epub_to_audiobook_ru.py book.epub -o output/ \
    --voice-ref my_voice.wav \
    --ref-text "Текст из референсного аудио"
```

### Reference Text from File

Instead of passing the reference text directly, you can save it to a file:

```bash
# Create a text file with your reference text
echo "Привет, это мой голос для озвучивания книг" > ref_text.txt

# Use the file
python3 epub_to_audiobook_ru.py book.epub -o output/ \
    --voice-ref my_voice.wav \
    --ref-text ref_text.txt
```

### Examples

```bash
# Full example with all options
python3 epub_to_audiobook_ru.py /path/to/russian_book.epub \
    -o ./audiobook/ \
    --voice-ref ./my_voice.wav \
    --ref-text ./ref_text.txt \
    --language Russian \
    --device cuda

# Skip synthesis (extract text only)
python3 epub_to_audiobook_ru.py book.epub -o output/ --skip-synthesize

# Skip export (don't convert to MP3)
python3 epub_to_audiobook_ru.py book.epub -o output/ --skip-export
```

### Options

| Option | Description |
|--------|-------------|
| `epub` | Path to epub file (required) |
| `-o, --output` | Output directory (default: ./output) |
| `--voice-ref` | Path to reference audio file for voice cloning |
| `--ref-text` | Text spoken in reference audio, or path to .txt file |
| `-l, --language` | Language (default: Russian) |
| `-d, --device` | Device (cuda or cpu, default: cuda) |
| `--no-flash` | Disable FlashAttention |
| `--skip-synthesize` | Skip TTS synthesis (extract text only) |
| `--skip-export` | Skip MP3 export |
| `--model` | Qwen3 model to use |

## Features

- **Number preprocessing**: Converts numbers to spoken words with proper Russian grammar
  - Years: "в 1985 году" → "в ...пятом году"
  - Chapters: "Глава 15" → "Глава пятнадцатая"
  - Dates: "01.01.2024" → "первое января две тысячи двадцать четыре года"

- **Variable pauses**: Different pause durations for sentence vs paragraph boundaries
  - Sentence → sentence: 600ms
  - Paragraph → paragraph: 1000ms

- **Voice cloning**: Uses Qwen3-TTS Base model with voice cloning from reference audio
