#!/usr/bin/env python3
"""
Russian Epub to Audiobook Converter using Qwen3-TTS Base with Voice Cloning
Uses voice cloning from a reference audio file instead of built-in speakers.

Requirements:
    pip install ebooklib beautifulsoup4 lxml pydub soundfile tqdm qwen-tts
    pip install num2words pymorphy3
    
    # Optional for mp3 export:
    pip install mutagen
    # ffmpeg must be installed on system
    
    # For FlashAttention 2 (recommended for GPU):
    pip install flash-attn --no-build-isolation

Environment:
    - Python 3.12
    - PyTorch 2.9
    - CUDA 12.x
    - FlashAttention 2.8.3

Usage:
    # With voice cloning (provide reference audio):
    python3 epub_to_audiobook_ru.py book.epub -o output/ --voice-ref my_voice.wav --ref-text "Text spoken in the reference audio"
    
    # With voice cloning (Russian language):
    python3 epub_to_audiobook_ru.py book.epub -o output/ --voice-ref my_voice.wav --ref-text "Текст из референсного аудио" --language Russian
    
    # Built-in voice (no cloning):
    python3 epub_to_audiobook_ru.py book.epub -o output/ -s aidar

Example:
    python3 epub_to_audiobook_ru.py /path/to/russian_book.epub -o ./audiobook/ \\
        --voice-ref ./my_voice.wav \\
        --ref-text "Привет, это мой голос" \\
        --language Russian
"""

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import soundfile as sf
import tqdm
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT, ITEM_COVER

# =============================================================================
# CONFIGURATION
# =============================================================================

SAMPLE_RATE = 24000
MAX_CHUNK_SIZE = 500  # chars per chunk
SENTENCE_PAUSE_MS = 600  # pause between sentences
PARAGRAPH_PAUSE_MS = 1000  # pause between paragraphs
PARA_MARKER = "{PARA}"  # marker for paragraph boundaries

# Content extraction config
CONTENT_TAGS = ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th"]
SKIP_TAGS = ["script", "style", "meta", "head", "link", "noscript", "nav", "header", "footer"]
MIN_CHAPTER_WORDS = 50

SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Chapter:
    index: int
    title: str
    text: str
    word_count: int

@dataclass
class Book:
    title: str
    author: str
    language: str
    chapters: list[Chapter]

# =============================================================================
# EPUB PARSING
# =============================================================================

def parse_epub(epub_path: str | Path) -> tuple[Book, Optional[bytes]]:
    """Parse epub file and return Book object with cover image."""
    book = epub.read_epub(str(epub_path))
    
    # Get metadata
    title = book.get_metadata('DC', 'title')
    title = title[0][0] if title else "Unknown Title"
    
    author = book.get_metadata('DC', 'creator')
    author = author[0][0] if author else "Unknown Author"
    
    language = book.get_metadata('DC', 'language')
    language = language[0][0] if language else "ru"  # Default to Russian
    
    # Get cover
    cover_data = None
    for item in book.get_items():
        if item.get_type() == ITEM_COVER:
            cover_data = item.get_content()
            break
    
    # Extract chapters
    chapters = []
    chapter_index = 0
    
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'lxml')
            text = extract_text_from_soup(soup)
            
            if text:
                # Try to get chapter title
                title_elem = soup.find(['h1', 'h2', 'h3', 'title'])
                chapter_title = title_elem.get_text(strip=True) if title_elem else f"Глава {chapter_index + 1}"
                
                # Skip if too short (probably not a real chapter)
                word_count = len(text.split())
                if word_count >= MIN_CHAPTER_WORDS:
                    chapters.append(Chapter(
                        index=chapter_index,
                        title=chapter_title,
                        text=text,
                        word_count=word_count
                    ))
                    chapter_index += 1
    
    return Book(title=title, author=author, language=language, chapters=chapters), cover_data


def extract_text_from_soup(soup: BeautifulSoup) -> str:
    """Extract text from HTML, preserving paragraph boundaries."""
    # Remove unwanted elements
    for tag in soup(SKIP_TAGS):
        tag.decompose()
    
    texts = []
    
    # Handle lists - flatten nested structure, each <li> becomes a paragraph
    for lst in soup.find_all(['ul', 'ol']):
        for li in lst.find_all('li', recursive=True):
            # Skip if this <li> has child <li> elements (they'll be processed separately)
            if li.find(['ul', 'ol']):
                continue
            text = li.get_text(strip=True)
            if text:
                texts.append(text)
    
    # Extract text from other content tags (excluding li, which we handled above)
    other_tags = [t for t in CONTENT_TAGS if t != 'li']
    for tag in soup.find_all(other_tags):
        # Skip if this tag is inside a list (already handled)
        if tag.find_parent(['ul', 'ol']):
            continue
        text = tag.get_text(strip=True)
        if text:
            texts.append(text)
    
    # Join paragraphs with marker
    return f' {PARA_MARKER} '.join(texts)


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def preprocess_text_v1(text: str) -> str:
    """Preprocess text to improve TTS pronunciation (basic version).
    
    Handles:
    - Ordinals (1-й, 2-я, 3-е etc.) → Russian ordinal words
    - Plain integers (years, quantities) → Russian words
    """
    import re
    from num2words import num2words
    
    # Handle ordinals: 1-й, 2-я, 3-е etc.
    ordinal_pattern = re.compile(r'(\d+)(-й|-я|-е|-го|-му|-ой|-ы|-ам|-ах|-ом)')
    def replace_ordinal(match):
        num = int(match.group(1))
        return num2words(num, lang='ru', to='ordinal')
    
    text = ordinal_pattern.sub(replace_ordinal, text)
    
    # Handle plain integers
    def replace_number(match):
        num_str = match.group(0)
        
        # Skip phones
        if '-' in num_str or num_str.startswith('+'):
            return num_str
        
        try:
            return num2words(int(num_str.replace(' ', '')), lang='ru')
        except ValueError:
            return num_str
    
    text = re.sub(r'(?<![.\d/+-])\b\d+\b(?![.\d/-])', replace_number, text)
    
    return text


def preprocess_text_v2(text: str) -> str:
    """Enhanced number preprocessing with Russian grammar rules (improved version).
    
    Handles:
    - Ordinals (1-й, 2-я, 3-е etc.) → Russian ordinal words
    - Plain integers (years, quantities) → Russian words
    - Dates (DD.MM.YYYY, DD/MM/YYYY) → full Russian date format
    
    Improvements over v1:
    - Proper date normalization with correct grammar cases
    """
    import re
    from num2words import num2words
    
    # Handle ordinals
    ordinal_pattern = re.compile(r'(\d+)(-й|-я|-е|-го|-му|-ой|-ы|-ам|-ах|-ом)')
    def replace_ordinal(match):
        num = int(match.group(1))
        return num2words(num, lang='ru', to='ordinal')
    
    text = ordinal_pattern.sub(replace_ordinal, text)
    
    # Handle dates: DD.MM.YYYY or DD/MM/YYYY
    def replace_date(match):
        date_str = match.group(0)
        parts = re.split(r'[.\s/]+', date_str)
        
        if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
            try:
                day = int(parts[0])
                month = int(parts[1])
                year = int(parts[2])
                
                # Day: neuter ordinal for dates
                day_word = num2words(day, lang='ru', to='ordinal')
                if day_word.endswith('ый'):
                    day_word = day_word[:-2] + 'ое'
                elif day_word.endswith('ий'):
                    day_word = day_word[:-2] + 'ое'
                
                # Month: genitive case
                months = ['', 'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
                         'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря']
                month_word = months[month] if 1 <= month <= 12 else parts[1]
                
                # Handle 2-digit years
                if year < 100:
                    year += 1900 if year > 50 else 2000
                
                year_word = num2words(year, lang='ru')
                
                return f'{day_word} {month_word} {year_word} года'
            except (ValueError, IndexError):
                pass
        
        return date_str
    
    text = re.sub(r'\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b', replace_date, text)
    
    # Handle plain integers
    def replace_number(match):
        num_str = match.group(0)
        if '-' in num_str or num_str.startswith('+'):
            return num_str
        try:
            return num2words(int(num_str.replace(' ', '')), lang='ru')
        except ValueError:
            return num_str
    
    text = re.sub(r'(?<![.\d/+-])\b\d+\b(?![.\d/-])', replace_number, text)
    
    return text


def preprocess_text(text: str) -> str:
    """Advanced number preprocessing with pymorphy3 for proper Russian grammar.
    
    Handles:
    - Ordinals (1-й, 2-я, 3-е etc.) → Russian ordinal words
    - Dates (DD.MM.YYYY) → full Russian date format
    - Years (в 1985 году) → ordinal + locative case
    - Chapters (Глава 15) → feminine ordinals
    - Plain integers → Russian words
    
    Requires: pip install pymorphy3 num2words
    """
    import re
    import pymorphy3
    from num2words import num2words
    
    morph = pymorphy3.MorphAnalyzer()
    
    # 1. Handle ordinals: 1-й, 2-я etc.
    ordinal_pattern = re.compile(r'(\d+)(-й|-я|-е|-го|-му|-ой|-ы|-ам|-ах|-ом)')
    text = ordinal_pattern.sub(lambda m: num2words(int(m.group(1)), lang='ru', to='ordinal'), text)
    
    # 2. Handle dates: DD.MM.YYYY or DD/MM/YYYY
    def replace_date(match):
        parts = re.split(r'[.\s/]+', match.group(0))
        if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
            try:
                day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                day_word = num2words(day, lang='ru', to='ordinal')
                if day_word.endswith('ый'):
                    day_word = day_word[:-2] + 'ое'
                elif day_word.endswith('ий'):
                    day_word = day_word[:-2] + 'ое'
                months = ['', 'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
                         'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря']
                month_word = months[month] if 1 <= month <= 12 else parts[1]
                if year < 100:
                    year += 1900 if year > 50 else 2000
                return f'{day_word} {month_word} {num2words(year, lang="ru")} года'
            except:
                pass
        return match.group(0)
    
    text = re.sub(r'\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b', replace_date, text)
    
    # 3. Handle years: 'в 1985 году' -> ordinal in locative case
    year_pattern = re.compile(r'в\s+(\d{4})\s*(году?|года)?', re.IGNORECASE)
    def replace_year(match):
        year = int(match.group(1))
        ord_year = num2words(year, lang='ru', to='ordinal')
        
        parsed = morph.parse(ord_year)[0]
        try:
            loc = parsed.inflect({'sing', 'loct'})
            if loc:
                ord_year = loc.word
            else:
                if ord_year.endswith('ый'):
                    ord_year = ord_year[:-2] + 'ом'
                elif ord_year.endswith('ий'):
                    ord_year = ord_year[:-2] + 'ем'
        except:
            if ord_year.endswith('ый'):
                ord_year = ord_year[:-2] + 'ом'
        
        return f'в {ord_year} году'
    
    text = year_pattern.sub(replace_year, text)
    
    # 4. Handle chapters: 'Глава 15' -> feminine ordinal
    chapter_pattern = re.compile(r'Глава\s+(\d+)', re.IGNORECASE)
    def replace_chapter(match):
        num = int(match.group(1))
        num_word = num2words(num, lang='ru', to='ordinal')
        parsed = morph.parse(num_word)[0]
        try:
            fem = parsed.inflect({'sing', 'femn', 'nomn'})
            if fem:
                return 'Глава ' + fem.word
        except:
            pass
        return 'Глава ' + num_word
    
    text = chapter_pattern.sub(replace_chapter, text)
    
    # 5. Handle plain numbers
    def replace_number(match):
        num_str = match.group(0)
        if '-' in num_str or num_str.startswith('+'):
            return num_str
        try:
            return num2words(int(num_str.replace(' ', '')), lang='ru')
        except ValueError:
            return num_str
    
    text = re.sub(r'(?<![.\d/+-])\b\d+\b(?![.\d/-])', replace_number, text)
    
    return text


# Use v3 (with pymorphy3) as default - best grammar handling
preprocess_text = preprocess_text


# =============================================================================
# TEXT CHUNKING
# =============================================================================

@dataclass
class Chunk:
    text: str
    pause_after: str  # "sentence", "paragraph", or "none"
    audio: any = None  # will hold numpy array after synthesis


def chunk_text(text: str, max_size: int = MAX_CHUNK_SIZE) -> list[Chunk]:
    """Split text into chunks at sentence and paragraph boundaries.
    
    Returns list of Chunks with pause_after indicating the pause type.
    """
    # First split by paragraph marker
    paragraphs = text.split(PARA_MARKER)
    
    all_chunks = []
    
    for para_idx, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # Split paragraph by sentences
        sentences = SENTENCE_ENDINGS.split(paragraph)
        
        current_chunk = []
        current_length = 0
        
        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_len = len(sentence)
            
            # If current chunk + new sentence is too big, finish current chunk
            if current_length + sentence_len > max_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                # Determine pause type: paragraph if it's the last chunk in paragraph
                is_last_in_para = (sent_idx == len(sentences) - 1) and (para_idx < len(paragraphs) - 1)
                pause_after = "paragraph" if is_last_in_para else "sentence"
                all_chunks.append(Chunk(text=chunk_text, pause_after=pause_after))
                current_chunk = []
                current_length = 0
            
            # If single sentence is still too big, force split it
            if sentence_len > max_size:
                words = sentence.split()
                current_word_chunk = []
                current_word_len = 0
                
                for word in words:
                    word_len = len(word)
                    if current_word_len + word_len + 1 > max_size:
                        chunk_text = " ".join(current_word_chunk)
                        # Inside a force-split sentence, all are sentence pauses
                        all_chunks.append(Chunk(text=chunk_text, pause_after="sentence"))
                        current_word_chunk = []
                        current_word_len = 0
                    current_word_chunk.append(word)
                    current_word_len += word_len + 1
                
                if current_word_chunk:
                    remainder = " ".join(current_word_chunk)
                    current_chunk.append(remainder)
                    current_length += len(remainder) + 1
                continue
            
            current_chunk.append(sentence)
            current_length += sentence_len + 1
        
        # Don't forget the last chunk in the paragraph
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            # If this isn't the last paragraph, it's a paragraph boundary
            is_last_para = (para_idx == len(paragraphs) - 1)
            pause_after = "none" if is_last_para else "paragraph"
            all_chunks.append(Chunk(text=chunk_text, pause_after=pause_after))
    
    return all_chunks


# =============================================================================
# AUDIO PROCESSING
# =============================================================================

def concatenate_audio(
    chunks: list,
    sample_rate: int = SAMPLE_RATE,
) -> list:
    """Concatenate audio arrays with variable pauses based on chunk metadata."""
    import numpy as np
    
    result = []
    
    for i, chunk in enumerate(chunks):
        # chunk.audio is the audio numpy array
        # chunk.pause_after tells us what pause to add
        
        result.append(chunk.audio)
        
        if i < len(chunks) - 1:
            # Determine pause based on pause_after
            pause_type = chunks[i].pause_after
            
            if pause_type == "paragraph":
                pause_ms = PARAGRAPH_PAUSE_MS
            else:  # sentence or none
                pause_ms = SENTENCE_PAUSE_MS
            
            pause_samples = int(sample_rate * pause_ms / 1000)
            pause = np.zeros(pause_samples, dtype=np.float32)
            result.append(pause)
    
    return np.concatenate(result)


def compute_hash(data: dict) -> str:
    """Compute hash for content-based caching."""
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


# =============================================================================
# TTS WRAPPER (requires GPU)
# =============================================================================

class QwenTTSEngine:
    """Wrapper for Qwen3-TTS Base with voice cloning."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        voice_ref_audio: Optional[str] = None,
        voice_ref_text: Optional[str] = None,
        language: str = "Russian",
        device: str = "cuda",
        use_flash_attention: bool = True
    ):
        self.model_name = model_name
        self.voice_ref_audio = voice_ref_audio
        self.voice_ref_text = voice_ref_text
        self.language = language
        self.device = device
        self.use_flash_attention = use_flash_attention
        self._model = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return
        
        import torch
        from qwen_tts import Qwen3TTSModel
        
        print(f"Loading {self.model_name} on {self.device}...")
        
        # Determine dtype and attention
        dtype = torch.float16  # FlashAttention works best with float16
        attn_impl = "flash_attention_2" if self.use_flash_attention and "cuda" in self.device else "sdpa"
        
        self._model = Qwen3TTSModel.from_pretrained(
            self.model_name,
            device_map=self.device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        print("Model loaded!")
    
    def synthesize(self, text: str) -> tuple:
        """Synthesize single text to audio."""
        self._load_model()
        
        if self.voice_ref_audio:
            # Voice cloning mode
            wavs, sr = self._model.generate_voice_clone(
                text=text,
                language=self.language,
                ref_audio=self.voice_ref_audio,
                ref_text=self.voice_ref_text,
                non_streaming_mode=True,
            )
        else:
            raise ValueError("Voice cloning requires --voice-ref and --ref-text")
        
        return wavs[0], sr
    
    def synthesize_chunks(self, chunks: list[Chunk]) -> tuple:
        """Synthesize multiple chunks and concatenate with variable pauses."""
        import numpy as np
        
        self._load_model()
        
        # Collect text from chunks for batch synthesis
        chunk_texts = [chunk.text for chunk in chunks]
        
        # Batch synthesis for efficiency
        batch_size = 16
        audio_results = []
        
        for i in tqdm.tqdm(range(0, len(chunk_texts), batch_size), desc="Synthesizing", unit="batch"):
            batch_texts = chunk_texts[i:i + batch_size]
            
            if self.voice_ref_audio:
                wavs, sr = self._model.generate_voice_clone(
                    text=batch_texts,
                    language=self.language,
                    ref_audio=self.voice_ref_audio,
                    ref_text=self.voice_ref_text,
                    non_streaming_mode=True,
                )
            else:
                raise ValueError("Voice cloning requires --voice-ref and --ref-text")
            
            if isinstance(wavs, np.ndarray):
                if len(wavs.shape) == 1:
                    wavs = [wavs]
                elif len(wavs.shape) > 1:
                    wavs = list(wavs)
            
            audio_results.extend(wavs)
        
        # Attach audio to chunks
        for i, chunk in enumerate(chunks):
            chunk.audio = audio_results[i]
        
        # Concatenate with variable pauses
        final_audio = concatenate_audio(chunks, sr)
        
        return final_audio, sr


# =============================================================================
# EXPORT
# =============================================================================

def export_mp3(
    audio_path: str | Path,
    output_path: str | Path,
    title: str,
    author: str,
    track: int = 1,
    bitrate: str = "192k"
):
    """Export wav to mp3 with ID3 tags."""
    import subprocess
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use ffmpeg to convert and add metadata
    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-b:a", bitrate,
        "-metadata", f"title={title}",
        "-metadata", f"artist={author}",
        "-metadata", f"track={track}",
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    
    print(f"Exported: {output_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def extract_epub(epub_path: str, output_dir: str) -> Book:
    """Extract epub to text files in output directory."""
    output_dir = Path(output_dir)
    extract_dir = output_dir / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Parsing epub: {epub_path}")
    book, cover_data = parse_epub(epub_path)
    
    print(f"Title: {book.title}")
    print(f"Author: {book.author}")
    print(f"Chapters: {len(book.chapters)}")
    
    # Save metadata
    metadata = {
        "title": book.title,
        "author": book.author,
        "language": book.language,
        "chapters": [
            {"index": ch.index, "title": ch.title, "word_count": ch.word_count}
            for ch in book.chapters
        ]
    }
    
    with open(extract_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save cover
    if cover_data:
        with open(extract_dir / "cover.jpg", "wb") as f:
            f.write(cover_data)
    
    # Save chapter texts
    for chapter in book.chapters:
        # Sanitize filename
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', chapter.title)[:50]
        filename = f"{chapter.index:02d}_{safe_title}.txt"
        
        with open(extract_dir / filename, "w", encoding="utf-8") as f:
            f.write(chapter.text)
    
    print(f"Extracted to: {extract_dir}")
    return book


def synthesize_book(
    book: Book,
    output_dir: str,
    voice_ref_audio: Optional[str] = None,
    voice_ref_text: Optional[str] = None,
    language: str = "Russian",
    device: str = "cuda",
    use_flash: bool = True
):
    """Synthesize book to audio (requires GPU)."""
    import numpy as np
    
    output_dir = Path(output_dir)
    synth_dir = output_dir / "synthesize"
    synth_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize TTS engine
    engine = QwenTTSEngine(
        voice_ref_audio=voice_ref_audio,
        voice_ref_text=voice_ref_text,
        language=language,
        device=device,
        use_flash_attention=use_flash
    )
    
    for chapter in tqdm.tqdm(book.chapters, desc="Chapters"):
        # Preprocess text (convert numbers to words)
        processed_text = preprocess_text(chapter.text)
        
        # Chunk text
        chunks = chunk_text(processed_text)
        
        # Synthesize
        audio, sr = engine.synthesize_chunks(chunks)
        
        # Save wav
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', chapter.title)[:50]
        filename = f"{chapter.index:02d}_{safe_title}.wav"
        
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        sf.write(synth_dir / filename, audio_int16, sr)
        
        print(f"Synthesized: {filename} ({len(audio)/sr:.1f}s)")
    
    print(f"Audio saved to: {synth_dir}")


def export_book(book: Book, output_dir: str, bitrate: str = "192k"):
    """Export audio to mp3 files with metadata."""
    output_dir = Path(output_dir)
    synth_dir = output_dir / "synthesize"
    export_dir = output_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    for chapter in book.chapters:
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', chapter.title)[:50]
        wav_file = synth_dir / f"{chapter.index:02d}_{safe_title}.wav"
        mp3_file = export_dir / f"{chapter.index:02d}_{safe_title}.mp3"
        
        if not wav_file.exists():
            print(f"Warning: {wav_file} not found, skipping")
            continue
        
        export_mp3(
            wav_file,
            mp3_file,
            title=chapter.title,
            author=book.author,
            track=chapter.index + 1,
            bitrate=bitrate
        )
    
    print(f"Exported to: {export_dir}")


def run_pipeline(
    epub_path: str,
    output_dir: str,
    voice_ref_audio: Optional[str] = None,
    voice_ref_text: Optional[str] = None,
    language: str = "Russian",
    device: str = "cuda",
    use_flash: bool = True,
    skip_synthesize: bool = False,
    skip_export: bool = False
):
    """Run full pipeline: extract -> synthesize -> export."""
    output_dir = Path(output_dir)
    
    # Step 1: Extract
    print("\n=== STEP 1: EXTRACT ===")
    book = extract_epub(epub_path, output_dir)
    
    if skip_synthesize:
        print("\n=== SKIPPING SYNTHESIZE (--skip-synthesize) ===")
    else:
        # Step 2: Synthesize (requires GPU)
        print("\n=== STEP 2: SYNTHESIZE (requires GPU) ===")
        synthesize_book(
            book, output_dir,
            voice_ref_audio=voice_ref_audio,
            voice_ref_text=voice_ref_text,
            language=language,
            device=device,
            use_flash=use_flash
        )
    
    if skip_export:
        print("\n=== SKIPPING EXPORT ===")
    else:
        # Step 3: Export
        print("\n=== STEP 3: EXPORT ===")
        export_book(book, output_dir)
    
    print("\n=== DONE ===")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Russian epub to audiobook using Qwen3-TTS Base with voice cloning"
    )
    parser.add_argument("epub", help="Path to epub file")
    parser.add_argument("-o", "--output", default="./output", help="Output directory")
    parser.add_argument("--voice-ref", dest="voice_ref", help="Path to reference audio file (.wav) for voice cloning")
    parser.add_argument("--ref-text", dest="ref_text", help="Text spoken in the reference audio, or path to a .txt file containing the text")
    parser.add_argument("-s", "--speaker", help="Speaker name (for built-in voices, use CustomVoice model instead)")
    parser.add_argument("-l", "--language", default="Russian", help="Language")
    parser.add_argument("-d", "--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--no-flash", action="store_true", help="Disable FlashAttention")
    parser.add_argument("--skip-synthesize", action="store_true", help="Skip TTS synthesis")
    parser.add_argument("--skip-export", action="store_true", help="Skip mp3 export")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base", help="Model name")
    
    args = parser.parse_args()
    
    # Handle ref-text: either direct text or read from file
    if args.ref_text:
        ref_text_path = Path(args.ref_text)
        if ref_text_path.exists():
            # Read from file
            args.ref_text = ref_text_path.read_text(encoding='utf-8').strip()
            print(f"Loaded reference text from: {ref_text_path}")
    
    # Validate inputs
    if not Path(args.epub).exists():
        print(f"Error: File not found: {args.epub}")
        sys.exit(1)
    
    # Voice cloning requires both voice-ref and ref-text
    if args.voice_ref and not args.ref_text:
        print("Error: --ref-text is required when using --voice-ref")
        sys.exit(1)
    
    if args.voice_ref and not Path(args.voice_ref).exists():
        print(f"Error: Voice reference file not found: {args.voice_ref}")
        sys.exit(1)
    
    use_flash = not args.no_flash
    
    run_pipeline(
        epub_path=args.epub,
        output_dir=args.output,
        voice_ref_audio=args.voice_ref,
        voice_ref_text=args.ref_text,
        language=args.language,
        device=args.device,
        use_flash=use_flash,
        skip_synthesize=args.skip_synthesize,
        skip_export=args.skip_export
    )


if __name__ == "__main__":
    main()
