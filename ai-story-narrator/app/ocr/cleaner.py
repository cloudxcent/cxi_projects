import re
import unicodedata

def clean_text(text):
    """Clean OCR text by removing garbage lines and fixing formatting."""
    def is_meaningful(line):
        # Skip empty or very short lines
        if len(line) < 10:
            return False

        # Skip lines with too few letter characters
        letter_count = sum(1 for c in line if c.isalpha())
        return letter_count >= len(line) * 0.4

    def normalize_unicode(s):
        # Remove control and non-printable characters
        return ''.join(c for c in s if unicodedata.category(c)[0] != 'C')

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = normalize_unicode(line.strip())
        if not is_meaningful(line):
            continue

        # Collapse multiple spaces and fix punctuation
        line = re.sub(r'\s+', ' ', line)
        line = re.sub(r'[.,:;_]{2,}', '.', line)
        line = re.sub(r'–+', '-', line)      # Normalize dashes
        line = re.sub(r'[“”]', '"', line)    # Normalize quotes
        line = re.sub(r"[‘’]", "'", line)

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)
