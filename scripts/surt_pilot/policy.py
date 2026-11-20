"""Versioned deterministic policy; no Hazm, numeric verbalization or audio imports."""
import re
import unicodedata

VERSION = 'fa-latin-v1'
TRANSLATE = str.maketrans({'ي': 'ی', 'ى': 'ی', 'ك': 'ک', 'ـ': '',
                          **{c: str(i) for i, c in enumerate('۰۱۲۳۴۵۶۷۸۹')},
                          **{c: str(i) for i, c in enumerate('٠١٢٣٤٥٦٧٨٩')}})


def normalize(raw):
    text = unicodedata.normalize('NFC', raw).translate(TRANSLATE)
    output = []
    for c in text:
        if '\u064b' <= c <= '\u065f' or c == '\u0670':
            continue
        # ZWNJ is retained inside words; other format/control characters separate words.
        if c == '\u200c':
            output.append(c)
        elif unicodedata.category(c)[0] in ('L', 'N'):
            output.append(c)
        else:
            output.append(' ')
    text = re.sub(r'\s+', ' ', ''.join(output)).strip()
    text = re.sub(r'\u200c+', '\u200c', text)
    text = re.sub(r'(?<!\w)\u200c|\u200c(?!\w)', '', text)
    return re.sub(r'\s+', ' ', text).strip()
