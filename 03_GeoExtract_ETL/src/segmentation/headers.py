import re

HEADER_PATTERNS = [
    r"^Abstract\b", r"^Keywords\b", r"^Introduction\b", r"^Background\b",
    r"^Overview\b", r"^Methodology\b", r"^Methods\b", r"^Data\b",
    r"^Results\b", r"^Discussion\b", r"^Analysis\b", r"^Conclusion\b",
    r"^Conclusions\b", r"^References\b", r"^Acknowledgments\b",
    r"^Appendix\b", r"^Supplementary\b", r"^Related Work\b",
    r"Seismic\b", r"^Geophysics\b", r"^Model\b", r"^Experiment\b",
    r"^Evaluation\b", r"^[A-Z][A-Za-z ]{2,40}$",
]

# -----------------------------
# Проверка: является ли строка заголовком
# -----------------------------
def is_header(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) > 90:
        return False
    if s.isupper() and len(s.split()) <= 10:
        return True
    if s.istitle() and 2 <= len(s.split()) <= 12:
        return True
    if 1 <= len(s.split()) <= 6 and "." not in s and s[0].isupper():
        return True
    for pat in HEADER_PATTERNS:
        if re.match(pat, s):
            return True
    return False