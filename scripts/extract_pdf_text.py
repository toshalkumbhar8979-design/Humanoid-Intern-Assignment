"""Extract text from a PDF with Flate-compressed content streams (no external deps)."""
import re
import sys
import zlib

STR_PAT = re.compile(r"\((?:[^()\\]|\\.)*\)")


def extract(path: str) -> str:
    raw = open(path, "rb").read()
    texts = []
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.S):
        try:
            data = zlib.decompress(m.group(1))
        except Exception:
            continue
        if b"Tj" not in data and b"TJ" not in data:
            continue
        out = []
        for t in STR_PAT.findall(data.decode("latin-1", "ignore")):
            s = t[1:-1]
            s = s.replace("\\(", "(").replace("\\)", ")").replace("\\\\", "\\")
            out.append(s)
        texts.append(" ".join(out))
    return "\n".join(texts)


if __name__ == "__main__":
    print(extract(sys.argv[1]))
