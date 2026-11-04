"""Check repository Markdown links without network access or external packages."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def prose(text: str) -> str:
    lines = []
    fence = None
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker[1][0]
            elif marker[1][0] == fence:
                fence = None
            continue
        if fence is None:
            lines.append(line)
    return "\n".join(lines)


def anchors(path: Path) -> set[str]:
    text = prose(path.read_text(encoding="utf-8"))
    result = set(re.findall(r'\bid=["\']([^"\']+)["\']', text))
    counts: dict[str, int] = {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, re.MULTILINE):
        heading = re.sub(r"<[^>]*>", "", heading).lower()
        slug = re.sub(r"[^\w\s-]", "", heading).replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        result.add(slug if count == 0 else f"{slug}-{count}")
    return result


def main() -> int:
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.md"],
        cwd=ROOT,
    ).decode().split("\0")
    checked = 0
    errors = []
    documents = sorted({name for name in names if name and (ROOT / name).is_file()})
    for name in documents:
        source = ROOT / name
        text = prose(source.read_text(encoding="utf-8"))
        links = re.findall(r"\]\(<?([^\s)>]+)>?(?:\s+[\"'][^)]*)?\)", text)
        links += re.findall(r'(?:src|href)=["\']([^"\']+)["\']', text)
        for raw in links:
            url = urlsplit(raw)
            if url.scheme or url.netloc:
                continue
            target = (source.parent / unquote(url.path)).resolve() if url.path else source
            if not target.is_relative_to(ROOT):
                errors.append(f"{name}: link escapes repository: {raw}")
            elif not target.exists():
                errors.append(f"{name}: missing target: {raw}")
            elif url.fragment and target.suffix == ".md" and unquote(url.fragment) not in anchors(target):
                errors.append(f"{name}: missing heading: {raw}")
            checked += 1
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print(f"Checked {checked} local links across {len(documents)} Markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
