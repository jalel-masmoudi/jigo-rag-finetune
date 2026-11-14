"""Download the pinned corpora and turn markdown into plain documents.

Two filters drop obvious non-content after the front matter is removed:

- empty: fewer than 40 letters or digits left (redirects, pages that
  are only a title in the front matter).
- nav stub: at least 8 markdown links, and fewer than 200 letters or
  digits outside those links. That is a table of contents. A short
  real page with a few links is kept, because the prose outside the
  links is what a question would be about.

Handbook files outside `pages/` are repository notes (license,
contributing guide, GitHub templates), not handbook chapters, so they
are not read. In the Jigo repo, every markdown file is read except
paths under `.github/`, which are pull-request templates.
"""

import re
import subprocess
from pathlib import Path

from jigo_rag.config import SOURCES

MIN_ALNUM = 40
NAV_MIN_LINKS = 8
NAV_MAX_OUTSIDE_ALNUM = 200

_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_LIQUID_COMMENT = re.compile(
    r"\{%\s*comment\s*%\}.*?\{%\s*endcomment\s*%\}", re.DOTALL | re.IGNORECASE
)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_LIQUID_TAG = re.compile(r"\{%.*?%\}", re.DOTALL)
# The label is group 1. URLs are dropped; the label is kept as text.
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)", re.DOTALL)


def alnum_count(text):
    return sum(character.isalnum() for character in text)


def prepare_markdown(text):
    """Return (keep, reason, cleaned_text).

    `reason` is `kept`, `empty`, or `nav_stub`.
    """
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    text = _FRONT_MATTER.sub("", text)
    text = _LIQUID_COMMENT.sub("", text)
    text = _HTML_COMMENT.sub("", text)
    text = _LIQUID_TAG.sub("", text)
    outside_links = _MARKDOWN_LINK.sub(" ", text)
    n_links = len(_MARKDOWN_LINK.findall(text))
    if alnum_count(text) < MIN_ALNUM:
        return False, "empty", ""
    if n_links >= NAV_MIN_LINKS and alnum_count(outside_links) < NAV_MAX_OUTSIDE_ALNUM:
        return False, "nav_stub", ""
    # Store the words of a link, not the URL.
    cleaned = _MARKDOWN_LINK.sub(r"\1", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if alnum_count(cleaned) < MIN_ALNUM:
        return False, "empty", ""
    return True, "kept", cleaned


def ensure_repo(url, sha, dest):
    """Clone `url` into `dest` and check out `sha`.

    If `dest` is already that commit, do nothing. If it is a different
    commit, stop, so a half-updated folder is not mixed into the corpus.
    """
    dest = Path(dest)
    if dest.exists() and (dest / ".git").exists():
        current = subprocess.check_output(
            ["git", "-C", str(dest), "rev-parse", "HEAD"], text=True
        ).strip()
        if current != sha:
            raise SystemExit(
                f"{dest} is at {current}, but the pinned commit is {sha}. "
                f"Delete {dest} and rerun."
            )
        return dest
    if dest.exists():
        raise SystemExit(f"{dest} exists and is not a git checkout. Delete it and rerun.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", url, str(dest)])
    subprocess.check_call(["git", "-C", str(dest), "checkout", "--detach", sha])
    return dest


def iter_markdown_files(repo_dir, source_name):
    repo_dir = Path(repo_dir)
    if source_name == "gsa-handbook":
        paths = sorted((repo_dir / "pages").rglob("*.md"))
    elif source_name == "jigo":
        paths = []
        for path in sorted(repo_dir.rglob("*.md")):
            relative_parts = path.relative_to(repo_dir).parts
            if ".github" in relative_parts or ".git" in relative_parts:
                continue
            paths.append(path)
    else:
        raise ValueError(f"unknown source {source_name}")
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(repo_dir).as_posix()
        yield relative, path.read_text(encoding="utf-8", errors="replace")


def load_documents(raw_root):
    """Return (documents, dropped).

    A document has doc_id, source, doc_path, and text.
    A dropped item has doc_id and reason.
    """
    raw_root = Path(raw_root)
    documents = []
    dropped = []
    for source in SOURCES:
        repo = ensure_repo(source["url"], source["sha"], raw_root / source["name"])
        for relative, text in iter_markdown_files(repo, source["name"]):
            doc_id = f"{source['name']}/{relative}"
            keep, reason, cleaned = prepare_markdown(text)
            if not keep:
                dropped.append({"doc_id": doc_id, "reason": reason})
                continue
            documents.append(
                {
                    "doc_id": doc_id,
                    "source": source["name"],
                    "doc_path": relative,
                    "text": cleaned,
                }
            )
    return documents, dropped
