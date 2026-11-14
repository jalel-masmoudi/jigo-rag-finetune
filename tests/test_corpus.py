"""Front matter, empty pages, and navigation stubs are removed."""

from jigo_rag.corpus import prepare_markdown


def test_front_matter_is_removed_and_prose_is_kept():
    text = """---
title: Travel card
questions:
  - travel
---

Employees must request a travel card before the first trip.
"""
    keep, reason, cleaned = prepare_markdown(text)
    assert keep and reason == "kept"
    assert "title:" not in cleaned
    assert "travel card" in cleaned.lower() or "request a travel card" in cleaned


def test_empty_redirect_is_dropped():
    text = """---
title: Moved
redirect_to: /somewhere/
---
"""
    keep, reason, cleaned = prepare_markdown(text)
    assert not keep
    assert reason == "empty"
    assert cleaned == ""


def test_table_of_contents_is_a_nav_stub():
    lines = ["# Contents", ""]
    for index in range(10):
        lines.append(f"- [Chapter {index}](/chapter-{index}/)")
    keep, reason, _cleaned = prepare_markdown("\n".join(lines))
    assert not keep
    assert reason == "nav_stub"


def test_a_short_page_with_a_few_links_is_kept():
    text = """---
title: Slack
---

Slack is the chat tool. New employees join the team channel on day one
and read the pinned note before posting. See the [guide](/slack/guide/)
and the [emoji list](/slack/emoji/) for the house style.
"""
    keep, reason, cleaned = prepare_markdown(text)
    assert keep and reason == "kept"
    assert "https://" not in cleaned
    assert "guide" in cleaned
