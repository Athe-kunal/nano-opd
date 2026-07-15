---
name: code-structure
description: Code style and structure conventions for this repo (nano-opd). Invoke BEFORE writing or reviewing any Python code change. Full reference is the Google Python Style Guide at assets/CODE.md — load it when a specific rule (docstring format, naming, imports, line length, type annotations) needs checking in detail.
---

# Code style

Follow the Google Python Style Guide for every code change: docstrings in
Google format, 4-space indentation, `snake_case`/`CapWords`/`ALL_CAPS`
naming conventions, type hints on public signatures. No unnecessary
abstractions — this codebase favors directly readable code over generic
frameworks. Prefer fewer lines, but never at the expense of readability.

The full style guide is in `assets/CODE.md` (mirrored from
https://google.github.io/styleguide/pyguide.html). Read it when you need the
precise rule for something not covered by the summary above — e.g. docstring
`Args:`/`Returns:`/`Raises:` formatting, import ordering, exception handling
conventions, or type-annotation line-breaking.
