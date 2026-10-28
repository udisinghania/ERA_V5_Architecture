"""G5 config checker — reports every setting with no value or no source.

Convention (see configs/config_v0.yaml header):
  * each mapping key needs a `# source: ...` comment on its own line or on
    comment line(s) directly above it (`source` matched case-insensitively);
  * each key needs a non-empty value. A key whose value is empty/null/TBD
    is reported, except keys in --allow-tbd (size values TSAI fills 31 Oct).

Exit code 0 + "0 errors"  <=>  Sat-31-Oct gate passes.

Usage:
    python harness/check_config.py configs/config_v0.yaml
    python harness/check_config.py configs/config_v0.yaml --allow-tbd ke2_shape_tbd size_tbd_1 size_tbd_2
"""

import re
import sys

MISSING_VALUES = {"", "null", "none", "~", "tbd", "todo", "todo:", "tbd:"}
KEY_RE = re.compile(r"^(\s*)([^#\s][^:]*):(\s*(.*))?$")


def check(path, allow_tbd):
    with open(path) as f:
        lines = f.read().splitlines()
    errors = []
    pending_comments = []  # comment lines directly above, no blank line between
    for i, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            if stripped.startswith("#"):
                pending_comments.append(stripped)
            else:
                pending_comments = []
            continue
        m = KEY_RE.match(raw)
        if not m:
            pending_comments = []
            continue
        key = m.group(2).strip()
        rest = (m.group(4) or "").strip()
        inline_comment = ""
        value = rest
        if "#" in rest:
            # split value from trailing comment (values here never contain '#')
            value, inline_comment = rest.split("#", 1)
            value = value.strip()
            inline_comment = inline_comment.strip()
        has_source = "source" in inline_comment.lower() or any(
            "source" in c.lower() for c in pending_comments
        )
        if not has_source and not key.startswith("-"):
            errors.append(f"line {i}: key '{key}' has no source comment")
        # a mapping key with no value on the line is fine iff nested keys follow
        if value == "" and not inline_comment:
            j = i  # look ahead: nested mapping?
            nested = False
            cur_indent = len(raw) - len(raw.lstrip())
            while j < len(lines):
                nxt = lines[j]
                if not nxt.strip() or nxt.strip().startswith("#"):
                    j += 1
                    continue
                nxt_indent = len(nxt) - len(nxt.lstrip())
                nested = nxt_indent > cur_indent and (":" in nxt or nxt.strip().startswith("-"))
                break
            if not nested and key not in allow_tbd and not key.startswith("-"):
                errors.append(f"line {i}: key '{key}' has no value")
        elif value.lower() in MISSING_VALUES:
            if key in allow_tbd:
                pass  # size values TSAI fills 31 Oct
            else:
                errors.append(f"line {i}: key '{key}' has no value (got '{value}')")
        pending_comments = []
    return errors


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    path = args[0]
    allow_tbd = set()
    if "--allow-tbd" in args:
        allow_tbd = set(args[args.index("--allow-tbd") + 1 :])
    errors = check(path, allow_tbd)
    for e in errors:
        print("ERROR:", e)
    print(f"{len(errors)} errors in {path}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
