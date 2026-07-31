"""robots.txt fetching and rule matching.

Python's stdlib ``urllib.robotparser`` does not implement the ``*`` and ``$``
wildcards, and several of the sites this tool targets rely on them (Financial
Express uses ``Disallow: /*?s=``, Business Line uses ``Disallow: /search/*``).
Under-matching a Disallow rule means fetching a page we were asked not to
fetch, so the matcher is implemented here against Google's robots.txt spec
rather than approximated.

Precedence follows the spec: the most specific rule wins (longest matching
pattern), and Allow beats Disallow on an exact-length tie.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class Rule:
    allow: bool
    pattern: str
    regex: re.Pattern = field(compare=False, repr=False)

    @property
    def specificity(self) -> int:
        # Google ranks by the character length of the raw pattern.
        return len(self.pattern)


def _compile(pattern: str) -> re.Pattern:
    """Translate a robots.txt path pattern into a regex.

    ``*`` matches any run of characters; a trailing ``$`` anchors the end of
    the path. Every other character is literal.
    """
    anchored_end = pattern.endswith("$")
    if anchored_end:
        pattern = pattern[:-1]
    out = ["^"]
    for ch in pattern:
        out.append(".*" if ch == "*" else re.escape(ch))
    if anchored_end:
        out.append("$")
    return re.compile("".join(out))


@dataclass
class GroupedRules:
    """The rule group that applies to one user-agent token."""

    agent: str
    rules: list[Rule] = field(default_factory=list)
    crawl_delay: float | None = None

    def allows(self, path: str) -> tuple[bool, str | None]:
        """Return ``(allowed, deciding_pattern)`` for a URL path."""
        best: Rule | None = None
        for rule in self.rules:
            if not rule.regex.match(path):
                continue
            if best is None or rule.specificity > best.specificity:
                best = rule
            elif rule.specificity == best.specificity and rule.allow and not best.allow:
                # Allow wins an exact tie.
                best = rule
        if best is None:
            return True, None  # Nothing matched: default allow.
        return best.allow, best.pattern


class RobotsPolicy:
    """Parsed robots.txt for a single origin.

    ``fetched_ok`` distinguishes "we read the policy and it permits this" from
    "we never got to read the policy". A 403 on robots.txt itself (which is
    what Business Standard returns) leaves us unable to establish permission,
    so :meth:`allows` fails closed in that case.
    """

    def __init__(self, origin: str, text: str | None, status: int | None) -> None:
        self.origin = origin
        self.status = status
        self.raw = text
        self.fetched_ok = status is not None and 200 <= status < 300 and text is not None
        self.groups: dict[str, GroupedRules] = {}
        self.sitemaps: list[str] = []
        if self.fetched_ok:
            self._parse(text or "")

    def _parse(self, text: str) -> None:
        current: list[str] = []
        # A blank line or a new User-agent after rules starts a fresh group.
        starting_group = True
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                continue
            field_name, _, value = line.partition(":")
            field_name = field_name.strip().lower()
            value = value.strip()

            if field_name == "sitemap":
                if value:
                    self.sitemaps.append(value)
                continue

            if field_name == "user-agent":
                token = value.lower()
                if not starting_group:
                    current = []
                    starting_group = True
                current.append(token)
                self.groups.setdefault(token, GroupedRules(agent=token))
                continue

            if field_name in ("allow", "disallow"):
                starting_group = False
                if not current:
                    continue
                if field_name == "disallow" and value == "":
                    # "Disallow:" with no value means allow everything.
                    continue
                if not value:
                    continue
                rule = Rule(field_name == "allow", value, _compile(value))
                for token in current:
                    self.groups[token].rules.append(rule)
                continue

            if field_name == "crawl-delay":
                starting_group = False
                try:
                    delay = float(value)
                except ValueError:
                    continue
                for token in current:
                    self.groups[token].crawl_delay = delay

    def group_for(self, user_agent_token: str) -> GroupedRules | None:
        """Pick the applicable group: exact token match, else ``*``.

        robots.txt matching is on a substring of the UA token, case-insensitive.
        We check whether any declared token appears in the configured agent
        string so that a UA of ``ClaudeBot/1.0`` matches a ``ClaudeBot`` group.
        """
        agent = user_agent_token.lower()
        best: GroupedRules | None = None
        for token, group in self.groups.items():
            if token == "*":
                continue
            if token in agent and (best is None or len(token) > len(best.agent)):
                best = group
        if best is not None:
            return best
        return self.groups.get("*")

    def allows(self, url: str, user_agent_token: str) -> tuple[bool, str]:
        """Return ``(allowed, human-readable reason)``."""
        if not self.fetched_ok:
            return False, (
                f"robots.txt for {self.origin} could not be read "
                f"(status={self.status}); treating as disallowed"
            )
        parsed = urllib.parse.urlsplit(url)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        group = self.group_for(user_agent_token)
        if group is None:
            return True, "no applicable group; default allow"
        allowed, pattern = group.allows(path)
        if pattern is None:
            return True, f"no rule matched in group '{group.agent}'; default allow"
        verb = "Allow" if allowed else "Disallow"
        return allowed, f"{verb}: {pattern} (group '{group.agent}')"

    def crawl_delay(self, user_agent_token: str) -> float | None:
        group = self.group_for(user_agent_token)
        return group.crawl_delay if group else None

    def blocks_entirely(self, tokens: Iterable[str]) -> list[str]:
        """Which of ``tokens`` are given a blanket ``Disallow: /``."""
        blocked = []
        for token in tokens:
            group = self.groups.get(token.lower())
            if group and any(not r.allow and r.pattern == "/" for r in group.rules):
                blocked.append(token)
        return blocked
