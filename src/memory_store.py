from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator based on character count.

    Approximately 4 characters per token for English/Vietnamese text.
    Returns 0 for empty or whitespace-only text.
    """
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, math.ceil(len(cleaned) / 4))


# ---------------------------------------------------------------------------
# Structured facts (entity extraction + confidence + conflict + decay)
# ---------------------------------------------------------------------------

# Fields that hold a bounded set of values (merged), not a single current value.
LIST_KEYS = {"Style trả lời", "Mối quan tâm"}
MAX_LIST_ITEMS = 5
# Identity facts do not go stale, so they are exempt from decay.
PERMANENT_KEYS = {"Tên"}

DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_HALF_LIFE_TURNS = 200.0
DEFAULT_MIN_SCORE = 0.25


@dataclass
class FactCandidate:
    """One fact proposed by the extractor, before the write gate."""

    key: str
    value: str
    confidence: float
    reason: str = ""

    @property
    def is_noise(self) -> bool:
        return self.confidence < DEFAULT_CONFIDENCE_THRESHOLD


@dataclass
class FactEntry:
    """A persisted fact with the metadata needed for decay and conflict audit."""

    value: str
    confidence: float = 1.0
    mentions: int = 1
    seen: int = 0  # logical clock of the last mention
    corrections: int = 0  # how many times a different value replaced this one


@dataclass
class ApplyResult:
    accepted: list[FactCandidate] = field(default_factory=list)
    rejected: list[FactCandidate] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)


_FACT_LINE = re.compile(r"^- \*\*(?P<key>.+?)\*\*:\s*(?P<value>.*?)\s*(?:<!--\s*(?P<meta>.*?)\s*-->)?\s*$")
_CLOCK_LINE = re.compile(r"<!--\s*clock=(\d+)\s*-->")


def _parse_meta(meta: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in (meta or "").split():
        name, _, raw = part.partition("=")
        try:
            out[name] = float(raw)
        except ValueError:
            continue
    return out


def effective_score(key: str, entry: FactEntry, clock: int, half_life: float = DEFAULT_HALF_LIFE_TURNS) -> float:
    """Confidence discounted by age; facts mentioned more often age more slowly."""
    if key in PERMANENT_KEYS:
        return entry.confidence
    age = max(0, clock - entry.seen)
    stretched = half_life * (1.0 + math.log(max(1, entry.mentions)))
    return entry.confidence * 0.5 ** (age / stretched)


def _merge_list(old: str, new: str) -> str:
    items = [p.strip() for p in old.split(",") if p.strip()]
    for p in (x.strip() for x in new.split(",")):
        if p and p not in items:
            items.append(p)
    return ", ".join(items[-MAX_LIST_ITEMS:])


@dataclass
class UserProfileStore:
    """Persistent storage for User.md per user."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Sanitize user_id and return path to User.md."""
        safe_id = re.sub(r"[^\w\-.]", "_", user_id)
        return self.root_dir / safe_id / "User.md"

    def read_text(self, user_id: str) -> str:
        """Return file content or default markdown profile."""
        profile_path = self.path_for(user_id)
        if profile_path.exists():
            return profile_path.read_text(encoding="utf-8")
        return f"# User Profile: {user_id}\n\n## Facts\n"

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown to disk and return the file path."""
        profile_path = self.path_for(user_id)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(content, encoding="utf-8")
        return profile_path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace first occurrence inside User.md and return True if modified."""
        content = self.read_text(user_id)
        if search_text in content:
            self.write_text(user_id, content.replace(search_text, replacement, 1))
            return True
        return False

    def file_size(self, user_id: str) -> int:
        """Return the current file size in bytes."""
        profile_path = self.path_for(user_id)
        return profile_path.stat().st_size if profile_path.exists() else 0

    # -- structured access -------------------------------------------------

    def clock(self, user_id: str) -> int:
        match = _CLOCK_LINE.search(self.read_text(user_id))
        return int(match.group(1)) if match else 0

    def get_entries(self, user_id: str) -> dict[str, FactEntry]:
        """Parse facts together with their metadata."""
        entries: dict[str, FactEntry] = {}
        for line in self.read_text(user_id).splitlines():
            m = _FACT_LINE.match(line.strip())
            if not m:
                continue
            meta = _parse_meta(m.group("meta"))
            entries[m.group("key").strip()] = FactEntry(
                value=m.group("value").strip(),
                confidence=meta.get("conf", 1.0),
                mentions=int(meta.get("n", 1)),
                seen=int(meta.get("seen", 0)),
                corrections=int(meta.get("corr", 0)),
            )
        return entries

    def get_facts(self, user_id: str) -> dict[str, str]:
        """Return stored facts as plain key -> value (no decay applied)."""
        return {k: e.value for k, e in self.get_entries(user_id).items()}

    def active_facts(
        self,
        user_id: str,
        half_life: float = DEFAULT_HALF_LIFE_TURNS,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> dict[str, str]:
        """Facts whose decayed score is still high enough to be used in a prompt."""
        clock = self.clock(user_id)
        return {
            k: e.value
            for k, e in self.get_entries(user_id).items()
            if effective_score(k, e, clock, half_life) >= min_score
        }

    def render_for_prompt(
        self,
        user_id: str,
        half_life: float = DEFAULT_HALF_LIFE_TURNS,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> str:
        """Clean markdown (no metadata, no decayed facts) that is injected into the prompt."""
        facts = self.active_facts(user_id, half_life, min_score)
        lines = [f"# User Profile: {user_id}", "", "## Facts"]
        lines += [f"- **{k}**: {v}" for k, v in sorted(facts.items())]
        return "\n".join(lines) + "\n"

    def _write_entries(self, user_id: str, entries: dict[str, FactEntry], clock: int) -> Path:
        lines = [f"# User Profile: {user_id}", f"<!-- clock={clock} -->", "", "## Facts"]
        for key, e in sorted(entries.items()):
            meta = f"conf={e.confidence:.2f} n={e.mentions} seen={e.seen} corr={e.corrections}"
            lines.append(f"- **{key}**: {e.value} <!-- {meta} -->")
        lines.append("")
        return self.write_text(user_id, "\n".join(lines))

    def apply_candidates(
        self,
        user_id: str,
        candidates: list[FactCandidate],
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> ApplyResult:
        """Write-gate: persist only confident facts; a new value replaces the old one.

        Every call advances the user's logical clock by one turn so that decay is
        deterministic (benchmarks and tests do not depend on wall-clock time).
        """
        entries = self.get_entries(user_id)
        clock = self.clock(user_id) + 1
        result = ApplyResult()

        for cand in candidates:
            if cand.confidence < threshold:
                result.rejected.append(cand)
                continue
            result.accepted.append(cand)
            current = entries.get(cand.key)
            if current is None:
                entries[cand.key] = FactEntry(cand.value, cand.confidence, 1, clock, 0)
            elif cand.key in LIST_KEYS:
                current.value = _merge_list(current.value, cand.value)
                current.confidence = max(current.confidence, cand.confidence)
                current.mentions += 1
                current.seen = clock
            elif current.value == cand.value:
                current.confidence = min(1.0, max(current.confidence, cand.confidence) + 0.02)
                current.mentions += 1
                current.seen = clock
            else:
                # Conflict: the newer, confident statement replaces the old fact so the
                # profile never carries two contradictory values at once.
                result.corrections.append(f"{cand.key}: {current.value} -> {cand.value}")
                entries[cand.key] = FactEntry(
                    cand.value, cand.confidence, 1, clock, current.corrections + 1
                )

        self._write_entries(user_id, entries, clock)
        return result

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> Path:
        """Legacy helper: write plain key/value facts with full confidence."""
        cands = [FactCandidate(k, v, 1.0, "manual") for k, v in updates.items()]
        self.apply_candidates(user_id, cands, threshold=0.0)
        return self.path_for(user_id)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

_CITIES = ["Hà Nội", "Huế", "Đà Nẵng", "Sài Gòn", "TP.HCM", "Hải Phòng", "Cần Thơ", "Nha Trang", "Đà Lạt"]
_CAP_WORD = r"[A-ZĐ][^\W\d_]*"
_PLACE = rf"{_CAP_WORD}(?:\s{_CAP_WORD})*"

_LOCATION_NOISE = ("họp", "công tác", "bay ra", "ghé", "du lịch", "chuyến đi", "tạm thời", "ví dụ cũ")
_JOB_NOISE = ("đùa", "giả sử", "nếu như", "ví dụ cũ", "tưởng tượng")
_NEGATION = ("không còn", "không phải", "chứ không", "đừng", "lúc đầu", "trước đây", "ban đầu")
_NOISE_CONFIDENCE = 0.3

_TITLE = re.compile(
    r"(?:làm|chuyển sang|chuyển thành|là|thành)\s+"
    r"((?:[A-Za-z][\w\-]*\s)?(?:engineer|manager|developer|scientist|designer|analyst))\b",
    re.IGNORECASE,
)
_NAME = re.compile(rf"tên (?:mình |tôi )?là\s+({_CAP_WORD}(?:\s{_CAP_WORD})*)")
_LOC_EXPLICIT = re.compile(rf"(?<!\w)(?:(hiện|đang|vẫn|còn)\s+(?:làm việc\s+)?)?ở\s+({_PLACE})")
_LOC_NAMED = re.compile(rf"nơi ở(?: hiện tại)?(?: của mình)? là\s+({_PLACE})")
_DRINK_FAV = re.compile(r"đồ uống yêu thích(?: của mình)? là\s+([^.,;:?!]+)", re.IGNORECASE)
_DRINK_HABIT = re.compile(r"(?:vẫn |hay |thường )uống\s+(.+?)(?:\s+(?:nhưng|như)\b|[.,;:?!]|$)", re.IGNORECASE)
_FOOD_FAV = re.compile(r"món ăn yêu thích(?: của mình)? là\s+([^.,;:?!]+)", re.IGNORECASE)
_PET = re.compile(r"nuôi\s+(?:một\s+)?(?:bé\s+|con\s+)?(corgi|chó|mèo)(?:\s+tên\s+(" + _CAP_WORD + r"))?")
_STYLE_GATE = ("trả lời", "giải thích", "style", "trình bày")
_INTEREST_GATE = ("thích", "quan tâm", "đang học", "ưu tiên")
_TECH_TERMS = [
    ("Python", re.compile(r"\bPython\b")),
    ("AI ứng dụng", re.compile(r"\bAI ứng dụng\b")),
    ("AI agent", re.compile(r"\bAI agent\b")),
    ("AI", re.compile(r"\bAI\b")),  # case-sensitive: the Vietnamese pronoun "ai" is lowercase
    ("MLOps", re.compile(r"\bMLOps\b(?!\s+engineer)")),
    ("RAG", re.compile(r"\bRAG\b")),
]


_QUESTION_CUES = re.compile(r"\b(?:là gì|ở đâu|nào|không nhỉ)\b|^(?:bạn\s+)?(?:thử\s+)?(?:nhắc|nhớ) lại|thử nhớ", re.IGNORECASE)


def _split_sentences(message: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n", message) if s.strip()]


def _is_negated(sentence: str, start: int) -> bool:
    window = sentence[max(0, start - 25):start].lower()
    return any(n in window for n in _NEGATION)


def _extract_sentence(sentence: str) -> list[FactCandidate]:
    lower = sentence.lower()
    out: list[FactCandidate] = []

    # Name
    m = _NAME.search(sentence)
    if m:
        out.append(FactCandidate("Tên", m.group(1).strip(), 0.95, "explicit self-introduction"))

    # Location: keep the last statement that is not negated / in the past
    location_noise = any(n in lower for n in _LOCATION_NOISE)
    loc: FactCandidate | None = None
    for m in _LOC_EXPLICIT.finditer(sentence):
        if _is_negated(sentence, m.start()):
            continue
        conf = 0.9 if m.group(1) else 0.75
        loc = FactCandidate("Nơi ở", m.group(2), _NOISE_CONFIDENCE if location_noise else conf, "ở <place>")
    for m in _LOC_NAMED.finditer(sentence):
        if not _is_negated(sentence, m.start()):
            loc = FactCandidate("Nơi ở", m.group(1), _NOISE_CONFIDENCE if location_noise else 0.95, "nơi ở là <place>")
    if loc is None and location_noise:
        for city in _CITIES:
            if city.lower() in lower and not _is_negated(sentence, lower.index(city.lower())):
                loc = FactCandidate("Nơi ở", city, _NOISE_CONFIDENCE, "city mentioned in a temporary/example context")
                break
    if loc:
        out.append(loc)

    # Profession
    job_noise = any(n in lower for n in _JOB_NOISE)
    job: FactCandidate | None = None
    for m in _TITLE.finditer(sentence):
        if _is_negated(sentence, m.start()):
            continue
        conf = _NOISE_CONFIDENCE if job_noise else 0.9
        job = FactCandidate("Nghề nghiệp", m.group(1).strip(), conf, "job title")
    if job:
        out.append(job)

    # Food / drink / pet
    m = _DRINK_FAV.search(sentence)
    if m:
        out.append(FactCandidate("Đồ uống yêu thích", m.group(1).strip(), 0.95, "explicit favourite"))
    else:
        m = _DRINK_HABIT.search(sentence)
        if m:
            out.append(FactCandidate("Đồ uống yêu thích", m.group(1).strip(), 0.7, "drinking habit"))
    m = _FOOD_FAV.search(sentence)
    if m:
        out.append(FactCandidate("Món ăn yêu thích", m.group(1).strip(), 0.95, "explicit favourite"))
    m = _PET.search(sentence)
    if m:
        value = m.group(1) + (f" tên {m.group(2)}" if m.group(2) else "")
        out.append(FactCandidate("Thú cưng", value, 0.9, "owns a pet"))

    # Answer style preferences (multi-valued)
    if any(g in lower for g in _STYLE_GATE):
        styles: list[str] = []
        if re.search(r"ngắn gọn|\bngắn\b|\bgọn\b", lower):
            styles.append("ngắn gọn")
        if "3 bullet" in lower:
            styles.append("3 bullet")
        elif "bullet" in lower:
            styles.append("bullet")
        if "ví dụ thực chiến" in lower:
            styles.append("ví dụ thực chiến")
        elif "ví dụ thực tế" in lower:
            styles.append("ví dụ thực tế")
        if "trade-off" in lower:
            styles.append("so sánh trade-off")
        if styles:
            out.append(FactCandidate("Style trả lời", ", ".join(styles), 0.85, "style preference"))

    # Interests (lexicon-based entity extraction)
    if any(g in lower for g in _INTEREST_GATE):
        found = [name for name, rx in _TECH_TERMS if rx.search(sentence)]
        if any(f.startswith("AI ") for f in found):
            found = [f for f in found if f != "AI"]
        if found:
            out.append(FactCandidate("Mối quan tâm", ", ".join(found), 0.8, "stated interest"))

    return out


def extract_fact_candidates(message: str) -> list[FactCandidate]:
    """Extract structured fact candidates with a confidence score for each.

    Noise (jokes, business trips, examples) is still returned, but with low
    confidence, so the write-gate in ``UserProfileStore.apply_candidates`` is the
    single place that decides what becomes persistent.
    """
    candidates: list[FactCandidate] = []
    for sentence in _split_sentences(message):
        if sentence.endswith("?") or _QUESTION_CUES.search(sentence):
            continue  # questions / recall requests never state facts
        candidates.extend(_extract_sentence(sentence))
    return candidates


def extract_profile_updates(message: str, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> dict[str, str]:
    """Plain key -> value view of the confident candidates (later statements win)."""
    updates: dict[str, str] = {}
    for cand in extract_fact_candidates(message):
        if cand.confidence < threshold:
            continue
        if cand.key in LIST_KEYS and cand.key in updates:
            updates[cand.key] = _merge_list(updates[cand.key], cand.value)
        else:
            updates[cand.key] = cand.value
    return updates


# ---------------------------------------------------------------------------
# Compact memory
# ---------------------------------------------------------------------------

_OPEN_THREAD_MARKERS = ("sẽ hỏi", "lát nữa", "sau đó mình", "mai mình", "tối nay mình", "sẽ quay lại", "sẽ còn hỏi")


def _topic_of(content: str) -> str:
    first = re.split(r"(?<=[.!?:])\s", content.strip())[0]
    return first if len(first) <= 60 else first[:57] + "..."


def summarize_messages(messages: list[dict[str, str]], max_items: int = 4) -> str:
    """Create a concise, bounded summary of older messages (no dataset-specific rules)."""
    topics: list[str] = []
    for msg in messages:
        content = msg.get("content", "").strip()
        if content and msg.get("role") != "assistant":
            topics.append(_topic_of(content))
    unique = list(dict.fromkeys(topics))[-max_items:]
    return "Tóm tắt: " + "; ".join(unique) if unique else ""


@dataclass
class CompactMemoryManager:
    """Manages short-term history and compacts older messages when threshold is exceeded."""

    threshold_tokens: int
    keep_messages: int
    max_topics: int = 4
    max_open_threads: int = 3
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "topics": [],
                "open_threads": [],
                "compactions": 0,
            }
        return self.state[thread_id]

    @staticmethod
    def _render_summary(topics: list[str], open_threads: list[str]) -> str:
        parts = []
        if topics:
            parts.append("Tóm tắt: " + "; ".join(topics))
        if open_threads:
            parts.append("Việc còn mở: " + "; ".join(open_threads))
        return " | ".join(parts)

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append message and trigger compaction if threshold exceeded."""
        t_state = self._ensure_thread(thread_id)
        messages: list[dict[str, str]] = t_state["messages"]  # type: ignore
        messages.append({"role": role, "content": content})

        total_tokens = estimate_tokens(str(t_state["summary"])) + sum(
            estimate_tokens(m["content"]) for m in messages
        )
        if total_tokens <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return

        older, kept = messages[: -self.keep_messages], messages[-self.keep_messages:]
        topics: list[str] = t_state["topics"]  # type: ignore
        open_threads: list[str] = t_state["open_threads"]  # type: ignore
        for msg in older:
            if msg["role"] == "assistant":
                continue
            topics.append(_topic_of(msg["content"]))
            for sentence in _split_sentences(msg["content"]):
                if any(k in sentence.lower() for k in _OPEN_THREAD_MARKERS):
                    open_threads.append(sentence if len(sentence) <= 70 else sentence[:67] + "...")
        t_state["topics"] = list(dict.fromkeys(topics))[-self.max_topics:]
        t_state["open_threads"] = list(dict.fromkeys(open_threads))[-self.max_open_threads:]
        t_state["summary"] = self._render_summary(t_state["topics"], t_state["open_threads"])  # type: ignore
        t_state["messages"] = kept
        t_state["compactions"] = int(t_state["compactions"]) + 1  # type: ignore

    def context(self, thread_id: str) -> dict[str, object]:
        """Return per-thread state."""
        return self._ensure_thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        """Return number of compactions for this thread."""
        return int(self._ensure_thread(thread_id).get("compactions", 0))
