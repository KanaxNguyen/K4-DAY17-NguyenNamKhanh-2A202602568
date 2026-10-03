from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_fact_candidates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: Advanced Agent.
    
    Equipped with 3 memory layers:
    1. Short-term (in-thread) memory
    2. Persistent memory stored in User.md
    3. Compact memory for long threads
    """

    def __init__(
        self,
        config: LabConfig | None = None,
        force_offline: bool = False,
        confidence_threshold: float | None = None,
    ) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.confidence_threshold = (
            self.config.confidence_threshold if confidence_threshold is None else confidence_threshold
        )
        self.rejected_candidates = 0
        self.corrections: list[str] = []
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None

        if not self.force_offline:
            try:
                self._maybe_build_langchain_agent()
            except Exception:
                self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route to live agent if configured or deterministic offline path."""
        if not self.force_offline and self.langchain_agent is not None:
            try:
                # Live LangChain invocation with injected User.md context
                profile_text = self._profile_prompt(user_id)
                system_prompt = f"Thông tin người dùng từ User.md:\n{profile_text}"
                res = self.langchain_agent.invoke(
                    {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": message},
                        ]
                    },
                    config={"configurable": {"thread_id": thread_id}},
                )
                output_content = str(res.get("messages", [{}])[-1].content)
                agent_tokens = estimate_tokens(output_content)
                self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens

                self._update_profile(user_id, message)

                self.compact_memory.append(thread_id, "user", message)
                self.compact_memory.append(thread_id, "assistant", output_content)

                turn_prompt = self._estimate_prompt_context_tokens(user_id, thread_id)
                self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + turn_prompt

                return {
                    "role": "assistant",
                    "content": output_content,
                    "token_usage": self.thread_tokens[thread_id],
                    "prompt_tokens_processed": self.thread_prompt_tokens[thread_id],
                }
            except Exception:
                pass

        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic advanced path with persistent and compact memory."""
        # 1. Extract profile updates and persist into User.md
        update = self._update_profile(user_id, message)

        # 2. Append incoming message to compact memory (may trigger compaction)
        self.compact_memory.append(thread_id, "user", message)

        # 3. Estimate prompt context load: User.md + summary + kept recent messages
        turn_prompt = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + turn_prompt

        # 4. Generate response using persistent memory facts
        response_text = self._offline_response(user_id, message, update)

        # 5. Append assistant reply to compact memory and track token counters
        self.compact_memory.append(thread_id, "assistant", response_text)
        agent_tokens = estimate_tokens(response_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens

        return {
            "role": "assistant",
            "content": response_text,
            "token_usage": self.thread_tokens[thread_id],
            "prompt_tokens_processed": self.thread_prompt_tokens[thread_id],
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the context carried into one turn."""
        user_md_tokens = estimate_tokens(self._profile_prompt(user_id))
        t_state = self.compact_memory.context(thread_id)
        summary_tokens = estimate_tokens(str(t_state.get("summary", "")))
        messages: list[dict[str, str]] = t_state.get("messages", [])  # type: ignore
        messages_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)
        return user_md_tokens + summary_tokens + messages_tokens

    def _profile_prompt(self, user_id: str) -> str:
        """User.md as injected into the prompt: no metadata, decayed facts removed."""
        return self.profile_store.render_for_prompt(
            user_id, self.config.decay_half_life_turns, self.config.decay_min_score
        )

    def _update_profile(self, user_id: str, message: str):
        """Extract candidates, pass them through the confidence gate, persist to User.md."""
        result = self.profile_store.apply_candidates(
            user_id, extract_fact_candidates(message), self.confidence_threshold
        )
        self.rejected_candidates += len(result.rejected)
        self.corrections.extend(result.corrections)
        return result

    # question keywords -> profile field. Answers are built ONLY from stored facts.
    _FIELD_QUERIES: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("Tên", ("tên",)),
        ("Nơi ở", ("nơi ở", "ở đâu", "còn ở")),
        ("Nghề nghiệp", ("nghề",)),
        ("Đồ uống yêu thích", ("đồ uống", "uống")),
        ("Món ăn yêu thích", ("món ăn",)),
        ("Thú cưng", ("nuôi", "thú cưng", "con gì")),
        ("Style trả lời", ("style", "kiểu trả lời", "trả lời như thế nào")),
        ("Mối quan tâm", ("quan tâm",)),
    )
    _RECALL_CUES = ("nhắc lại", "nhớ lại", "tóm tắt", "là ai", "biết")

    def _offline_response(self, user_id: str, message: str, update=None) -> str:
        """Deterministic answer built strictly from persisted facts (never from defaults)."""
        lower = message.lower()
        is_question = message.strip().endswith("?") or any(c in lower for c in self._RECALL_CUES)
        facts = self.profile_store.active_facts(
            user_id, self.config.decay_half_life_turns, self.config.decay_min_score
        )

        wanted = [
            key
            for key, kws in self._FIELD_QUERIES
            if any(re.search(rf"(?<!\w){re.escape(k)}(?!\w)", lower) for k in kws)
        ]
        if is_question and ("là ai" in lower or "tóm tắt" in lower):
            wanted += [k for k, _ in self._FIELD_QUERIES if k not in wanted and k in ("Tên", "Nghề nghiệp", "Mối quan tâm")]

        if is_question and wanted:
            lines = []
            for key in wanted:
                if key in facts:
                    lines.append(f"- {key}: {facts[key]}")
                else:
                    lines.append(f"- {key}: chưa có thông tin trong User.md")
            return "Theo User.md:\n" + "\n".join(lines)

        # Plain statement: acknowledge what was actually persisted / rejected.
        if update is not None and update.corrections:
            return "Đã cập nhật: " + "; ".join(update.corrections)
        if update is not None and update.accepted:
            saved = ", ".join(f"{c.key}={c.value}" for c in update.accepted[:3])
            return f"Đã ghi nhận: {saved}"
        if update is not None and update.rejected:
            return "Mình bỏ qua chi tiết này vì chưa đủ chắc chắn đó là thông tin ổn định."
        return "Đã nghe, mình giữ ngữ cảnh gần nhất."

    def _maybe_build_langchain_agent(self):
        """Optionally wire live LangChain / LangGraph agent."""
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.prebuilt import create_react_agent

        model = build_chat_model(self.config.model)
        checkpointer = MemorySaver()
        self.langchain_agent = create_react_agent(
            model=model,
            tools=[],
            checkpointer=checkpointer,
        )
