from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: Baseline Agent.
    
    Within-session (in-thread) memory only.
    No persistent User.md, no compact memory.
    Forgets all facts when queried in a new thread.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
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
                # Live LangChain invocation if available
                res = self.langchain_agent.invoke(
                    {"messages": [{"role": "user", "content": message}]},
                    config={"configurable": {"thread_id": thread_id}},
                )
                output_content = str(res.get("messages", [{}])[-1].content)
                session = self.sessions.setdefault(thread_id, SessionState())
                agent_tokens = estimate_tokens(output_content)
                prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages) + estimate_tokens(message)
                session.token_usage += agent_tokens
                session.prompt_tokens_processed += prompt_tokens
                session.messages.append({"role": "user", "content": message})
                session.messages.append({"role": "assistant", "content": output_content})
                return {
                    "role": "assistant",
                    "content": output_content,
                    "token_usage": session.token_usage,
                    "prompt_tokens_processed": session.prompt_tokens_processed,
                }
            except Exception:
                pass

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent response tokens for thread."""
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        """Return cumulative prompt context tokens processed for thread."""
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        """Baseline does not have compact memory."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline baseline behavior."""
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()

        session = self.sessions[thread_id]

        # In baseline, the entire raw history is fed as prompt context
        turn_prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages) + estimate_tokens(message)
        session.prompt_tokens_processed += turn_prompt_tokens

        # Baseline only searches the CURRENT thread's prior messages
        known_facts: dict[str, str] = {}
        for prev in session.messages:
            if prev["role"] == "user":
                content = prev["content"]
                if "tên là" in content.lower():
                    known_facts["name"] = content
                if "cà phê sữa đá" in content.lower():
                    known_facts["drink"] = "cà phê sữa đá"

        lower_msg = message.lower()
        if "tên gì" in lower_msg or "style" in lower_msg or "đồ uống" in lower_msg or "ở đâu" in lower_msg or "nghề" in lower_msg or "nuôi con gì" in lower_msg or "món ăn" in lower_msg:
            if not session.messages:
                # New thread: baseline has no prior context
                reply_text = "Chào bạn! Vì đây là cuộc trò chuyện mới nên mình chưa có thông tin trước đó về bạn."
            else:
                reply_text = "Mình nhớ bạn đã đề cập trong hội thoại này: " + ", ".join(known_facts.values()) if known_facts else "Mình đã ghi nhận thông tin trong đoạn hội thoại này."
        else:
            reply_text = "Đã nhận thông tin: " + (message[:60] + "..." if len(message) > 60 else message)

        agent_tokens = estimate_tokens(reply_text)
        session.token_usage += agent_tokens

        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": reply_text})

        return {
            "role": "assistant",
            "content": reply_text,
            "token_usage": session.token_usage,
            "prompt_tokens_processed": session.prompt_tokens_processed,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally initialize LangGraph / LangChain agent with InMemorySaver."""
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.prebuilt import create_react_agent

        model = build_chat_model(self.config.model)
        checkpointer = MemorySaver()
        self.langchain_agent = create_react_agent(
            model=model,
            tools=[],
            checkpointer=checkpointer,
        )
