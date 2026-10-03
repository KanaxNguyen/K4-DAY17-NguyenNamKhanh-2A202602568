from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for testing with small compact threshold."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    dummy_model = ProviderConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.0,
    )

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=50,
        compact_keep_messages=2,
        model=dummy_model,
        judge_model=dummy_model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify User.md can be created, updated, and edited."""
    profiles_dir = tmp_path / "profiles"
    store = UserProfileStore(profiles_dir)

    user_id = "test_user"
    init_content = "# User Profile: test_user\n\n- **Tên**: Nam Khanh\n"
    written_path = store.write_text(user_id, init_content)

    assert written_path.exists()
    assert store.read_text(user_id) == init_content
    assert store.file_size(user_id) > 0

    # Test editing
    edited = store.edit_text(user_id, "Nam Khanh", "DũngCT")
    assert edited is True
    assert "DũngCT" in store.read_text(user_id)

    # Test upsert_facts
    store.upsert_facts(user_id, {"Nơi ở": "Huế", "Nghề nghiệp": "MLOps engineer"})
    facts = store.get_facts(user_id)
    assert facts.get("Nơi ở") == "Huế"
    assert facts.get("Nghề nghiệp") == "MLOps engineer"


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)

    thread_id = "test_compact_thread"
    for i in range(6):
        msg = f"Tin nhắn số {i}: nội dung hội thoại dài nhằm làm tăng lượng token và kích hoạt compact memory."
        agent.reply("user_compact", thread_id, msg)

    assert agent.compaction_count(thread_id) > 0
    t_context = agent.compact_memory.context(thread_id)
    assert t_context.get("summary") != ""


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions/threads and baseline does not."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    user_id = "dungct_recall_test"
    turn_1 = "Chào bạn, mình tên là DũngCT và đồ uống yêu thích là cà phê sữa đá."

    # Thread 1: provide facts
    baseline.reply(user_id, "thread_1", turn_1)
    advanced.reply(user_id, "thread_1", turn_1)

    # Thread 2: ask recall question in fresh thread
    recall_q = "Mình tên gì và đồ uống yêu thích là gì?"
    base_res = baseline.reply(user_id, "thread_2", recall_q)
    adv_res = advanced.reply(user_id, "thread_2", recall_q)

    # Baseline forgot across threads
    assert "DũngCT" not in base_res["content"] or "chưa có thông tin" in base_res["content"]

    # Advanced remembered across threads from User.md
    assert "DũngCT" in adv_res["content"]
    assert "cà phê sữa đá" in adv_res["content"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    thread_id = "long_thread_comparison"
    user_id = "stress_test_user"

    for i in range(8):
        msg = (
            f"Bản tin số {i}: NASA Artemis III cùng máy bay siêu thanh X-59 và cảnh báo El Nino từ WMO. "
            f"Dự báo nhu cầu năng lượng của British Columbia tăng trưởng mạnh vào năm 2030."
        )
        baseline.reply(user_id, thread_id, msg)
        advanced.reply(user_id, thread_id, msg)

    base_prompt_load = baseline.prompt_token_usage(thread_id)
    adv_prompt_load = advanced.prompt_token_usage(thread_id)

    # Advanced compact memory must significantly reduce the prompt load compared to raw baseline accumulation
    assert adv_prompt_load < base_prompt_load


def test_noise_filtering_and_correction(tmp_path: Path) -> None:
    """Verify agent filters noise (jokes, meeting trips) and accepts true corrections."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)
    user_id = "test_noise_user"

    # Initial facts
    agent.reply(user_id, "t1", "Chào bạn, mình tên là DũngCT, ở Huế và làm backend engineer.")
    facts_1 = agent.profile_store.get_facts(user_id)
    assert facts_1.get("Nơi ở") == "Huế"
    assert facts_1.get("Nghề nghiệp") == "backend engineer"

    # Correction: moved to Đà Nẵng, changed to MLOps
    agent.reply(user_id, "t1", "Mình đính chính nhé: giờ mình đang làm việc ở Đà Nẵng và chuyển sang MLOps engineer.")
    facts_2 = agent.profile_store.get_facts(user_id)
    assert facts_2.get("Nơi ở") == "Đà Nẵng"
    assert facts_2.get("Nghề nghiệp") == "MLOps engineer"

    # Noise: joke about product manager and trip to Hà Nội
    agent.reply(
        user_id,
        "t1",
        "Có lúc mình đùa chuyển sang product manager, nhưng đó chỉ là câu đùa. Hà Nội chỉ là nơi mình vừa bay ra họp 2 ngày.",
    )
    facts_3 = agent.profile_store.get_facts(user_id)
    # Must NOT change to product manager or Hà Nội
    assert facts_3.get("Nơi ở") == "Đà Nẵng"
    assert facts_3.get("Nghề nghiệp") == "MLOps engineer"


def test_provider_normalization() -> None:
    """Verify provider aliases are normalized properly."""
    from model_provider import normalize_provider

    assert normalize_provider("anthorpic") == "anthropic"
    assert normalize_provider("google-genai") == "gemini"
    assert normalize_provider("open_router") == "openrouter"
    assert normalize_provider("openai-compatible") == "custom"
    assert normalize_provider("local") == "ollama"
    assert normalize_provider("chatgpt") == "openai"



# ---------------------------------------------------------------------------
# Guardrail / bonus tests: confidence, conflict, decay, entity extraction
# ---------------------------------------------------------------------------

from memory_store import (  # noqa: E402
    CompactMemoryManager,
    extract_fact_candidates,
    extract_profile_updates,
    estimate_tokens,
)


def test_baseline_forgets_in_new_thread(tmp_path: Path) -> None:
    """Baseline remembers inside a thread but must know nothing in a fresh one."""
    baseline = BaselineAgent(config=make_config(tmp_path), force_offline=True)
    baseline.reply("u", "t1", "Chào bạn, mình tên là DũngCT.")
    fresh = baseline.reply("u", "t2", "Mình tên gì?")["content"]
    assert "DũngCT" not in fresh
    assert len(baseline.sessions["t2"].messages) == 2


def test_advanced_does_not_fabricate_unknown_facts(tmp_path: Path) -> None:
    """With an empty User.md the agent must say it does not know, not guess a default."""
    agent = AdvancedAgent(config=make_config(tmp_path), force_offline=True)
    answer = agent.reply("brand_new_user", "t1", "Mình tên gì và ở đâu?")["content"]
    assert "chưa có thông tin" in answer
    assert "DũngCT" not in answer and "Huế" not in answer


def test_confidence_threshold_blocks_noise(tmp_path: Path) -> None:
    """A joke and a business trip are extracted as low-confidence and never persisted."""
    noisy = "Mình đùa là chuyển sang product manager. Hà Nội chỉ là nơi mình bay ra họp 2 ngày."
    cands = extract_fact_candidates(noisy)
    assert cands and all(c.confidence < 0.6 for c in cands)
    assert extract_profile_updates(noisy) == {}

    gated = AdvancedAgent(config=make_config(tmp_path / "a"), force_offline=True)
    gated.reply("u", "t", "Mình ở Huế và đang làm MLOps engineer.")
    gated.reply("u", "t", noisy)
    facts = gated.profile_store.get_facts("u")
    assert facts["Nơi ở"] == "Huế" and facts["Nghề nghiệp"] == "MLOps engineer"
    assert gated.rejected_candidates >= 2

    # Without the gate the same noise overwrites correct facts: this is the risk the gate removes.
    ungated = AdvancedAgent(config=make_config(tmp_path / "b"), force_offline=True, confidence_threshold=0.0)
    ungated.reply("u", "t", "Mình ở Huế và đang làm MLOps engineer.")
    ungated.reply("u", "t", noisy)
    assert ungated.profile_store.get_facts("u")["Nơi ở"] == "Hà Nội"


def test_conflict_handling_keeps_single_current_fact(tmp_path: Path) -> None:
    """A correction replaces the old value; User.md never holds both."""
    agent = AdvancedAgent(config=make_config(tmp_path), force_offline=True)
    agent.reply("u", "t", "Mình đang ở Đà Nẵng.")
    agent.reply("u", "t", "Mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa.")
    text = agent.profile_store.read_text("u")
    assert text.count("**Nơi ở**") == 1
    assert "Huế" in text and "Đà Nẵng" not in text
    assert agent.profile_store.get_entries("u")["Nơi ở"].corrections == 1
    assert any("Đà Nẵng -> Huế" in c for c in agent.corrections)


def test_negated_old_value_is_not_stored() -> None:
    updates = extract_profile_updates("Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")
    assert updates["Nghề nghiệp"] == "MLOps engineer"
    updates = extract_profile_updates("Bạn nhớ là mình làm MLOps engineer chứ không còn là backend engineer nữa nhé.")
    assert updates["Nghề nghiệp"] == "MLOps engineer"


def test_questions_do_not_write_facts() -> None:
    assert extract_profile_updates("Bạn thử nhớ lại xem đồ uống yêu thích của mình là gì.") == {}
    assert extract_profile_updates("Mình tên gì và ở đâu?") == {}


def test_entity_extraction_structured_and_pronoun_disambiguation() -> None:
    facts = extract_profile_updates("Mình thích Python, AI ứng dụng và cà phê sữa đá.")
    assert facts["Mối quan tâm"] == "Python, AI ứng dụng"
    # Vietnamese pronoun "ai" (who) must not be read as the AI topic.
    assert "Mối quan tâm" not in extract_profile_updates("Mình thích hỏi xem ai đó nhớ gì.")
    pet = extract_profile_updates("Mình nuôi một bé corgi tên Bơ.")
    assert pet["Thú cưng"] == "corgi tên Bơ"


def test_memory_decay_drops_stale_facts_but_keeps_identity(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    store.apply_candidates("u", extract_fact_candidates("Mình tên là Lan. Mình thích ăn mì Quảng. Mình vẫn ở Huế."))
    # 40 unrelated turns later, with a short half-life, the unrepeated facts fade.
    for _ in range(40):
        store.apply_candidates("u", [])
    active = store.active_facts("u", half_life=5, min_score=0.25)
    assert active == {"Tên": "Lan"}
    assert "Huế" in store.get_facts("u").values()  # still on disk for audit, just not prompted
    # Repetition slows decay: a fact mentioned often survives the same gap.
    store2 = UserProfileStore(tmp_path / "profiles2")
    for _ in range(8):
        store2.apply_candidates("u", extract_fact_candidates("Mình vẫn ở Huế."))
    for _ in range(12):
        store2.apply_candidates("u", [])
    assert "Nơi ở" in store2.active_facts("u", half_life=5, min_score=0.25)


def test_prompt_view_is_smaller_than_file(tmp_path: Path) -> None:
    """Metadata lives on disk only; the prompt gets the clean view."""
    store = UserProfileStore(tmp_path / "profiles")
    store.apply_candidates("u", extract_fact_candidates("Mình tên là Lan. Mình ở Huế."))
    assert "<!--" in store.read_text("u")
    prompt = store.render_for_prompt("u")
    assert "<!--" not in prompt and estimate_tokens(prompt) < estimate_tokens(store.read_text("u"))


def test_compact_summary_is_bounded_and_keeps_open_threads(tmp_path: Path) -> None:
    mgr = CompactMemoryManager(threshold_tokens=40, keep_messages=2)
    for i in range(12):
        mgr.append("t", "user", f"Chủ đề số {i} khá dài để vượt ngưỡng token của compact memory.")
    mgr.append("t", "user", "Lát nữa mình sẽ hỏi lại về nơi ở hiện tại. Cảm ơn nhé.")
    for i in range(4):
        mgr.append("t", "user", f"Thêm một đoạn dài số {i} để ép compact thêm lần nữa trong thread này.")
    ctx = mgr.context("t")
    assert len(ctx["topics"]) <= 4  # bounded, does not grow forever
    assert "Việc còn mở" in ctx["summary"] and "sẽ hỏi lại" in ctx["summary"]
    assert len(ctx["messages"]) <= 3


def test_short_conversation_compact_does_not_win(tmp_path: Path) -> None:
    """On a short thread no compaction fires and User.md is pure overhead."""
    config = make_config(tmp_path)
    config.compact_threshold_tokens = 600
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)
    for turn in ("Chào bạn, mình tên là DũngCT.", "Mình ở Huế.", "Mình thích Python."):
        baseline.reply("u", "short", turn)
        advanced.reply("u", "short", turn)
    assert advanced.compaction_count("short") == 0
    assert advanced.prompt_token_usage("short") > baseline.prompt_token_usage("short")


def test_recall_points_is_boundary_and_acronym_aware() -> None:
    from benchmark import recall_points

    assert recall_points("Bạn biết ai không?", ["AI"]) == 0.0
    assert recall_points("Mình thích AI ứng dụng", ["AI"]) == 1.0
    assert recall_points("hai ngày", ["ai"]) == 0.0
