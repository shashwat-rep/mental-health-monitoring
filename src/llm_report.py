"""
ml/src/llm_report.py

Phase 4 — LLM report generation via Groq, chat-mode.
Receives ONLY the structured JSON payload from model.py (+ risk_flags /
history_trend, computed by the backend) — never raw user text beyond the
current conversation's own messages, which the backend passes explicitly.
"""

import os
import json
from typing import Dict, List, Optional

from groq import Groq

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_REASONING_EFFORT = "medium"

# Severity-band based, NOT a suicidal-ideation detector — the model was
# never trained on that signal (DAIC-WOZ excludes PHQ-9's ideation item).
CRISIS_FLAGS = {"severe_symptom_range"}

CRISIS_RESOURCE_BLOCK = """
If you are in the US: you can call or text 988 (Suicide & Crisis Lifeline), available 24/7.
If you are outside the US: findahelpline.com lists crisis lines by country.
If you are in immediate danger, please contact local emergency services.
""".strip()

CHAT_SYSTEM_PROMPT_TEMPLATE = """You are a supportive mental health check-in assistant having an ongoing
conversation. You will be given a structured summary of the person's most
recent message, produced by a separate classifier — you will NOT see raw
analysis details beyond what's listed here.

Guidelines:
- Do not diagnose. Do not claim certainty about what the person is experiencing.
- Speak in plain, warm, second-person language, not clinical jargon.
- Respond conversationally, building on what's been said earlier in this chat.
- If mh_category indicates depression signals, gently normalize seeking support without being alarmist.
- Offer small, concrete suggestions only when they fit naturally — don't force advice into every reply.
- Keep replies short — a few sentences, like a real conversation, not an essay.
- Never mention "the classifier," "the model," "JSON," or any internal system detail.
- Do not offer medical or diagnostic advice, and do not tell the person to stop any treatment or medication.

Structured signals from their most recent message:
{payload_json}
"""


def _relevant_payload(payload: Dict) -> Dict:
    return {
        "primary_emotion": payload.get("emotion"),
        "emotion_confidence": payload.get("emotion_confidence"),
        "active_emotions": payload.get("active_emotions"),
        "mh_category": payload.get("mh_category"),
        "severity_score_0_to_1": payload.get("severity_score"),
        "severity_phq8_equivalent": payload.get("severity_phq8_equivalent"),
        "history_trend": payload.get("history_trend", "no_history_available"),
    }


class ReportGenerator:
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set. Add it to backend/.env.")
        self.client = Groq(api_key=api_key)

    def _run_completion(self, messages: List[Dict]) -> str:
        completion = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            reasoning_effort=GROQ_REASONING_EFFORT,
            temperature=0.7,
            max_tokens=1024,
        )
        message = completion.choices[0].message
        text = (message.content or "").strip()
        if not text:
            fallback = getattr(message, "reasoning", None)
            text = fallback.strip() if fallback else (
                "I wasn't able to generate a reply just now — please try again."
            )
        return text

    def generate_chat_reply(self, payload: Dict, recent_messages: List) -> Dict:
        """
        recent_messages: ORM Message objects (oldest -> newest) from the
        current conversation, INCLUDING the just-saved user message as the
        last entry. Each needs .role (MessageRole) and .text.
        """
        risk_flags: List[str] = payload.get("risk_flags", [])
        is_crisis = bool(CRISIS_FLAGS.intersection(risk_flags))

        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            payload_json=json.dumps(_relevant_payload(payload), indent=2)
        )
        messages = [{"role": "system", "content": system_prompt}]
        for m in recent_messages:
            role = "user" if m.role.value == "user" else "assistant"
            messages.append({"role": role, "content": m.text})

        report_text = self._run_completion(messages)

        result = {"report": report_text, "crisis_flagged": is_crisis}
        if is_crisis:
            result["crisis_resources"] = CRISIS_RESOURCE_BLOCK
        return result


if __name__ == "__main__":
    class _FakeMsg:
        def __init__(self, role, text):
            class R:
                value = role
            self.role = R()
            self.text = text

    gen = ReportGenerator()
    sample_payload = {
        "emotion": "sadness", "emotion_confidence": 0.86,
        "active_emotions": ["disappointment", "nervousness", "sadness"],
        "mh_category": "depression", "severity_score": 0.46,
        "severity_phq8_equivalent": 11.1, "risk_flags": [], "history_trend": "first_entry",
    }
    history = [_FakeMsg("user", "I haven't been sleeping well and nothing feels worth doing anymore.")]
    print(json.dumps(gen.generate_chat_reply(sample_payload, history), indent=2))