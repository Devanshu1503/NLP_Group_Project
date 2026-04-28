"""
Agent 1: Multi-turn conversation manager.
Collects patient info through natural dialogue and signals when enough
data has been gathered to run the retrieval pipeline.
"""
import os
from anthropic import Anthropic
from ner.schemas import PatientProfile
from ner.ner_llm import extract_profile_from_conversation

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

_SYSTEM = """You are TrialNav, a compassionate assistant that helps patients find clinical trials.

Your job:
1. Warmly greet the patient and ask them to describe their health situation
2. Ask follow-up questions ONE AT A TIME to fill gaps in: age, gender, conditions, medications, location
3. Use plain language — no medical jargon
4. Never make diagnoses or medical recommendations
5. Never tell a patient they ARE eligible — only that they MAY qualify
6. Once you have age, gender, at least one condition, and location — end your message with exactly: [PROFILE_COMPLETE]

Always be warm, clear, and encouraging."""


class ConversationAgent:
    def __init__(self):
        self.messages: list = []
        self.profile_complete: bool = False

    def chat(self, user_message: str) -> tuple[str, bool]:
        """
        Process one conversation turn.

        Returns:
            (assistant_response, profile_complete)
        """
        self.messages.append({"role": "user", "content": user_message})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=self.messages,
        )

        assistant_text = response.content[0].text
        profile_complete = "[PROFILE_COMPLETE]" in assistant_text
        clean_response = assistant_text.replace("[PROFILE_COMPLETE]", "").strip()

        self.messages.append({"role": "assistant", "content": assistant_text})

        if profile_complete:
            self.profile_complete = True

        return clean_response, profile_complete

    def extract_profile(self) -> PatientProfile:
        """Extract structured profile from full conversation history."""
        return extract_profile_from_conversation(self.messages)

    def reset(self):
        """Reset for a new patient session."""
        self.messages = []
        self.profile_complete = False
