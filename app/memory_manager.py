import logging
from typing import List, Tuple, Dict
from datetime import datetime
import uuid

import memory  # low-level memory
from behavior_profile import get_behavioral_summary, update_behavioral_traits

logger = logging.getLogger(__name__)

# === Configuration ===
DEFAULT_CONTEXT_SIZE = 5
EPISODE_LOG_PATH = "data/episodes.log"
BEHAVIOR_PROFILE_PATH = "data/behavior_profile.json"

# === Runtime ===
CURRENT_SESSION_ID = str(uuid.uuid4())


# === High-Level Memory Management ===

def add_to_memory(user_input: str, spark_output: str):
    """
    Saves the conversation to long-term memory, logs to session, and updates behavior traits.
    """
    try:
        timestamp = datetime.utcnow().isoformat()

        # 1. Save to persistent memory
        memory.save_to_memory(user_input, spark_output)

        # 2. Append to episodic session log
        with open(EPISODE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}][Session: {CURRENT_SESSION_ID}]\nUser: {user_input}\nSpark: {spark_output}\n\n")

        # 3. Update behavioral traits from this interaction
        update_behavioral_traits(user_input, spark_output)

        logger.info("Memory, episode, and behavior saved successfully.")
    except Exception as e:
        logger.error(f"Error in memory saving: {e}")


def recall_context(query: str, k: int = DEFAULT_CONTEXT_SIZE) -> str:
    """
    Retrieves the most relevant memories to the current query.
    """
    try:
        pairs = memory.retrieve_context(query, k=k)
        return memory.format_context(pairs)
    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        return ""


def summarize_recent_memories(n: int = 10) -> str:
    """
    Summarizes the last n memories into a digestible form.
    """
    try:
        pairs = memory.retrieve_context("recent summary", k=n)
        return "\n".join([f"- {u.strip()} → {s.strip()}" for u, s in pairs])
    except Exception as e:
        logger.warning(f"Memory summary failed: {e}")
        return ""


def inject_memory_context(user_input: str, base_prompt: str) -> str:
    """
    Builds a full prompt using:
    - long-term memory context
    - behavioral profile
    - user input
    """
    context = recall_context(user_input)
    behavior_summary = get_behavioral_summary()

    full_prompt = (
        f"{base_prompt}\n"
        f"\n[BEHAVIORAL TRAITS]\n{behavior_summary}"
        f"\n[MEMORY CONTEXT]\n{context}"
        f"\nUser: {user_input}\nSpark:"
    )

    return full_prompt.strip()


def clear_all_memory():
    """
    Clears all memory and behavioral logs.
    """
    try:
        memory.clear_memory()
        open(EPISODE_LOG_PATH, "w").close()
        open(BEHAVIOR_PROFILE_PATH, "w").write("{}")
        logger.info("All memory and behavior logs cleared.")
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
