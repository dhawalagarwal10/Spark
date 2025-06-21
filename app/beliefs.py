import sqlite3
import os
from datetime import datetime

DB_PATH = "db/beliefs.db"

def _init_belief_db():
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS beliefs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_input TEXT,
            response TEXT,
            belief_summary TEXT
        )
    """)
    conn.commit()
    conn.close()

_init_belief_db()

class BeliefSystem:
    """
    A class to manage belief tracking and storage.
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        # Ensure database is initialized
        _init_belief_db()
    
    async def update_beliefs(self, user_input: str, response, context: dict = None):
        """Update beliefs based on user input and response."""
        # Extract content from response object if it has a content attribute
        response_text = getattr(response, 'content', str(response))
        belief_summary = self.extract_beliefs_from_response(response_text, context)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO beliefs (timestamp, user_input, response, belief_summary)
            VALUES (?, ?, ?, ?)
        """, (datetime.now().isoformat(), user_input, response_text, belief_summary))
        conn.commit()
        conn.close()
    
    def extract_beliefs_from_response(self, response: str, context: dict = None) -> str:
        """
        Very basic belief extractor – can be replaced with a more intelligent LLM-based pattern extractor.
        Context can provide additional information for belief extraction.
        """
        # Example heuristic: anything stated confidently becomes a belief
        lines = response.splitlines()
        beliefs = [line for line in lines if "I believe" in line or "It is true that" in line]
        
        # Enhanced belief detection with context if available
        if context:
            # You can add context-aware belief extraction here
            # For example, check if context contains belief-related information
            pass
        
        # Additional belief patterns
        belief_patterns = [
            "I think that", "In my opinion", "I'm convinced that", 
            "It seems clear that", "I understand that", "My view is"
        ]
        
        for pattern in belief_patterns:
            beliefs.extend([line for line in lines if pattern in line])
        
        # Remove duplicates while preserving order
        beliefs = list(dict.fromkeys(beliefs))
        
        return "\n".join(beliefs) if beliefs else "No explicit belief found."
    
    def get_all_beliefs(self):
        """Retrieve all beliefs from the database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM beliefs ORDER BY timestamp DESC")
        beliefs = c.fetchall()
        conn.close()
        return beliefs
    
    def get_recent_beliefs(self, limit: int = 10):
        """Get the most recent beliefs."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM beliefs ORDER BY timestamp DESC LIMIT ?", (limit,))
        beliefs = c.fetchall()
        conn.close()
        return beliefs

async def update_beliefs(user_input: str, response, context: dict = None):
    """Legacy function wrapper for backward compatibility."""
    belief_system = BeliefSystem()
    await belief_system.update_beliefs(user_input, response, context)

def extract_beliefs_from_response(response: str, context: dict = None) -> str:
    """Legacy function wrapper for backward compatibility."""
    belief_system = BeliefSystem()
    return belief_system.extract_beliefs_from_response(response, context)