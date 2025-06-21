import sqlite3
import os
from datetime import datetime

DB_PATH = "db/goals.db"

def _init_goals_db():
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            evaluation TEXT,
            relevant_response TEXT
        )
    """)
    conn.commit()
    conn.close()

_init_goals_db()

class GoalManager:
    """
    A class to manage goal tracking, evaluation, and storage.
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        # Ensure database is initialized
        _init_goals_db()
    
    async def evaluate_goals(self, response, context: dict = None):
        """
        Evaluate if any internal goals have been mentioned, modified, or achieved.
        """
        # Extract content from response object if it has a content attribute
        response_text = getattr(response, 'content', str(response))
        evaluation = self.analyze_goal_impact(response_text, context)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO goals (timestamp, evaluation, relevant_response)
            VALUES (?, ?, ?)
        """, (datetime.now().isoformat(), evaluation, response_text))
        conn.commit()
        conn.close()
    
    def analyze_goal_impact(self, response: str, context: dict = None) -> str:
        """
        Simple rule-based analysis – replace later with LLM-based reasoning if needed.
        Context can provide additional information for goal analysis.
        """
        response_lower = response.lower()
        
        # Enhanced analysis with context if available
        if context:
            # You can add context-aware analysis here
            # For example, check if context contains goal-related information
            pass
        
        if "goal achieved" in response_lower:
            return "Detected completion of a goal."
        elif "I will try" in response or "my objective is" in response:
            return "Detected goal formulation or planning."
        elif "working towards" in response_lower or "progress on" in response_lower:
            return "Detected goal progress tracking."
        elif "new goal" in response_lower or "setting a goal" in response_lower:
            return "Detected new goal creation."
        return "No major goal-related update."
    
    def get_all_goals(self):
        """Retrieve all goal evaluations from the database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM goals ORDER BY timestamp DESC")
        goals = c.fetchall()
        conn.close()
        return goals
    
    def get_recent_goals(self, limit: int = 10):
        """Get the most recent goal evaluations."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM goals ORDER BY timestamp DESC LIMIT ?", (limit,))
        goals = c.fetchall()
        conn.close()
        return goals
    
    def get_goals_by_type(self, evaluation_type: str):
        """Get goals filtered by evaluation type (e.g., 'completion', 'formulation')."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM goals WHERE evaluation LIKE ? ORDER BY timestamp DESC", (f"%{evaluation_type}%",))
        goals = c.fetchall()
        conn.close()
        return goals
    
    def clear_goals(self):
        """Clear all goal records from the database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM goals")
        conn.commit()
        conn.close()

async def evaluate_goals(response, context: dict = None):
    """Legacy function wrapper for backward compatibility."""
    goal_manager = GoalManager()
    await goal_manager.evaluate_goals(response, context)

def analyze_goal_impact(response: str, context: dict = None) -> str:
    """Legacy function wrapper for backward compatibility."""
    goal_manager = GoalManager()
    return goal_manager.analyze_goal_impact(response, context)