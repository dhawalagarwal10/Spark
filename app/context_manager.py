import json
import sqlite3
from typing import Dict, Any, List
from datetime import datetime, timedelta

class ContextManager:
    """
    Advanced context management system
    - Maintains conversation context across sessions
    - Manages context relevance and decay
    - Provides context summarization
    - Handles multi-user context isolation
    """
    
    def __init__(self, db_path: str = "db/context.db"):
        self.db_path = db_path
        self._init_db()
        self.context_cache = {}
        self.relevance_threshold = 0.3
    
    def _init_db(self):
        """Initialize context database"""
        import os
        os.makedirs("db", exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS context_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                context_type TEXT,
                context_data TEXT,
                relevance_score REAL,
                timestamp TEXT,
                expires_at TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def update_context(self, session_id: str, context_type: str, 
                      context_data: Dict[str, Any], relevance_score: float = 1.0):
        """Update context for a session"""
        expires_at = (datetime.now() + timedelta(hours=24)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO context_sessions 
            (session_id, context_type, context_data, relevance_score, timestamp, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, context_type, json.dumps(context_data), 
              relevance_score, datetime.now().isoformat(), expires_at))
        conn.commit()
        conn.close()
        
        # Update cache
        if session_id not in self.context_cache:
            self.context_cache[session_id] = {}
        self.context_cache[session_id][context_type] = context_data
    
    def get_context(self, session_id: str, context_types: List[str] = None) -> Dict[str, Any]:
        """Retrieve context for a session"""
        # Check cache first
        if session_id in self.context_cache:
            if context_types:
                return {ct: self.context_cache[session_id].get(ct, {}) 
                       for ct in context_types}
            return self.context_cache[session_id]
        
        # Query database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT context_type, context_data, relevance_score 
            FROM context_sessions 
            WHERE session_id = ? AND datetime(expires_at) > datetime('now')
            AND relevance_score > ?
        """
        params = [session_id, self.relevance_threshold]
        
        if context_types:
            placeholders = ','.join('?' * len(context_types))
            query += f" AND context_type IN ({placeholders})"
            params.extend(context_types)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        context = {}
        for context_type, context_data, relevance_score in results:
            context[context_type] = json.loads(context_data)
        
        return context
