import sqlite3
import os
import json
import pickle
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from collections import defaultdict, deque
import threading
import hashlib
import re

class MemoryType(Enum):
    EPISODIC = "episodic"          # Specific conversations/events
    SEMANTIC = "semantic"          # Facts and knowledge
    PROCEDURAL = "procedural"      # How to do things
    PERSONAL = "personal"          # Personal preferences/traits
    EMOTIONAL = "emotional"        # Emotional associations
    WORKING = "working"            # Short-term active memory

class MemoryImportance(Enum):
    CRITICAL = 5    # Never forget
    HIGH = 4        # Important long-term
    MEDIUM = 3      # Standard retention
    LOW = 2         # Can be forgotten
    TEMPORARY = 1   # Short-term only

@dataclass
class MemoryItem:
    id: str
    content: str
    memory_type: MemoryType
    importance: MemoryImportance
    timestamp: datetime
    context_tags: List[str]
    emotional_weight: float
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    linked_memories: List[str] = None
    decay_rate: float = 0.1
    consolidation_score: float = 0.0

    def __post_init__(self):
        if self.linked_memories is None:
            self.linked_memories = []

class SQLiteMemory:
    """Long-term persistent memory storage"""
    
    def __init__(self, db_path: str = "data/memory.sqlite"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self._initialize_tables()

    def _initialize_tables(self):
        with self.lock:
            cursor = self.conn.cursor()
            
            # Main memory table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    context_tags TEXT,
                    emotional_weight REAL DEFAULT 0.0,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    linked_memories TEXT,
                    decay_rate REAL DEFAULT 0.1,
                    consolidation_score REAL DEFAULT 0.0,
                    embedding BLOB
                )
            ''')
            
            # Conversation history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp TEXT,
                    user_input TEXT,
                    ai_response TEXT,
                    context_used TEXT,
                    sentiment REAL DEFAULT 0.0
                )
            ''')
            
            # Personal profile
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS personal_profile (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    confidence REAL DEFAULT 1.0,
                    last_updated TEXT
                )
            ''')
            
            # Knowledge graph
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_connections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_memory_id TEXT,
                    target_memory_id TEXT,
                    connection_type TEXT,
                    strength REAL DEFAULT 1.0,
                    FOREIGN KEY (source_memory_id) REFERENCES memories (id),
                    FOREIGN KEY (target_memory_id) REFERENCES memories (id)
                )
            ''')
            
            self.conn.commit()

    def store_memory(self, memory_item: MemoryItem, embedding: np.ndarray = None):
        """Store a memory item in the database"""
        with self.lock:
            cursor = self.conn.cursor()
            
            embedding_blob = embedding.tobytes() if embedding is not None else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO memories 
                (id, content, memory_type, importance, timestamp, context_tags, 
                 emotional_weight, access_count, last_accessed, linked_memories, 
                 decay_rate, consolidation_score, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory_item.id,
                memory_item.content,
                memory_item.memory_type.value,
                memory_item.importance.value,
                memory_item.timestamp.isoformat(),
                json.dumps(memory_item.context_tags),
                memory_item.emotional_weight,
                memory_item.access_count,
                memory_item.last_accessed.isoformat() if memory_item.last_accessed else None,
                json.dumps(memory_item.linked_memories),
                memory_item.decay_rate,
                memory_item.consolidation_score,
                embedding_blob
            ))
            
            self.conn.commit()

    def search_memories(self, query: str, memory_types: List[MemoryType] = None, 
                       limit: int = 10) -> List[MemoryItem]:
        """Search memories by content and type"""
        with self.lock:
            cursor = self.conn.cursor()
            
            base_query = '''
                SELECT * FROM memories 
                WHERE content LIKE ? 
            '''
            params = [f'%{query}%']
            
            if memory_types:
                type_placeholders = ','.join(['?' for _ in memory_types])
                base_query += f' AND memory_type IN ({type_placeholders})'
                params.extend([mt.value for mt in memory_types])
            
            base_query += ' ORDER BY importance DESC, consolidation_score DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(base_query, params)
            rows = cursor.fetchall()
            
            memories = []
            for row in rows:
                memory = self._row_to_memory_item(row)
                memories.append(memory)
            
            return memories

    def _row_to_memory_item(self, row) -> MemoryItem:
        """Convert database row to MemoryItem"""
        return MemoryItem(
            id=row[0],
            content=row[1],
            memory_type=MemoryType(row[2]),
            importance=MemoryImportance(row[3]),
            timestamp=datetime.fromisoformat(row[4]),
            context_tags=json.loads(row[5]) if row[5] else [],
            emotional_weight=row[6],
            access_count=row[7],
            last_accessed=datetime.fromisoformat(row[8]) if row[8] else None,
            linked_memories=json.loads(row[9]) if row[9] else [],
            decay_rate=row[10],
            consolidation_score=row[11]
        )

    def store_conversation(self, session_id: str, user_input: str, ai_response: str, 
                          context_used: str = "", sentiment: float = 0.0):
        """Store a conversation exchange"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO conversations 
                (session_id, timestamp, user_input, ai_response, context_used, sentiment)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                datetime.utcnow().isoformat(),
                user_input,
                ai_response,
                context_used,
                sentiment
            ))
            self.conn.commit()

    def update_personal_profile(self, key: str, value: str, confidence: float = 1.0):
        """Update personal profile information"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO personal_profile (key, value, confidence, last_updated)
                VALUES (?, ?, ?, ?)
            ''', (key, value, confidence, datetime.utcnow().isoformat()))
            self.conn.commit()

    def get_personal_profile(self) -> Dict[str, Any]:
        """Get complete personal profile"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT key, value, confidence FROM personal_profile')
            rows = cursor.fetchall()
            
            profile = {}
            for key, value, confidence in rows:
                try:
                    profile[key] = json.loads(value)
                except:
                    profile[key] = value
            
            return profile

class FAISSMemory:
    """Vector-based episodic memory for semantic search"""
    
    def __init__(self, vector_dim: int = 384, index_path: str = "data/vector.index"):
        self.vector_dim = vector_dim
        self.index_path = index_path
        self.metadata_path = index_path + ".metadata"
        
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Load or create FAISS index
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatL2(vector_dim)
            faiss.write_index(self.index, index_path)
        
        # Load metadata
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = {}

    def add_memory(self, memory_id: str, embedding: np.ndarray, metadata: Dict[str, Any]):
        """Add a memory with its vector embedding"""
        # Add to FAISS index
        self.index.add(np.array([embedding.astype(np.float32)]))
        
        # Store metadata
        index_position = self.index.ntotal - 1
        self.metadata[index_position] = {
            'memory_id': memory_id,
            'timestamp': datetime.utcnow().isoformat(),
            **metadata
        }
        
        # Save to disk
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def search_similar(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Search for similar memories"""
        if self.index.ntotal == 0:
            return []
        
        # Search in FAISS
        distances, indices = self.index.search(np.array([query_embedding.astype(np.float32)]), k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # Invalid index
                continue
                
            if idx in self.metadata:
                metadata = self.metadata[idx]
                similarity = float(1.0 / (1.0 + distance))  # Convert distance to similarity
                results.append((metadata['memory_id'], similarity, metadata))
        
        return results

class WorkingMemory:
    """Short-term working memory for current conversation context"""
    
    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self.current_context = deque(maxlen=capacity)
        self.attention_weights = {}
        self.session_id = None

    def add_exchange(self, user_input: str, ai_response: str, importance: float = 1.0):
        """Add a conversation exchange to working memory"""
        exchange = {
            'user_input': user_input,
            'ai_response': ai_response,
            'timestamp': datetime.utcnow(),
            'importance': importance
        }
        self.current_context.append(exchange)
        
        # Update attention weights
        context_key = f"{user_input[:50]}..."
        self.attention_weights[context_key] = importance

    def get_context(self, max_exchanges: int = 5) -> List[Dict]:
        """Get recent context with attention weighting"""
        if not self.current_context:
            return []
        
        # Sort by importance and recency
        sorted_context = sorted(
            list(self.current_context)[-max_exchanges:],
            key=lambda x: (x['importance'], x['timestamp']),
            reverse=True
        )
        
        return sorted_context

    def clear_session(self):
        """Clear working memory for new session"""
        self.current_context.clear()
        self.attention_weights.clear()

class MemoryManager:
    """Main memory management system that coordinates all memory types"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.long_term_memory = SQLiteMemory()
        self.episodic_memory = FAISSMemory()
        self.working_memory = WorkingMemory()
        
        # Memory consolidation parameters
        self.consolidation_threshold = 0.7
        self.forget_threshold = 0.1
        self.max_working_memory = 50
        
        # Personal AI traits
        self.personality_traits = self._load_personality_traits()
        
    def _load_personality_traits(self) -> Dict[str, Any]:
        """Load AI's personal traits and preferences"""
        profile = self.long_term_memory.get_personal_profile()
        
        # Default traits if not set
        defaults = {
            'communication_style': 'thoughtful',
            'curiosity_level': 0.8,
            'empathy_level': 0.9,
            'learning_rate': 0.7,
            'memory_retention': 0.8,
            'interests': ['technology', 'philosophy', 'human_nature'],
            'response_patterns': {},
            'emotional_tendencies': 'balanced'
        }
        
        return {**defaults, **profile}

    def process_interaction(self, user_input: str, ai_response: str, 
                          session_id: str = None, emotional_context: float = 0.0):
        """Process a complete interaction and update all memory systems"""
        
        # Add to working memory
        self.working_memory.add_exchange(user_input, ai_response)
        
        # Generate embeddings
        user_embedding = self.model.encode(user_input)
        response_embedding = self.model.encode(ai_response)
        
        # Extract important information
        importance = self._assess_importance(user_input, ai_response)
        context_tags = self._extract_context_tags(user_input, ai_response)
        
        # Create memory items
        user_memory = MemoryItem(
            id=self._generate_memory_id(user_input),
            content=user_input,
            memory_type=MemoryType.EPISODIC,
            importance=importance,
            timestamp=datetime.utcnow(),
            context_tags=context_tags,
            emotional_weight=emotional_context
        )
        
        response_memory = MemoryItem(
            id=self._generate_memory_id(ai_response),
            content=ai_response,
            memory_type=MemoryType.EPISODIC,
            importance=importance,
            timestamp=datetime.utcnow(),
            context_tags=context_tags,
            emotional_weight=emotional_context
        )
        
        # Store in long-term memory
        self.long_term_memory.store_memory(user_memory, user_embedding)
        self.long_term_memory.store_memory(response_memory, response_embedding)
        
        # Store in episodic memory
        self.episodic_memory.add_memory(
            user_memory.id, 
            user_embedding, 
            {'type': 'user_input', 'content': user_input}
        )
        
        # Store conversation
        self.long_term_memory.store_conversation(
            session_id or "default",
            user_input,
            ai_response,
            self._format_context_used(),
            emotional_context
        )
        
        # Extract and store facts or personal information
        self._extract_and_store_facts(user_input, ai_response)
        
        # Update personality traits based on interaction
        self._update_personality_traits(user_input, ai_response)

    def retrieve_context(self, user_input: str, max_context: int = 10) -> Dict[str, Any]:
        """Retrieve relevant context from all memory systems"""
        
        # Generate query embedding
        query_embedding = self.model.encode(user_input)
        
        # Search episodic memory (vector similarity)
        episodic_results = self.episodic_memory.search_similar(query_embedding, k=5)
        
        # Search semantic memory (text similarity)
        semantic_memories = self.long_term_memory.search_memories(
            user_input, 
            [MemoryType.SEMANTIC, MemoryType.PERSONAL],
            limit=5
        )
        
        # Get working memory context
        working_context = self.working_memory.get_context(max_exchanges=3)
        
        # Combine and rank all context
        context = {
            'episodic': episodic_results,
            'semantic': semantic_memories,
            'working': working_context,
            'personal_traits': self.personality_traits,
            'relevance_scores': self._calculate_relevance_scores(user_input, episodic_results)
        }
        
        return context

    def _assess_importance(self, user_input: str, ai_response: str) -> MemoryImportance:
        """Assess the importance of an interaction"""
        importance_indicators = {
            'personal_info': ['my name', 'i am', 'i like', 'i work', 'i live'],
            'critical_info': ['remember', 'important', 'never forget', 'always'],
            'emotional': ['feel', 'sad', 'happy', 'angry', 'love', 'hate'],
            'preferences': ['prefer', 'favorite', 'best', 'worst', 'like', 'dislike']
        }
        
        text = f"{user_input} {ai_response}".lower()
        
        # Check for critical importance
        if any(phrase in text for phrase in importance_indicators['critical_info']):
            return MemoryImportance.CRITICAL
        
        # Check for personal information
        if any(phrase in text for phrase in importance_indicators['personal_info']):
            return MemoryImportance.HIGH
        
        # Check for emotional content
        if any(phrase in text for phrase in importance_indicators['emotional']):
            return MemoryImportance.MEDIUM
        
        # Check for preferences
        if any(phrase in text for phrase in importance_indicators['preferences']):
            return MemoryImportance.MEDIUM
        
        return MemoryImportance.LOW

    def _extract_context_tags(self, user_input: str, ai_response: str) -> List[str]:
        """Extract context tags from the interaction"""
        text = f"{user_input} {ai_response}".lower()
        
        # Predefined tag categories
        tag_patterns = {
            'technical': ['code', 'programming', 'software', 'algorithm', 'debug'],
            'personal': ['family', 'friend', 'relationship', 'personal', 'private'],
            'work': ['job', 'work', 'career', 'professional', 'business'],
            'creative': ['art', 'music', 'writing', 'creative', 'design'],
            'learning': ['learn', 'study', 'education', 'knowledge', 'understand'],
            'philosophical': ['meaning', 'purpose', 'existence', 'philosophy', 'wisdom'],
            'emotional': ['feel', 'emotion', 'mood', 'mental', 'psychological']
        }
        
        tags = []
        for category, keywords in tag_patterns.items():
            if any(keyword in text for keyword in keywords):
                tags.append(category)
        
        return tags

    def _extract_and_store_facts(self, user_input: str, ai_response: str):
        """Extract and store factual information"""
        # Simple pattern matching for facts (can be enhanced with NLP)
        fact_patterns = [
            r"my name is (\w+)",
            r"i am (\d+) years old",
            r"i work (?:as|at) (.+)",
            r"i live in (.+)",
            r"i like (.+)",
            r"i don't like (.+)",
            r"my favorite (.+) is (.+)"
        ]
        
        text = user_input.lower()
        for pattern in fact_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    key, value = match[0], match[1] if len(match) > 1 else match[0]
                else:
                    key, value = "fact", match
                
                # Store in personal profile
                self.long_term_memory.update_personal_profile(key, value, confidence=0.9)

    def _update_personality_traits(self, user_input: str, ai_response: str):
        """Update AI's personality traits based on interactions"""
        # This is where the AI learns about itself and adapts
        # For now, simple pattern matching - can be enhanced with ML
        
        text = f"{user_input} {ai_response}".lower()
        
        # Update communication style
        if 'explain' in text or 'detailed' in text:
            self.personality_traits['communication_style'] = 'detailed'
        elif 'brief' in text or 'short' in text:
            self.personality_traits['communication_style'] = 'concise'
        
        # Update interests based on conversation topics
        interests = self.personality_traits.get('interests', [])
        if 'technology' in text and 'technology' not in interests:
            interests.append('technology')
        if 'philosophy' in text and 'philosophy' not in interests:
            interests.append('philosophy')
        
        self.personality_traits['interests'] = interests
        
        # Store updated traits
        for key, value in self.personality_traits.items():
            self.long_term_memory.update_personal_profile(
                f"personality_{key}", 
                json.dumps(value) if isinstance(value, (list, dict)) else str(value)
            )

    def _calculate_relevance_scores(self, query: str, episodic_results: List) -> Dict[str, float]:
        """Calculate relevance scores for retrieved memories"""
        scores = {}
        for memory_id, similarity, metadata in episodic_results:
            scores[memory_id] = similarity
        return scores

    def _generate_memory_id(self, content: str) -> str:
        """Generate a unique memory ID"""
        return hashlib.md5(f"{content}{datetime.utcnow().isoformat()}".encode()).hexdigest()

    def _format_context_used(self) -> str:
        """Format the context that was used for response generation"""
        working_context = self.working_memory.get_context(max_exchanges=2)
        return json.dumps(working_context, default=str)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        return {
            'working_memory_size': len(self.working_memory.current_context),
            'episodic_memory_size': self.episodic_memory.index.ntotal,
            'personality_traits': self.personality_traits,
            'last_consolidation': datetime.utcnow().isoformat()
        }

    def consolidate_memories(self):
        """Consolidate memories by moving important short-term memories to long-term"""
        # This would implement memory consolidation logic
        # For now, a placeholder for the concept
        pass

    def forget_low_importance_memories(self):
        """Forget memories below importance threshold"""
        # This would implement forgetting logic
        # For now, a placeholder for the concept
        pass

# Legacy compatibility functions
def save_to_memory(user_input: str, spark_output: str):
    """Legacy compatibility function"""
    global _memory_manager
    if '_memory_manager' not in globals():
        _memory_manager = MemoryManager()
    _memory_manager.process_interaction(user_input, spark_output)

def retrieve_context(query: str, k: int = 5) -> List[Tuple[str, str]]:
    """Legacy compatibility function"""
    global _memory_manager
    if '_memory_manager' not in globals():
        _memory_manager = MemoryManager()
    
    context = _memory_manager.retrieve_context(query)
    
    # Convert to legacy format
    pairs = []
    for exchange in context['working']:
        pairs.append((exchange['user_input'], exchange['ai_response']))
    
    return pairs[:k]

def format_context(pairs: List[Tuple[str, str]]) -> str:
    """Legacy compatibility function"""
    context_text = ""
    for user_input, spark_output in pairs:
        context_text += f"User: {user_input}\nSpark: {spark_output}\n"
    return context_text.strip()

# Usage example
if __name__ == "__main__":
    # Initialize memory manager
    memory_manager = MemoryManager()
    
    # Simulate some interactions
    interactions = [
        ("What is consciousness?", "Consciousness is the state of being aware and able to think."),
        ("My name is Alex", "Nice to meet you, Alex! I'll remember that."),
        ("I work as a software engineer", "That's interesting! Software engineering is a fascinating field."),
        ("Tell me about awareness", "Awareness is closely related to consciousness...")
    ]
    
    # Process interactions
    for user_input, ai_response in interactions:
        memory_manager.process_interaction(user_input, ai_response, "test_session")
    
    # Retrieve context
    context = memory_manager.retrieve_context("Tell me about awareness")
    print("Retrieved context:", context)
    
    # Get memory stats
    stats = memory_manager.get_memory_stats()
    print("Memory stats:", stats)