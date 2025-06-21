import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict

class LearningEngine:
    """
    Continuous learning system
    - Pattern recognition from interactions
    - Preference learning
    - Performance optimization
    - Adaptation strategies
    """
    
    def __init__(self, learning_data_path: str = "data/learning.json"):
        self.learning_data_path = learning_data_path
        self.learning_data = self._load_learning_data()
        self.interaction_patterns = defaultdict(list)
        self.user_preferences = defaultdict(dict)
        
    def _load_learning_data(self) -> Dict[str, Any]:
        """Load existing learning data"""
        if os.path.exists(self.learning_data_path):
            with open(self.learning_data_path, 'r') as f:
                return json.load(f)
        return {
            "interaction_patterns": {},
            "user_preferences": {},
            "performance_metrics": {},
            "adaptation_rules": []
        }
    
    def _save_learning_data(self):
        """Save learning data to disk"""
        os.makedirs(os.path.dirname(self.learning_data_path), exist_ok=True)
        with open(self.learning_data_path, 'w') as f:
            json.dump(self.learning_data, f, indent=2, default=str)
    
    def record_interaction(self, user_input: str, response: str, 
                          feedback: Optional[Dict[str, Any]] = None):
        """Record interaction for learning"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response,
            "feedback": feedback
        }
        
        # Analyze patterns
        patterns = self._extract_patterns(user_input, response)
        for pattern in patterns:
            self.learning_data["interaction_patterns"].setdefault(pattern, []).append(interaction)
        
        # Update user preferences if feedback available
        if feedback:
            self._update_preferences(user_input, response, feedback)
        
        self._save_learning_data()
    
    def _extract_patterns(self, user_input: str, response: str) -> List[str]:
        """Extract patterns from interaction"""
        patterns = []
        
        # Input patterns
        if "?" in user_input:
            patterns.append("question_pattern")
        if any(word in user_input.lower() for word in ["help", "assist", "support"]):
            patterns.append("help_request_pattern")
        
        # Response patterns
        if len(response) > 200:
            patterns.append("detailed_response_pattern")
        elif len(response) < 50:
            patterns.append("brief_response_pattern")
        
        return patterns
    
    def _update_preferences(self, user_input: str, response: str, feedback: Dict[str, Any]):
        """Update user preferences based on feedback"""
        if feedback.get("rating", 0) > 3:  # Assuming 1-5 rating
            response_style = "detailed" if len(response) > 100 else "brief"
            self.learning_data["user_preferences"][response_style] = \
                self.learning_data["user_preferences"].get(response_style, 0) + 1
    
    def get_recommendations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations based on learned patterns"""
        recommendations = {
            "response_style": "balanced",
            "confidence_adjustment": 0.0,
            "suggested_improvements": []
        }
        
        # Analyze user preferences
        prefs = self.learning_data.get("user_preferences", {})
        if prefs.get("detailed", 0) > prefs.get("brief", 0):
            recommendations["response_style"] = "detailed"
        elif prefs.get("brief", 0) > prefs.get("detailed", 0):
            recommendations["response_style"] = "brief"
        
        return recommendations