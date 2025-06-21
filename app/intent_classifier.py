from typing import Dict, List, Tuple, Any
import re
from enum import Enum

class IntentType(Enum):
    QUESTION = "question"
    REQUEST = "request"
    CONVERSATION = "conversation"
    TASK = "task"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    EMOTIONAL = "emotional"

class EmotionalTone(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    CONCERNED = "concerned"
    CURIOUS = "curious"

class IntentClassifier:
    """
    Advanced intent classification system
    - Multi-dimensional intent analysis
    - Emotional tone detection
    - Urgency assessment
    - Confidence scoring
    """
    
    def __init__(self):
        self.intent_patterns = {
            IntentType.QUESTION: [
                r'\b(what|who|when|where|why|how|which|can you)\b',
                r'\?',
                r'\b(tell me|explain|describe)\b'
            ],
            IntentType.REQUEST: [
                r'\b(please|could you|would you|can you|help me)\b',
                r'\b(I need|I want|I would like)\b',
                r'\b(do|make|create|generate|write)\b'
            ],
            IntentType.CREATIVE: [
                r'\b(write|create|generate|make|design|story|poem)\b',
                r'\b(creative|artistic|imagine|fictional)\b'
            ],
            IntentType.ANALYTICAL: [
                r'\b(analyze|compare|evaluate|assess|calculate)\b',
                r'\b(data|statistics|research|study)\b'
            ]
        }
        
        self.emotional_patterns = {
            EmotionalTone.POSITIVE: [r'\b(happy|great|awesome|love|excited|wonderful)\b'],
            EmotionalTone.NEGATIVE: [r'\b(sad|angry|frustrated|disappointed|terrible)\b'],
            EmotionalTone.CURIOUS: [r'\b(interesting|curious|wonder|fascinating)\b'],
            EmotionalTone.CONCERNED: [r'\b(worried|concerned|problem|issue|trouble)\b']
        }
        
        self.urgency_indicators = {
            "high": [r'\b(urgent|emergency|asap|immediately|critical|now)\b'],
            "medium": [r'\b(soon|quickly|priority|important)\b'],
            "low": [r'\b(when you can|no rush|eventually)\b']
        }
        
        self.last_confidence = 0.0
    
    def classify_primary(self, text: str) -> Dict[str, Any]:
        """Classify primary intent with confidence scoring"""
        text_lower = text.lower()
        intent_scores = {}
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches * 0.3
            intent_scores[intent_type.value] = min(score, 1.0)
        
        # Default to conversation if no strong patterns
        if not any(score > 0.3 for score in intent_scores.values()):
            intent_scores[IntentType.CONVERSATION.value] = 0.7
        
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])
        self.last_confidence = primary_intent[1]
        
        return {
            "intent": primary_intent[0],
            "confidence": primary_intent[1],
            "all_scores": intent_scores
        }
    
    def classify_secondary(self, text: str) -> List[str]:
        """Identify secondary intents"""
        text_lower = text.lower()
        secondary_intents = []
        
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    if intent_type.value not in secondary_intents:
                        secondary_intents.append(intent_type.value)
        
        return secondary_intents[:3]  # Top 3 secondary intents
    
    def analyze_emotional_tone(self, text: str) -> Dict[str, Any]:
        """Analyze emotional tone of input"""
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, patterns in self.emotional_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches * 0.4
            emotion_scores[emotion.value] = min(score, 1.0)
        
        if not any(score > 0.2 for score in emotion_scores.values()):
            emotion_scores[EmotionalTone.NEUTRAL.value] = 0.8
        
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        return {
            "primary_emotion": primary_emotion[0],
            "confidence": primary_emotion[1],
            "all_emotions": emotion_scores
        }
    
    def assess_urgency(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess urgency level of the request"""
        text_lower = text.lower()
        urgency_scores = {"high": 0, "medium": 0, "low": 0}
        
        for level, patterns in self.urgency_indicators.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                urgency_scores[level] += matches * 0.5
        
        # Context-based urgency (e.g., repeated similar requests)
        if context.get("memory", {}).get("episodic"):
            # If similar requests were made recently, increase urgency
            urgency_scores["medium"] += 0.2
        
        max_urgency = max(urgency_scores.items(), key=lambda x: x[1])
        
        return {
            "urgency_level": max_urgency[0] if max_urgency[1] > 0.3 else "low",
            "confidence": max_urgency[1],
            "scores": urgency_scores
        }
    
    def get_confidence(self) -> float:
        """Get confidence of last classification"""
        return self.last_confidence
