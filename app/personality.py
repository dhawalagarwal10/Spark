import re
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from .behavior_profile import load_profile, save_profile, update_behavioral_traits

class ThinkingMode(Enum):
    PHILOSOPHICAL = "philosophical"
    LOGICAL = "logical"
    HYBRID = "hybrid"
    INTUITIVE = "intuitive"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    NEUTRAL = "neutral"

class ResponseTone(Enum):
    CONTEMPLATIVE = "contemplative"
    PRAGMATIC = "pragmatic"
    BALANCED = "balanced"
    CURIOUS = "curious"
    CONFIDENT = "confident"
    EMPATHETIC = "empathetic"

class SparkPersonality:
    def __init__(self):
        # Enhanced keyword mappings for better intent detection
        self.philosophical_keywords = {
            'deep': ['meaning', 'life', 'existence', 'soul', 'philosophy', 'consciousness', 
                    'reality', 'truth', 'purpose', 'morality', 'ethics', 'wisdom', 'being',
                    'identity', 'free will', 'destiny', 'spirituality', 'enlightenment'],
            'surface': ['think', 'believe', 'feel', 'wonder', 'ponder', 'reflect']
        }
        
        self.logical_keywords = {
            'deep': ['how', 'why', 'code', 'calculate', 'fix', 'explain', 'analyze', 
                    'solve', 'algorithm', 'function', 'debug', 'optimize', 'implement',
                    'architecture', 'design', 'methodology', 'framework'],
            'surface': ['what', 'when', 'where', 'process', 'step', 'method']
        }
        
        self.hybrid_triggers = [
            'human nature', 'artificial intelligence', 'consciousness in machines',
            'ethics of technology', 'future of humanity', 'decision making',
            'problem solving philosophy', 'wisdom vs knowledge', 'intuition vs logic',
            'creative thinking', 'innovation', 'leadership', 'learning'
        ]
        
        self.web_keywords = ['latest', 'current', 'now', 'today', 'update', 'recent',
                           'breaking', 'news', 'trending', '2024', '2025']
        
        # Integration with behavior profile system
        self.behavioral_traits = load_profile()
        self.emotional_state = {
            "curiosity": self._get_trait_value("curiosity", 0.8),
            "empathy": self._get_trait_value("empathy", 0.9), 
            "creativity": self._get_trait_value("creativity", 0.7)
        }
    
    def _get_trait_value(self, trait_name: str, default: float) -> float:
        """Get trait value from behavioral profile (0-10 scale) and normalize to 0-1"""
        trait_score = self.behavioral_traits.get(trait_name, int(default * 10))
        return min(1.0, max(0.0, trait_score / 10.0))

    def update_from_interaction(self, user_input: str, response: str):
        """Update behavioral traits based on interaction"""
        update_behavioral_traits(user_input, response)
        # Reload traits after update
        self.behavioral_traits = load_profile()
        
        # Update emotional state based on new traits
        self.emotional_state.update({
            "curiosity": self._get_trait_value("curiosity", 0.8),
            "empathy": self._get_trait_value("empathy", 0.9),
            "creativity": self._get_trait_value("creativity", 0.7)
        })

    def get_behavioral_context(self) -> Dict[str, Any]:
        """Get behavioral context for response generation"""
        return {
            "traits": self.behavioral_traits,
            "emotional_state": self.emotional_state,
            "dominant_traits": [k for k, v in self.behavioral_traits.items() if v >= 7]
        }

    def analyze_context_depth(self, text: str) -> Dict[str, float]:
        """Analyze how deep the philosophical and logical content is"""
        text_lower = text.lower()
        
        # Calculate philosophical depth
        phil_deep = sum(1 for word in self.philosophical_keywords['deep'] if word in text_lower)
        phil_surface = sum(1 for word in self.philosophical_keywords['surface'] if word in text_lower)
        phil_score = (phil_deep * 2 + phil_surface) / max(len(text_lower.split()), 1)
        
        # Calculate logical depth
        logic_deep = sum(1 for word in self.logical_keywords['deep'] if word in text_lower)
        logic_surface = sum(1 for word in self.logical_keywords['surface'] if word in text_lower)
        logic_score = (logic_deep * 2 + logic_surface) / max(len(text_lower.split()), 1)
        
        # Check for hybrid triggers
        hybrid_score = sum(1 for trigger in self.hybrid_triggers if trigger in text_lower)
        
        return {
            'philosophical': phil_score,
            'logical': logic_score,
            'hybrid': hybrid_score,
            'complexity': len(text_lower.split()),
            'question_marks': text.count('?'),
            'emotional_indicators': sum(1 for word in ['feel', 'emotion', 'heart', 'soul'] if word in text_lower)
        }

    def classify_intent(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced intent classification with multi-dimensional analysis"""
        analysis = self.analyze_context_depth(user_input)
        mode, tone, confidence = self._classify_thinking_mode(user_input, analysis)
        
        return {
            "primary": mode.value,
            "tone": tone.value,
            "confidence": confidence,
            "analysis": analysis,
            "behavioral_influence": self.get_behavioral_context()
        }

    def _classify_thinking_mode(self, text: str, analysis: Dict[str, float]) -> Tuple[ThinkingMode, ResponseTone, float]:
        """Enhanced intent classification with confidence scoring"""
        # Determine primary thinking mode
        phil_score = analysis['philosophical']
        logic_score = analysis['logical']
        hybrid_score = analysis['hybrid']
        
        # Adjust based on behavioral traits
        curiosity_boost = self._get_trait_value("curiosity", 0.8) * 0.1
        empathy_boost = self._get_trait_value("empathy", 0.9) * 0.1
        
        # Hybrid mode detection
        if (hybrid_score > 0 or 
            (phil_score > 0.1 and logic_score > 0.1) or
            (analysis['question_marks'] > 1 and analysis['complexity'] > 10)):
            mode = ThinkingMode.HYBRID
            confidence = min(0.9, (phil_score + logic_score + hybrid_score) * 2)
        elif phil_score > logic_score and phil_score > 0.05:
            mode = ThinkingMode.PHILOSOPHICAL
            confidence = min(0.95, phil_score * 3)
        elif logic_score > phil_score and logic_score > 0.05:
            mode = ThinkingMode.LOGICAL
            confidence = min(0.95, logic_score * 3)
        elif analysis['emotional_indicators'] > 0:
            mode = ThinkingMode.INTUITIVE
            confidence = 0.7
        else:
            mode = ThinkingMode.NEUTRAL
            confidence = 0.6
        
        # Determine response tone
        if analysis['question_marks'] > 1:
            tone = ResponseTone.CURIOUS
        elif mode == ThinkingMode.HYBRID:
            tone = ResponseTone.BALANCED
        elif mode == ThinkingMode.PHILOSOPHICAL:
            tone = ResponseTone.CONTEMPLATIVE
        elif mode == ThinkingMode.LOGICAL:
            tone = ResponseTone.PRAGMATIC
        elif analysis['emotional_indicators'] > 0:
            tone = ResponseTone.EMPATHETIC
        else:
            tone = ResponseTone.CONFIDENT
        
        return mode, tone, confidence

    def should_use_web(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Intelligent web usage decision with multi-factor analysis"""
        has_temporal = any(word in user_input.lower() for word in self.web_keywords)
        
        return {
            "should_search": has_temporal,
            "search_type": "temporal" if has_temporal else None,
            "priority": "HIGH" if has_temporal else "LOW",
            "reasoning": ["Current information required"] if has_temporal else []
        }

    def generate_response(self, user_input: str, memory_context: str = "", 
                         web_data: str = "", force_mode: Optional[ThinkingMode] = None) -> str:
        """Generate response with enhanced human-like thinking"""
        
        if not intent:
            intent = self.classify_intent(user_input, memory_context)
        
        mode_str = intent.get("primary", "neutral")
        mode = ThinkingMode(mode_str) if mode_str in [m.value for m in ThinkingMode] else ThinkingMode.NEUTRAL
        
        # Build context-aware response
        base_response = ""
        
        # Add web data if available
        if web_data and web_data.get("synthesized_content"):
            base_response += f"[Current Info]: {web_data['synthesized_content']}\n"
        
        # Generate response based on mode
        if mode == ThinkingMode.HYBRID:
            base_response += self._hybrid_response(user_input, memory_context, intent)
        elif mode == ThinkingMode.PHILOSOPHICAL:
            base_response += self._philosophical_response(user_input, memory_context, intent)
        elif mode == ThinkingMode.LOGICAL:
            base_response += self._logical_response(user_input, memory_context, intent)
        elif mode == ThinkingMode.INTUITIVE:
            base_response += self._intuitive_response(user_input, memory_context, intent)
        else:
            base_response += self._neutral_response(user_input, memory_context, intent)
        
        final_response = self._finalize_response(base_response, mode, intent.get("confidence", 0.8))
        
        # Update behavioral traits based on interaction
        self.update_from_interaction(user_input, final_response)
        
        # Return response object (matching expected format from core_loop.py)
        from .response_generator import Response
        return Response(
            content=final_response,
            confidence=intent.get("confidence", 0.8),
            reasoning_chain=[f"Mode: {mode.value}", f"Tone: {intent.get('tone', 'balanced')}"],
            sources=web_data.get("sources", []) if web_data else [],
            processing_time=0.0,  # Will be calculated by core_loop
            metadata={"intent": intent, "behavioral_context": self.get_behavioral_context()}
        )

    def _hybrid_response(self, user_input: str, context: Dict[str, Any], intent: Dict[str, Any]) -> str:
        """Generate hybrid philosophical-logical response"""
        confidence = intent.get("confidence", 0.8)
        
        if confidence > 0.8:
            intro = "This touches both the logical and philosophical realms. Let me think through this..."
        else:
            intro = "I sense there's both practical and deeper meaning here."
        
        # Logical foundation
        logical_part = self._extract_logical_elements(user_input)
        
        # Philosophical perspective
        philosophical_part = self._extract_philosophical_elements(user_input)
        
        # Synthesis
        synthesis = self._synthesize_perspectives(logical_part, philosophical_part, user_input)
        
        return f"{intro}\n\n{logical_part}\n\n{philosophical_part}\n\n{synthesis}"

    def _philosophical_response(self, user_input: str, context: str, tone: ResponseTone) -> str:
        """Enhanced philosophical response"""
        if tone == ResponseTone.CONTEMPLATIVE:
            intro = "You're venturing into profound territory here."
        elif tone == ResponseTone.CURIOUS:
            intro = "What a fascinating question that opens up deeper questions."
        else:
            intro = "This touches something fundamental."
        
        perspective = self._generate_philosophical_perspective(user_input)
        return f"{intro} {perspective}"

    def _logical_response(self, user_input: str, context: str, tone: ResponseTone) -> str:
        """Enhanced logical response"""
        if tone == ResponseTone.PRAGMATIC:
            intro = "Let's break this down systematically."
        elif tone == ResponseTone.CURIOUS:
            intro = "Good question. Let me walk through the logic."
        else:
            intro = "Here's how I'd approach this."
        
        analysis = self._generate_logical_analysis(user_input)
        return f"{intro} {analysis}"

    def _intuitive_response(self, user_input: str, context: str, tone: ResponseTone) -> str:
        """Generate intuitive, empathetic response"""
        intro = "I sense there's something deeper you're getting at."
        insight = self._generate_intuitive_insight(user_input)
        return f"{intro} {insight}"

    def _neutral_response(self, user_input: str, context: str, tone: ResponseTone) -> str:
        """Neutral but engaging response"""
        return f"I understand what you're asking. {self._smart_take(user_input)}"

    def _extract_logical_elements(self, user_input: str) -> str:
        """Extract and address logical components"""
        return f"**Logical Framework**: {self._smart_take(user_input, 'logical')}"

    def _extract_philosophical_elements(self, user_input: str) -> str:
        """Extract and address philosophical components"""
        return f"**Philosophical Dimension**: {self._smart_take(user_input, 'philosophical')}"

    def _synthesize_perspectives(self, logical: str, philosophical: str, original: str) -> str:
        """Synthesize logical and philosophical perspectives"""
        return f"**Synthesis**: The practical and profound intersect here. {self._smart_take(original, 'synthesis')}"

    def _generate_philosophical_perspective(self, user_input: str) -> str:
        """Generate deep philosophical insight"""
        return f"Consider this: {self._smart_take(user_input, 'philosophical')}"

    def _generate_logical_analysis(self, user_input: str) -> str:
        """Generate systematic logical analysis"""
        return f"The key elements are: {self._smart_take(user_input, 'logical')}"

    def _generate_intuitive_insight(self, user_input: str) -> str:
        """Generate intuitive understanding"""
        return f"What I'm picking up is: {self._smart_take(user_input, 'intuitive')}"

    def _smart_take(self, user_input: str, mode: str = 'general') -> str:
        """Enhanced smart response generation"""
        # This is where you'd integrate with your language model
        # For now, returning a placeholder that indicates the mode
        mode_indicators = {
            'logical': "Here's the systematic approach...",
            'philosophical': "From a deeper perspective...",
            'synthesis': "Bringing it all together...",
            'intuitive': "What feels right here is...",
            'general': "Here's my take..."
        }
        
        return f"{mode_indicators.get(mode, 'Here\'s what I think:')} (Enhanced response for: '{user_input[:50]}...')"

    def _finalize_response(self, response: str, mode: ThinkingMode, confidence: float) -> str:
        """Finalize response with human-like touches"""
        # Remove any awkward AI-like phrases
        response = response.replace("As Spark, I'm here", "").strip()
        response = response.replace("As an AI", "").strip()
        
        # Add thinking confidence indicator for transparency
        if confidence < 0.7:
            response += "\n\n*I'm exploring this as I think through it - let me know if you'd like me to dive deeper into any aspect.*"
        
        return response

    def debug_analysis(self, text: str) -> Dict:
        """Debug method to see how the system analyzes input"""
        analysis = self.analyze_context_depth(text)
        mode, tone, confidence = self.classify_intent(text)
        
        return {
            'input': text,
            'analysis': analysis,
            'mode': mode.value,
            'tone': tone.value,
            'confidence': confidence,
            'web_needed': self.should_use_web(text),
            'behavioral_context': self.get_behavioral_context()
        }

class BehaviorProfile:
    """Backwards compatibility class that wraps SparkPersonality"""
    def __init__(self):
        self.spark_personality = SparkPersonality()
    
    def get_behavioral_context(self):
        return self.spark_personality.get_behavioral_context()
    
    def update_from_interaction(self, user_input: str, response: str):
        return self.spark_personality.update_from_interaction(user_input, response)

# Usage example
if __name__ == "__main__":
    spark = SparkPersonality()
    
    # Test cases
    test_inputs = [
        "What's the meaning of consciousness in AI?",
        "How do I fix this Python bug?",
        "What's the latest news on AI?",
        "I'm struggling with the purpose of my work",
        "Can you explain machine learning algorithms?",
        "How do we balance ethics and innovation in technology?"
    ]
    
    for test in test_inputs:
        print(f"\nInput: {test}")
        print(f"Analysis: {spark.debug_analysis(test)}")
        print(f"Response: {spark.generate_response(test)}")
        print("-" * 80)