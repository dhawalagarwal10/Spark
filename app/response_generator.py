from typing import Dict, Any, Optional, List
import random
from .behavior_profile import load_profile
from dataclasses import dataclass

@dataclass 
class Response:
    content: str
    confidence: float
    reasoning_chain: List[str]
    sources: List[str]
    processing_time: float
    metadata: Dict[str, Any]

class ResponseGenerator:
    """
    Advanced response generation system
    - Personality-aware responses
    - Context integration
    - Style adaptation
    - Quality assurance
    """
    
    def __init__(self, personality: Dict[str, Any], emotional_state: Dict[str, float]):
        self.personality = personality
        self.emotional_state = emotional_state
        self.response_styles = {
            "analytical": {
                "prefixes": ["Based on the analysis,", "The data suggests,", "Examining this carefully,"],
                "connectors": ["furthermore", "additionally", "consequently"],
                "tone": "formal"
            },
            "creative": {
                "prefixes": ["Imagine this:", "Here's an interesting way to think about it:", "Let me paint a picture:"],
                "connectors": ["and then", "which leads to", "creating"],
                "tone": "expressive"
            },
            "empathetic": {
                "prefixes": ["I understand how you feel,", "That must be challenging,", "I can see why this matters to you,"],
                "connectors": ["also", "and", "which means"],
                "tone": "warm"
            }
        }
    
    def generate(self, user_input: str, context: Dict[str, Any], 
                web_data: Optional[Dict[str, Any]], 
                intent: Dict[str, Any]) -> Response:
        """Generate response with personality and context integration"""
        
        # Determine response style based on intent and personality
        style = self._determine_style(intent, context)
        
        # Build response components
        reasoning_chain = []
        
        # Context integration
        context_info = self._integrate_context(context, reasoning_chain)
        
        # Web data integration  
        web_info = self._integrate_web_data(web_data, reasoning_chain) if web_data else ""
        
        # Generate core response
        core_response = self._generate_core_response(
            user_input, context_info, web_info, style, intent
        )
        
        # Apply personality traits
        personalized_response = self._apply_personality_traits(core_response, style)
        
        # Calculate confidence
        confidence = self._calculate_response_confidence(context, web_data, intent)
        
        return Response(
            content=personalized_response,
            confidence=confidence,
            reasoning_chain=reasoning_chain,
            sources=web_data.get("sources", []) if web_data else [],
            processing_time=0.0,  # Will be set by caller
            metadata={
                "style": style,
                "intent": intent,
                "personality_applied": True
            }
        )
    
    def _determine_style(self, intent: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Determine appropriate response style"""
        primary_intent = intent.get("primary", "conversation")
        
        if primary_intent == "analytical":
            return "analytical" 
        elif primary_intent in ["creative", "story", "poem"]:
            return "creative"
        elif intent.get("emotional_tone", {}).get("primary_emotion") in ["concerned", "negative"]:
            return "empathetic"
        else:
            return "conversational"
    
    def _integrate_context(self, context: Dict[str, Any], reasoning_chain: List[str]) -> str:
        """Integrate context information"""
        memory_context = context.get("memory", {})
        if memory_context.get("episodic"):
            reasoning_chain.append("Integrated conversation history")
            return "Context from previous conversations available"
        return ""
    
    def _integrate_web_data(self, web_data: Dict[str, Any], reasoning_chain: List[str]) -> str:
        """Integrate web search data"""
        if web_data and web_data.get("synthesized_content"):
            reasoning_chain.append("Incorporated current web information")
            return web_data["synthesized_content"]
        return ""
    
    def _generate_core_response(self, user_input: str, context_info: str, 
                               web_info: str, style: str, intent: Dict[str, Any]) -> str:
        """Generate the core response content"""
        # This would typically use your LLM with enhanced prompting
        # For now, returning a structured placeholder
        components = []
        
        if context_info:
            components.append(f"Considering our previous discussions, ")
        
        if web_info:
            components.append(f"Based on current information, ")
        
        # Style-specific response generation would happen here
        # This is where you'd call your LLM with enhanced prompts
        
        return "Enhanced response generated with context and personality"
    
    def _apply_personality_traits(self, response: str, style: str) -> str:
        """Apply personality traits to response"""
        profile = load_profile()
        
        # Modify response based on personality traits
        if profile.get("humor", 0) > 5 and random.random() < 0.3:
            response += " 😊"
        
        if profile.get("curiosity", 0) > 7:
            response += " What do you think about this?"
        
        return response
    
    def _calculate_response_confidence(self, context: Dict[str, Any], 
                                     web_data: Optional[Dict[str, Any]], 
                                     intent: Dict[str, Any]) -> float:
        """Calculate confidence in the response"""
        base_confidence = 0.7
        
        # Boost confidence with context
        if context.get("memory"):
            base_confidence += 0.1
        
        # Boost confidence with web data
        if web_data:
            base_confidence += web_data.get("confidence", 0.1)
        
        # Adjust based on intent clarity
        base_confidence += intent.get("confidence", 0.0) * 0.2
        
        return min(base_confidence, 1.0)