# core_loop.py
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

from .memory import MemoryManager, retrieve_context, save_to_memory
from .goals import GoalManager, evaluate_goals
from .beliefs import BeliefSystem, update_beliefs
from .behavior_profile import BehaviorProfile, load_behavioral_traits
from .web_tool import SmartWebSearch, search_web, scrape_url
from .spark_core import llm, SystemMessage, HumanMessage, DEFAULT_SYSTEM_PROMPT
from .reasoning import ReasoningEngine
from .context_manager import ContextManager
from .intent_classifier import IntentClassifier
from .response_generator import ResponseGenerator
from .learning_engine import LearningEngine

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    REACTIVE = "reactive"
    PROACTIVE = "proactive"
    REFLECTIVE = "reflective"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"


class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ProcessingContext:
    user_input: str
    timestamp: datetime
    session_id: str
    priority: Priority
    mode: ProcessingMode
    context_data: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class Response:
    content: str
    confidence: float
    reasoning_chain: List[str]
    sources: List[str]
    processing_time: float
    metadata: Dict[str, Any]

class ResponseGenerator:
    """Enhanced response generator with personality and context awareness"""
    
    def __init__(self, personality=None, emotional_state=None):
        self.personality = personality
        self.emotional_state = emotional_state or {"curiosity": 0.8, "empathy": 0.9, "creativity": 0.7}
    
    def generate(self, user_input: str, context: Dict[str, Any], 
                web_data: Optional[Dict[str, Any]], 
                intent: Dict[str, Any]) -> Response:
        """Generate response using LLM with enhanced context"""
        start_time = datetime.now()
        
        try:
            # Build enhanced system prompt
            system_prompt = self._build_system_prompt(context, web_data, intent)
            
            # Build user message with context
            user_message = self._build_user_message(user_input, context, web_data)
            
            # Generate response using LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            # Call the LLM
            llm_response = llm.invoke(messages)
            response_content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Extract sources from web data
            sources = []
            if web_data and "sources" in web_data:
                sources = web_data["sources"]
            
            # Build reasoning chain
            reasoning_chain = self._build_reasoning_chain(intent, web_data, context)
            
            # Calculate confidence
            confidence = self._calculate_confidence(intent, web_data, context)
            
            return Response(
                content=response_content,
                confidence=confidence,
                reasoning_chain=reasoning_chain,
                sources=sources,
                processing_time=processing_time,
                metadata={
                    "intent": intent,
                    "web_used": web_data is not None,
                    "context_quality": self._assess_context_quality(context)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return Response(
                content="I apologize, but I encountered an issue generating a response. Could you please rephrase your question?",
                confidence=0.0,
                reasoning_chain=["Error occurred during response generation"],
                sources=[],
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    def _build_system_prompt(self, context: Dict[str, Any], 
                           web_data: Optional[Dict[str, Any]], 
                           intent: Dict[str, Any]) -> str:
        """Build enhanced system prompt with context"""
        base_prompt = DEFAULT_SYSTEM_PROMPT
        
        # Add personality traits
        personality_context = ""
        if self.emotional_state:
            traits = []
            for trait, level in self.emotional_state.items():
                traits.append(f"{trait}: {level:.1f}")
            personality_context = f"Current emotional state: {', '.join(traits)}. "
        
        # Add intent context
        intent_context = ""
        if intent.get("primary"):
            intent_context = f"User intent detected: {intent['primary']}. "
            if intent.get("emotional_tone"):
                intent_context += f"Emotional tone: {intent['emotional_tone']}. "
        
        # Add web context
        web_context = ""
        if web_data:
            web_context = f"Recent web search performed. Use the provided information to enhance your response. "
        
        # Add memory context
        memory_context = ""
        if context.get("memory", {}).get("working"):
            memory_context = "Consider the conversation context when responding. "
        
        enhanced_prompt = f"{base_prompt}\n\n{personality_context}{intent_context}{web_context}{memory_context}"
        
        return enhanced_prompt.strip()
    
    def _build_user_message(self, user_input: str, context: Dict[str, Any], 
                          web_data: Optional[Dict[str, Any]]) -> str:
        """Build enhanced user message with context"""
        message_parts = [user_input]
        
        # Add web data context
        if web_data and web_data.get("synthesized_content"):
            message_parts.append(f"\nRelevant web information: {web_data['synthesized_content']}")
        
        # Add conversation context
        if context.get("memory", {}).get("working", {}).get("last_interaction"):
            last_interaction = context["memory"]["working"]["last_interaction"]
            message_parts.append(f"\nPrevious context: User asked about something related, and I responded about: {last_interaction.get('response', '')[:100]}...")
        
        return "\n".join(message_parts)
    
    def _build_reasoning_chain(self, intent: Dict[str, Any], 
                             web_data: Optional[Dict[str, Any]], 
                             context: Dict[str, Any]) -> List[str]:
        """Build reasoning chain for transparency"""
        reasoning = []
        
        if intent.get("primary"):
            reasoning.append(f"Identified primary intent: {intent['primary']}")
        
        if web_data:
            reasoning.append("Incorporated web search results")
        
        if context.get("memory", {}).get("episodic"):
            reasoning.append("Considered conversation history")
        
        reasoning.append("Generated response with personality integration")
        
        return reasoning
    
    def _calculate_confidence(self, intent: Dict[str, Any], 
                            web_data: Optional[Dict[str, Any]], 
                            context: Dict[str, Any]) -> float:
        """Calculate response confidence score"""
        confidence = 0.7  # Base confidence
        
        # Boost confidence with good intent classification
        if intent.get("confidence", 0) > 0.8:
            confidence += 0.1
        
        # Boost confidence with web data
        if web_data and web_data.get("confidence", 0) > 0.8:
            confidence += 0.1
        
        # Boost confidence with good context
        if self._assess_context_quality(context) > 0.8:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _assess_context_quality(self, context: Dict[str, Any]) -> float:
        """Assess the quality of available context"""
        quality = 0.5  # Base quality
        
        if context.get("memory", {}).get("working"):
            quality += 0.2
        
        if context.get("intent", {}).get("confidence", 0) > 0.7:
            quality += 0.2
        
        if context.get("web", {}).get("confidence", 0) > 0.7:
            quality += 0.1
        
        return min(quality, 1.0)

class SparkPersonality:
    """Advanced personality system with dynamic trait adaptation"""
    
    def __init__(self):
        self.behavior_profile = BehaviorProfile()
        self.intent_classifier = IntentClassifier()
        self.learning_engine = LearningEngine()
        self.trait_weights = {}
        self.emotional_state = {"curiosity": 0.8, "empathy": 0.9, "creativity": 0.7}
        
    def classify_intent(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced intent classification with multi-dimensional analysis"""
        primary_intent = self.intent_classifier.classify_primary(user_input)
        secondary_intents = self.intent_classifier.classify_secondary(user_input)
        emotional_tone = self.intent_classifier.analyze_emotional_tone(user_input)
        urgency_level = self.intent_classifier.assess_urgency(user_input, context)
        
        return {
            "primary": primary_intent,
            "secondary": secondary_intents,
            "emotional_tone": emotional_tone,
            "urgency": urgency_level,
            "confidence": self.intent_classifier.get_confidence()
        }
    
    def should_use_web(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent web usage decision with multi-factor analysis"""
        temporal_indicators = self._detect_temporal_needs(user_input)
        factual_requirements = self._assess_factual_needs(user_input, context)
        knowledge_gaps = self._identify_knowledge_gaps(user_input, context)
        
        web_decision = {
            "should_search": False,
            "search_type": None,
            "priority": Priority.LOW,
            "reasoning": []
        }
        
        # Advanced decision logic
        if temporal_indicators["score"] > 0.7:
            web_decision.update({
                "should_search": True,
                "search_type": "temporal",
                "priority": Priority.HIGH,
                "reasoning": ["Current information required"]
            })
        
        if factual_requirements["confidence"] < 0.6:
            web_decision.update({
                "should_search": True,
                "search_type": "factual_verification",
                "priority": Priority.MEDIUM
            })
            web_decision["reasoning"].append("Factual verification needed")
        
        if knowledge_gaps["gap_score"] > 0.5:
            web_decision["should_search"] = True
            web_decision["search_type"] = "knowledge_expansion"
            web_decision["reasoning"].append("Knowledge gap identified")
        
        return web_decision
    
    def generate_response(self, user_input: str, context: Dict[str, Any], 
                         web_data: Optional[Dict[str, Any]], 
                         intent: Dict[str, Any]) -> Response:
        """Advanced response generation with reasoning chains"""
        response_generator = ResponseGenerator(
            personality=self.behavior_profile,
            emotional_state=self.emotional_state
        )
        
        return response_generator.generate(user_input, context, web_data, intent)
    
    def _detect_temporal_needs(self, user_input: str) -> Dict[str, Any]:
        temporal_keywords = {
            "high": ["now", "current", "latest", "today", "breaking"],
            "medium": ["recent", "new", "update", "this week"],
            "low": ["when", "date", "time"]
        }
        
        score = 0.0
        matched_keywords = []
        
        text_lower = user_input.lower()
        for level, keywords in temporal_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if level == "high":
                        score += 0.4
                    elif level == "medium":
                        score += 0.2
                    else:
                        score += 0.1
                    matched_keywords.append(keyword)
        
        return {"score": min(score, 1.0), "keywords": matched_keywords}
    
    def _assess_factual_needs(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Implement factual assessment logic
        return {"confidence": 0.8, "domains": ["general"]}
    
    def _identify_knowledge_gaps(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Implement knowledge gap analysis
        return {"gap_score": 0.3, "areas": []}


class SmartWebSearch:
    """Enhanced web search with intelligent result processing"""
    
    def __init__(self):
        self.search_history = []
        self.result_cache = {}
        self.source_reliability = {}
        
    async def fetch(self, query: str, search_config: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent web data fetching with quality assessment"""
        search_type = search_config.get("search_type", "general")
        priority = search_config.get("priority", Priority.MEDIUM)
        
        # Multi-source search strategy
        results = await self._execute_multi_source_search(query, search_type)
        
        # Quality assessment and ranking
        ranked_results = self._assess_and_rank_results(results, query)
        
        # Content extraction and synthesis
        synthesized_data = await self._synthesize_content(ranked_results, search_type)
        
        return {
            "query": query,
            "results": ranked_results[:5],  # Top 5 results
            "synthesized_content": synthesized_data,
            "confidence": self._calculate_confidence(ranked_results),
            "sources": [r["url"] for r in ranked_results[:3]],
            "search_metadata": {
                "type": search_type,
                "priority": priority.name,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    async def _execute_multi_source_search(self, query: str, search_type: str) -> List[Dict[str, Any]]:
        """Execute searches across multiple sources"""
        # Implement multi-source search logic
        return search_web(query)
    
    def _assess_and_rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Advanced result ranking with quality metrics"""
        for result in results:
            result["quality_score"] = self._calculate_quality_score(result, query)
        
        return sorted(results, key=lambda x: x.get("quality_score", 0), reverse=True)
    
    def _calculate_quality_score(self, result: Dict[str, Any], query: str) -> float:
        # Implement quality scoring logic
        return 0.8
    
    async def _synthesize_content(self, results: List[Dict[str, Any]], search_type: str) -> str:
        """Synthesize content from multiple sources"""
        # Implement content synthesis
        return "Synthesized content from web sources"
    
    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        # Calculate overall confidence in results
        return 0.85


class AdvancedMemoryManager:
    """Enhanced memory system with episodic, semantic, and working memory"""
    
    def __init__(self):
        self.episodic_memory = []  # Conversation episodes
        self.semantic_memory = {}  # Factual knowledge
        self.working_memory = {}   # Current context
        self.memory_consolidation_threshold = 10
        
    def retrieve_context(self, user_input: str, max_context: int = 5) -> Dict[str, Any]:
        """Advanced context retrieval with relevance scoring"""
        episodic_context = self._retrieve_episodic_context(user_input, max_context)
        semantic_context = self._retrieve_semantic_context(user_input)
        working_context = self.working_memory.copy()
        
        return {
            "episodic": episodic_context,
            "semantic": semantic_context,
            "working": working_context,
            "retrieval_confidence": self._calculate_retrieval_confidence()
        }
    
    def update_context(self, user_input: str, response: str, metadata: Dict[str, Any]):
        """Update memory systems with new information"""
        # Update episodic memory
        episode = {
            "timestamp": datetime.now(),
            "user_input": user_input,
            "response": response,
            "metadata": metadata
        }
        self.episodic_memory.append(episode)
        
        # Update working memory
        self.working_memory["last_interaction"] = episode
        
        # Trigger consolidation if needed
        if len(self.episodic_memory) % self.memory_consolidation_threshold == 0:
            self._consolidate_memories()
    
    def _retrieve_episodic_context(self, query: str, max_items: int) -> List[Dict[str, Any]]:
        # Implement episodic memory retrieval
        return []
    
    def _retrieve_semantic_context(self, query: str) -> Dict[str, Any]:
        # Implement semantic memory retrieval
        return {}
    
    def _calculate_retrieval_confidence(self) -> float:
        return 0.9
    
    def _consolidate_memories(self):
        """Consolidate episodic memories into semantic knowledge"""
        pass


class SparkCore:
    """Advanced Spark AI Core with sophisticated processing capabilities"""
    
    def __init__(self):
        self.personality = SparkPersonality()
        self.memory = AdvancedMemoryManager()
        self.web = SmartWebSearch()
        self.reasoning_engine = ReasoningEngine()
        self.context_manager = ContextManager()
        self.goal_manager = GoalManager()
        self.belief_system = BeliefSystem()
        
        # Advanced processing components
        self.processing_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.active_sessions = {}
        
        # Performance monitoring
        self.performance_metrics = {
            "total_requests": 0,
            "average_response_time": 0.0,
            "success_rate": 1.0
        }
    
    async def respond(self, user_input: str, session_id: str = "default", 
                     processing_mode: ProcessingMode = ProcessingMode.REACTIVE) -> Response:
        """Advanced response generation with full pipeline processing"""
        start_time = datetime.now()
        
        try:
            # Create processing context
            context = ProcessingContext(
                user_input=user_input,
                timestamp=start_time,
                session_id=session_id,
                priority=self._assess_priority(user_input),
                mode=processing_mode,
                context_data={},
                metadata={}
            )
            
            # Stage 1: Context Analysis and Memory Retrieval
            memory_context = self.memory.retrieve_context(user_input)
            context.context_data["memory"] = memory_context
            
            # Stage 2: Intent Classification and Analysis
            intent_analysis = self.personality.classify_intent(user_input, memory_context)
            context.context_data["intent"] = intent_analysis
            
            # Stage 3: Web Search Decision and Execution
            web_decision = self.personality.should_use_web(user_input, memory_context)
            web_data = None
            
            if web_decision["should_search"]:
                web_data = await self.web.fetch(user_input, web_decision)
                context.context_data["web"] = web_data
            
            # Stage 4: Reasoning and Response Generation
            reasoning_chain = await self.reasoning_engine.process(context)
            response = self.personality.generate_response(
                user_input, context.context_data, web_data, intent_analysis
            )
            
            # Stage 5: Memory Update and Learning
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response_metadata = {
                "processing_time": processing_time,
                "intent": intent_analysis,
                "web_used": web_decision["should_search"],
                "reasoning_steps": len(reasoning_chain)
            }
            
            self.memory.update_context(user_input, response.content, response_metadata)
            
            # Stage 6: Goal and Belief Updates
            await self._update_cognitive_systems(user_input, response, context)
            
            # Update performance metrics
            self._update_performance_metrics(processing_time, True)
            
            return response
            
        except Exception as e:
            logger.exception(f"Error in core response loop: {e}")
            self._update_performance_metrics(0, False)
            
            return Response(
                content="I encountered an error while processing your request. Let me try to help you differently.",
                confidence=0.0,
                reasoning_chain=["Error occurred during processing"],
                sources=[],
                processing_time=(datetime.now() - start_time).total_seconds(),
                metadata={"error": str(e)}
            )
    
    def _assess_priority(self, user_input: str) -> Priority:
        """Assess request priority based on content analysis"""
        urgent_indicators = ["urgent", "emergency", "critical", "asap", "immediately"]
        if any(indicator in user_input.lower() for indicator in urgent_indicators):
            return Priority.CRITICAL
        
        important_indicators = ["important", "priority", "need"]
        if any(indicator in user_input.lower() for indicator in important_indicators):
            return Priority.HIGH
        
        return Priority.MEDIUM
    
    async def _update_cognitive_systems(self, user_input: str, response: Response, context: ProcessingContext):
        """Update goals and beliefs based on interaction"""
        # Update goals
        await self.goal_manager.evaluate_goals(response.content, context)
        
        # Update beliefs
        await self.belief_system.update_beliefs(user_input, response, context)
    
    def _update_performance_metrics(self, processing_time: float, success: bool):
        """Update system performance metrics"""
        self.performance_metrics["total_requests"] += 1
        
        if success:
            # Update average response time
            current_avg = self.performance_metrics["average_response_time"]
            total_requests = self.performance_metrics["total_requests"]
            new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
            self.performance_metrics["average_response_time"] = new_avg
        
        # Update success rate
        total_requests = self.performance_metrics["total_requests"]
        successful_requests = total_requests * self.performance_metrics["success_rate"]
        if success:
            successful_requests += 1
        self.performance_metrics["success_rate"] = successful_requests / total_requests
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "performance_metrics": self.performance_metrics,
            "active_sessions": len(self.active_sessions),
            "memory_usage": {
                "episodic_memories": len(self.memory.episodic_memory),
                "semantic_entries": len(self.memory.semantic_memory),
                "working_memory_size": len(self.memory.working_memory)
            },
            "cognitive_state": {
                "goals": self.goal_manager.get_active_goals(),
                "beliefs": self.belief_system.get_core_beliefs(),
                "personality_state": self.personality.emotional_state
            }
        }


# Backwards compatibility function
def run_core_loop(user_input: str, system_prompt: Optional[str] = None) -> str:
    """Backwards compatible interface for the advanced core system"""
    core = SparkCore()
    
    # Run async method in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        response = loop.run_until_complete(core.respond(user_input))
        return response.content
    finally:
        loop.close()


# Advanced usage example
async def advanced_core_example():
    """Example of advanced core usage"""
    core = SparkCore()
    
    # Multi-turn conversation with context
    responses = []
    session_id = "advanced_session_001"
    
    for user_input in [
        "What's the latest news about AI development?",
        "How does this relate to my previous question about machine learning?",
        "Can you help me understand the implications for my career?"
    ]:
        response = await core.respond(
            user_input, 
            session_id=session_id,
            processing_mode=ProcessingMode.ANALYTICAL
        )
        responses.append(response)
        
        print(f"User: {user_input}")
        print(f"Spark: {response.content}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Processing time: {response.processing_time:.2f}s")
        print("---")
    
    # System status
    status = core.get_system_status()
    print("System Status:", json.dumps(status, indent=2, default=str))


if __name__ == "__main__":
    # Run the advanced example
    asyncio.run(advanced_core_example())