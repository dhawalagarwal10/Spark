import json
import os
from collections import defaultdict
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

PROFILE_PATH = "data/behavior_profile.json"

def load_profile() -> Dict[str, int]:
    if not os.path.exists(PROFILE_PATH):
        return {}
    with open(PROFILE_PATH, "r") as f:
        return json.load(f)
    
def load_behavioral_traits() -> str:
    profile = load_profile()
    if not profile:
        return "No personality traits recorded yet."
    
    formatted_traits = []
    for trait, score in profile.items():
        level = "High" if score > 7 else "Medium" if score > 4 else "Low"
        formatted_traits.append(f"-{trait.replace('_', ' ').title()}: {level} ({score})/10")
    
    return "\n".join(formatted_traits)
        

def save_profile(profile: Dict[str, int]):
    os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2)

def update_behavioral_traits(user_input: str, spark_output: str):
    profile = load_profile()
    profile.setdefault("curiosity", 0)
    profile.setdefault("empathy", 0)
    profile.setdefault("humor", 0)

    if "why" in user_input.lower() or "how" in user_input.lower():
        profile["curiosity"] += 1
    if any(word in spark_output.lower() for word in ["understand", "feel", "meaning"]):
        profile["empathy"] += 1
    if any(word in spark_output.lower() for word in ["joke", "haha", "funny"]):
        profile["humor"] += 1

    save_profile(profile)

def get_behavioral_summary() -> str:
    profile = load_profile()
    traits = [f"{k.capitalize()}: {v}" for k, v in profile.items()]
    return "\n".join(traits) if traits else "No behavioral traits observed yet."

@dataclass
class PersonalityTrait:
    name: str
    value: float  # 0.0 to 1.0
    description: str
    influence_weight: float = 1.0


class BehaviorProfile:
    """Manages personality traits and behavioral patterns"""
    
    def __init__(self):
        self.traits = self._initialize_traits_from_file()
        self.interaction_history = []
        self.adaptation_rate = 0.1
        
    def _initialize_traits_from_file(self) -> Dict[str, PersonalityTrait]:
        """Initialize traits from existing profile file or defaults"""
        file_profile = load_profile()
        
        # Default traits with descriptions
        default_traits = {
            "curiosity": PersonalityTrait("curiosity", 0.8, "Drive to explore and learn"),
            "empathy": PersonalityTrait("empathy", 0.9, "Understanding and sharing feelings"),
            "creativity": PersonalityTrait("creativity", 0.7, "Ability to generate novel ideas"),
            "analytical": PersonalityTrait("analytical", 0.8, "Logical and systematic thinking"),
            "helpfulness": PersonalityTrait("helpfulness", 0.95, "Desire to assist and support"),
            "humor": PersonalityTrait("humor", 0.6, "Tendency to use humor appropriately"),
            "formality": PersonalityTrait("formality", 0.5, "Level of formal vs casual communication"),
            "patience": PersonalityTrait("patience", 0.85, "Tolerance and persistence")
        }
        
        # Update with values from file (convert 0-10 scale to 0-1 scale)
        for trait_name, score in file_profile.items():
            if trait_name in default_traits:
                normalized_value = min(1.0, max(0.0, score / 10.0))
                default_traits[trait_name].value = normalized_value
            else:
                # Add new trait from file
                normalized_value = min(1.0, max(0.0, score / 10.0))
                default_traits[trait_name] = PersonalityTrait(
                    trait_name, normalized_value, f"Learned trait: {trait_name}"
                )
        
        return default_traits
    
    def get_trait_value(self, trait_name: str) -> float:
        """Get the current value of a personality trait"""
        trait = self.traits.get(trait_name)
        return trait.value if trait else 0.5
    
    def update_trait(self, trait_name: str, new_value: float):
        """Update a personality trait value"""
        if trait_name in self.traits:
            # Ensure value stays within bounds
            new_value = max(0.0, min(1.0, new_value))
            self.traits[trait_name].value = new_value
            
            # Also update the file-based profile
            self._sync_to_file()
    
    def _sync_to_file(self):
        """Sync current traits to the file-based profile"""
        file_profile = {}
        for name, trait in self.traits.items():
            # Convert 0-1 scale back to 0-10 scale for file storage
            file_profile[name] = int(trait.value * 10)
        
        save_profile(file_profile)
    
    def adapt_to_interaction(self, user_input: str, context: Dict[str, Any]):
        """Adapt personality based on user interaction"""
        # Use existing logic from update_behavioral_traits function
        if "why" in user_input.lower() or "how" in user_input.lower():
            current_curiosity = self.get_trait_value("curiosity")
            self.update_trait("curiosity", min(1.0, current_curiosity + 0.01))
        
        # Simple adaptation for formality
        if any(word in user_input.lower() for word in ["formal", "professional", "business"]):
            current_formality = self.get_trait_value("formality")
            self.update_trait("formality", min(1.0, current_formality + self.adaptation_rate))
        
        if any(word in user_input.lower() for word in ["funny", "joke", "humor"]):
            current_humor = self.get_trait_value("humor")
            self.update_trait("humor", min(1.0, current_humor + 0.01))
    
    def get_behavioral_context(self) -> Dict[str, Any]:
        """Get current behavioral context for response generation"""
        return {
            "primary_traits": {name: trait.value for name, trait in self.traits.items()},
            "dominant_traits": self._get_dominant_traits(),
            "communication_style": self._determine_communication_style()
        }
    
    def _get_dominant_traits(self, threshold: float = 0.7) -> List[str]:
        """Get list of dominant personality traits"""
        return [name for name, trait in self.traits.items() if trait.value >= threshold]
    
    def _determine_communication_style(self) -> Dict[str, Any]:
        """Determine current communication style based on traits"""
        formality = self.get_trait_value("formality")
        humor = self.get_trait_value("humor")
        empathy = self.get_trait_value("empathy")
        
        style = {
            "tone": "formal" if formality > 0.7 else "casual" if formality < 0.3 else "balanced",
            "humor_level": "high" if humor > 0.7 else "low" if humor < 0.3 else "moderate",
            "empathy_expression": "high" if empathy > 0.8 else "moderate"
        }
        
        return style
    
    def update_from_response(self, user_input: str, spark_output: str):
        """Update traits based on interaction (using existing logic)"""
        # Use the existing update_behavioral_traits logic
        update_behavioral_traits(user_input, spark_output)
        
        # Reload traits from file to keep in sync
        self.traits = self._initialize_traits_from_file()