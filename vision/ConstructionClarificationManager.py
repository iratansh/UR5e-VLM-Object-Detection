#!/usr/bin/env python3
"""
Construction Clarification Manager for Intelligent HRI Research.

This module implements five distinct clarification strategies for construction HRI:
1. Direct Strategy - Simple binary questions
2. History-Aware Strategy - Context from previous interactions  
3. Confidence-Based Strategy - Graduated uncertainty expression
4. Options-Based Strategy - Multiple choice clarifications
5. Expertise-Adaptive Strategy - Adjusts to user skill level

Core research framework for trust formation under cognitive load and
expertise inversion scenarios in construction robotics.
"""

import logging
import random
import time
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClarificationStrategy(Enum):
    """Five clarification strategies for construction HRI research"""
    DIRECT = "direct"
    HISTORY_AWARE = "history_aware" 
    CONFIDENCE_BASED = "confidence_based"
    OPTIONS_BASED = "options_based"
    EXPERTISE_ADAPTIVE = "expertise_adaptive"

class UserExpertiseLevel(Enum):
    """Construction worker expertise levels"""
    APPRENTICE = "apprentice"      # 0-2 years experience
    JOURNEYMAN = "journeyman"      # 3-7 years experience  
    FOREMAN = "foreman"           # 8-15 years experience
    MASTER = "master"             # 15+ years experience

@dataclass
class TaskMemory:
    """Shared task history for master-apprentice mental model"""
    tool_name: str
    action: str
    location: Optional[str] = None
    success: bool = True
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    user_feedback: Optional[str] = None
    strategy_used: Optional[str] = None

@dataclass
class ClarificationResponse:
    """Response from clarification strategy"""
    text: str
    strategy: ClarificationStrategy
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    tts_enabled: bool = True

class ConstructionClarificationManager:
    """
    Intelligent clarification system for construction HRI research.
    
    Implements five distinct clarification strategies to study trust formation,
    cognitive load, and expertise inversion in construction robotics scenarios.
    
    Parameters
    ----------
    default_strategy : ClarificationStrategy, optional
        Default clarification approach, by default DIRECT
    user_expertise : UserExpertiseLevel, optional  
        Assessed user expertise level, by default JOURNEYMAN
    memory_size : int, optional
        Task memory window size, by default 10
    confidence_threshold : float, optional
        Threshold for triggering clarifications, by default 0.7
        
    Attributes
    ----------
    task_memory : deque
        Shared task history for context-aware responses
    user_expertise : UserExpertiseLevel
        Current user expertise assessment
    interaction_count : int
        Total number of clarification interactions
    strategy_performance : Dict
        Performance metrics per strategy for A/B testing
    """
    
    def __init__(self, 
                 default_strategy: ClarificationStrategy = ClarificationStrategy.DIRECT,
                 user_expertise: UserExpertiseLevel = UserExpertiseLevel.JOURNEYMAN,
                 memory_size: int = 10,
                 confidence_threshold: float = 0.7,
                 enable_rag: bool = True):
        
        self.logger = logging.getLogger(__name__)
        self.default_strategy = default_strategy
        self.user_expertise = user_expertise
        self.memory_size = memory_size
        self.confidence_threshold = confidence_threshold
        self.enable_rag = enable_rag
        
        # Task memory for history-aware responses
        self.task_memory: deque = deque(maxlen=memory_size)
        
        # Interaction tracking for research metrics
        self.interaction_count = 0
        self.strategy_performance = {
            strategy.value: {
                'usage_count': 0,
                'success_rate': 0.0,
                'avg_response_time': 0.0,
                'user_satisfaction': 0.0
            } for strategy in ClarificationStrategy
        }
        
        # Construction jargon database
        self.construction_jargon = {
            # Tool terminology
            "framing hammer": "16 oz straight claw hammer for rough carpentry",
            "speed square": "triangular aluminum square for marking and measuring",
            "crescent wrench": "adjustable open-end wrench",
            "skill saw": "handheld circular saw (brand name Skilsaw)",
            "sawzall": "reciprocating saw (brand name Milwaukee Sawzall)",
            
            # Spatial references in construction
            "on the deck": "on the floor/subfloor", 
            "up top": "on the upper level or roof",
            "rough opening": "framed opening for door or window",
            "plumb": "perfectly vertical",
            "square": "at perfect right angle",
            "level": "perfectly horizontal",
            
            # Construction processes
            "rough-in": "initial installation before finishing",
            "finish work": "final detail work and trim",
            "mockup": "test assembly to check fit",
            "punch list": "final cleanup and correction tasks"
        }
        
        # Initialize RAG manager if enabled
        self.rag_manager = None
        if self.enable_rag:
            try:
                from ConstructionRAGManager import ConstructionRAGManager
                self.rag_manager = ConstructionRAGManager()
                self.logger.info("✅ RAG integration enabled")
            except ImportError as e:
                self.logger.warning(f"RAG manager not available: {e}")
                self.enable_rag = False
        
        self.logger.info(f"✅ Construction Clarification Manager initialized")
        self.logger.info(f"   Strategy: {default_strategy.value}")
        self.logger.info(f"   User expertise: {user_expertise.value}")
        self.logger.info(f"   RAG enhancement: {'Enabled' if self.rag_manager else 'Disabled'}")

    def request_clarification(self, 
                            tool_request: str,
                            detected_objects: List[Dict],
                            confidence_scores: List[float],
                            strategy: Optional[ClarificationStrategy] = None) -> ClarificationResponse:
        """
        Generate intelligent clarification based on detection uncertainty.
        
        Parameters
        ----------
        tool_request : str
            User's original tool request (e.g., "get me the hammer")
        detected_objects : List[Dict]  
            List of detected objects with metadata
        confidence_scores : List[float]
            Detection confidence scores for each object
        strategy : ClarificationStrategy, optional
            Override default strategy for A/B testing
            
        Returns
        -------
        ClarificationResponse
            Clarification response with strategy metadata
        """
        
        strategy = strategy or self.default_strategy
        self.interaction_count += 1
        
        # Track strategy usage
        self.strategy_performance[strategy.value]['usage_count'] += 1
        
        # Route to appropriate strategy
        if strategy == ClarificationStrategy.DIRECT:
            response = self._direct_clarification(tool_request, detected_objects, confidence_scores)
        elif strategy == ClarificationStrategy.HISTORY_AWARE:
            response = self._history_aware_clarification(tool_request, detected_objects, confidence_scores)
        elif strategy == ClarificationStrategy.CONFIDENCE_BASED:
            response = self._confidence_based_clarification(tool_request, detected_objects, confidence_scores)
        elif strategy == ClarificationStrategy.OPTIONS_BASED:
            response = self._options_based_clarification(tool_request, detected_objects, confidence_scores)
        elif strategy == ClarificationStrategy.EXPERTISE_ADAPTIVE:
            response = self._expertise_adaptive_clarification(tool_request, detected_objects, confidence_scores)
        else:
            raise ValueError(f"Unknown clarification strategy: {strategy}")
            
        response.strategy = strategy
        
        # Apply RAG enhancement if enabled
        if self.rag_manager and self.enable_rag:
            try:
                rag_response = self.rag_manager.enhance_clarification(
                    original_text=response.text,
                    tool_request=tool_request,
                    detected_tools=detected_objects,
                    user_expertise=self.user_expertise.value,
                    task_history=[
                        {
                            'tool_name': mem.tool_name,
                            'action': mem.action,
                            'success': mem.success,
                            'timestamp': mem.timestamp
                        } for mem in list(self.task_memory)
                    ]
                )
                
                # Update response with RAG enhancement
                if rag_response.confidence > 0.3:  # Only use if confident
                    response.text = rag_response.enhanced_text
                    response.metadata['rag_enhanced'] = True
                    response.metadata['rag_confidence'] = rag_response.confidence
                    response.metadata['rag_items'] = len(rag_response.retrieved_items)
                    
                    self.logger.info(f"🧠 RAG enhanced (confidence: {rag_response.confidence:.2f})")
                
            except Exception as e:
                self.logger.error(f"RAG enhancement failed: {e}")
        
        self.logger.info(f"🤖 Clarification ({strategy.value}): {response.text}")
        return response

    def _direct_clarification(self, tool_request: str, detected_objects: List[Dict], 
                            confidence_scores: List[float]) -> ClarificationResponse:
        """
        Strategy 1: Direct binary questions - simple and efficient.
        
        Best for high cognitive load situations where users need quick decisions.
        Uses construction trade terminology appropriately.
        """
        
        if not detected_objects:
            return ClarificationResponse(
                text=f"I don't see any {tool_request} in the work area. Should I look somewhere else?",
                strategy=ClarificationStrategy.DIRECT,
                confidence=0.0,
                metadata={'no_detections': True}
            )
        
        if len(detected_objects) == 1:
            obj = detected_objects[0]
            tool_name = obj.get('trade_term', obj.get('label', 'tool'))
            confidence = confidence_scores[0] if confidence_scores else 0.5
            
            return ClarificationResponse(
                text=f"I see a {tool_name}. Is that the right tool?",
                strategy=ClarificationStrategy.DIRECT,
                confidence=confidence,
                metadata={'single_detection': True, 'tool': tool_name}
            )
        
        # Multiple detections - ask about the first/best one
        best_idx = max(range(len(confidence_scores)), key=lambda i: confidence_scores[i])
        best_tool = detected_objects[best_idx].get('trade_term', detected_objects[best_idx].get('label'))
        
        return ClarificationResponse(
            text=f"I see multiple tools. Do you want the {best_tool}?",
            strategy=ClarificationStrategy.DIRECT,
            confidence=max(confidence_scores),
            metadata={'multiple_detections': len(detected_objects), 'selected_tool': best_tool}
        )

    def _history_aware_clarification(self, tool_request: str, detected_objects: List[Dict],
                                   confidence_scores: List[float]) -> ClarificationResponse:
        """
        Strategy 2: History-aware responses using shared task memory.
        
        Leverages previous interactions for context. Supports transactive memory
        theory by referencing shared task history.
        """
        
        # Check recent task memory for context
        recent_tools = [mem.tool_name for mem in list(self.task_memory)[-3:]]
        
        if not detected_objects:
            if recent_tools:
                last_tool = recent_tools[-1]
                return ClarificationResponse(
                    text=f"No {tool_request} visible. Like the {last_tool} from earlier, should I check the tool crib?",
                    strategy=ClarificationStrategy.HISTORY_AWARE,
                    confidence=0.0,
                    metadata={'referenced_history': True, 'last_tool': last_tool}
                )
            else:
                return ClarificationResponse(
                    text=f"I don't see a {tool_request} in our current work area. Where should I look?",
                    strategy=ClarificationStrategy.HISTORY_AWARE,
                    confidence=0.0
                )
        
        # Single detection with history context
        if len(detected_objects) == 1:
            current_tool = detected_objects[0].get('trade_term', detected_objects[0].get('label'))
            
            # Check if similar tool used recently
            for mem in reversed(self.task_memory):
                if mem.tool_name.lower() in current_tool.lower() or current_tool.lower() in mem.tool_name.lower():
                    return ClarificationResponse(
                        text=f"Found a {current_tool}, same type as the {mem.tool_name} we used {self._time_since(mem.timestamp)}. This the one?",
                        strategy=ClarificationStrategy.HISTORY_AWARE,
                        confidence=confidence_scores[0] if confidence_scores else 0.5,
                        metadata={'referenced_tool': mem.tool_name, 'similarity_match': True}
                    )
            
            # No history match
            return ClarificationResponse(
                text=f"I found a {current_tool}. We haven't used this type yet today - is this what you need?",
                strategy=ClarificationStrategy.HISTORY_AWARE,
                confidence=confidence_scores[0] if confidence_scores else 0.5,
                metadata={'new_tool_type': True}
            )
        
        # Multiple detections with history
        if recent_tools:
            # Prefer tools similar to recent usage
            for i, obj in enumerate(detected_objects):
                tool_name = obj.get('trade_term', obj.get('label'))
                for recent_tool in recent_tools:
                    if recent_tool.lower() in tool_name.lower():
                        return ClarificationResponse(
                            text=f"I see several tools including a {tool_name}, similar to the {recent_tool} we used before. Want this one?",
                            strategy=ClarificationStrategy.HISTORY_AWARE,
                            confidence=confidence_scores[i] if i < len(confidence_scores) else 0.5,
                            metadata={'history_preference': recent_tool, 'similar_tool': tool_name}
                        )
        
        # Fallback for multiple with no strong history match
        tool_names = [obj.get('trade_term', obj.get('label')) for obj in detected_objects[:2]]
        return ClarificationResponse(
            text=f"I see {len(detected_objects)} options: {', '.join(tool_names)}. Which matches what we need for this job?",
            strategy=ClarificationStrategy.HISTORY_AWARE,
            confidence=max(confidence_scores) if confidence_scores else 0.5,
            metadata={'multiple_options': tool_names}
        )

    def _confidence_based_clarification(self, tool_request: str, detected_objects: List[Dict],
                                      confidence_scores: List[float]) -> ClarificationResponse:
        """
        Strategy 3: Graduated uncertainty expression with percentages.
        
        Aligns with construction's "measure twice, cut once" culture by
        expressing confidence levels numerically.
        """
        
        if not detected_objects:
            return ClarificationResponse(
                text=f"I'm 0% confident I can find a {tool_request} in this area. Should I expand the search?",
                strategy=ClarificationStrategy.CONFIDENCE_BASED,
                confidence=0.0,
                metadata={'confidence_expression': '0%'}
            )
        
        if len(detected_objects) == 1:
            confidence = confidence_scores[0] if confidence_scores else 0.5
            confidence_pct = int(confidence * 100)
            tool_name = detected_objects[0].get('trade_term', detected_objects[0].get('label'))
            
            if confidence_pct >= 80:
                return ClarificationResponse(
                    text=f"I'm {confidence_pct}% confident this {tool_name} is what you need. Proceed with pickup?",
                    strategy=ClarificationStrategy.CONFIDENCE_BASED,
                    confidence=confidence,
                    metadata={'confidence_level': 'high', 'confidence_pct': confidence_pct}
                )
            elif confidence_pct >= 60:
                return ClarificationResponse(
                    text=f"I'm about {confidence_pct}% sure this {tool_name} is correct. Worth double-checking?",
                    strategy=ClarificationStrategy.CONFIDENCE_BASED,
                    confidence=confidence,
                    metadata={'confidence_level': 'medium', 'confidence_pct': confidence_pct}
                )
            else:
                return ClarificationResponse(
                    text=f"Only {confidence_pct}% confident this {tool_name} is right. Should I keep looking?",
                    strategy=ClarificationStrategy.CONFIDENCE_BASED, 
                    confidence=confidence,
                    metadata={'confidence_level': 'low', 'confidence_pct': confidence_pct}
                )
        
        # Multiple detections with confidence ranking
        if confidence_scores:
            sorted_indices = sorted(range(len(confidence_scores)), key=lambda i: confidence_scores[i], reverse=True)
            best_idx = sorted_indices[0]
            best_confidence = confidence_scores[best_idx]
            best_tool = detected_objects[best_idx].get('trade_term', detected_objects[best_idx].get('label'))
            
            confidence_pct = int(best_confidence * 100)
            
            return ClarificationResponse(
                text=f"Best match is {best_tool} at {confidence_pct}% confidence. The other options are lower. Go with this one?",
                strategy=ClarificationStrategy.CONFIDENCE_BASED,
                confidence=best_confidence,
                metadata={
                    'confidence_ranking': True,
                    'best_confidence': confidence_pct,
                    'alternatives_count': len(detected_objects) - 1
                }
            )
        
        # Fallback
        return ClarificationResponse(
            text=f"Found {len(detected_objects)} possible matches but confidence is uncertain. Need me to describe what I see?",
            strategy=ClarificationStrategy.CONFIDENCE_BASED,
            confidence=0.5
        )

    def _options_based_clarification(self, tool_request: str, detected_objects: List[Dict],
                                   confidence_scores: List[float]) -> ClarificationResponse:
        """
        Strategy 4: Multiple choice clarifications with construction context.
        
        Reduces cognitive load by presenting clear options with professional
        terminology and spatial context.
        """
        
        if not detected_objects:
            return ClarificationResponse(
                text=f"No {tool_request} visible. Options: A) Check tool box, B) Look in truck, C) Ask crew. Which?",
                strategy=ClarificationStrategy.OPTIONS_BASED,
                confidence=0.0,
                metadata={'search_options': ['tool box', 'truck', 'ask crew']}
            )
        
        if len(detected_objects) == 1:
            tool_name = detected_objects[0].get('trade_term', detected_objects[0].get('label'))
            location = self._get_spatial_description(detected_objects[0])
            
            return ClarificationResponse(
                text=f"Found one {tool_name} {location}. Options: A) Take this one, B) Look for another. Your choice?",
                strategy=ClarificationStrategy.OPTIONS_BASED,
                confidence=confidence_scores[0] if confidence_scores else 0.5,
                metadata={'binary_choice': True, 'location': location}
            )
        
        # Multiple options - present up to 3 with letters
        options = []
        option_letters = ['A', 'B', 'C', 'D']
        
        for i, obj in enumerate(detected_objects[:3]):  # Limit to 3 options
            tool_name = obj.get('trade_term', obj.get('label'))
            location = self._get_spatial_description(obj)
            confidence_pct = int(confidence_scores[i] * 100) if i < len(confidence_scores) else 50
            
            options.append(f"{option_letters[i]}) {tool_name} {location} ({confidence_pct}% match)")
        
        if len(detected_objects) > 3:
            options.append("D) See more options")
        
        options_text = ", ".join(options)
        
        return ClarificationResponse(
            text=f"Multiple {tool_request} options: {options_text}. Which one?",
            strategy=ClarificationStrategy.OPTIONS_BASED,
            confidence=max(confidence_scores) if confidence_scores else 0.5,
            metadata={'option_count': len(options), 'truncated': len(detected_objects) > 3}
        )

    def _expertise_adaptive_clarification(self, tool_request: str, detected_objects: List[Dict],
                                        confidence_scores: List[float]) -> ClarificationResponse:
        """
        Strategy 5: Expertise-level adaptive communication.
        
        Adjusts terminology and detail level based on assessed user expertise.
        Supports expertise inversion research scenarios.
        """
        
        if self.user_expertise == UserExpertiseLevel.APPRENTICE:
            return self._apprentice_level_response(tool_request, detected_objects, confidence_scores)
        elif self.user_expertise == UserExpertiseLevel.JOURNEYMAN:
            return self._journeyman_level_response(tool_request, detected_objects, confidence_scores)
        elif self.user_expertise == UserExpertiseLevel.FOREMAN:
            return self._foreman_level_response(tool_request, detected_objects, confidence_scores)
        elif self.user_expertise == UserExpertiseLevel.MASTER:
            return self._master_level_response(tool_request, detected_objects, confidence_scores)
        else:
            # Fallback to journeyman level
            return self._journeyman_level_response(tool_request, detected_objects, confidence_scores)

    def _apprentice_level_response(self, tool_request: str, detected_objects: List[Dict],
                                 confidence_scores: List[float]) -> ClarificationResponse:
        """Detailed explanations for apprentice-level users"""
        
        if not detected_objects:
            return ClarificationResponse(
                text=f"I can't find a {tool_request}. These are typically stored in the tool crib or gang box. Should I check there?",
                strategy=ClarificationStrategy.EXPERTISE_ADAPTIVE,
                confidence=0.0,
                metadata={'expertise_level': 'apprentice', 'educational_context': True}
            )
        
        if len(detected_objects) == 1:
            obj = detected_objects[0]
            tool_name = obj.get('trade_term', obj.get('label'))
            category = obj.get('category', 'tool')
            
            # Add educational context
            description = self.construction_jargon.get(tool_name.lower(), '')
            
            return ClarificationResponse(
                text=f"Found a {tool_name} ({description}). This is a {category.replace('_', ' ')}. Is this what you need for the task?",
                strategy=ClarificationStrategy.EXPERTISE_ADAPTIVE,
                confidence=confidence_scores[0] if confidence_scores else 0.5,
                metadata={
                    'expertise_level': 'apprentice',
                    'educational_description': description,
                    'category_explanation': True
                }
            )
        
        # Multiple detections - explain differences
        tool_names = [obj.get('trade_term', obj.get('label')) for obj in detected_objects[:2]]
        return ClarificationResponse(
            text=f"I see {len(detected_objects)} tools: {', '.join(tool_names)}. Different tools work better for different tasks. Which fits your current job?",
            strategy=ClarificationStrategy.EXPERTISE_ADAPTIVE,
            confidence=max(confidence_scores) if confidence_scores else 0.5,
            metadata={'expertise_level': 'apprentice', 'tool_education': True}
        )

    def _journeyman_level_response(self, tool_request: str, detected_objects: List[Dict],
                                 confidence_scores: List[float]) -> ClarificationResponse:
        """Standard professional communication for journeyman-level users"""
        
        if not detected_objects:
            return ClarificationResponse(
                text=f"No {tool_request} in sight. Check the truck or send an apprentice to get one?",
                strategy=ClarificationStrategy.EXPERTISE_ADAPTIVE,
                confidence=0.0,
                metadata={'expertise_level': 'journeyman'}
            )
        
        if len(detected_objects) == 1:
            tool_name = detected_objects[0].get('trade_term', detected_objects[0].get('label'))
            return ClarificationResponse(
                text=f"Got a {tool_name} here. This work for you?",
                strategy=ClarificationStrategy.EXPERTISE_ADAPTIVE,
                confidence=confidence_scores[0] if confidence_scores else 0.5,
                metadata={'expertise_level': 'journeyman', 'concise_communication': True}
            )
        
        # Multiple - let them choose
        tool_names = [obj.get('trade_term', obj.get('label')) for obj in detected_objects[:3]]
        return ClarificationResponse(
            text=f"Got {len(detected_objects)} options: {', '.join(tool_names)}. Your call.",
            strategy=ClarificationStrategy.EXPERTISE_ADAPTIVE,
            confidence=max(confidence_scores) if confidence_scores else 0.5,
            metadata={'expertise_level': 'journeyman', 'options_presented': len(tool_names)}
        )

    def _foreman_level_response(self, tool_request: str, detected_objects: List[Dict],
                              confidence_scores: List[float]) -> ClarificationResponse:
        """Workflow-focused communication for foreman-level users"""
        
        if not detected_objects:
            return ClarificationResponse(
                text=f"No {tool_request} available. This'll hold up the crew. Should I flag this for procurement?",
                strategy=ClarificationStrategy.EXPERTISE_ADAPTIVE,
                confidence=0.0,
                metadata={'expertise_level': 'foreman', 'workflow_impact': True}
            )
        
        if len(detected_objects) == 1:
            tool_name = detected_objects[0].get('trade_term', detected_objects[0].get('label'))
            confidence = confidence_scores[0] if confidence_scores else 0.5
            
            return ClarificationResponse(
                text=f"Located {tool_name}. Confidence {int(confidence*100)}%. Fits the schedule?",
                strategy=ClarificationStrategy.EXPERTISE_ADAPTIVE,
                confidence=confidence,
                metadata={'expertise_level': 'foreman', 'schedule_focused': True}
            )
        
        # Multiple - efficiency focus
        best_idx = max(range(len(confidence_scores)), key=lambda i: confidence_scores[i]) if confidence_scores else 0
        best_tool = detected_objects[best_idx].get('trade_term', detected_objects[best_idx].get('label'))
        
        return ClarificationResponse(
            text=f"Multiple tools available. Recommend {best_tool} for efficiency. Approve?",
            strategy=ClarificationStrategy.EXPERTISE_ADAPTIVE,
            confidence=max(confidence_scores) if confidence_scores else 0.5,
            metadata={'expertise_level': 'foreman', 'efficiency_recommendation': best_tool}
        )

    def _master_level_response(self, tool_request: str, detected_objects: List[Dict],
                             confidence_scores: List[float]) -> ClarificationResponse:
        """Minimal, technical communication for master-level users"""
        
        if not detected_objects:
            return ClarificationResponse(
                text=f"No {tool_request}. Procurement issue?",
                strategy=ClarificationStrategy.EXPERTISE_ADAPTIVE,
                confidence=0.0,
                metadata={'expertise_level': 'master', 'minimal_communication': True}
            )
        
        if len(detected_objects) == 1:
            tool_name = detected_objects[0].get('trade_term', detected_objects[0].get('label'))
            return ClarificationResponse(
                text=f"{tool_name}. Proceed?",
                strategy=ClarificationStrategy.EXPERTISE_ADAPTIVE,
                confidence=confidence_scores[0] if confidence_scores else 0.5,
                metadata={'expertise_level': 'master', 'minimal_words': True}
            )
        
        # Multiple - just present count and best option
        best_idx = max(range(len(confidence_scores)), key=lambda i: confidence_scores[i]) if confidence_scores else 0
        best_tool = detected_objects[best_idx].get('trade_term', detected_objects[best_idx].get('label'))
        
        return ClarificationResponse(
            text=f"{len(detected_objects)} options. Recommend {best_tool}.",
            strategy=ClarificationStrategy.EXPERTISE_ADAPTIVE,
            confidence=max(confidence_scores) if confidence_scores else 0.5,
            metadata={'expertise_level': 'master', 'technical_brevity': True}
        )

    def update_task_memory(self, tool_name: str, action: str, success: bool = True, 
                          user_feedback: Optional[str] = None, strategy_used: Optional[str] = None):
        """
        Update shared task memory for history-aware responses.
        
        Parameters
        ----------
        tool_name : str
            Name of the tool used
        action : str  
            Action performed (pickup, place, etc.)
        success : bool, optional
            Whether the action succeeded, by default True
        user_feedback : str, optional
            User feedback on the interaction
        strategy_used : str, optional
            Which clarification strategy was used
        """

        memory = TaskMemory(
            tool_name=tool_name,
            action=action,
            success=success,
            user_feedback=user_feedback,
            strategy_used=strategy_used
        )
        
        self.task_memory.append(memory)
        self.logger.info(f"📝 Updated task memory: {tool_name} -> {action} ({'✅' if success else '❌'})")

    def update_user_expertise(self, new_level: UserExpertiseLevel):
        """Update assessed user expertise level for adaptive responses"""
        old_level = self.user_expertise
        self.user_expertise = new_level
        self.logger.info(f"👷 User expertise updated: {old_level.value} -> {new_level.value}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for research analysis.
        
        Returns
        -------
        Dict[str, Any]
            Performance data for A/B testing and research analysis
        """
        
        return {
            'total_interactions': self.interaction_count,
            'strategy_performance': self.strategy_performance.copy(),
            'user_expertise': self.user_expertise.value,
            'memory_size': len(self.task_memory),
            'current_strategy': self.default_strategy.value
        }

    def _get_spatial_description(self, obj: Dict) -> str:
        """Generate spatial description for construction context"""

        # Mock spatial reasoning - in real implementation would use bbox center
        locations = ["on the left", "on the right", "in the center", "on the workbench", "by the wall"]
        return random.choice(locations)

    def _time_since(self, timestamp: float) -> str:
        """Human-readable time since timestamp"""
        
        elapsed = time.time() - timestamp
        if elapsed < 60:
            return "just now"
        elif elapsed < 3600:
            return f"{int(elapsed/60)} minutes ago"
        else:
            return f"{int(elapsed/3600)} hours ago"

    def export_research_data(self, filepath: str):
        """Export interaction data for research analysis"""
        
        data = {
            'performance_metrics': self.get_performance_metrics(),
            'task_memory': [
                {
                    'tool_name': mem.tool_name,
                    'action': mem.action, 
                    'success': mem.success,
                    'confidence': mem.confidence,
                    'timestamp': mem.timestamp,
                    'strategy_used': mem.strategy_used
                } for mem in self.task_memory
            ],
            'export_timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"📊 Research data exported to {filepath}")