#!/usr/bin/env python3
"""
Construction RAG (Retrieval-Augmented Generation) Manager.

This module implements RAG capabilities for construction HRI by retrieving
relevant contextual information to enhance clarification responses.

Supports the History-Aware clarification strategy and Context-Aware Memory
framework by providing:
- Construction knowledge base retrieval
- Task context retrieval from memory
- Tool usage pattern analysis
- Expertise-level content filtering
- Transactive memory theory implementation

Integrates with ConstructionClarificationManager to provide enriched,
contextually-aware responses for construction robotics research.
"""

import logging
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import pickle
import os

# For embeddings and similarity search
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available - RAG will use mock implementation")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class ConstructionKnowledgeItem:
    """Individual knowledge item in construction domain"""
    id: str
    content: str
    category: str  # tool_usage, safety, procedure, terminology
    expertise_level: str  # apprentice, journeyman, foreman, master
    tools_involved: List[str]
    embedding: Optional[np.ndarray] = None
    usage_count: int = 0
    last_accessed: float = field(default_factory=time.time)

@dataclass
class RAGResponse:
    """Response from RAG retrieval with context"""
    enhanced_text: str
    original_text: str
    retrieved_items: List[ConstructionKnowledgeItem]
    confidence: float
    context_type: str  # knowledge_base, task_memory, usage_patterns

class ConstructionRAGManager:
    """
    RAG manager for construction domain knowledge retrieval and response enhancement.
    
    Implements Transactive Memory Theory by maintaining and accessing shared
    construction knowledge and task history for context-aware clarifications.
    
    Parameters
    ----------
    knowledge_base_path : str, optional
        Path to construction knowledge base file
    embedding_model : str, optional
        Sentence transformer model for embeddings, by default 'all-MiniLM-L6-v2'
    max_retrieved_items : int, optional
        Maximum items to retrieve per query, by default 3
    similarity_threshold : float, optional
        Minimum similarity for relevant items, by default 0.3
        
    Attributes
    ----------
    knowledge_base : List[ConstructionKnowledgeItem]
        Construction domain knowledge items
    embedding_model : SentenceTransformer or None
        Model for generating embeddings
    tool_usage_patterns : Dict
        Patterns of tool usage and sequences
    task_context_memory : Dict
        Contextual information from recent tasks
    """
    
    def __init__(self, 
                 knowledge_base_path: Optional[str] = None,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 max_retrieved_items: int = 3,
                 similarity_threshold: float = 0.3):
        
        self.logger = logging.getLogger(__name__)
        self.knowledge_base_path = knowledge_base_path
        self.max_retrieved_items = max_retrieved_items
        self.similarity_threshold = similarity_threshold
        
        # Initialize knowledge base
        self.knowledge_base: List[ConstructionKnowledgeItem] = []
        
        # Tool usage patterns for context
        self.tool_usage_patterns = defaultdict(list)
        self.task_context_memory = {}
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                self.logger.info(f"✅ RAG embedding model loaded: {embedding_model}")
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
            self.logger.warning("Using mock RAG implementation")
        
        # Load or create knowledge base
        self._initialize_knowledge_base()
        
        self.logger.info(f"✅ Construction RAG Manager initialized")
        self.logger.info(f"   Knowledge base: {len(self.knowledge_base)} items")
        self.logger.info(f"   Embedding model: {'Available' if self.embedding_model else 'Mock'}")

    def _initialize_knowledge_base(self):
        """Initialize construction knowledge base with domain expertise"""
        
        if self.knowledge_base_path and os.path.exists(self.knowledge_base_path):
            self._load_knowledge_base()
        else:
            self._create_default_knowledge_base()
        
        # Generate embeddings for all knowledge items
        if self.embedding_model:
            self._generate_embeddings()

    def _create_default_knowledge_base(self):
        """Create default construction knowledge base"""
        
        knowledge_items = [
            # Tool Usage Knowledge
            ConstructionKnowledgeItem(
                id="hammer_framing_001",
                content="Framing hammers are best for rough carpentry work. The 16 oz weight provides good balance for driving framing nails into lumber. Use straight claw for general framing work.",
                category="tool_usage",
                expertise_level="apprentice",
                tools_involved=["framing hammer", "nails", "lumber"]
            ),
            
            ConstructionKnowledgeItem(
                id="hammer_finish_002", 
                content="Finish hammers have smooth faces to prevent marring. Use for trim work and finish carpentry where appearance matters. Lighter weight provides better control.",
                category="tool_usage",
                expertise_level="journeyman",
                tools_involved=["finish hammer", "trim", "finish nails"]
            ),
            
            # Safety Knowledge
            ConstructionKnowledgeItem(
                id="safety_eye_001",
                content="Always wear safety glasses when using power tools or striking tools. Flying debris can cause serious eye injury. This is non-negotiable on construction sites.",
                category="safety",
                expertise_level="apprentice",
                tools_involved=["power tools", "hammer", "chisel"]
            ),
            
            ConstructionKnowledgeItem(
                id="safety_hearing_002",
                content="Hearing protection required when noise levels exceed 85dB. Power saws, drills, and pneumatic tools typically exceed this threshold. Use ear plugs or muffs.",
                category="safety",
                expertise_level="journeyman",
                tools_involved=["circular saw", "drill", "air tools"]
            ),
            
            # Procedures
            ConstructionKnowledgeItem(
                id="procedure_measuring_001",
                content="Measure twice, cut once. Always verify measurements before cutting expensive materials. Use a quality tape measure and mark clearly with a sharp pencil.",
                category="procedure",
                expertise_level="apprentice",
                tools_involved=["tape measure", "pencil", "saw"]
            ),
            
            ConstructionKnowledgeItem(
                id="procedure_square_002",
                content="Check for square using the 3-4-5 triangle method or a framing square. Critical for door and window rough openings. Out of square openings cause installation problems.",
                category="procedure", 
                expertise_level="journeyman",
                tools_involved=["framing square", "tape measure"]
            ),
            
            # Advanced Techniques
            ConstructionKnowledgeItem(
                id="technique_layout_001",
                content="Use a speed square for quick angle cuts and rafter layout. The pivot point allows for consistent angle marking. Essential for roof framing work.",
                category="procedure",
                expertise_level="foreman",
                tools_involved=["speed square", "circular saw", "rafters"]
            ),
            
            # Terminology and Jargon
            ConstructionKnowledgeItem(
                id="terminology_oc_001",
                content="O.C. means 'on center' - the distance between the centers of studs or joists. Standard framing uses 16 inch O.C. or 24 inch O.C. spacing for structural members.",
                category="terminology",
                expertise_level="apprentice", 
                tools_involved=["studs", "joists", "tape measure"]
            ),
            
            ConstructionKnowledgeItem(
                id="terminology_plumb_002",
                content="Plumb means perfectly vertical, level means perfectly horizontal. Use a level to check both. Critical for professional-looking construction work.",
                category="terminology",
                expertise_level="apprentice",
                tools_involved=["level", "plumb bob"]
            ),
            
            # Tool Combinations and Workflows
            ConstructionKnowledgeItem(
                id="workflow_rough_frame_001",
                content="Rough framing sequence: layout with chalk line and square, cut studs to length, assemble walls on deck, then raise and brace. Work systematically to maintain quality.",
                category="procedure",
                expertise_level="foreman",
                tools_involved=["chalk line", "circular saw", "framing hammer", "level"]
            ),
            
            # Troubleshooting
            ConstructionKnowledgeItem(
                id="troubleshoot_drill_001", 
                content="If drill bit wanders, start with a center punch or awl to create a starting dimple. Use proper cutting speed - too fast burns the bit, too slow can cause binding.",
                category="procedure",
                expertise_level="journeyman",
                tools_involved=["drill", "center punch", "drill bits"]
            ),
            
            # Material Knowledge
            ConstructionKnowledgeItem(
                id="material_lumber_001",
                content="2x4 actual dimensions are 1.5 x 3.5 inches. Nominal sizes vs actual sizes can confuse beginners. Always plan using actual dimensions for accuracy.",
                category="terminology",
                expertise_level="apprentice",
                tools_involved=["lumber", "tape measure"]
            )
        ]
        
        self.knowledge_base = knowledge_items
        self.logger.info(f"Created default knowledge base with {len(knowledge_items)} items")

    def _generate_embeddings(self):
        """Generate embeddings for all knowledge base items"""
        
        if not self.embedding_model:
            return
        
        try:
            texts = [item.content for item in self.knowledge_base]
            embeddings = self.embedding_model.encode(texts)
            
            for i, item in enumerate(self.knowledge_base):
                item.embedding = embeddings[i]
            
            self.logger.info(f"Generated embeddings for {len(self.knowledge_base)} knowledge items")
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")

    def enhance_clarification(self, 
                            original_text: str,
                            tool_request: str,
                            detected_tools: List[Dict],
                            user_expertise: str,
                            task_history: List[Dict] = None) -> RAGResponse:
        """
        Enhance clarification text with retrieved construction knowledge.
        
        Parameters
        ----------
        original_text : str
            Original clarification text
        tool_request : str
            User's tool request
        detected_tools : List[Dict] 
            Detected tools with metadata
        user_expertise : str
            User expertise level (apprentice, journeyman, foreman, master)
        task_history : List[Dict], optional
            Recent task history for context
            
        Returns
        -------
        RAGResponse
            Enhanced response with retrieved context
        """
        
        # Retrieve relevant knowledge
        relevant_items = self._retrieve_relevant_knowledge(
            query=f"{tool_request} {original_text}",
            tools=[tool.get('trade_term', '') for tool in detected_tools],
            expertise_level=user_expertise
        )
        
        # Enhance the response based on retrieved knowledge
        enhanced_text = self._generate_enhanced_response(
            original_text=original_text,
            relevant_items=relevant_items,
            user_expertise=user_expertise,
            context_type="knowledge_base"
        )
        
        # Calculate confidence based on retrieval quality
        confidence = self._calculate_enhancement_confidence(relevant_items)
        
        return RAGResponse(
            enhanced_text=enhanced_text,
            original_text=original_text,
            retrieved_items=relevant_items,
            confidence=confidence,
            context_type="knowledge_base"
        )

    def _retrieve_relevant_knowledge(self, 
                                   query: str,
                                   tools: List[str],
                                   expertise_level: str) -> List[ConstructionKnowledgeItem]:
        """Retrieve relevant knowledge items using semantic similarity"""
        
        if not self.embedding_model:
            # Mock retrieval for testing
            return self._mock_retrieve_knowledge(tools, expertise_level)
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Calculate similarities
            similarities = []
            for item in self.knowledge_base:
                if item.embedding is not None:
                    similarity = np.dot(query_embedding, item.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(item.embedding)
                    )
                    similarities.append((item, similarity))
            
            # Filter by similarity threshold and expertise level
            relevant_items = []
            for item, similarity in similarities:
                if similarity >= self.similarity_threshold:
                    # Check if any tools match
                    tool_match = any(tool.lower() in item.content.lower() for tool in tools)
                    
                    # Check expertise level compatibility
                    expertise_match = self._is_expertise_compatible(item.expertise_level, expertise_level)
                    
                    if tool_match or expertise_match:
                        item.usage_count += 1
                        item.last_accessed = time.time()
                        relevant_items.append(item)
            
            # Sort by similarity and limit results
            relevant_items.sort(key=lambda x: similarities[self.knowledge_base.index(x)][1], reverse=True)
            return relevant_items[:self.max_retrieved_items]
            
        except Exception as e:
            self.logger.error(f"Knowledge retrieval failed: {e}")
            return []

    def _mock_retrieve_knowledge(self, tools: List[str], expertise_level: str) -> List[ConstructionKnowledgeItem]:
        """Mock knowledge retrieval for testing without embeddings"""
        
        relevant_items = []
        
        for item in self.knowledge_base:
            # Simple keyword matching
            tool_match = any(tool.lower() in item.content.lower() for tool in tools if tool)
            expertise_match = self._is_expertise_compatible(item.expertise_level, expertise_level)
            
            if tool_match and len(relevant_items) < self.max_retrieved_items:
                item.usage_count += 1
                item.last_accessed = time.time()
                relevant_items.append(item)
        
        return relevant_items

    def _is_expertise_compatible(self, item_level: str, user_level: str) -> bool:
        """Check if knowledge item is appropriate for user expertise level"""
        
        levels = ['apprentice', 'journeyman', 'foreman', 'master']
        
        try:
            item_idx = levels.index(item_level.lower())
            user_idx = levels.index(user_level.lower())
            
            # Allow items at user level or below (simpler concepts)
            return item_idx <= user_idx
            
        except ValueError:
            return True  # Default to compatible if levels not recognized

    def _generate_enhanced_response(self, 
                                  original_text: str,
                                  relevant_items: List[ConstructionKnowledgeItem],
                                  user_expertise: str,
                                  context_type: str) -> str:
        """Generate enhanced response using retrieved knowledge"""
        
        if not relevant_items:
            return original_text
        
        # Select best knowledge item
        best_item = relevant_items[0]
        
        # Craft enhancement based on expertise level
        if user_expertise.lower() == 'apprentice':
            # Provide educational context
            if best_item.category == 'safety':
                enhancement = f"{original_text} Quick safety tip: {best_item.content.split('.')[0]}."
            elif best_item.category == 'terminology':
                enhancement = f"{original_text} Note: {best_item.content.split('.')[0]}."
            else:
                enhancement = f"{original_text} Pro tip: {best_item.content.split('.')[0]}."
                
        elif user_expertise.lower() == 'journeyman':
            # Provide contextual information
            if 'procedure' in best_item.category:
                enhancement = f"{original_text} Remember: {best_item.content.split('.')[0]}."
            else:
                enhancement = original_text
                
        elif user_expertise.lower() in ['foreman', 'master']:
            # Minimal enhancement for experienced users
            if best_item.category == 'safety' and 'safety' not in original_text.lower():
                enhancement = f"{original_text} Safety check required."
            else:
                enhancement = original_text
        else:
            enhancement = original_text
        
        return enhancement

    def _calculate_enhancement_confidence(self, retrieved_items: List[ConstructionKnowledgeItem]) -> float:
        """Calculate confidence score for the enhancement"""
        
        if not retrieved_items:
            return 0.0
        
        # Base confidence on number and quality of retrieved items
        base_confidence = min(len(retrieved_items) / self.max_retrieved_items, 1.0)
        
        # Adjust for item categories (safety gets higher confidence)
        category_boost = 0.0
        for item in retrieved_items:
            if item.category == 'safety':
                category_boost += 0.2
            elif item.category == 'procedure':
                category_boost += 0.1
        
        return min(base_confidence + category_boost, 1.0)

    def update_tool_usage_pattern(self, tool_sequence: List[str], task_type: str):
        """Update tool usage patterns for future context"""
        
        self.tool_usage_patterns[task_type].append({
            'sequence': tool_sequence,
            'timestamp': time.time()
        })
        
        # Keep only recent patterns (last 100)
        if len(self.tool_usage_patterns[task_type]) > 100:
            self.tool_usage_patterns[task_type] = self.tool_usage_patterns[task_type][-100:]

    def get_contextual_tool_suggestions(self, current_tool: str, task_type: str = "general") -> List[str]:
        """Get tool suggestions based on usage patterns"""
        
        if task_type not in self.tool_usage_patterns:
            return []
        
        # Find patterns containing current tool
        relevant_patterns = []
        for pattern in self.tool_usage_patterns[task_type]:
            if current_tool in pattern['sequence']:
                relevant_patterns.append(pattern)
        
        # Extract commonly used next tools
        next_tools = []
        for pattern in relevant_patterns:
            seq = pattern['sequence']
            try:
                current_idx = seq.index(current_tool)
                if current_idx < len(seq) - 1:
                    next_tools.append(seq[current_idx + 1])
            except ValueError:
                continue
        
        # Return most common next tools
        from collections import Counter
        tool_counts = Counter(next_tools)
        return [tool for tool, count in tool_counts.most_common(3)]

    def add_knowledge_item(self, item: ConstructionKnowledgeItem):
        """Add new knowledge item to the knowledge base"""
        
        self.knowledge_base.append(item)
        
        # Generate embedding if model available
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode([item.content])[0]
                item.embedding = embedding
            except Exception as e:
                self.logger.error(f"Failed to generate embedding for new item: {e}")
        
        self.logger.info(f"Added knowledge item: {item.id}")

    def save_knowledge_base(self, filepath: str):
        """Save knowledge base to file"""
        
        try:
            # Convert to serializable format
            kb_data = []
            for item in self.knowledge_base:
                item_dict = {
                    'id': item.id,
                    'content': item.content,
                    'category': item.category,
                    'expertise_level': item.expertise_level,
                    'tools_involved': item.tools_involved,
                    'usage_count': item.usage_count,
                    'last_accessed': item.last_accessed,
                    'embedding': item.embedding.tolist() if item.embedding is not None else None
                }
                kb_data.append(item_dict)
            
            with open(filepath, 'w') as f:
                json.dump(kb_data, f, indent=2)
            
            self.logger.info(f"Knowledge base saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save knowledge base: {e}")

    def _load_knowledge_base(self):
        """Load knowledge base from file"""
        
        try:
            with open(self.knowledge_base_path, 'r') as f:
                kb_data = json.load(f)
            
            self.knowledge_base = []
            for item_dict in kb_data:
                embedding = None
                if item_dict.get('embedding'):
                    embedding = np.array(item_dict['embedding'])
                
                item = ConstructionKnowledgeItem(
                    id=item_dict['id'],
                    content=item_dict['content'],
                    category=item_dict['category'],
                    expertise_level=item_dict['expertise_level'],
                    tools_involved=item_dict['tools_involved'],
                    embedding=embedding,
                    usage_count=item_dict.get('usage_count', 0),
                    last_accessed=item_dict.get('last_accessed', time.time())
                )
                self.knowledge_base.append(item)
            
            self.logger.info(f"Loaded knowledge base from {self.knowledge_base_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load knowledge base: {e}")
            self._create_default_knowledge_base()

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        
        stats = {
            'total_items': len(self.knowledge_base),
            'categories': defaultdict(int),
            'expertise_levels': defaultdict(int),
            'most_used_items': [],
            'recent_items': []
        }
        
        for item in self.knowledge_base:
            stats['categories'][item.category] += 1
            stats['expertise_levels'][item.expertise_level] += 1
        
        # Most used items
        sorted_by_usage = sorted(self.knowledge_base, key=lambda x: x.usage_count, reverse=True)
        stats['most_used_items'] = [(item.id, item.usage_count) for item in sorted_by_usage[:5]]
        
        # Recently accessed items
        sorted_by_access = sorted(self.knowledge_base, key=lambda x: x.last_accessed, reverse=True) 
        stats['recent_items'] = [(item.id, item.last_accessed) for item in sorted_by_access[:5]]
        
        return stats