#!/usr/bin/env python3
"""
Enhanced Construction RAG with ChromaDB and Sentence Transformers.

This advanced RAG implementation uses:
- ChromaDB for vector database storage
- Sentence Transformers for high-quality embeddings
- Comprehensive construction knowledge base
- Semantic search and retrieval
- Contextual response enhancement

Provides sophisticated knowledge retrieval for construction HRI research.
"""

import logging
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import uuid

# Vector database and embeddings
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class ConstructionKnowledgeItem:
    """Enhanced construction knowledge item"""
    id: str
    content: str
    category: str  
    expertise_level: str
    tools_involved: List[str]
    safety_critical: bool = False
    workflow_stage: Optional[str] = None  # planning, rough_work, finish_work, inspection
    trade_specialty: Optional[str] = None  # carpentry, electrical, plumbing, general
    common_mistakes: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    related_codes: List[str] = field(default_factory=list)  # Building codes
    usage_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RAGResponse:
    """Enhanced RAG response"""
    enhanced_text: str
    original_text: str
    retrieved_items: List[ConstructionKnowledgeItem]
    confidence: float
    context_type: str
    semantic_similarity_scores: List[float] = field(default_factory=list)
    retrieved_count: int = 0

class EnhancedConstructionRAG:
    """
    Advanced RAG system using ChromaDB and Sentence Transformers.
    
    Features:
    - Vector database with ChromaDB
    - Semantic search with sentence transformers
    - Comprehensive construction knowledge base
    - Multi-modal retrieval (text + metadata)
    - Contextual response enhancement
    
    Parameters
    ----------
    db_path : str, optional
        Path to ChromaDB database directory
    embedding_model_name : str, optional
        Sentence transformer model name
    collection_name : str, optional
        ChromaDB collection name
    """
    
    def __init__(self,
                 db_path: str = "./construction_knowledge_db",
                 embedding_model_name: str = "all-mpnet-base-v2",
                 collection_name: str = "construction_knowledge"):
        
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        
        # Initialize components
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        self.knowledge_items = {}  # Cache of knowledge items
        
        # Initialize vector database
        if CHROMADB_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
            self._initialize_vector_db()
            self._initialize_embedding_model()
            self._populate_knowledge_base()
        else:
            self.logger.warning("ChromaDB or SentenceTransformers not available - using fallback")
            self._initialize_fallback()
        
        self.logger.info("✅ Enhanced Construction RAG initialized")
        self.logger.info(f"   Database: {db_path}")
        self.logger.info(f"   Embedding Model: {embedding_model_name}")
        self.logger.info(f"   Knowledge Items: {len(self.knowledge_items)}")

    def _initialize_vector_db(self):
        """Initialize ChromaDB vector database"""
        
        try:
            # Create ChromaDB client
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                self.logger.info(f"✅ Loaded existing collection: {self.collection_name}")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Construction domain knowledge for HRI"}
                )
                self.logger.info(f"✅ Created new collection: {self.collection_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None

    def _initialize_embedding_model(self):
        """Initialize sentence transformer model"""
        
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.logger.info(f"✅ Embedding model loaded: {self.embedding_model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None

    def _initialize_fallback(self):
        """Initialize fallback implementation when dependencies unavailable"""
        
        self.knowledge_items = self._create_comprehensive_knowledge_base()
        self.logger.info("✅ Fallback RAG implementation initialized")

    def _create_comprehensive_knowledge_base(self) -> Dict[str, ConstructionKnowledgeItem]:
        """Create comprehensive construction knowledge base"""
        
        knowledge_items = {}
        
        # ============= TOOL KNOWLEDGE =============
        
        # Hammers
        knowledge_items["hammer_framing_001"] = ConstructionKnowledgeItem(
            id="hammer_framing_001",
            content="Framing hammers (16-20 oz) are essential for rough carpentry. The straight claw design excels at pulling nails and prying. Heavier weight drives nails efficiently but can cause fatigue. Proper grip prevents slippage and injury.",
            category="tool_usage",
            expertise_level="apprentice",
            tools_involved=["framing hammer", "nails", "lumber"],
            workflow_stage="rough_work",
            trade_specialty="carpentry",
            common_mistakes=["Using finish hammer for framing", "Improper grip causing blisters", "Over-swinging and missing nails"],
            prerequisites=["Basic hammer safety", "Nail selection knowledge"]
        )
        
        knowledge_items["hammer_finish_002"] = ConstructionKnowledgeItem(
            id="hammer_finish_002",
            content="Finish hammers (8-16 oz) have smooth faces to prevent wood marring. Lighter weight provides precision control for trim work. Curved claw design offers better leverage for delicate nail removal without surface damage.",
            category="tool_usage",
            expertise_level="journeyman",
            tools_involved=["finish hammer", "finish nails", "trim"],
            workflow_stage="finish_work",
            trade_specialty="carpentry",
            common_mistakes=["Using on rough framing", "Not pre-drilling hardwoods", "Wrong nail angle"]
        )
        
        # Power Tools
        knowledge_items["circular_saw_001"] = ConstructionKnowledgeItem(
            id="circular_saw_001",
            content="Circular saws require proper blade selection, depth setting, and guide usage. Blade depth should extend 1/4 inch below material. Always support both sides of cut to prevent binding. Kickback occurs when blade binds - maintain firm grip.",
            category="tool_usage",
            expertise_level="journeyman",
            tools_involved=["circular saw", "saw blades", "cutting guide"],
            safety_critical=True,
            workflow_stage="rough_work",
            trade_specialty="carpentry",
            common_mistakes=["Wrong blade for material", "Insufficient support", "Cutting without measuring twice"],
            prerequisites=["Power tool safety training", "PPE knowledge", "Material handling"]
        )
        
        knowledge_items["drill_techniques_001"] = ConstructionKnowledgeItem(
            id="drill_techniques_001",
            content="Drilling success depends on bit selection, speed control, and proper technique. Use pilot holes for screws near board ends. Variable speed prevents bit burning in metal. Clutch settings prevent over-driving screws.",
            category="tool_usage",
            expertise_level="journeyman",
            tools_involved=["cordless drill", "drill bits", "screws"],
            workflow_stage="rough_work",
            trade_specialty="general",
            common_mistakes=["Wrong bit for material", "Too fast on metal", "No pilot holes"],
            prerequisites=["Bit selection knowledge", "Material properties understanding"]
        )
        
        # ============= SAFETY KNOWLEDGE =============
        
        knowledge_items["ppe_requirements_001"] = ConstructionKnowledgeItem(
            id="ppe_requirements_001",
            content="Personal Protective Equipment is mandatory on construction sites. Minimum: hard hat, safety glasses, work boots, high-vis vest. Tool-specific PPE includes hearing protection (>85dB), respirators (dust/fumes), cut-resistant gloves.",
            category="safety",
            expertise_level="apprentice",
            tools_involved=["all_power_tools", "cutting_tools", "grinding_tools"],
            safety_critical=True,
            workflow_stage="all",
            trade_specialty="general",
            related_codes=["OSHA 1926", "Local safety codes"],
            common_mistakes=["Removing PPE for comfort", "Wrong PPE for task", "Damaged PPE use"]
        )
        
        knowledge_items["electrical_safety_001"] = ConstructionKnowledgeItem(
            id="electrical_safety_001",
            content="Electrical hazards are invisible and deadly. Always assume wires are live until tested. Use GFCI protection in wet areas. Lock-out/tag-out procedures prevent accidental energization. Only qualified electricians work on electrical systems.",
            category="safety", 
            expertise_level="foreman",
            tools_involved=["power_tools", "extension_cords", "electrical_equipment"],
            safety_critical=True,
            workflow_stage="all",
            trade_specialty="electrical",
            related_codes=["NEC", "OSHA 1926 Subpart K"],
            common_mistakes=["Using damaged cords", "Wet conditions", "Bypassing GFCI"]
        )
        
        # ============= PROCEDURES & TECHNIQUES =============
        
        knowledge_items["measuring_accuracy_001"] = ConstructionKnowledgeItem(
            id="measuring_accuracy_001",
            content="'Measure twice, cut once' prevents costly mistakes. Use quality measuring tools, mark with sharp pencil, and verify critical dimensions. For long measurements, use chalk line or laser level. Account for material thickness in calculations.",
            category="procedure",
            expertise_level="apprentice",
            tools_involved=["tape_measure", "pencil", "square", "chalk_line"],
            workflow_stage="planning",
            trade_specialty="general",
            common_mistakes=["Dull pencil marks", "Not accounting for kerf", "Rushed measurements"],
            prerequisites=["Basic math skills", "Tool familiarity"]
        )
        
        knowledge_items["layout_techniques_001"] = ConstructionKnowledgeItem(
            id="layout_techniques_001",
            content="Accurate layout is foundation of quality work. Use 3-4-5 triangle method for square corners. String lines for long straight runs. Laser levels for precise elevation work. Mark clearly and consistently throughout project.",
            category="procedure",
            expertise_level="journeyman", 
            tools_involved=["tape_measure", "string_line", "laser_level", "square"],
            workflow_stage="planning",
            trade_specialty="carpentry",
            common_mistakes=["Not checking for square", "Unclear markings", "Accumulating errors"],
            prerequisites=["Geometry basics", "Tool proficiency"]
        )
        
        # ============= MATERIAL KNOWLEDGE =============
        
        knowledge_items["lumber_properties_001"] = ConstructionKnowledgeItem(
            id="lumber_properties_001",
            content="Dimensional lumber actual sizes differ from nominal: 2x4 is actually 1.5x3.5 inches. Moisture content affects shrinkage. Grade stamps indicate strength properties. Pressure-treated lumber resists decay but requires special fasteners.",
            category="material",
            expertise_level="apprentice",
            tools_involved=["lumber", "fasteners"],
            workflow_stage="planning",
            trade_specialty="carpentry",
            related_codes=["IRC", "Building codes"],
            common_mistakes=["Using nominal dimensions", "Wrong lumber grade", "Improper storage"],
            prerequisites=["Material properties understanding"]
        )
        
        knowledge_items["fastener_selection_001"] = ConstructionKnowledgeItem(
            id="fastener_selection_001",
            content="Fastener selection affects structural integrity. Framing nails for structural connections, finish nails for trim. Screw types: wood screws for wood, self-tapping for metal. Length should penetrate at least 1.5 times material thickness.",
            category="material",
            expertise_level="journeyman",
            tools_involved=["nails", "screws", "bolts"],
            workflow_stage="rough_work",
            trade_specialty="general",
            related_codes=["IRC Table R602.3", "Fastener specifications"],
            common_mistakes=["Insufficient penetration", "Wrong fastener type", "Over-driven fasteners"]
        )
        
        # ============= TROUBLESHOOTING =============
        
        knowledge_items["common_errors_001"] = ConstructionKnowledgeItem(
            id="common_errors_001",
            content="Common framing errors: walls out of plumb, incorrect stud spacing, missing fire blocking. Prevention: double-check measurements, use proper bracing during construction, follow structural drawings carefully.",
            category="troubleshooting",
            expertise_level="foreman",
            tools_involved=["level", "tape_measure", "plumb_bob"],
            workflow_stage="rough_work", 
            trade_specialty="carpentry",
            common_mistakes=["Rushing framing process", "Not checking for plumb", "Missing structural elements"],
            prerequisites=["Framing knowledge", "Blueprint reading"]
        )
        
        knowledge_items["tool_maintenance_001"] = ConstructionKnowledgeItem(
            id="tool_maintenance_001",
            content="Tool maintenance extends life and ensures safety. Keep cutting tools sharp, oil moving parts, clean after use. Replace damaged components immediately. Proper storage prevents rust and damage. Regular inspection identifies problems early.",
            category="maintenance",
            expertise_level="journeyman",
            tools_involved=["all_tools"],
            workflow_stage="all",
            trade_specialty="general",
            safety_critical=True,
            common_mistakes=["Using dull blades", "Neglecting maintenance", "Improper storage"],
            prerequisites=["Tool knowledge", "Maintenance procedures"]
        )
        
        # ============= TRADE-SPECIFIC KNOWLEDGE =============
        
        knowledge_items["electrical_basics_001"] = ConstructionKnowledgeItem(
            id="electrical_basics_001",
            content="Basic electrical: black=hot, white=neutral, green/bare=ground. 15A circuits for lighting, 20A for outlets. GFCI required in wet areas. Arc-fault protection for bedrooms. Only licensed electricians do electrical work.",
            category="trade_specific",
            expertise_level="foreman",
            tools_involved=["electrical_tools", "wire", "outlets"],
            workflow_stage="rough_work",
            trade_specialty="electrical",
            safety_critical=True,
            related_codes=["NEC", "Local electrical codes"],
            common_mistakes=["Wrong wire gauge", "Missing GFCI", "Improper grounding"]
        )
        
        knowledge_items["plumbing_basics_001"] = ConstructionKnowledgeItem(
            id="plumbing_basics_001",
            content="Plumbing basics: slope drain lines 1/4 inch per foot minimum. Vent every fixture. Use appropriate materials for application. PVC for drain/waste/vent, copper or PEX for supply. Pressure test all connections.",
            category="trade_specific",
            expertise_level="foreman",
            tools_involved=["pipe", "fittings", "pipe_wrench"],
            workflow_stage="rough_work",
            trade_specialty="plumbing",
            related_codes=["IPC", "Local plumbing codes"],
            common_mistakes=["Insufficient slope", "Missing vents", "Wrong materials"],
            prerequisites=["Plumbing code knowledge", "Material compatibility"]
        )
        
        # ============= ADVANCED TECHNIQUES =============
        
        knowledge_items["advanced_framing_001"] = ConstructionKnowledgeItem(
            id="advanced_framing_001",
            content="Advanced framing techniques reduce lumber usage while maintaining strength. Optimum Value Engineering (OVE): 2-stud corners, single top plate with engineered connections, 24-inch on-center spacing where allowed by code.",
            category="advanced_technique",
            expertise_level="master",
            tools_involved=["framing_lumber", "structural_connectors"],
            workflow_stage="rough_work",
            trade_specialty="carpentry",
            related_codes=["IRC", "Engineered lumber specifications"],
            prerequisites=["Advanced framing knowledge", "Engineering principles"],
            common_mistakes=["Applying without proper design", "Code violations", "Inadequate connections"]
        )
        
        knowledge_items["energy_efficiency_001"] = ConstructionKnowledgeItem(
            id="energy_efficiency_001",
            content="Energy-efficient construction requires attention to thermal bridging, air sealing, and insulation continuity. Continuous insulation reduces thermal bridging. Blower door testing verifies air sealing effectiveness.",
            category="advanced_technique",
            expertise_level="master",
            tools_involved=["insulation", "air_barrier_materials", "caulk"],
            workflow_stage="finish_work",
            trade_specialty="general",
            related_codes=["IECC", "Energy codes"],
            prerequisites=["Building science knowledge", "Energy code familiarity"]
        )
        
        return knowledge_items

    def _populate_knowledge_base(self):
        """Populate ChromaDB with knowledge items"""
        
        if not (self.collection and self.embedding_model):
            self.logger.warning("Cannot populate knowledge base - missing components")
            return
        
        # Check if collection already has data
        try:
            count = self.collection.count()
            if count > 0:
                self.logger.info(f"Collection already contains {count} items")
                return
        except:
            pass
        
        # Create comprehensive knowledge base
        knowledge_items = self._create_comprehensive_knowledge_base()
        self.knowledge_items = knowledge_items
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for item_id, item in knowledge_items.items():
            documents.append(item.content)
            
            metadata = {
                "category": item.category,
                "expertise_level": item.expertise_level,
                "tools_involved": ",".join(item.tools_involved),
                "safety_critical": item.safety_critical,
                "workflow_stage": item.workflow_stage or "",
                "trade_specialty": item.trade_specialty or "",
                "common_mistakes": ",".join(item.common_mistakes),
                "prerequisites": ",".join(item.prerequisites),
                "related_codes": ",".join(item.related_codes)
            }
            
            metadatas.append(metadata)
            ids.append(item_id)
        
        # Generate embeddings
        self.logger.info("Generating embeddings for knowledge base...")
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        self.logger.info(f"✅ Added {len(documents)} items to knowledge base")

    def retrieve_relevant_knowledge(self, 
                                  query: str,
                                  tool_context: List[str] = None,
                                  expertise_level: str = "journeyman",
                                  n_results: int = 3) -> List[ConstructionKnowledgeItem]:
        """Retrieve relevant knowledge using semantic search"""
        
        if self.collection and self.embedding_model:
            return self._chromadb_retrieve(query, tool_context, expertise_level, n_results)
        else:
            return self._fallback_retrieve(query, tool_context, expertise_level, n_results)

    def _chromadb_retrieve(self, query: str, tool_context: List[str], 
                          expertise_level: str, n_results: int) -> List[ConstructionKnowledgeItem]:
        """Retrieve using ChromaDB semantic search"""
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Build metadata filter
            where_filter = {}
            if tool_context:
                # Filter by tools involved - ChromaDB doesn't support $contains
                # We'll do a broader search and filter in Python
                # For now, skip metadata filtering for tools and filter after retrieval
                pass  # No where filter for tools - we'll filter in post-processing
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert results to ConstructionKnowledgeItem objects
            retrieved_items = []
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )):
                item_id = results['ids'][0][i]
                
                # Get cached item or create new one
                if item_id in self.knowledge_items:
                    item = self.knowledge_items[item_id]
                else:
                    item = ConstructionKnowledgeItem(
                        id=item_id,
                        content=doc,
                        category=metadata.get('category', ''),
                        expertise_level=metadata.get('expertise_level', ''),
                        tools_involved=metadata.get('tools_involved', '').split(','),
                        safety_critical=metadata.get('safety_critical', False),
                        workflow_stage=metadata.get('workflow_stage', ''),
                        trade_specialty=metadata.get('trade_specialty', ''),
                        common_mistakes=metadata.get('common_mistakes', '').split(','),
                        prerequisites=metadata.get('prerequisites', '').split(','),
                        related_codes=metadata.get('related_codes', '').split(',')
                    )
                
                # Filter by tool context if specified
                if tool_context:
                    # Check if any of the requested tools are in this item's tools
                    item_tools = [tool.strip().lower() for tool in item.tools_involved]
                    context_tools = [tool.strip().lower() for tool in tool_context]
                    
                    if not any(tool in item_tools for tool in context_tools):
                        continue  # Skip this item if no tool match
                
                item.usage_count += 1
                item.last_accessed = time.time()
                retrieved_items.append(item)
            
            return retrieved_items
            
        except Exception as e:
            self.logger.error(f"ChromaDB retrieval failed: {e}")
            return self._fallback_retrieve(query, tool_context, expertise_level, n_results)

    def _fallback_retrieve(self, query: str, tool_context: List[str], 
                          expertise_level: str, n_results: int) -> List[ConstructionKnowledgeItem]:
        """Fallback retrieval using simple text matching"""
        
        if not self.knowledge_items:
            self.knowledge_items = self._create_comprehensive_knowledge_base()
        
        # Simple keyword-based matching
        query_words = set(query.lower().split())
        scored_items = []
        
        for item in self.knowledge_items.values():
            score = 0
            content_words = set(item.content.lower().split())
            
            # Content similarity
            score += len(query_words.intersection(content_words)) * 2
            
            # Tool context bonus
            if tool_context:
                for tool in tool_context:
                    if tool.lower() in item.content.lower():
                        score += 5
                    if tool in item.tools_involved:
                        score += 3
            
            # Expertise level matching
            if item.expertise_level == expertise_level:
                score += 2
            
            if score > 0:
                scored_items.append((score, item))
        
        # Sort by score and return top results
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored_items[:n_results]]

    def enhance_clarification(self, 
                            original_text: str,
                            tool_request: str,
                            detected_tools: List[Dict],
                            user_expertise: str = "journeyman",
                            task_history: List[Dict] = None) -> RAGResponse:
        """Enhance clarification with retrieved knowledge"""
        
        # Extract tool context
        tool_context = []
        if detected_tools:
            tool_context.extend([tool.get('trade_term', tool.get('label', '')) for tool in detected_tools])
        
        # Retrieve relevant knowledge
        retrieved_items = self.retrieve_relevant_knowledge(
            query=f"{tool_request} {original_text}",
            tool_context=tool_context,
            expertise_level=user_expertise,
            n_results=2
        )
        
        # Enhance response based on retrieved knowledge
        enhanced_text = original_text
        context_type = "knowledge_base"
        
        if retrieved_items:
            # Add relevant context from knowledge base
            relevant_info = []
            
            for item in retrieved_items:
                if item.safety_critical:
                    relevant_info.append(f"Safety note: {item.content.split('.')[0]}.")
                elif item.category == "tool_usage" and tool_context:
                    relevant_info.append(f"Pro tip: {item.content.split('.')[0]}.")
                elif item.category == "procedure":
                    relevant_info.append(f"Remember: {item.content.split('.')[0]}.")
            
            if relevant_info:
                enhanced_text = f"{original_text} {' '.join(relevant_info[:1])}"
        
        return RAGResponse(
            enhanced_text=enhanced_text,
            original_text=original_text,
            retrieved_items=retrieved_items,
            confidence=0.7 if retrieved_items else 0.3,
            context_type=context_type,
            retrieved_count=len(retrieved_items)
        )

    # ============= HRI INTEGRATION METHODS =============
    
    def process_asr_for_object_detection(self, 
                                       spoken_command: str,
                                       user_expertise: str = "journeyman") -> Dict[str, Any]:
        """
        Process ASR output to aid OWL-ViT object detection with construction lingo.
        
        This method translates construction slang/terms into standard object labels
        that OWL-ViT can better recognize, and provides context for disambiguation.
        
        Args:
            spoken_command: Raw ASR text from construction worker
            user_expertise: Worker's expertise level for appropriate translation
            
        Returns:
            Dict with processed command, target objects, and detection hints
        """
        
        # Construction lingo mapping for OWL-ViT
        construction_lingo_map = {
            # Lumber terminology
            "2x4": ["2x4 lumber", "wooden stud", "framing lumber"],
            "two by four": ["2x4 lumber", "wooden stud", "framing lumber"], 
            "stud": ["2x4 lumber", "wooden stud", "vertical lumber"],
            "joist": ["floor joist", "ceiling joist", "horizontal lumber"],
            "beam": ["support beam", "wooden beam", "structural lumber"],
            "plank": ["wooden plank", "lumber board", "construction wood"],
            
            # Tool slang
            "framing hammer": ["framing hammer", "16 oz framing hammer", "straight claw hammer"],
            "sawzall": ["reciprocating saw", "demolition saw"],
            "skilsaw": ["circular saw", "power saw"],
            "cat's paw": ["nail puller", "pry bar"],
            "pig": ["nail gun", "pneumatic nailer"],
            "chopsaw": ["miter saw", "cut-off saw"],
            "grinder": ["angle grinder", "disc grinder"],
            
            # Fastener terminology
            "sixteen penny": ["16d nail", "3.5 inch nail", "framing nail"],
            "eightpenny": ["8d nail", "2.5 inch nail", "common nail"],
            "sinker": ["sinker nail", "coated nail", "framing nail"],
            "duplex": ["duplex nail", "double-head nail", "form nail"],
            
            # Hardware
            "lag": ["lag screw", "lag bolt", "heavy duty screw"],
            "carriage bolt": ["round head bolt", "carriage bolt"],
            "toggle": ["toggle bolt", "hollow wall anchor"],
            
            # Electrical
            "romex": ["electrical cable", "NM cable", "house wire"],
            "pig tail": ["wire connector", "electrical splice"],
            
            # General construction
            "sheet rock": ["drywall", "gypsum board", "wall board"],
            "rock": ["drywall", "gypsum board"],
        }
        
        # Extract construction terms and convert to detection targets
        command_lower = spoken_command.lower()
        detected_terms = []
        target_objects = []
        
        for slang, standard_terms in construction_lingo_map.items():
            if slang in command_lower:
                detected_terms.append(slang)
                target_objects.extend(standard_terms)
        
        # Get relevant knowledge for context
        relevant_knowledge = self.retrieve_relevant_knowledge(
            query=spoken_command,
            tool_context=target_objects,
            expertise_level=user_expertise,
            n_results=2
        )
        
        # Generate detection hints based on knowledge
        detection_hints = []
        safety_notes = []
        
        for item in relevant_knowledge:
            if item.safety_critical:
                safety_notes.append(f"Safety: {item.content.split('.')[0]}")
            
            # Add specific visual features for better detection
            if "hammer" in item.content.lower():
                detection_hints.append("Look for: claw hammer, framing hammer, handle and head")
            elif "saw" in item.content.lower():
                detection_hints.append("Look for: blade, handle, power cord (if electric)")
            elif "lumber" in item.content.lower() or "wood" in item.content.lower():
                detection_hints.append("Look for: rectangular wood pieces, grain pattern")
        
        return {
            "original_command": spoken_command,
            "processed_command": spoken_command,  # Could add grammar correction here
            "detected_construction_terms": detected_terms,
            "target_objects_for_owlvit": list(set(target_objects)),  # Remove duplicates
            "detection_hints": detection_hints,
            "safety_notes": safety_notes,
            "relevant_knowledge_items": len(relevant_knowledge),
            "expertise_context": user_expertise
        }
    
    def generate_camera_clarification_response(self,
                                             detected_objects: List[Dict],
                                             original_request: str,
                                             camera_fov_info: Dict,
                                             user_expertise: str = "journeyman") -> str:
        """
        Generate TTS-ready clarification responses when camera sees multiple objects
        or needs to ask for clarification about what the construction worker wants.
        
        Args:
            detected_objects: List of objects detected by OWL-ViT with confidence scores
            original_request: Original spoken command from worker
            camera_fov_info: Information about camera field of view and object positions
            user_expertise: Worker expertise level for appropriate response complexity
            
        Returns:
            TTS-ready clarification text
        """
        
        if not detected_objects:
            # No objects detected
            base_response = f"I don't see any {self._extract_target_from_request(original_request)} in my current view."
            
            # Add helpful suggestions based on knowledge
            suggestions = self._get_search_suggestions(original_request, user_expertise)
            if suggestions:
                base_response += f" {suggestions}"
            
            return base_response
        
        elif len(detected_objects) == 1:
            # One object detected - confirm
            obj = detected_objects[0]
            confidence_level = "clearly" if obj.get('confidence', 0) > 0.8 else "possibly"
            
            # Get knowledge about proper handling
            knowledge_items = self.retrieve_relevant_knowledge(
                query=f"{obj.get('label', '')} handling safety",
                tool_context=[obj.get('label', '')],
                expertise_level=user_expertise,
                n_results=1
            )
            
            response = f"I can see {confidence_level} a {obj.get('label', 'item')}."
            
            # Add safety note if relevant
            if knowledge_items and knowledge_items[0].safety_critical:
                safety_tip = knowledge_items[0].content.split('.')[0]
                response += f" Safety reminder: {safety_tip}."
            
            response += " Should I pick it up?"
            return response
            
        else:
            # Multiple objects detected - ask for clarification
            object_descriptions = []
            
            for i, obj in enumerate(detected_objects, 1):
                label = obj.get('label', 'item')
                position = self._describe_position(obj.get('bbox', []), camera_fov_info)
                confidence = obj.get('confidence', 0)
                
                if confidence > 0.8:
                    object_descriptions.append(f"{label} {position}")
                else:
                    object_descriptions.append(f"what appears to be a {label} {position}")
            
            # Construct clarification based on expertise level
            if user_expertise in ["apprentice", "journeyman"]:
                # More descriptive for less experienced workers
                if len(object_descriptions) == 2:
                    response = f"I can see two options: a {object_descriptions[0]} and a {object_descriptions[1]}. Which one would you like me to get?"
                else:
                    response = f"I can see {len(object_descriptions)} options: "
                    response += ", ".join(object_descriptions[:-1]) + f", and a {object_descriptions[-1]}. "
                    response += "Which specific one do you need?"
            else:
                # More concise for experienced workers (foreman/master)
                response = f"I see {len(detected_objects)} {self._extract_target_from_request(original_request)}s. "
                response += "Which one - " + " or ".join([desc.split()[0] for desc in object_descriptions]) + "?"
            
            # Add relevant context from knowledge base
            target = self._extract_target_from_request(original_request)
            knowledge_items = self.retrieve_relevant_knowledge(
                query=f"{target} selection criteria",
                tool_context=[target],
                expertise_level=user_expertise,
                n_results=1
            )
            
            if knowledge_items and "selection" not in knowledge_items[0].content.lower():
                # Add helpful selection tip
                selection_tip = knowledge_items[0].content.split('.')[0]
                response += f" Pro tip: {selection_tip}."
            
            return response
    
    def _extract_target_from_request(self, request: str) -> str:
        """Extract the main target object from a spoken request"""
        # Simple extraction - could be enhanced with more sophisticated NLP
        request_lower = request.lower()
        
        # Common construction request patterns
        if "2x4" in request_lower or "two by four" in request_lower:
            return "2x4"
        elif "hammer" in request_lower:
            return "hammer"
        elif "saw" in request_lower:
            return "saw"
        elif "drill" in request_lower:
            return "drill"
        elif "screw" in request_lower:
            return "screw"
        elif "nail" in request_lower:
            return "nail"
        else:
            # Fallback - extract noun after "get", "grab", "bring"
            words = request_lower.split()
            for i, word in enumerate(words):
                if word in ["get", "grab", "bring", "fetch"] and i + 1 < len(words):
                    return words[i + 1]
            return "item"
    
    def _get_search_suggestions(self, original_request: str, expertise_level: str) -> str:
        """Generate helpful search suggestions when no objects are detected"""
        target = self._extract_target_from_request(original_request)
        
        suggestions = {
            "apprentice": f"Try looking in the tool area or checking if the {target} might be on the workbench.",
            "journeyman": f"Check the usual spots for {target}s - tool rack, workbench, or material storage.",
            "foreman": f"The {target} might be in a different work area.",
            "master": f"Consider if the {target} is stored elsewhere."
        }
        
        return suggestions.get(expertise_level, suggestions["journeyman"])
    
    def _describe_position(self, bbox: List[float], camera_info: Dict) -> str:
        """Describe object position in camera field of view in construction-friendly terms"""
        if not bbox or len(bbox) < 4:
            return "in view"
        
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Assume normalized coordinates or convert based on camera_info
        image_width = camera_info.get('width', 1.0)
        image_height = camera_info.get('height', 1.0)
        
        # Horizontal position
        if center_x < 0.33:
            horizontal = "on the left"
        elif center_x > 0.67:
            horizontal = "on the right"
        else:
            horizontal = "in the center"
        
        # Vertical position  
        if center_y < 0.33:
            vertical = "towards the top"
        elif center_y > 0.67:
            vertical = "towards the bottom"
        else:
            vertical = "in the middle"
        
        # Combine for natural description
        if horizontal == "in the center" and vertical == "in the middle":
            return "right in front"
        elif horizontal == "in the center":
            return vertical
        elif vertical == "in the middle":
            return horizontal
        else:
            return f"{horizontal}, {vertical}"

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        
        if not self.knowledge_items:
            return {"error": "Knowledge base not initialized"}
        
        stats = {
            "total_items": len(self.knowledge_items),
            "categories": defaultdict(int),
            "expertise_levels": defaultdict(int),
            "trade_specialties": defaultdict(int),
            "safety_critical_items": 0,
            "most_accessed": []
        }
        
        for item in self.knowledge_items.values():
            stats["categories"][item.category] += 1
            stats["expertise_levels"][item.expertise_level] += 1
            if item.trade_specialty:
                stats["trade_specialties"][item.trade_specialty] += 1
            if item.safety_critical:
                stats["safety_critical_items"] += 1
        
        # Convert defaultdicts to regular dicts
        stats["categories"] = dict(stats["categories"])
        stats["expertise_levels"] = dict(stats["expertise_levels"])
        stats["trade_specialties"] = dict(stats["trade_specialties"])
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    # Create enhanced RAG system
    rag = EnhancedConstructionRAG()
    
    # Test retrieval
    query = "How to use a framing hammer safely?"
    results = rag.retrieve_relevant_knowledge(query, ["framing hammer"], "apprentice")
    
    print(f"Query: {query}")
    print(f"Retrieved {len(results)} items:")
    for item in results:
        print(f"- {item.id}: {item.content[:100]}...")
    
    # Test enhancement
    response = rag.enhance_clarification(
        "I found a hammer. Is this correct?",
        "get me a hammer",
        [{"label": "framing hammer", "trade_term": "framing hammer"}],
        "apprentice"
    )
    
    print(f"\nOriginal: {response.original_text}")
    print(f"Enhanced: {response.enhanced_text}")
    
    # Print statistics
    stats = rag.get_knowledge_stats()
    print(f"\nKnowledge Base Stats:")
    print(f"Total items: {stats['total_items']}")
    print(f"Categories: {stats['categories']}")
    print(f"Safety critical: {stats['safety_critical_items']}")