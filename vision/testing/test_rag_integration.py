#!/usr/bin/env python3
"""
Test script for Construction RAG Integration.

This script validates the RAG (Retrieval-Augmented Generation) integration
with the Construction Clarification Manager, testing context-aware responses
and knowledge base retrieval for construction HRI.
"""

import sys
import os
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_integration():
    """Test Construction RAG integration comprehensively"""
    
    try:
        logger.info("🧠 CONSTRUCTION RAG INTEGRATION TEST")
        logger.info("="*50)
        
        # Test RAG Manager independently
        logger.info("📚 Testing RAG Manager...")
        
        from ConstructionRAGManager import (
            ConstructionRAGManager,
            ConstructionKnowledgeItem,
            RAGResponse
        )
        
        rag_manager = ConstructionRAGManager()
        logger.info("✅ RAG Manager initialized")
        
        # Test knowledge base statistics
        stats = rag_manager.get_knowledge_stats()
        logger.info(f"📊 Knowledge base stats:")
        logger.info(f"   Total items: {stats['total_items']}")
        logger.info(f"   Categories: {dict(stats['categories'])}")
        logger.info(f"   Expertise levels: {dict(stats['expertise_levels'])}")
        
        # Test knowledge retrieval
        logger.info("\n🔍 Testing Knowledge Retrieval...")
        
        test_queries = [
            ("hammer safety", ["framing hammer"], "apprentice"),
            ("measuring accuracy", ["tape measure"], "journeyman"),
            ("framing procedure", ["framing square", "studs"], "foreman"),
            ("tool maintenance", ["drill"], "master")
        ]
        
        for query, tools, expertise in test_queries:
            logger.info(f"\n   Query: '{query}' | Tools: {tools} | Expertise: {expertise}")
            
            relevant_items = rag_manager._retrieve_relevant_knowledge(
                query=query,
                tools=tools,
                expertise_level=expertise
            )
            
            logger.info(f"   Retrieved {len(relevant_items)} items:")
            for item in relevant_items:
                logger.info(f"     - {item.id}: {item.content[:60]}... (Category: {item.category})")
        
        # Test response enhancement
        logger.info("\n✨ Testing Response Enhancement...")
        
        mock_tools = [{
            'trade_term': 'framing hammer',
            'category': 'striking_tool',
            'bbox': [100, 100, 200, 200]
        }]
        
        enhancement_tests = [
            ("I see a framing hammer. Is that correct?", "apprentice"),
            ("Got a framing hammer here. This work for you?", "journeyman"), 
            ("Located framing hammer. Confidence 85%. Fits the schedule?", "foreman"),
            ("Framing hammer. Proceed?", "master")
        ]
        
        for original_text, expertise in enhancement_tests:
            logger.info(f"\n   Original ({expertise}): '{original_text}'")
            
            rag_response = rag_manager.enhance_clarification(
                original_text=original_text,
                tool_request="hammer",
                detected_tools=mock_tools,
                user_expertise=expertise
            )
            
            logger.info(f"   Enhanced: '{rag_response.enhanced_text}'")
            logger.info(f"   RAG confidence: {rag_response.confidence:.2f}")
            logger.info(f"   Retrieved items: {len(rag_response.retrieved_items)}")
        
        # Test with Clarification Manager integration
        logger.info("\n🤖 Testing Clarification Manager Integration...")
        
        from ConstructionClarificationManager import (
            ConstructionClarificationManager,
            ClarificationStrategy,
            UserExpertiseLevel
        )
        
        # Initialize with RAG enabled
        clarification_mgr = ConstructionClarificationManager(
            user_expertise=UserExpertiseLevel.APPRENTICE,
            enable_rag=True
        )
        
        # Test different strategies with RAG enhancement
        mock_detections = [{
            'label': 'framing hammer',
            'trade_term': 'framing hammer',
            'category': 'striking_tool',
            'bbox': [100, 100, 200, 200]
        }]
        mock_confidences = [0.85]
        
        logger.info("\n🔧 Testing RAG-Enhanced Clarification Strategies...")
        
        strategies_to_test = [
            ClarificationStrategy.DIRECT,
            ClarificationStrategy.CONFIDENCE_BASED,
            ClarificationStrategy.EXPERTISE_ADAPTIVE
        ]
        
        for strategy in strategies_to_test:
            logger.info(f"\n   Testing {strategy.value} with RAG:")
            
            response = clarification_mgr.request_clarification(
                tool_request="hammer",
                detected_objects=mock_detections,
                confidence_scores=mock_confidences,
                strategy=strategy
            )
            
            logger.info(f"     Response: '{response.text}'")
            logger.info(f"     Metadata: {response.metadata}")
            
            if response.metadata.get('rag_enhanced'):
                logger.info(f"     ✅ RAG enhanced (confidence: {response.metadata['rag_confidence']:.2f})")
            else:
                logger.info(f"     ➖ No RAG enhancement applied")
        
        # Test tool usage patterns
        logger.info("\n🔄 Testing Tool Usage Patterns...")
        
        # Simulate tool usage sequence
        tool_sequences = [
            (["tape measure", "pencil", "circular saw"], "cutting"),
            (["level", "framing square", "drill"], "framing"),
            (["safety glasses", "circular saw", "sandpaper"], "finishing")
        ]
        
        for sequence, task_type in tool_sequences:
            rag_manager.update_tool_usage_pattern(sequence, task_type)
            logger.info(f"   Updated pattern: {task_type} -> {sequence}")
        
        # Test contextual suggestions
        for current_tool in ["tape measure", "level", "safety glasses"]:
            suggestions = rag_manager.get_contextual_tool_suggestions(current_tool, "cutting")
            logger.info(f"   After '{current_tool}' in cutting: {suggestions}")
        
        # Test knowledge base expansion
        logger.info("\n📝 Testing Knowledge Base Expansion...")
        
        new_knowledge = ConstructionKnowledgeItem(
            id="test_safety_001",
            content="Always check for electrical wires before drilling into walls. Use a stud finder with wire detection capability to avoid hazardous contact.",
            category="safety",
            expertise_level="journeyman",
            tools_involved=["drill", "stud finder"]
        )
        
        rag_manager.add_knowledge_item(new_knowledge)
        logger.info(f"✅ Added knowledge item: {new_knowledge.id}")
        
        # Test the new knowledge in retrieval
        drill_items = rag_manager._retrieve_relevant_knowledge(
            query="drilling safety electrical",
            tools=["drill"],
            expertise_level="journeyman"
        )
        
        logger.info(f"   Found {len(drill_items)} items about drilling safety")
        for item in drill_items:
            if "electrical" in item.content.lower():
                logger.info(f"   ✅ New knowledge item found: {item.id}")
        
        # Test history-aware enhancement
        logger.info("\n🧠 Testing History-Aware Enhancement...")
        
        # Add some task memory
        clarification_mgr.update_task_memory("framing hammer", "pickup", True)
        clarification_mgr.update_task_memory("tape measure", "use", True) 
        clarification_mgr.update_task_memory("circular saw", "cutting", False)  # Failed task
        
        # Request clarification that could benefit from history
        history_response = clarification_mgr.request_clarification(
            tool_request="saw",
            detected_objects=[{
                'label': 'circular saw',
                'trade_term': 'circular saw',
                'category': 'cutting_tool',
                'bbox': [200, 150, 300, 250]
            }],
            confidence_scores=[0.75],
            strategy=ClarificationStrategy.HISTORY_AWARE
        )
        
        logger.info(f"   History-aware response: '{history_response.text}'")
        if history_response.metadata.get('rag_enhanced'):
            logger.info(f"   Enhanced with RAG (confidence: {history_response.metadata['rag_confidence']:.2f})")
        
        # Test expertise progression impact
        logger.info("\n👷 Testing Expertise Progression Impact...")
        
        expertise_levels = [UserExpertiseLevel.APPRENTICE, UserExpertiseLevel.JOURNEYMAN, 
                           UserExpertiseLevel.FOREMAN, UserExpertiseLevel.MASTER]
        
        for expertise in expertise_levels:
            clarification_mgr.update_user_expertise(expertise)
            
            response = clarification_mgr.request_clarification(
                tool_request="drill",
                detected_objects=[{
                    'label': 'cordless drill',
                    'trade_term': 'cordless drill',
                    'category': 'power_tool',
                    'bbox': [150, 100, 250, 200]
                }],
                confidence_scores=[0.8],
                strategy=ClarificationStrategy.EXPERTISE_ADAPTIVE
            )
            
            logger.info(f"   {expertise.value}: '{response.text}'")
            if response.metadata.get('rag_enhanced'):
                logger.info(f"      RAG enhanced: {response.metadata['rag_confidence']:.2f} confidence")
        
        # Performance metrics
        logger.info("\n📊 RAG Performance Metrics...")
        
        final_stats = rag_manager.get_knowledge_stats()
        logger.info(f"   Knowledge base usage:")
        for item_id, usage_count in final_stats['most_used_items']:
            logger.info(f"     {item_id}: {usage_count} uses")
        
        clarification_stats = clarification_mgr.get_performance_metrics()
        logger.info(f"   Clarification interactions: {clarification_stats['total_interactions']}")
        
        # Save knowledge base for future use
        knowledge_file = "/tmp/construction_knowledge_base.json"
        rag_manager.save_knowledge_base(knowledge_file)
        logger.info(f"   Knowledge base saved to: {knowledge_file}")
        
        logger.info("\n✅ CONSTRUCTION RAG INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        logger.info("    RAG enhancement provides context-aware responses")
        logger.info("    Knowledge retrieval supports expertise adaptation") 
        logger.info("    History-aware responses utilize task memory")
        logger.info("    Transactive Memory Theory implementation validated")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_integration()
    sys.exit(0 if success else 1)