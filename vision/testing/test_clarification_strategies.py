#!/usr/bin/env python3
"""
Test script for Construction Clarification Manager.

This script validates all five clarification strategies for construction HRI research:
1. Direct Strategy - Simple binary questions
2. History-Aware Strategy - Context from previous interactions
3. Confidence-Based Strategy - Graduated uncertainty expression  
4. Options-Based Strategy - Multiple choice clarifications
5. Expertise-Adaptive Strategy - User skill level adaptation
"""

import sys
import os
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_detections() -> tuple:
    """Create mock detection data for testing"""
    
    # No detections scenario
    no_detections = ([], [])
    
    # Single detection scenario
    single_detection = (
        [{'label': 'framing hammer', 'trade_term': 'framing hammer', 'category': 'striking_tool', 'bbox': [100, 100, 200, 200]}],
        [0.85]
    )
    
    # Multiple detections scenario
    multiple_detections = (
        [
            {'label': 'claw hammer', 'trade_term': 'claw hammer', 'category': 'striking_tool', 'bbox': [50, 50, 150, 150]},
            {'label': 'ball-peen hammer', 'trade_term': 'ball-peen hammer', 'category': 'striking_tool', 'bbox': [200, 100, 300, 200]},
            {'label': 'sledgehammer', 'trade_term': 'sledgehammer', 'category': 'striking_tool', 'bbox': [350, 80, 450, 180]}
        ],
        [0.72, 0.68, 0.45]
    )
    
    return no_detections, single_detection, multiple_detections

def test_all_strategies():
    """Test all five clarification strategies comprehensively"""
    
    try:
        from ConstructionClarificationManager import (
            ConstructionClarificationManager, 
            ClarificationStrategy,
            UserExpertiseLevel
        )
        
        logger.info("✅ ConstructionClarificationManager imported successfully")
        
        # Initialize manager
        manager = ConstructionClarificationManager(
            user_expertise=UserExpertiseLevel.JOURNEYMAN,
            confidence_threshold=0.6
        )
        
        # Get mock detection scenarios
        no_detections, single_detection, multiple_detections = create_mock_detections()
        
        # Test scenarios
        test_scenarios = [
            ("hammer", no_detections[0], no_detections[1], "No detections"),
            ("hammer", single_detection[0], single_detection[1], "Single detection"),
            ("hammer", multiple_detections[0], multiple_detections[1], "Multiple detections")
        ]
        
        logger.info("\n" + "="*60)
        logger.info("TESTING ALL CLARIFICATION STRATEGIES")
        logger.info("="*60)
        
        # Test each strategy with each scenario
        for strategy in ClarificationStrategy:
            logger.info(f"\n🔧 TESTING STRATEGY: {strategy.value.upper()}")
            logger.info("-" * 40)
            
            for tool_request, objects, confidences, scenario_name in test_scenarios:
                logger.info(f"\n📝 Scenario: {scenario_name}")
                
                response = manager.request_clarification(
                    tool_request=tool_request,
                    detected_objects=objects,
                    confidence_scores=confidences,
                    strategy=strategy
                )
                
                logger.info(f"   Response: '{response.text}'")
                logger.info(f"   Confidence: {response.confidence:.2f}")
                logger.info(f"   Metadata: {response.metadata}")
        
        logger.info("\n" + "="*60)
        logger.info("TESTING HISTORY-AWARE FUNCTIONALITY")
        logger.info("="*60)
        
        # Test history-aware functionality
        logger.info("\n📚 Building task memory...")
        manager.update_task_memory("framing hammer", "pickup", True, "good choice", "direct")
        manager.update_task_memory("Phillips screwdriver", "pickup", True, "worked well", "confidence_based")
        manager.update_task_memory("adjustable wrench", "pickup", False, "too small", "options_based")
        
        # Test history-aware response
        response = manager.request_clarification(
            tool_request="hammer",
            detected_objects=single_detection[0], 
            confidence_scores=single_detection[1],
            strategy=ClarificationStrategy.HISTORY_AWARE
        )
        logger.info(f"🧠 History-aware response: '{response.text}'")
        
        logger.info("\n" + "="*60)
        logger.info("TESTING EXPERTISE ADAPTATION")
        logger.info("="*60)
        
        # Test all expertise levels
        expertise_levels = [
            UserExpertiseLevel.APPRENTICE,
            UserExpertiseLevel.JOURNEYMAN, 
            UserExpertiseLevel.FOREMAN,
            UserExpertiseLevel.MASTER
        ]
        
        for expertise in expertise_levels:
            logger.info(f"\n👷 Testing expertise level: {expertise.value}")
            manager.update_user_expertise(expertise)
            
            response = manager.request_clarification(
                tool_request="screwdriver",
                detected_objects=single_detection[0],
                confidence_scores=single_detection[1],
                strategy=ClarificationStrategy.EXPERTISE_ADAPTIVE
            )
            
            logger.info(f"   Response: '{response.text}'")
            logger.info(f"   Metadata: {response.metadata}")
        
        logger.info("\n" + "="*60)
        logger.info("TESTING CONFIDENCE EXPRESSIONS")
        logger.info("="*60)
        
        # Test different confidence levels
        confidence_test_cases = [
            ([0.95], "High confidence"),
            ([0.75], "Medium confidence"), 
            ([0.45], "Low confidence"),
            ([0.15], "Very low confidence")
        ]
        
        for confidences, desc in confidence_test_cases:
            logger.info(f"\n📊 Testing {desc}: {confidences[0]:.2f}")
            
            response = manager.request_clarification(
                tool_request="drill",
                detected_objects=[single_detection[0][0]],  # Reuse single object
                confidence_scores=confidences,
                strategy=ClarificationStrategy.CONFIDENCE_BASED
            )
            
            logger.info(f"   Response: '{response.text}'")
        
        logger.info("\n" + "="*60)
        logger.info("TESTING CONSTRUCTION JARGON INTEGRATION")
        logger.info("="*60)
        
        # Test construction-specific terminology
        construction_terms = ["speed square", "skill saw", "crescent wrench", "sawzall"]
        
        for term in construction_terms:
            test_obj = {
                'label': term, 
                'trade_term': term, 
                'category': 'construction_tool',
                'bbox': [100, 100, 200, 200]
            }
            
            response = manager.request_clarification(
                tool_request=term,
                detected_objects=[test_obj],
                confidence_scores=[0.8],
                strategy=ClarificationStrategy.EXPERTISE_ADAPTIVE
            )
            
            logger.info(f"🔨 {term}: '{response.text}'")
        
        # Get performance metrics
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE METRICS")
        logger.info("="*60)
        
        metrics = manager.get_performance_metrics()
        logger.info(f"Total interactions: {metrics['total_interactions']}")
        logger.info(f"User expertise: {metrics['user_expertise']}")
        logger.info(f"Memory size: {metrics['memory_size']}")
        
        for strategy, perf in metrics['strategy_performance'].items():
            logger.info(f"{strategy}: {perf['usage_count']} uses")
        
        # Export research data
        export_path = "/tmp/clarification_test_data.json"
        manager.export_research_data(export_path)
        logger.info(f"📊 Research data exported to: {export_path}")
        
        logger.info("\n✅ Phase 1C Clarification Strategy test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_all_strategies()
    sys.exit(0 if success else 1)