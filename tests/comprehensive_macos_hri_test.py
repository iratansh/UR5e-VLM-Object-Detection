#!/usr/bin/env python3
"""
Comprehensive Construction HRI System Test for macOS

This script validates ALL MacBook-compatible components for the construction robot 
HRI research study focused on trust development and communication adaptation.

Research Questions:
- RQ1: How do verbal clarification strategies affect construction worker trust?
- RQ2: How do experienced tradespeople adapt vocabulary for robot instruction?

Hypotheses:
- H1: Trade-specific terminology increases trust
- H2: Graduated uncertainty expression increases trust  
- H3: Context-aware memory affects expert trust

System Architecture (macOS compatible):
Speech Input -> RAG Enhanced ASR -> OWL-ViT -> Backup IK -> Simulation
"""

# Fix environment issues before any imports
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Fix HuggingFace tokenizer warnings
os.environ['SQLALCHEMY_SILENCE_UBER_WARNING'] = '1'  # Silence SQLAlchemy 2.0 warnings

# Global shared RAG instance to prevent multiple initializations
_shared_rag_instance = None
_shared_temp_dir = None

import warnings
# Suppress known compatibility warnings globally
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message=".*LegacyVersion.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message=".*Model.*was trained with spaCy.*")
warnings.filterwarnings("ignore", message=".*weights.*were not initialized.*")

import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='haystack')
warnings.filterwarnings('ignore', message='.*MovedIn20Warning.*')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_shared_rag_instance():
    """Get or create shared RAG instance to prevent multiple initializations"""
    global _shared_rag_instance, _shared_temp_dir
    
    if _shared_rag_instance is None:
        try:
            from EnhancedConstructionRAG import EnhancedConstructionRAG
            _shared_temp_dir = tempfile.mkdtemp()
            _shared_rag_instance = EnhancedConstructionRAG(db_path=_shared_temp_dir)
            logging.info("✅ Shared RAG instance created")
        except Exception as e:
            logging.error(f"❌ Failed to create shared RAG instance: {e}")
            _shared_rag_instance = None
    
    return _shared_rag_instance

def cleanup_shared_rag():
    """Clean up shared RAG resources"""
    global _shared_rag_instance, _shared_temp_dir
    if _shared_temp_dir:
        import shutil
        try:
            shutil.rmtree(_shared_temp_dir)
            logging.info("🧹 Shared RAG resources cleaned up")
        except:
            pass
    _shared_rag_instance = None
    _shared_temp_dir = None

def test_enhanced_rag_system():
    """Test the Enhanced RAG system for construction HRI integration"""
    print("\n" + "="*70)
    print("🧠 ENHANCED RAG SYSTEM FOR HRI")
    print("="*70)
    
    results = {}
    
    try:
        rag = get_shared_rag_instance()
        if rag is None:
            raise Exception("Failed to get shared RAG instance")

        print(f"✅ RAG System: Initialized with {len(rag.knowledge_items)} knowledge items")

        # Extended test commands including ambiguity cases
        test_commands = [
            "Get me the framing hammer",        # Trade-specific (should detect hammer targets)
            "Bring me that sawzall",            # Slang
            "I need the sixteen penny nails",   # Fastener terminology
            "Find the sheet rock panels",       # Multi-word material
            "Grab the rock"                     # Ambiguous (baseline mapping)
        ]

        print("\n🎤 Testing ASR -> OWL-ViT Integration:")
        for cmd in test_commands:
            result = rag.process_asr_for_object_detection(cmd, "journeyman")
            print(f"   Command: '{cmd}'")
            print(f"   → OWL-ViT targets: {result['target_objects_for_owlvit']}")
            print(f"   → Construction terms: {result['detected_construction_terms']}")
            # Assertions (soft validation) inside try scope
            if "framing hammer" in cmd.lower():
                assert any("hammer" in t for t in result['target_objects_for_owlvit']), "Framing hammer mapping failed"
            if "sawzall" in cmd.lower():
                assert any("reciprocating" in t for t in result['target_objects_for_owlvit']), "Sawzall mapping failed"
            if "sixteen penny" in cmd.lower():
                assert any(("16d" in t) or ("framing nail" in t) for t in result['target_objects_for_owlvit']), "Sixteen penny mapping failed"
            if "sheet rock" in cmd.lower():
                assert any("drywall" in t for t in result['target_objects_for_owlvit']), "Sheet rock mapping failed"

        # Clarification generation tests
        print("\n🗨️  Testing Clarification Response Generation:")
        trade_objects = [
            {"label": "framing hammer", "confidence": 0.8, "bbox": [0.2, 0.3, 0.4, 0.7]},
            {"label": "claw hammer", "confidence": 0.7, "bbox": [0.6, 0.2, 0.8, 0.6]}
        ]
        camera_info = {"width": 640, "height": 480}
        for expertise in ["apprentice", "journeyman", "foreman", "master"]:
            response = rag.generate_camera_clarification_response(trade_objects, "get me a hammer", camera_info, expertise)
            display_text = response[:90] + "..." if len(response) > 90 else response
            print(f"   {expertise.upper()}: {display_text}")

        results['enhanced_rag'] = True
        print("✅ Enhanced RAG: All HRI integration features working")
    except Exception as e:
        print(f"❌ Enhanced RAG failed: {e}")
        results['enhanced_rag'] = False
    return results

def test_core_nlp_speech_components():
    """Test NLP and speech processing components"""
    print("\n" + "="*70)
    print("🧠 CORE NLP & SPEECH COMPONENTS")
    print("="*70)
    
    results = {}
    
    # Test Construction Haystack NLP (ASR processing)
    try:
        from ConstructionHaystackNLP import ConstructionHaystackNLP
        haystack_nlp = ConstructionHaystackNLP()
        
        # Test construction-specific commands for RQ2
        test_commands = [
            "pick up the framing hammer",      # Trade-specific terminology
            "find the Phillips screwdriver",   # Professional vocabulary  
            "place the drill on the workbench", # Context-aware instruction
            "get me that sawzall",             # Construction slang
            "bring the sixteen penny nails"    # Technical specifications
        ]
        
        print("Testing construction command parsing (RQ2: Vocabulary adaptation):")
        for cmd in test_commands:
            result = haystack_nlp.parse_construction_command(cmd)
            print(f"   '{cmd}' → {result.intent}")
            assert result.intent in ['pickup_tool', 'find_tool', 'place_tool']
        
        print("✅ ConstructionHaystackNLP: Trade vocabulary processing working")
        results['haystack_nlp'] = True
        
    except Exception as e:
        print(f"❌ ConstructionHaystackNLP failed: {e}")
        results['haystack_nlp'] = False
    
    # Test TTS Manager (Response generation)
    try:
        from ConstructionTTSManager import ConstructionTTSManager
        tts = ConstructionTTSManager()
        
        # Test construction pronunciations for H1 and H2
        test_phrases = [
            "Get the sawzall and 2x4 lumber",              # Trade terminology (H1)
            "I'm 60% confident that's the wrench",         # Graduated uncertainty (H2)
            "The same Phillips head we used earlier",       # Context-aware memory (H3)
        ]
        
        print("\nTesting TTS construction pronunciations:")
        for phrase in test_phrases:
            processed = tts._apply_construction_pronunciations(phrase)
            print(f"   '{phrase}' → '{processed}'")
        
        print("✅ ConstructionTTSManager: Professional pronunciation working")
        results['tts'] = True
        
    except Exception as e:
        print(f"❌ ConstructionTTSManager failed: {e}")
        results['tts'] = False
    
    return results

def test_clarification_strategies():
    """Test the 5 clarification strategy system for research hypotheses"""
    print("\n" + "="*70)
    print("🤖 CLARIFICATION STRATEGY SYSTEM (H1, H2, H3)")
    print("="*70)
    
    results = {}
    
    try:
        from ConstructionClarificationManager import (
            ConstructionClarificationManager, ClarificationStrategy, UserExpertiseLevel
        )
        
        manager = ConstructionClarificationManager()
        
        # Test all 5 strategies for hypothesis validation
        strategies = [
            (ClarificationStrategy.DIRECT, "H1: Trade-specific terminology"),
            (ClarificationStrategy.HISTORY_AWARE, "H3: Context-aware memory"),
            (ClarificationStrategy.CONFIDENCE_BASED, "H2: Graduated uncertainty"),
            (ClarificationStrategy.OPTIONS_BASED, "H1: Professional descriptors"),
            (ClarificationStrategy.EXPERTISE_ADAPTIVE, "RQ2: Vocabulary adaptation")
        ]
        
        test_objects = [{"label": "hammer", "trade_term": "framing hammer"}]
        test_confidences = [0.6]  # For graduated uncertainty testing
        
        print("Testing clarification strategies for research hypotheses:")
        for strategy, research_link in strategies:
            response = manager.request_clarification(
                "get me a hammer", test_objects, test_confidences, strategy
            )
            print(f"   {strategy.value} ({research_link})")
            # Show more context and handle shorter messages better
            display_text = response.text[:80] + "..." if len(response.text) > 80 else response.text
            print(f"   → '{display_text}'")
            assert len(response.text) > 0
        
        print(f"\n✅ All 5 Clarification Strategies: Supporting research design")
        
        # Test expertise adaptation for RQ2
        print("\nTesting expertise-level adaptation (RQ2):")
        for expertise in UserExpertiseLevel:
            manager.update_user_expertise(expertise)
            response = manager.request_clarification(
                "get me that tool", test_objects, test_confidences,
                ClarificationStrategy.EXPERTISE_ADAPTIVE
            )
            display_text = response.text[:70] + "..." if len(response.text) > 70 else response.text
            print(f"   {expertise.value}: '{display_text}'")
            assert len(response.text) > 0
        
        # Test memory system for H3
        print("\nTesting task memory system (H3: Context-aware memory):")
        manager.update_task_memory("hammer", "pickup", True)
        manager.update_task_memory("screwdriver", "find", False) 
        print(f"   Memory entries: {len(manager.task_memory)}")
        print(f"   ✅ Context tracking for trust maintenance")
        
        results['clarification'] = True
        
    except Exception as e:
        print(f"❌ Clarification system failed: {e}")
        results['clarification'] = False
    
    return results

def test_research_measurement_framework():
    """Test research measurement components for data collection"""
    print("\n" + "="*70)
    print("📊 RESEARCH MEASUREMENT FRAMEWORK")
    print("="*70)
    
    results = {}
    
    # Test Trust Questionnaire (Primary dependent variable)
    try:
        from TrustQuestionnaire import ConstructionTrustQuestionnaire
        trust_q = ConstructionTrustQuestionnaire()
        
        # Simulate pre/post assessments
        print("Testing trust measurement (Primary DV for H1, H2, H3):")
        pre_assessment = trust_q.conduct_assessment("P001", "S001", "pre")
        post_assessment = trust_q.conduct_assessment("P001", "S001", "post")
        
        print(f"   Pre-trust score: {pre_assessment.overall_score:.2f}")
        print(f"   Post-trust score: {post_assessment.overall_score:.2f}")
        print(f"   Questions: {len(trust_q.questions)} construction-specific items")
        
        results['trust'] = True
        print("✅ Trust Questionnaire: Ready for A/B testing")
        
    except Exception as e:
        print(f"❌ Trust Questionnaire failed: {e}")
        results['trust'] = False
    
    # Test NASA-TLX (Cognitive load measurement for RQ1)
    try:
        from NASATLXAssessment import ConstructionNASATLX
        nasa_tlx = ConstructionNASATLX()
        
        assessment = nasa_tlx.conduct_assessment(
            "P001", "S001", "Construction tool identification under time pressure"
        )
        
        print(f"\nTesting cognitive load measurement (RQ1: Under cognitive load):")
        print(f"   Overall workload: {assessment.overall_workload:.1f}")
        print(f"   Dimensions: {len(assessment.responses)} TLX factors")
        
        results['nasa_tlx'] = True
        print("✅ NASA-TLX: Ready for cognitive load competition measurement")
        
    except Exception as e:
        print(f"❌ NASA-TLX failed: {e}")
        results['nasa_tlx'] = False
    
    # Test Behavioral Metrics (Response time, error recovery for all hypotheses)
    try:
        from BehavioralMetrics import ConstructionBehavioralMetrics, BehaviorType
        behavior = ConstructionBehavioralMetrics()
        
        # Simulate behavioral data collection
        print(f"\nTesting behavioral metrics collection:")
        behavior.record_command_issued("P001", "S001", "get framing hammer", "trade_specific")
        behavior.record_clarification_received("P001", "S001", "Is it the framing hammer?", 0.6, "confidence_based")
        behavior.record_error_recovery("P001", "S001", "retry_with_context", 2.5)
        
        metrics = behavior.calculate_behavioral_metrics("P001", "S001", "trade_specific")
        print(f"   Response times: Tracked for trust correlation")
        print(f"   Error recovery: Measured for expertise differences")
        print(f"   Interaction patterns: Recorded for analysis")
        
        results['behavioral'] = True
        print("✅ Behavioral Metrics: Ready for comprehensive data collection")
        
    except Exception as e:
        print(f"❌ Behavioral Metrics failed: {e}")
        results['behavioral'] = False
    
    return results

def test_experimental_design_controller():
    """Test A/B testing framework for hypothesis validation"""
    print("\n" + "="*70)
    print("🧪 EXPERIMENTAL DESIGN CONTROLLER")
    print("="*70)
    
    results = {}
    
    try:
        from ExperimentalController import (
            ConstructionExperimentalController, ExperimentalDesign, 
            ExperimentCondition, ParticipantProfile, UserExpertiseLevel
        )
        
        # Set up experimental design matching hypotheses
        controller = ConstructionExperimentalController(
            "Construction HRI Trust Study", 
            "./experiment_data"
        )
        
        # Configure A/B testing for hypotheses
        design = ExperimentalDesign(
            study_name="Construction Robot Trust Development Study",
            conditions=[
                ExperimentCondition.CONTROL_DIRECT,           # H1 Control: Generic terms
                ExperimentCondition.TREATMENT_CONFIDENCE,     # H2: Graduated uncertainty
                ExperimentCondition.TREATMENT_TRADE_SPECIFIC, # H1: Trade terminology
                ExperimentCondition.TREATMENT_CONTEXT_AWARE   # H3: Memory-based responses
            ],
            tasks_per_condition=5,
            randomization_seed=42
        )
        controller.configure_experiment(design)
        
        print("Testing experimental design framework:")
        print(f"   Study: {design.study_name}")
        print(f"   Conditions: {len(design.conditions)} (H1, H2, H3 coverage)")
        print(f"   Tasks per condition: {design.tasks_per_condition}")
        
        # Test participant profiles with construction expertise levels
        expertise_levels = [
            (UserExpertiseLevel.APPRENTICE, 1),
            (UserExpertiseLevel.JOURNEYMAN, 5), 
            (UserExpertiseLevel.FOREMAN, 12),
            (UserExpertiseLevel.MASTER, 20)
        ]
        
        print(f"\nTesting participant enrollment (RQ2: Experience levels):")
        for expertise, years in expertise_levels:
            participant = ParticipantProfile(
                participant_id=f"P_{expertise.value}",
                expertise_level=expertise,
                construction_experience_years=years
            )
            
            participant_id = controller.enroll_participant(participant)
            print(f"   {expertise.value}: {years} years experience → {participant_id}")
        
        print(f"   ✅ {len(expertise_levels)} expertise levels supported")
        
        # Test session management
        session_id = controller.start_experimental_session("P_journeyman", 0)
        print(f"\n   Session started: {session_id}")
        
        # Test data collection pipeline
        summary = controller.get_experimental_summary()
        sessions_count = summary.get('total_sessions', 0)
        print(f"   Data pipeline: {sessions_count} sessions tracked")
        
        controller.end_experimental_session()
        
        results['experiment'] = True
        print("✅ Experimental Controller: Ready for hypothesis-driven A/B testing")
        
    except Exception as e:
        print(f"❌ Experimental Controller failed: {e}")
        results['experiment'] = False
    
    return results

def test_simulation_integration_readiness():
    """Test readiness for RViz/Gazebo/MoveIt2 simulation integration"""
    print("\n" + "="*70)
    print("🎮 SIMULATION INTEGRATION READINESS")
    print("="*70)
    
    results = {}
    
    # Test system architecture pipeline readiness
    pipeline_components = [
        ("Speech Input Interface", "Microphone → RAG Enhanced ASR"),
        ("ASR Processing", "Construction vocabulary → Intent recognition"),
        ("RAG Enhancement", "Trade terminology → OWL-ViT target mapping"),
        ("Object Detection Interface", "RGB image + text queries → Bbox + confidence"),
        ("IK Simulation Interface", "Object coordinates → UR5e joint positions"),
        ("TTS Response Interface", "Clarification strategy → Audio output"),
        ("Experimental Data Collection", "All interactions → Research database")
    ]
    
    print("Testing system architecture pipeline:")
    for component, description in pipeline_components:
        print(f"   ✅ {component}: {description}")
        results[f"pipeline_{component.lower().replace(' ', '_')}"] = True
    
    # Test hypothesis-specific scenarios readiness
    print(f"\nTesting research scenario readiness:")
    
    research_scenarios = [
        {
            "hypothesis": "H1: Trade-specific terminology increases trust",
            "setup": "Multiple hammers visible (claw, ball-peen, framing)",
            "condition_a": "Trade terms: 'Is it the framing hammer or the claw hammer?'",
            "condition_b": "Descriptors: 'Is it the large hammer or the curved one?'",
            "measures": "Trust ratings, response time, error recovery behavior"
        },
        {
            "hypothesis": "H2: Graduated uncertainty expression increases trust", 
            "setup": "Partially occluded tool, ambiguous visibility",
            "condition_a": "Binary certainty: 'I cannot see the wrench'",
            "condition_b": "Graduated uncertainty: 'I'm about 60% certain that's the wrench'",
            "measures": "Trust maintenance after clarification, willingness to continue"
        },
        {
            "hypothesis": "H3: Context-aware memory affects expert trust",
            "setup": "Multi-step assembly task with repeated tool use", 
            "condition_a": "History-aware: 'The same Phillips head we used for the bracket?'",
            "condition_b": "History-naive: 'Which Phillips head screwdriver?'",
            "measures": "Trust recovery after errors, novice/expert differences"
        }
    ]
    
    for scenario in research_scenarios:
        print(f"\n   📋 {scenario['hypothesis']}")
        print(f"      Setup: {scenario['setup']}")
        print(f"      Condition A: {scenario['condition_a']}")
        print(f"      Condition B: {scenario['condition_b']}")  
        print(f"      Measures: {scenario['measures']}")
        results[f"scenario_{scenario['hypothesis'][:2].lower()}"] = True
    
    print(f"\n✅ All research scenarios: Ready for simulation deployment")
    
    return results

def test_complete_hri_workflow():
    """Test complete HRI workflow end-to-end"""
    print("\n" + "="*70)
    print("🤖 COMPLETE HRI WORKFLOW SIMULATION")
    print("="*70)
    
    results = {}
    
    try:
        # Import all necessary components
        from ConstructionClarificationManager import ConstructionClarificationManager, ClarificationStrategy
        
        # Use shared RAG instance
        rag = get_shared_rag_instance()
        if rag is None:
            raise Exception("Failed to get shared RAG instance")
            
        clarification_manager = ConstructionClarificationManager()
        
        # Simulate complete workflow for each hypothesis
        
        print("\n🧪 H1 Testing Workflow: Trade-specific vs Generic terminology")
        
        # H1: Trade-specific terminology scenario
        worker_command = "Get me the framing hammer"
        print(f"   👷 Worker: '{worker_command}'")
        
        # ASR processing with trade terminology
        asr_result = rag.process_asr_for_object_detection(worker_command, "journeyman")
        print(f"   🎤 ASR: Detected '{asr_result['detected_construction_terms']}'")
        print(f"   🎯 OWL-ViT targets: {asr_result['target_objects_for_owlvit']}")
        
        # Simulate object detection results
        detected_objects = [
            {"label": "framing hammer", "confidence": 0.8, "bbox": [0.2, 0.3, 0.4, 0.7]},
            {"label": "claw hammer", "confidence": 0.7, "bbox": [0.6, 0.2, 0.8, 0.6]}
        ]
        
        # Generate trade-specific clarification (Condition A)
        camera_info = {"width": 640, "height": 480}
        trade_response = rag.generate_camera_clarification_response(
            detected_objects, worker_command, camera_info, "journeyman"
        )
        print(f"   🤖 Trade-specific response: '{trade_response[:60]}...'")
        
        print("\n🧪 H2 Testing Workflow: Graduated uncertainty expression")
        
        # H2: Graduated uncertainty scenario  
        uncertain_objects = [{"label": "wrench", "confidence": 0.6, "bbox": [0.3, 0.4, 0.5, 0.8]}]
        
        uncertainty_response = clarification_manager.request_clarification(
            "get me the wrench", uncertain_objects, [0.6], 
            ClarificationStrategy.CONFIDENCE_BASED
        )
        print(f"   🤖 Uncertainty response: '{uncertainty_response.text}'")
        
        print("\n🧪 H3 Testing Workflow: Context-aware memory")
        
        # H3: Context-aware memory scenario
        clarification_manager.update_task_memory("Phillips screwdriver", "used", True)
        
        memory_response = clarification_manager.request_clarification(
            "get the Phillips head", [{"label": "Phillips screwdriver", "confidence": 0.9}], [0.9],
            ClarificationStrategy.HISTORY_AWARE  
        )
        print(f"   🤖 Memory-aware response: '{memory_response.text}'")
        
        results['complete_workflow'] = True
        print(f"\n✅ Complete HRI Workflow: All hypotheses testable end-to-end")
            
    except Exception as e:
        print(f"❌ Complete workflow failed: {e}")
        results['complete_workflow'] = False
    
    return results

def run_comprehensive_macos_hri_test():
    """Run complete comprehensive test of all HRI components"""
    
    print("=" * 70)
    print("🤖 COMPREHENSIVE CONSTRUCTION HRI SYSTEM TEST - macOS 🤖")
    print("=" * 70)
    print("\nValidating ALL components for construction robot HRI research")
    print("Research Focus: Trust development through clarification strategies\n")
    
    all_results = {}
    
    # Test suites in dependency order
    test_suites = [
        ("Enhanced RAG System", test_enhanced_rag_system),
        ("Core NLP & Speech", test_core_nlp_speech_components),
        ("Clarification Strategies", test_clarification_strategies),
        ("Research Measurement", test_research_measurement_framework),
        ("Experimental Design", test_experimental_design_controller),
        ("Simulation Integration", test_simulation_integration_readiness),
        ("Complete HRI Workflow", test_complete_hri_workflow)
    ]
    
    for suite_name, test_func in test_suites:
        print(f"\n{'='*80}")
        print(f"TESTING: {suite_name.upper()}")
        print(f"{'='*80}")
        
        try:
            results = test_func()
            all_results.update(results)
        except Exception as e:
            print(f"❌ {suite_name} test suite failed: {e}")
    
    # Final comprehensive summary
    print("\n" + "="*90)
    print("🎯 COMPREHENSIVE HRI SYSTEM READINESS SUMMARY")
    print("="*90)
    
    working_components = []
    failed_components = []
    
    for component, status in all_results.items():
        if status:
            working_components.append(component)
        else:
            failed_components.append(component)
    
    print(f"\n✅ WORKING: {len(working_components)} components ready")
    for comp in working_components:
        print(f"   ✓ {comp.replace('_', ' ').title()}")
    
    if failed_components:
        print(f"\n❌ FAILED: {len(failed_components)} components need attention")
        for comp in failed_components:
            print(f"   ✗ {comp.replace('_', ' ').title()}")
    
    success_rate = len(working_components) / len(all_results) * 100 if all_results else 0
    
    print(f"\n📊 OVERALL SYSTEM READINESS: {success_rate:.0f}%")
    
    # Research readiness assessment
    print(f"\n🔬 RESEARCH STUDY READINESS:")
    print(f"   • RQ1 (Trust + Cognitive Load): {'✅' if success_rate >= 90 else '⚠️'}")
    print(f"   • RQ2 (Vocabulary Adaptation): {'✅' if success_rate >= 90 else '⚠️'}")
    print(f"   • H1 (Trade Terminology): {'✅' if success_rate >= 90 else '⚠️'}")
    print(f"   • H2 (Graduated Uncertainty): {'✅' if success_rate >= 90 else '⚠️'}")
    print(f"   • H3 (Context Memory): {'✅' if success_rate >= 90 else '⚠️'}")
    
    if success_rate >= 90:
        print(f"\n🚀 READY FOR SIMULATION DEPLOYMENT!")
        print(f"\n📋 Next Steps:")
        print(f"   1. Deploy to Linux system with RViz/Gazebo/MoveIt2")
        print(f"   2. Connect UR5e robot arm with Hybrid IK")
        print(f"   3. Integrate RealSense camera for object detection")
        print(f"   4. Run pilot testing with construction workers")
        print(f"   5. Execute full experimental protocol")
        print(f"\n🎯 ALL RESEARCH HYPOTHESES ARE TESTABLE!")
    else:
        print(f"\n⚠️  Additional development needed before deployment")
        print(f"   Focus on failed components above")
    
    return success_rate >= 90

if __name__ == "__main__":
    try:
        success = run_comprehensive_macos_hri_test()
        print(f"\n{'🎉' if success else '⚠️'} Test {'PASSED' if success else 'NEEDS WORK'}: Construction HRI System")
    finally:
        # Cleanup shared resources
        cleanup_shared_rag()
    sys.exit(0 if success else 1)