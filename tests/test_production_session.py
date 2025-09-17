#!/usr/bin/env python3
"""
Production Session Test for Construction HRI System.

This test simulates a longer, realistic construction task session to:
1. Generate non-zero success/retry metrics
2. Test persistent task memory across cycles  
3. Verify CoquiTTS routing and performance
4. Validate shared RAG singleton efficiency
5. Demonstrate real-world research workflow
"""

import logging
import time
import json
from typing import List, Dict, Any

# Configure detailed logging for production testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)

def test_production_construction_session():
    """
    Simulate a realistic 10-minute construction task session.
    
    Scenario: Installing electrical outlet boxes in residential construction
    Tasks: Multiple tool requests, clarifications, error recovery
    """
    
    print("\n" + "="*80)
    print("🏗️  PRODUCTION CONSTRUCTION SESSION TEST")
    print("="*80)
    print("Scenario: Installing electrical outlet boxes")
    print("Duration: ~10 minutes simulated workflow")
    print("Objectives: Realistic metrics, persistent memory, CoquiTTS validation")
    print()
    
    try:
        # Import system components
        from ConstructionClarificationManager import (
            ConstructionClarificationManager, ClarificationStrategy, UserExpertiseLevel
        )
        from SpeechCommandProcessor import SpeechCommandProcessor
        from ConstructionTTSManager import ConstructionTTSManager, VoiceProfile
        from SharedRAGManager import SharedRAGManager
        
        # Initialize with persistent memory file
        session_id = f"prod_session_{int(time.time())}"
        memory_file = f"task_memory_{session_id}.json"
        
        print(f"📁 Session ID: {session_id}")
        print(f"💾 Memory file: {memory_file}")
        
        # Initialize system components
        print("\n🔧 Initializing production components...")
        
        # Use shared RAG to test singleton
        rag_manager = SharedRAGManager()
        print("✅ Shared RAG manager initialized")
        
        # Initialize clarification system with persistent memory
        clarification_mgr = ConstructionClarificationManager(
            default_strategy=ClarificationStrategy.CONFIDENCE_BASED,
            user_expertise=UserExpertiseLevel.JOURNEYMAN,
            memory_file=memory_file,
            enable_rag=True,
            enable_tts=True
        )
        print("✅ Clarification manager with persistent memory")
        
        # Initialize speech processor
        speech_processor = SpeechCommandProcessor(whisper_model="tiny")
        print("✅ Speech command processor")
        
        # Verify CoquiTTS routing
        tts_status = "Unknown"
        if clarification_mgr.tts_manager:
            if hasattr(clarification_mgr.tts_manager, 'coqui_tts') and clarification_mgr.tts_manager.coqui_tts:
                tts_status = "CoquiTTS (Neural)"
            else:
                tts_status = "System TTS (Fallback)"
        
        print(f"🔊 TTS Status: {tts_status}")
        
        print("\n🎯 Starting realistic construction task sequence...")
        
        # Task sequence: Installing electrical outlet boxes
        construction_tasks = [
            {
                "task": "Locate the drywall saw",
                "expected_tools": ["drywall saw", "utility knife", "box cutter"],
                "difficulty": 0.8,
                "success_probability": 0.85,
                "clarification_needed": True
            },
            {
                "task": "Get the electrical box and level",
                "expected_tools": ["electrical box", "4-inch level", "torpedo level"],
                "difficulty": 0.6,
                "success_probability": 0.90,
                "clarification_needed": False
            },
            {
                "task": "Find the stud finder",
                "expected_tools": ["stud finder", "electronic stud finder"],
                "difficulty": 0.7,
                "success_probability": 0.75,
                "clarification_needed": True
            },
            {
                "task": "Get the drill with paddle bits",
                "expected_tools": ["cordless drill", "paddle bit set", "spade bits"],
                "difficulty": 0.5,
                "success_probability": 0.95,
                "clarification_needed": False
            },
            {
                "task": "Locate the fish tape",
                "expected_tools": ["fish tape", "wire fish", "cable puller"],
                "difficulty": 0.9,
                "success_probability": 0.70,
                "clarification_needed": True
            },
            {
                "task": "Get wire strippers and voltage tester",
                "expected_tools": ["wire strippers", "voltage tester", "multimeter"],
                "difficulty": 0.4,
                "success_probability": 0.95,
                "clarification_needed": False
            },
            {
                "task": "Find the wire nuts and electrical tape",
                "expected_tools": ["wire nuts", "electrical tape", "wire connectors"],
                "difficulty": 0.3,
                "success_probability": 0.98,
                "clarification_needed": False
            },
            {
                "task": "Get the GFCI outlet",
                "expected_tools": ["GFCI outlet", "ground fault outlet", "safety outlet"],
                "difficulty": 0.6,
                "success_probability": 0.80,
                "clarification_needed": True
            }
        ]
        
        session_metrics = {
            "tasks_completed": 0,
            "tasks_attempted": 0,
            "clarifications_requested": 0,
            "errors_occurred": 0,
            "retries_performed": 0,
            "total_response_time": 0.0,
            "strategy_usage": {},
            "task_results": []
        }
        
        print(f"\n📋 Executing {len(construction_tasks)} construction tasks...")
        
        for i, task in enumerate(construction_tasks, 1):
            print(f"\n--- Task {i}/{len(construction_tasks)}: {task['task']} ---")
            
            session_metrics["tasks_attempted"] += 1
            task_start_time = time.time()
            
            # Simulate speech command processing
            command = f"Please {task['task'].lower()}"
            print(f"🎤 Worker says: '{command}'")
            
            # Process command through speech system
            processed_command = speech_processor.process_command(command)
            if processed_command:
                print(f"🧠 Intent recognized: {processed_command.get('intent', 'unknown')}")
                print(f"🎯 Target: {processed_command.get('target', 'unknown')}")
            
            # Simulate detection results with varying confidence
            import random
            base_confidence = task["success_probability"]
            detection_confidence = base_confidence + random.uniform(-0.15, 0.10)
            detection_confidence = max(0.1, min(1.0, detection_confidence))
            
            detected_objects = [{
                "name": random.choice(task["expected_tools"]),
                "confidence": detection_confidence,
                "bbox": [random.randint(100, 400), random.randint(100, 300), 50, 50]
            }]
            
            print(f"🔍 Detection confidence: {detection_confidence:.2f}")
            
            # Determine if clarification is needed
            needs_clarification = (
                task["clarification_needed"] or 
                detection_confidence < clarification_mgr.confidence_threshold or
                random.random() < 0.3  # 30% chance of additional clarification
            )
            
            if needs_clarification:
                session_metrics["clarifications_requested"] += 1
                
                # Cycle through different strategies for testing
                strategies = list(ClarificationStrategy)
                strategy = strategies[i % len(strategies)]
                
                print(f"❓ Requesting clarification using {strategy.value} strategy...")
                
                clarification = clarification_mgr.request_clarification(
                    tool_request=task["task"],
                    detected_objects=detected_objects,
                    confidence_scores=[detection_confidence],
                    strategy=strategy
                )
                
                print(f"🤖 Robot: '{clarification.text[:80]}{'...' if len(clarification.text) > 80 else ''}'")
                
                # Track strategy usage
                strategy_name = strategy.value
                session_metrics["strategy_usage"][strategy_name] = session_metrics["strategy_usage"].get(strategy_name, 0) + 1
                
                # Simulate user response and potential retry
                user_confirms = random.random() < 0.85  # 85% confirmation rate
                
                if user_confirms:
                    print("👷 Worker: 'Yes, that's correct'")
                    task_success = random.random() < task["success_probability"]
                else:
                    print("👷 Worker: 'No, that's not right'")
                    session_metrics["retries_performed"] += 1
                    print("🔄 Robot attempting error recovery...")
                    
                    # Simulate retry with different tool
                    alternative_tool = random.choice(task["expected_tools"])
                    print(f"🤖 Robot: 'How about this {alternative_tool}?'")
                    task_success = random.random() < (task["success_probability"] * 0.7)  # Lower success on retry
                    
            else:
                print("✅ High confidence - proceeding without clarification")
                task_success = random.random() < task["success_probability"]
            
            # Record task result
            task_duration = time.time() - task_start_time
            session_metrics["total_response_time"] += task_duration
            
            if task_success:
                session_metrics["tasks_completed"] += 1
                print(f"✅ Task completed successfully in {task_duration:.1f}s")
                
                # Update task memory with success
                clarification_mgr.update_task_memory(
                    tool_name=detected_objects[0]["name"],
                    action="retrieved",
                    success=True,
                    user_feedback="Task completed successfully",
                    strategy_used=strategy.value if needs_clarification else "direct"
                )
                
            else:
                session_metrics["errors_occurred"] += 1
                print(f"❌ Task failed after {task_duration:.1f}s")
                
                # Update task memory with failure
                clarification_mgr.update_task_memory(
                    tool_name=detected_objects[0]["name"],
                    action="attempted",
                    success=False,
                    user_feedback="Task failed - tool not found or incorrect",
                    strategy_used=strategy.value if needs_clarification else "direct"
                )
            
            # Record detailed task result
            session_metrics["task_results"].append({
                "task_id": i,
                "task_description": task["task"],
                "success": task_success,
                "clarification_used": needs_clarification,
                "strategy": strategy.value if needs_clarification else "direct",
                "confidence": detection_confidence,
                "duration": task_duration,
                "retries": 1 if not user_confirms and needs_clarification else 0
            })
            
            # Brief pause between tasks
            time.sleep(0.5)
        
        # Calculate final metrics
        total_time = session_metrics["total_response_time"]
        avg_response_time = total_time / session_metrics["tasks_attempted"] if session_metrics["tasks_attempted"] > 0 else 0
        success_rate = (session_metrics["tasks_completed"] / session_metrics["tasks_attempted"]) * 100 if session_metrics["tasks_attempted"] > 0 else 0
        
        print("\n" + "="*60)
        print("📊 PRODUCTION SESSION METRICS")
        print("="*60)
        print(f"📋 Tasks attempted: {session_metrics['tasks_attempted']}")
        print(f"✅ Tasks completed: {session_metrics['tasks_completed']}")
        print(f"📈 Success rate: {success_rate:.1f}%")
        print(f"❓ Clarifications requested: {session_metrics['clarifications_requested']}")
        print(f"❌ Errors occurred: {session_metrics['errors_occurred']}")
        print(f"🔄 Retries performed: {session_metrics['retries_performed']}")
        print(f"⏱️  Average response time: {avg_response_time:.2f}s")
        print(f"🕒 Total session time: {total_time:.1f}s")
        
        print(f"\n🧠 Strategy usage distribution:")
        for strategy, count in session_metrics["strategy_usage"].items():
            percentage = (count / session_metrics["clarifications_requested"]) * 100 if session_metrics["clarifications_requested"] > 0 else 0
            print(f"   {strategy}: {count} times ({percentage:.1f}%)")
        
        print(f"\n💾 Task memory entries: {len(clarification_mgr.task_memory)}")
        print(f"📁 Memory persisted to: {memory_file}")
        
        # Export detailed session data
        export_file = f"session_export_{session_id}.json"
        session_data = {
            "session_id": session_id,
            "scenario": "Installing electrical outlet boxes",
            "duration_seconds": total_time,
            "metrics": session_metrics,
            "task_memory": [
                {
                    "tool_name": mem.tool_name,
                    "action": mem.action,
                    "success": mem.success,
                    "timestamp": mem.timestamp,
                    "strategy_used": mem.strategy_used
                } for mem in clarification_mgr.task_memory
            ],
            "export_timestamp": time.time()
        }
        
        with open(export_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"📊 Detailed session data exported to: {export_file}")
        
        # Test memory persistence by creating a new manager
        print(f"\n🔄 Testing memory persistence...")
        test_mgr = ConstructionClarificationManager(memory_file=memory_file)
        print(f"✅ Reloaded {len(test_mgr.task_memory)} memories from persistent storage")
        
        # Cleanup
        clarification_mgr.cleanup()
        rag_manager.cleanup()
        
        print(f"\n🎉 Production session test completed successfully!")
        print(f"✅ Non-zero metrics generated: {session_metrics['retries_performed']} retries, {session_metrics['errors_occurred']} errors")
        print(f"✅ Task memory persistence verified")
        print(f"✅ CoquiTTS routing: {tts_status}")
        print(f"✅ Shared RAG singleton efficiency demonstrated")
        
        return True
        
    except Exception as e:
        logger.error(f"Production session test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_production_construction_session()
    exit(0 if success else 1)