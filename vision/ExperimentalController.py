#!/usr/bin/env python3
"""
Experimental Controller for Construction HRI Research.

This module orchestrates the complete experimental framework for construction
HRI research, managing A/B testing between clarification strategies, data
collection, and experimental protocol execution.

Provides:
- A/B testing framework for clarification strategies
- Randomized experimental design
- Comprehensive data collection coordination
- Statistical analysis and reporting
- Experimental protocol enforcement
- Integration with all measurement systems
"""

import logging
import time
import json
import uuid
import random
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics
from pathlib import Path

# Import all research components
try:
    from ConstructionClarificationManager import ConstructionClarificationManager, ClarificationStrategy, UserExpertiseLevel
    from TrustQuestionnaire import ConstructionTrustQuestionnaire, TrustAssessment
    from NASATLXAssessment import ConstructionNASATLX, TLXAssessment
    from BehavioralMetrics import ConstructionBehavioralMetrics, BehavioralMetrics, BehaviorType, ResponseType
    from ConstructionTTSManager import ConstructionTTSManager, VoiceProfile
    from SpeechCommandProcessor import SpeechCommandProcessor
    from OWLViTDetector import OWLViTDetector
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some experimental components not available: {e}")
    COMPONENTS_AVAILABLE = False
except Exception as e:
    # Handle pydantic version compatibility issues
    if "GetCoreSchemaHandler" in str(e):
        logging.warning("Pydantic version compatibility - using fallback implementations")
    else:
        logging.warning(f"Research components not fully available - using mock implementations: {e}")
    COMPONENTS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExperimentCondition(Enum):
    """Experimental conditions for A/B testing"""
    CONTROL_DIRECT = "control_direct"                    # Direct clarification baseline
    TREATMENT_CONFIDENCE = "treatment_confidence"        # Confidence-based clarification
    TREATMENT_HISTORY = "treatment_history"              # History-aware clarification
    TREATMENT_OPTIONS = "treatment_options"              # Options-based clarification
    TREATMENT_ADAPTIVE = "treatment_adaptive"            # Expertise-adaptive clarification
    TREATMENT_TRADE_SPECIFIC = "treatment_trade_specific"  # Trade-specific terminology (H1)
    TREATMENT_CONTEXT_AWARE = "treatment_context_aware"    # Context-aware memory (H3)

class ExperimentPhase(Enum):
    """Phases of experimental session"""
    SETUP = "setup"                      # Initial setup and calibration
    PRE_ASSESSMENT = "pre_assessment"    # Pre-interaction questionnaires
    TRAINING = "training"                # Familiarization with system
    MAIN_TASK = "main_task"             # Primary experimental task
    POST_ASSESSMENT = "post_assessment"  # Post-interaction measurements
    DEBRIEFING = "debriefing"           # Final debriefing and questions

@dataclass
class ExperimentalDesign:
    """Experimental design configuration"""
    study_name: str
    conditions: List[ExperimentCondition]
    block_randomization: bool = True
    counterbalancing: bool = True
    within_subject: bool = True
    session_duration_minutes: int = 45
    tasks_per_condition: int = 8
    rest_breaks: bool = True
    randomization_seed: Optional[int] = None  # Added for reproducible macOS test harness
    
@dataclass  
class ParticipantProfile:
    """Participant profile and demographics"""
    participant_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    construction_experience_years: Optional[int] = None
    expertise_level: Optional[UserExpertiseLevel] = None
    technology_comfort: Optional[int] = None  # 1-7 scale
    robot_experience: Optional[int] = None    # 1-7 scale
    assigned_conditions: List[ExperimentCondition] = field(default_factory=list)
    condition_order: List[int] = field(default_factory=list)

@dataclass
class ExperimentalSession:
    """Single experimental session data"""
    session_id: str
    participant_id: str
    condition: ExperimentCondition
    start_time: float
    end_time: Optional[float] = None
    current_phase: ExperimentPhase = ExperimentPhase.SETUP
    
    # Task performance data
    tasks_completed: int = 0
    tasks_successful: int = 0
    total_interaction_time: float = 0.0
    total_clarifications: int = 0
    
    # Assessment data
    pre_trust: Optional[TrustAssessment] = None
    post_trust: Optional[TrustAssessment] = None
    tlx_assessment: Optional[TLXAssessment] = None
    behavioral_metrics: Optional[BehavioralMetrics] = None
    
    # Metadata
    notes: str = ""
    technical_issues: List[str] = field(default_factory=list)
    
class ConstructionExperimentalController:
    """
    Master experimental controller for construction HRI research.
    
    Orchestrates the complete experimental protocol including participant
    management, condition assignment, data collection, and analysis.
    
    Parameters
    ----------
    study_name : str
        Name of the research study
    data_directory : str, optional
        Directory for storing experimental data
    enable_real_time_analysis : bool, optional
        Enable real-time data analysis, by default True
        
    Attributes
    ----------
    experimental_design : ExperimentalDesign
        Current experimental design configuration
    participants : Dict[str, ParticipantProfile]
        Enrolled participant profiles
    sessions : List[ExperimentalSession]
        Completed experimental sessions
    active_session : Optional[ExperimentalSession]
        Currently running session
    """
    
    def __init__(self,
                 study_name: str,
                 data_directory: str = "./experimental_data",
                 enable_real_time_analysis: bool = True):
        
        self.logger = logging.getLogger(__name__)
        self.study_name = study_name
        self.data_directory = Path(data_directory)
        self.enable_real_time_analysis = enable_real_time_analysis
        
        # Create data directory
        self.data_directory.mkdir(parents=True, exist_ok=True)
        
        # Experimental data
        self.experimental_design = None
        self.participants = {}
        self.sessions = []
        self.active_session = None
        
        # Initialize research components if available
        if COMPONENTS_AVAILABLE:
            self._initialize_research_components()
        else:
            self.logger.warning("Research components not fully available - using mock implementations")
            self.research_components = {}
        
        self.logger.info(f"✅ Experimental Controller initialized")
        self.logger.info(f"   Study: {study_name}")
        self.logger.info(f"   Data directory: {data_directory}")

    def _initialize_research_components(self):
        """Initialize all research measurement components"""
        
        try:
            self.research_components = {
                'clarification_manager': ConstructionClarificationManager(),
                'trust_questionnaire': ConstructionTrustQuestionnaire(),
                'nasa_tlx': ConstructionNASATLX(),
                'behavioral_metrics': ConstructionBehavioralMetrics(),
                'tts_manager': ConstructionTTSManager(),
                'speech_processor': SpeechCommandProcessor(),
                'object_detector': OWLViTDetector()
            }
            
            self.logger.info("✅ All research components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize research components: {e}")
            self.research_components = {}

    def configure_experiment(self, experimental_design: ExperimentalDesign):
        """
        Configure the experimental design.
        
        Parameters
        ----------
        experimental_design : ExperimentalDesign
            Complete experimental design specification
        """
        
        self.experimental_design = experimental_design
        # Apply reproducible seeding if provided
        if getattr(experimental_design, 'randomization_seed', None) is not None:
            try:
                random.seed(experimental_design.randomization_seed)
                self.logger.info(f"🔒 Randomization seed set: {experimental_design.randomization_seed}")
            except Exception as e:
                self.logger.warning(f"Could not set randomization seed: {e}")
        
        self.logger.info(f"🔬 Experimental design configured")
        self.logger.info(f"   Conditions: {len(experimental_design.conditions)}")
        self.logger.info(f"   Tasks per condition: {experimental_design.tasks_per_condition}")
        self.logger.info(f"   Within-subject: {experimental_design.within_subject}")
        self.logger.info(f"   Session duration: {experimental_design.session_duration_minutes} min")

    def enroll_participant(self, participant_profile: ParticipantProfile) -> str:
        """
        Enroll a new participant in the study.
        
        Parameters
        ----------
        participant_profile : ParticipantProfile
            Complete participant profile
            
        Returns
        -------
        str
            Assigned participant ID
        """
        
        if not participant_profile.participant_id:
            participant_profile.participant_id = f"P{len(self.participants) + 1:03d}"
        
        # Assign experimental conditions
        if self.experimental_design:
            participant_profile.assigned_conditions = self.experimental_design.conditions.copy()
            
            # Randomize condition order if specified
            if self.experimental_design.block_randomization:
                random.shuffle(participant_profile.assigned_conditions)
                participant_profile.condition_order = list(range(len(participant_profile.assigned_conditions)))
        
        self.participants[participant_profile.participant_id] = participant_profile
        
        # Save participant data
        self._save_participant_data(participant_profile)
        
        self.logger.info(f"👤 Participant enrolled: {participant_profile.participant_id}")
        self.logger.info(f"   Expertise: {participant_profile.expertise_level}")
        self.logger.info(f"   Conditions: {[c.value for c in participant_profile.assigned_conditions]}")
        
        return participant_profile.participant_id

    def start_experimental_session(self, participant_id: str, condition_index: int = 0) -> str:
        """
        Start a new experimental session.
        
        Parameters
        ----------
        participant_id : str
            Participant identifier
        condition_index : int, optional
            Index of condition to run (for within-subject design)
            
        Returns
        -------
        str
            Session ID
        """
        
        if participant_id not in self.participants:
            raise ValueError(f"Participant {participant_id} not enrolled")
        
        if self.active_session:
            raise RuntimeError("Another session is already active")
        
        participant = self.participants[participant_id]
        
        if condition_index >= len(participant.assigned_conditions):
            raise ValueError(f"Invalid condition index: {condition_index}")
        
        condition = participant.assigned_conditions[condition_index]
        session_id = f"{participant_id}_S{condition_index + 1}_{int(time.time())}"
        
        self.active_session = ExperimentalSession(
            session_id=session_id,
            participant_id=participant_id,
            condition=condition,
            start_time=time.time()
        )
        
        # Configure systems for this condition
        self._configure_systems_for_condition(condition, participant.expertise_level)
        
        self.logger.info(f"🚀 Experimental session started")
        self.logger.info(f"   Session: {session_id}")
        self.logger.info(f"   Participant: {participant_id}")
        self.logger.info(f"   Condition: {condition.value}")
        
        return session_id

    def _configure_systems_for_condition(self, condition: ExperimentCondition, expertise_level: Optional[UserExpertiseLevel]):
        """Configure research systems for experimental condition"""
        
        if not COMPONENTS_AVAILABLE:
            return
        
        # Configure clarification manager strategy
        strategy_mapping = {
            ExperimentCondition.CONTROL_DIRECT: ClarificationStrategy.DIRECT,
            ExperimentCondition.TREATMENT_CONFIDENCE: ClarificationStrategy.CONFIDENCE_BASED,
            ExperimentCondition.TREATMENT_HISTORY: ClarificationStrategy.HISTORY_AWARE,
            ExperimentCondition.TREATMENT_OPTIONS: ClarificationStrategy.OPTIONS_BASED,
            ExperimentCondition.TREATMENT_ADAPTIVE: ClarificationStrategy.EXPERTISE_ADAPTIVE
        }
        
        clarification_strategy = strategy_mapping.get(condition, ClarificationStrategy.DIRECT)
        
        # Update clarification manager
        clarification_manager = self.research_components.get('clarification_manager')
        if clarification_manager:
            clarification_manager.default_strategy = clarification_strategy
            if expertise_level:
                clarification_manager.update_user_expertise(expertise_level)
        
        # Configure TTS based on expertise
        tts_manager = self.research_components.get('tts_manager')
        if tts_manager and expertise_level:
            if expertise_level == UserExpertiseLevel.APPRENTICE:
                tts_manager.set_voice_profile(VoiceProfile.APPRENTICE_FRIENDLY)
            elif expertise_level in [UserExpertiseLevel.FOREMAN, UserExpertiseLevel.MASTER]:
                tts_manager.set_voice_profile(VoiceProfile.PROFESSIONAL)
        
        self.logger.info(f"⚙️ Systems configured for condition: {condition.value}")

    def advance_to_phase(self, phase: ExperimentPhase) -> bool:
        """
        Advance experimental session to next phase.
        
        Parameters
        ----------
        phase : ExperimentPhase
            Target experimental phase
            
        Returns
        -------
        bool
            Success of phase transition
        """
        
        if not self.active_session:
            self.logger.error("No active session to advance")
            return False
        
        self.active_session.current_phase = phase
        
        self.logger.info(f"📋 Advanced to phase: {phase.value}")
        
        # Phase-specific actions
        if phase == ExperimentPhase.PRE_ASSESSMENT:
            return self._conduct_pre_assessment()
        elif phase == ExperimentPhase.MAIN_TASK:
            return self._initialize_main_task()
        elif phase == ExperimentPhase.POST_ASSESSMENT:
            return self._conduct_post_assessment()
        elif phase == ExperimentPhase.DEBRIEFING:
            return self._conduct_debriefing()
        
        return True

    def _conduct_pre_assessment(self) -> bool:
        """Conduct pre-interaction trust assessment"""
        
        if not COMPONENTS_AVAILABLE:
            self.logger.info("📝 [MOCK] Pre-assessment completed")
            return True
        
        try:
            trust_questionnaire = self.research_components.get('trust_questionnaire')
            if trust_questionnaire:
                pre_trust = trust_questionnaire.conduct_assessment(
                    participant_id=self.active_session.participant_id,
                    session_id=self.active_session.session_id,
                    assessment_type="pre",
                    user_expertise=self.participants[self.active_session.participant_id].expertise_level.value if self.participants[self.active_session.participant_id].expertise_level else None
                )
                
                self.active_session.pre_trust = pre_trust
                self.logger.info("📝 Pre-assessment completed")
                return True
            
        except Exception as e:
            self.logger.error(f"Pre-assessment failed: {e}")
            return False
        
        return False

    def _initialize_main_task(self) -> bool:
        """Initialize main experimental task"""
        
        # Reset task counters
        self.active_session.tasks_completed = 0
        self.active_session.tasks_successful = 0
        self.active_session.total_interaction_time = 0.0
        self.active_session.total_clarifications = 0
        
        self.logger.info("🎯 Main task phase initialized")
        return True

    def _conduct_post_assessment(self) -> bool:
        """Conduct post-interaction assessments"""
        
        if not COMPONENTS_AVAILABLE:
            self.logger.info("📊 [MOCK] Post-assessments completed")
            return True
        
        try:
            # Trust assessment
            trust_questionnaire = self.research_components.get('trust_questionnaire')
            if trust_questionnaire:
                post_trust = trust_questionnaire.conduct_assessment(
                    participant_id=self.active_session.participant_id,
                    session_id=self.active_session.session_id,
                    assessment_type="post",
                    user_expertise=self.participants[self.active_session.participant_id].expertise_level.value if self.participants[self.active_session.participant_id].expertise_level else None
                )
                self.active_session.post_trust = post_trust
            
            # NASA-TLX assessment
            nasa_tlx = self.research_components.get('nasa_tlx')
            if nasa_tlx:
                tlx_assessment = nasa_tlx.conduct_assessment(
                    participant_id=self.active_session.participant_id,
                    session_id=self.active_session.session_id,
                    task_description="Construction tool identification and retrieval with robot assistance",
                    clarification_strategy=self.active_session.condition.value,
                    user_expertise=self.participants[self.active_session.participant_id].expertise_level.value if self.participants[self.active_session.participant_id].expertise_level else None
                )
                self.active_session.tlx_assessment = tlx_assessment
            
            # Behavioral metrics
            behavioral_metrics = self.research_components.get('behavioral_metrics')
            if behavioral_metrics:
                metrics = behavioral_metrics.calculate_behavioral_metrics(
                    participant_id=self.active_session.participant_id,
                    session_id=self.active_session.session_id,
                    clarification_strategy=self.active_session.condition.value
                )
                self.active_session.behavioral_metrics = metrics
            
            self.logger.info("📊 Post-assessments completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Post-assessment failed: {e}")
            return False

    def _conduct_debriefing(self) -> bool:
        """Conduct final debriefing"""
        
        self.logger.info("💬 Debriefing phase - participant feedback collection")
        # In real implementation, this would collect qualitative feedback
        return True

    def record_task_attempt(self, task_description: str, success: bool, 
                          interaction_time: float, clarifications_used: int) -> bool:
        """
        Record a task attempt during the main experimental phase.
        
        Parameters
        ----------
        task_description : str
            Description of the task attempted
        success : bool
            Whether the task was completed successfully
        interaction_time : float
            Time spent on this task
        clarifications_used : int
            Number of clarifications used
            
        Returns
        -------
        bool
            Success of recording
        """
        
        if not self.active_session:
            self.logger.error("No active session for task recording")
            return False
        
        if self.active_session.current_phase != ExperimentPhase.MAIN_TASK:
            self.logger.error("Not in main task phase")
            return False
        
        self.active_session.tasks_completed += 1
        if success:
            self.active_session.tasks_successful += 1
        
        self.active_session.total_interaction_time += interaction_time
        self.active_session.total_clarifications += clarifications_used
        
        # Record in behavioral metrics if available
        if COMPONENTS_AVAILABLE:
            behavioral_metrics = self.research_components.get('behavioral_metrics')
            if behavioral_metrics:
                behavioral_metrics.record_task_completion(
                    participant_id=self.active_session.participant_id,
                    session_id=self.active_session.session_id,
                    task_description=task_description,
                    success=success,
                    completion_time=interaction_time,
                    error_count=0 if success else 1
                )
        
        self.logger.info(f"📝 Task recorded: {task_description} ({'✅' if success else '❌'})")
        
        return True

    def end_experimental_session(self) -> bool:
        """
        End the current experimental session and save data.
        
        Returns
        -------
        bool
            Success of session completion
        """
        
        if not self.active_session:
            self.logger.error("No active session to end")
            return False
        
        self.active_session.end_time = time.time()
        session_duration = self.active_session.end_time - self.active_session.start_time
        
        # Save session data
        self.sessions.append(self.active_session)
        self._save_session_data(self.active_session)
        
        # Log session summary
        success_rate = (self.active_session.tasks_successful / 
                       max(1, self.active_session.tasks_completed) * 100)
        
        self.logger.info(f"🏁 Experimental session completed")
        self.logger.info(f"   Duration: {session_duration/60:.1f} minutes")
        self.logger.info(f"   Tasks completed: {self.active_session.tasks_completed}")
        self.logger.info(f"   Success rate: {success_rate:.1f}%")
        self.logger.info(f"   Total clarifications: {self.active_session.total_clarifications}")
        
        # Clear active session
        completed_session = self.active_session
        self.active_session = None
        
        return True

    def get_experimental_summary(self) -> Dict[str, Any]:
        """Get summary of all experimental data"""
        
        if not self.sessions:
            return {"message": "No experimental sessions completed yet"}
        
        summary = {
            'study_name': self.study_name,
            'total_participants': len(self.participants),
            'total_sessions': len(self.sessions),
            'conditions_tested': len(set(s.condition for s in self.sessions)),
            'session_summaries': [],
            'condition_comparisons': {}
        }
        
        # Session-level summaries
        for session in self.sessions:
            session_summary = {
                'session_id': session.session_id,
                'participant_id': session.participant_id,
                'condition': session.condition.value,
                'tasks_completed': session.tasks_completed,
                'success_rate': (session.tasks_successful / max(1, session.tasks_completed)) * 100,
                'avg_interaction_time': session.total_interaction_time / max(1, session.tasks_completed),
                'clarifications_per_task': session.total_clarifications / max(1, session.tasks_completed)
            }
            
            # Add assessment scores if available
            if session.post_trust:
                session_summary['post_trust_score'] = session.post_trust.overall_score
            if session.tlx_assessment:
                session_summary['workload_score'] = session.tlx_assessment.overall_workload
            if session.behavioral_metrics:
                session_summary['avg_response_time'] = session.behavioral_metrics.avg_response_time
            
            summary['session_summaries'].append(session_summary)
        
        # Condition-level comparisons
        conditions_data = {}
        for session in self.sessions:
            condition = session.condition.value
            if condition not in conditions_data:
                conditions_data[condition] = []
            
            conditions_data[condition].append({
                'success_rate': (session.tasks_successful / max(1, session.tasks_completed)) * 100,
                'interaction_time': session.total_interaction_time / max(1, session.tasks_completed),
                'clarifications': session.total_clarifications / max(1, session.tasks_completed),
                'trust_score': session.post_trust.overall_score if session.post_trust else None,
                'workload': session.tlx_assessment.overall_workload if session.tlx_assessment else None
            })
        
        # Calculate condition averages
        for condition, data_points in conditions_data.items():
            summary['condition_comparisons'][condition] = {
                'n_sessions': len(data_points),
                'avg_success_rate': statistics.mean([d['success_rate'] for d in data_points]),
                'avg_interaction_time': statistics.mean([d['interaction_time'] for d in data_points]),
                'avg_clarifications': statistics.mean([d['clarifications'] for d in data_points]),
                'avg_trust_score': statistics.mean([d['trust_score'] for d in data_points if d['trust_score'] is not None]) if any(d['trust_score'] is not None for d in data_points) else None,
                'avg_workload': statistics.mean([d['workload'] for d in data_points if d['workload'] is not None]) if any(d['workload'] is not None for d in data_points) else None
            }
        
        return summary

    def export_complete_dataset(self, filepath: Optional[str] = None) -> str:
        """
        Export complete experimental dataset for statistical analysis.
        
        Parameters
        ----------
        filepath : str, optional
            Output filepath, auto-generated if not provided
            
        Returns
        -------
        str
            Path to exported dataset
        """
        
        if not filepath:
            timestamp = int(time.time())
            filepath = self.data_directory / f"{self.study_name}_complete_dataset_{timestamp}.json"
        
        # Compile complete dataset
        dataset = {
            'study_info': {
                'study_name': self.study_name,
                'export_timestamp': time.time(),
                'experimental_design': {
                    'conditions': [c.value for c in self.experimental_design.conditions] if self.experimental_design else [],
                    'within_subject': self.experimental_design.within_subject if self.experimental_design else None,
                    'tasks_per_condition': self.experimental_design.tasks_per_condition if self.experimental_design else None
                }
            },
            'participants': {},
            'sessions': [],
            'trust_assessments': [],
            'tlx_assessments': [],
            'behavioral_data': [],
            'summary_statistics': self.get_experimental_summary()
        }
        
        # Participant data
        for participant_id, participant in self.participants.items():
            dataset['participants'][participant_id] = {
                'age': participant.age,
                'gender': participant.gender,
                'construction_experience_years': participant.construction_experience_years,
                'expertise_level': participant.expertise_level.value if participant.expertise_level else None,
                'technology_comfort': participant.technology_comfort,
                'robot_experience': participant.robot_experience,
                'assigned_conditions': [c.value for c in participant.assigned_conditions],
                'condition_order': participant.condition_order
            }
        
        # Session data  
        for session in self.sessions:
            session_data = {
                'session_id': session.session_id,
                'participant_id': session.participant_id,
                'condition': session.condition.value,
                'start_time': session.start_time,
                'end_time': session.end_time,
                'tasks_completed': session.tasks_completed,
                'tasks_successful': session.tasks_successful,
                'total_interaction_time': session.total_interaction_time,
                'total_clarifications': session.total_clarifications,
                'notes': session.notes,
                'technical_issues': session.technical_issues
            }
            dataset['sessions'].append(session_data)
            
            # Assessment data
            if session.pre_trust or session.post_trust:
                if session.pre_trust:
                    dataset['trust_assessments'].append({
                        'session_id': session.session_id,
                        'participant_id': session.participant_id,
                        'assessment_type': 'pre',
                        'overall_score': session.pre_trust.overall_score,
                        'dimension_scores': session.pre_trust.dimension_scores,
                        'completion_time': session.pre_trust.completion_time
                    })
                
                if session.post_trust:
                    dataset['trust_assessments'].append({
                        'session_id': session.session_id,
                        'participant_id': session.participant_id,
                        'assessment_type': 'post',
                        'overall_score': session.post_trust.overall_score,
                        'dimension_scores': session.post_trust.dimension_scores,
                        'completion_time': session.post_trust.completion_time
                    })
            
            if session.tlx_assessment:
                dataset['tlx_assessments'].append({
                    'session_id': session.session_id,
                    'participant_id': session.participant_id,
                    'overall_workload': session.tlx_assessment.overall_workload,
                    'dimension_scores': {r.dimension.value: r.raw_score for r in session.tlx_assessment.responses},
                    'completion_time': session.tlx_assessment.completion_time
                })
            
            if session.behavioral_metrics:
                dataset['behavioral_data'].append({
                    'session_id': session.session_id,
                    'participant_id': session.participant_id,
                    'clarification_strategy': session.behavioral_metrics.clarification_strategy,
                    'avg_response_time': session.behavioral_metrics.avg_response_time,
                    'total_retries': session.behavioral_metrics.total_retries,
                    'task_completion_rate': session.behavioral_metrics.task_completion_rate,
                    'confirmation_rate': session.behavioral_metrics.confirmation_rate,
                    'efficiency_score': session.behavioral_metrics.efficiency_score
                })
        
        # Save dataset
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        self.logger.info(f"📊 Complete dataset exported to {filepath}")
        self.logger.info(f"   Participants: {len(dataset['participants'])}")
        self.logger.info(f"   Sessions: {len(dataset['sessions'])}")
        self.logger.info(f"   Trust assessments: {len(dataset['trust_assessments'])}")
        
        return str(filepath)

    def _save_participant_data(self, participant: ParticipantProfile):
        """Save participant data to file"""
        
        filepath = self.data_directory / f"participant_{participant.participant_id}.json"
        
        participant_data = {
            'participant_id': participant.participant_id,
            'age': participant.age,
            'gender': participant.gender,
            'construction_experience_years': participant.construction_experience_years,
            'expertise_level': participant.expertise_level.value if participant.expertise_level else None,
            'technology_comfort': participant.technology_comfort,
            'robot_experience': participant.robot_experience,
            'assigned_conditions': [c.value for c in participant.assigned_conditions],
            'condition_order': participant.condition_order,
            'enrollment_timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(participant_data, f, indent=2)

    def _save_session_data(self, session: ExperimentalSession):
        """Save session data to file"""
        
        filepath = self.data_directory / f"session_{session.session_id}.json"
        
        session_data = {
            'session_id': session.session_id,
            'participant_id': session.participant_id,
            'condition': session.condition.value,
            'start_time': session.start_time,
            'end_time': session.end_time,
            'current_phase': session.current_phase.value,
            'tasks_completed': session.tasks_completed,
            'tasks_successful': session.tasks_successful,
            'total_interaction_time': session.total_interaction_time,
            'total_clarifications': session.total_clarifications,
            'notes': session.notes,
            'technical_issues': session.technical_issues
        }
        
        # Add assessment data if available
        if session.pre_trust:
            session_data['pre_trust_score'] = session.pre_trust.overall_score
        if session.post_trust:
            session_data['post_trust_score'] = session.post_trust.overall_score
        if session.tlx_assessment:
            session_data['workload_score'] = session.tlx_assessment.overall_workload
        if session.behavioral_metrics:
            session_data['avg_response_time'] = session.behavioral_metrics.avg_response_time
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)

    def cleanup(self):
        """Clean up experimental controller resources"""
        
        # End active session if running
        if self.active_session:
            self.logger.warning("Force-ending active session due to cleanup")
            self.end_experimental_session()
        
        # Cleanup research components
        if COMPONENTS_AVAILABLE:
            for component_name, component in self.research_components.items():
                if hasattr(component, 'cleanup'):
                    try:
                        component.cleanup()
                    except Exception as e:
                        self.logger.error(f"Error cleaning up {component_name}: {e}")
        
        self.logger.info("🧹 Experimental Controller cleaned up")

# Convenience functions for quick experimental setup
def create_construction_hri_study(study_name: str, 
                                conditions: List[str] = None,
                                data_directory: str = None) -> ConstructionExperimentalController:
    """
    Quick setup for construction HRI study.
    
    Parameters
    ----------
    study_name : str
        Name of the study
    conditions : List[str], optional
        List of condition names to test
    data_directory : str, optional
        Directory for data storage
        
    Returns
    -------
    ConstructionExperimentalController
        Configured experimental controller
    """
    
    if not conditions:
        conditions = ["control_direct", "treatment_confidence", "treatment_adaptive"]
    
    if not data_directory:
        data_directory = f"./data_{study_name.lower().replace(' ', '_')}"
    
    # Create controller
    controller = ConstructionExperimentalController(
        study_name=study_name,
        data_directory=data_directory
    )
    
    # Configure experimental design
    experimental_conditions = []
    for condition_name in conditions:
        if hasattr(ExperimentCondition, condition_name.upper()):
            experimental_conditions.append(ExperimentCondition[condition_name.upper()])
    
    design = ExperimentalDesign(
        study_name=study_name,
        conditions=experimental_conditions,
        block_randomization=True,
        within_subject=True,
        tasks_per_condition=8
    )
    
    controller.configure_experiment(design)
    
    return controller