#!/usr/bin/env python3
"""
Behavioral Metrics for Construction HRI Research.

This module implements behavioral measurement systems for analyzing user
behavior patterns during human-robot interaction in construction scenarios.
Focuses on objective behavioral indicators of trust, cognitive load, and
adaptation to different clarification strategies.

Provides:
- Retry speed and error recovery behavior analysis
- Interaction pattern recognition
- Response time analysis
- Task completion behavior tracking
- Statistical analysis and visualization
- Integration with experimental controller
"""

import logging
import time
import json
import threading
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import deque, defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BehaviorType(Enum):
    """Types of behavioral events to track"""
    COMMAND_ISSUED = "command_issued"
    CLARIFICATION_RECEIVED = "clarification_received"
    USER_RESPONSE = "user_response"
    RETRY_ATTEMPT = "retry_attempt"
    ERROR_CORRECTION = "error_correction"
    TASK_COMPLETION = "task_completion"
    HESITATION_PAUSE = "hesitation_pause"
    INTERRUPTION = "interruption"
    STRATEGY_CHANGE = "strategy_change"

class ResponseType(Enum):
    """User response types to robot clarifications"""
    CONFIRMATION = "confirmation"      # "Yes", "Correct", "That's right"
    REJECTION = "rejection"           # "No", "Wrong", "Not that one"
    CLARIFICATION_REQUEST = "clarification_request"  # "What do you mean?", "Can you explain?"
    OVERRIDE = "override"             # User proceeds despite robot uncertainty
    TIMEOUT = "timeout"               # No response within expected time

@dataclass
class BehavioralEvent:
    """Single behavioral event measurement"""
    event_id: str
    participant_id: str
    session_id: str
    event_type: BehaviorType
    timestamp: float
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Context information
    clarification_strategy: Optional[str] = None
    user_expertise: Optional[str] = None
    task_context: Optional[str] = None
    
    # Measurement specifics
    response_time: Optional[float] = None
    error_count: int = 0
    retry_count: int = 0
    confidence_level: Optional[float] = None

@dataclass
class InteractionSequence:
    """Sequence of related behavioral events"""
    sequence_id: str
    participant_id: str
    session_id: str
    start_time: float
    end_time: float
    events: List[BehavioralEvent] = field(default_factory=list)
    outcome: Optional[str] = None  # "success", "failure", "timeout"
    total_retries: int = 0
    total_corrections: int = 0

@dataclass 
class BehavioralMetrics:
    """Calculated behavioral metrics for analysis"""
    participant_id: str
    session_id: str
    clarification_strategy: str
    
    # Response timing metrics
    avg_response_time: float = 0.0
    median_response_time: float = 0.0
    response_time_variance: float = 0.0
    
    # Error and retry metrics
    total_errors: int = 0
    total_retries: int = 0
    avg_retry_speed: float = 0.0  # Time between error and retry attempt
    error_recovery_success_rate: float = 0.0
    
    # Interaction pattern metrics
    hesitation_frequency: float = 0.0  # Hesitations per minute
    interruption_frequency: float = 0.0
    confirmation_rate: float = 0.0     # % of clarifications confirmed
    override_rate: float = 0.0         # % of times user overrode robot
    
    # Task completion metrics
    task_completion_rate: float = 0.0
    avg_task_duration: float = 0.0
    efficiency_score: float = 0.0      # Tasks completed per unit time
    
    # Adaptation metrics
    strategy_adaptation_count: int = 0
    learning_curve_slope: float = 0.0  # Improvement over time
    
    timestamp: float = field(default_factory=time.time)

class ConstructionBehavioralMetrics:
    """
    Behavioral metrics collection and analysis for construction HRI.
    
    Tracks objective behavioral indicators during human-robot interaction
    to measure trust formation, cognitive load, and adaptation patterns.
    
    Parameters
    ----------
    real_time_analysis : bool, optional
        Enable real-time behavioral analysis, by default True
    buffer_size : int, optional
        Size of rolling analysis buffer, by default 1000
    hesitation_threshold : float, optional
        Threshold for detecting hesitation pauses, by default 2.0 seconds
        
    Attributes
    ----------
    behavioral_events : List[BehavioralEvent]
        All recorded behavioral events
    interaction_sequences : List[InteractionSequence]
        Grouped interaction sequences
    real_time_buffer : deque
        Rolling buffer for real-time analysis
    metrics_history : List[BehavioralMetrics]
        Calculated metrics over time
    """
    
    def __init__(self,
                 real_time_analysis: bool = True,
                 buffer_size: int = 1000,
                 hesitation_threshold: float = 2.0):
        
        self.logger = logging.getLogger(__name__)
        self.real_time_analysis = real_time_analysis
        self.buffer_size = buffer_size
        self.hesitation_threshold = hesitation_threshold
        
        # Data storage
        self.behavioral_events = []
        self.interaction_sequences = []
        self.real_time_buffer = deque(maxlen=buffer_size)
        self.metrics_history = []
        
        # Real-time analysis
        self.current_sequences = {}  # Active interaction sequences
        self.response_timers = {}    # Timing ongoing responses
        
        # Analysis thread
        self.analysis_thread = None
        self.stop_analysis = False
        
        if self.real_time_analysis:
            self._start_analysis_thread()
        
        self.logger.info("✅ Construction Behavioral Metrics initialized")
        self.logger.info(f"   Real-time analysis: {real_time_analysis}")
        self.logger.info(f"   Buffer size: {buffer_size}")
        self.logger.info(f"   Hesitation threshold: {hesitation_threshold}s")

    def record_command_issued(self, participant_id: str, session_id: str,
                             command: str, clarification_strategy: str,
                             user_expertise: str = None) -> str:
        """
        Record when user issues a command to the robot.
        
        Parameters
        ----------
        participant_id : str
            Participant identifier
        session_id : str
            Session identifier
        command : str
            The command issued
        clarification_strategy : str
            Active clarification strategy
        user_expertise : str, optional
            User expertise level
            
        Returns
        -------
        str
            Event ID for tracking
        """
        
        event_id = f"cmd_{int(time.time() * 1000)}"
        
        event = BehavioralEvent(
            event_id=event_id,
            participant_id=participant_id,
            session_id=session_id,
            event_type=BehaviorType.COMMAND_ISSUED,
            timestamp=time.time(),
            clarification_strategy=clarification_strategy,
            user_expertise=user_expertise,
            metadata={'command': command}
        )
        
        self._add_event(event)
        
        # Start tracking response time
        self.response_timers[event_id] = time.time()
        
        return event_id

    def record_clarification_received(self, participant_id: str, session_id: str,
                                    clarification_text: str, confidence: float,
                                    clarification_strategy: str,
                                    command_event_id: str = None) -> str:
        """
        Record when robot provides clarification to user.
        
        Parameters
        ----------
        participant_id : str
            Participant identifier
        session_id : str
            Session identifier
        clarification_text : str
            The clarification provided
        confidence : float
            Robot's confidence in the clarification
        clarification_strategy : str
            Strategy used for clarification
        command_event_id : str, optional
            Related command event ID
            
        Returns
        -------
        str
            Event ID for tracking
        """
        
        event_id = f"clar_{int(time.time() * 1000)}"
        
        # Calculate response time if related command exists
        response_time = None
        if command_event_id and command_event_id in self.response_timers:
            response_time = time.time() - self.response_timers[command_event_id]
            del self.response_timers[command_event_id]
        
        event = BehavioralEvent(
            event_id=event_id,
            participant_id=participant_id,
            session_id=session_id,
            event_type=BehaviorType.CLARIFICATION_RECEIVED,
            timestamp=time.time(),
            response_time=response_time,
            confidence_level=confidence,
            clarification_strategy=clarification_strategy,
            metadata={
                'clarification_text': clarification_text,
                'related_command': command_event_id
            }
        )
        
        self._add_event(event)
        
        # Start tracking user response time
        self.response_timers[event_id] = time.time()
        
        return event_id

    def record_user_response(self, participant_id: str, session_id: str,
                           response_type: ResponseType, response_text: str,
                           clarification_event_id: str = None) -> str:
        """
        Record user's response to robot clarification.
        
        Parameters
        ----------
        participant_id : str
            Participant identifier
        session_id : str
            Session identifier
        response_type : ResponseType
            Type of user response
        response_text : str
            Actual response text
        clarification_event_id : str, optional
            Related clarification event ID
            
        Returns
        -------
        str
            Event ID for tracking
        """
        
        event_id = f"resp_{int(time.time() * 1000)}"
        
        # Calculate response time
        response_time = None
        if clarification_event_id and clarification_event_id in self.response_timers:
            response_time = time.time() - self.response_timers[clarification_event_id]
            del self.response_timers[clarification_event_id]
        
        event = BehavioralEvent(
            event_id=event_id,
            participant_id=participant_id,
            session_id=session_id,
            event_type=BehaviorType.USER_RESPONSE,
            timestamp=time.time(),
            response_time=response_time,
            metadata={
                'response_type': response_type.value,
                'response_text': response_text,
                'related_clarification': clarification_event_id
            }
        )
        
        self._add_event(event)
        
        return event_id

    def record_retry_attempt(self, participant_id: str, session_id: str,
                           original_command: str, new_command: str,
                           retry_reason: str = None) -> str:
        """
        Record when user retries after an error or clarification.
        
        Parameters
        ----------
        participant_id : str
            Participant identifier
        session_id : str
            Session identifier
        original_command : str
            Original command that failed
        new_command : str
            New/modified command
        retry_reason : str, optional
            Reason for retry
            
        Returns
        -------
        str
            Event ID for tracking
        """
        
        event_id = f"retry_{int(time.time() * 1000)}"
        
        event = BehavioralEvent(
            event_id=event_id,
            participant_id=participant_id,
            session_id=session_id,
            event_type=BehaviorType.RETRY_ATTEMPT,
            timestamp=time.time(),
            retry_count=1,
            metadata={
                'original_command': original_command,
                'new_command': new_command,
                'retry_reason': retry_reason
            }
        )
        
        self._add_event(event)
        
        return event_id

    def record_hesitation_pause(self, participant_id: str, session_id: str,
                              pause_duration: float, context: str = None) -> str:
        """
        Record hesitation pauses that may indicate cognitive load or uncertainty.
        
        Parameters
        ----------
        participant_id : str
            Participant identifier
        session_id : str
            Session identifier
        pause_duration : float
            Length of hesitation pause in seconds
        context : str, optional
            Context where hesitation occurred
            
        Returns
        -------
        str
            Event ID for tracking
        """
        
        event_id = f"hesit_{int(time.time() * 1000)}"
        
        event = BehavioralEvent(
            event_id=event_id,
            participant_id=participant_id,
            session_id=session_id,
            event_type=BehaviorType.HESITATION_PAUSE,
            timestamp=time.time(),
            duration=pause_duration,
            metadata={
                'context': context,
                'exceeds_threshold': pause_duration > self.hesitation_threshold
            }
        )
        
        self._add_event(event)
        
        return event_id

    def record_task_completion(self, participant_id: str, session_id: str,
                             task_description: str, success: bool,
                             completion_time: float, error_count: int = 0) -> str:
        """
        Record task completion with success/failure and metrics.
        
        Parameters
        ----------
        participant_id : str
            Participant identifier
        session_id : str
            Session identifier
        task_description : str
            Description of completed task
        success : bool
            Whether task was completed successfully
        completion_time : float
            Time taken to complete task
        error_count : int, optional
            Number of errors during task
            
        Returns
        -------
        str
            Event ID for tracking
        """
        
        event_id = f"task_{int(time.time() * 1000)}"
        
        event = BehavioralEvent(
            event_id=event_id,
            participant_id=participant_id,
            session_id=session_id,
            event_type=BehaviorType.TASK_COMPLETION,
            timestamp=time.time(),
            duration=completion_time,
            error_count=error_count,
            metadata={
                'task_description': task_description,
                'success': success,
                'errors_during_task': error_count
            }
        )
        
        self._add_event(event)
        
        return event_id

    def _add_event(self, event: BehavioralEvent):
        """Add event to storage and real-time buffer"""
        
        self.behavioral_events.append(event)
        self.real_time_buffer.append(event)
        
        if self.real_time_analysis:
            self._update_real_time_analysis(event)

    def _update_real_time_analysis(self, event: BehavioralEvent):
        """Update real-time behavioral analysis"""
        
        # Detect hesitation patterns
        if event.event_type == BehaviorType.USER_RESPONSE and event.response_time:
            if event.response_time > self.hesitation_threshold:
                self.record_hesitation_pause(
                    event.participant_id, event.session_id, 
                    event.response_time, "response_delay"
                )
        
        # Group related events into sequences
        sequence_key = f"{event.participant_id}_{event.session_id}"
        if sequence_key not in self.current_sequences:
            self.current_sequences[sequence_key] = InteractionSequence(
                sequence_id=f"seq_{int(time.time() * 1000)}",
                participant_id=event.participant_id,
                session_id=event.session_id,
                start_time=event.timestamp,
                end_time=event.timestamp
            )
        
        # Add event to current sequence
        sequence = self.current_sequences[sequence_key]
        sequence.events.append(event)
        sequence.end_time = event.timestamp
        
        # Update sequence counters
        if event.event_type == BehaviorType.RETRY_ATTEMPT:
            sequence.total_retries += event.retry_count
        elif event.event_type == BehaviorType.ERROR_CORRECTION:
            sequence.total_corrections += 1

    def calculate_behavioral_metrics(self, participant_id: str, session_id: str,
                                   clarification_strategy: str) -> BehavioralMetrics:
        """
        Calculate comprehensive behavioral metrics for analysis.
        
        Parameters
        ----------
        participant_id : str
            Participant identifier
        session_id : str
            Session identifier
        clarification_strategy : str
            Clarification strategy used
            
        Returns
        -------
        BehavioralMetrics
            Calculated behavioral metrics
        """
        
        # Filter events for this participant/session
        session_events = [
            e for e in self.behavioral_events
            if e.participant_id == participant_id and e.session_id == session_id
        ]
        
        if not session_events:
            return BehavioralMetrics(
                participant_id=participant_id,
                session_id=session_id,
                clarification_strategy=clarification_strategy
            )
        
        metrics = BehavioralMetrics(
            participant_id=participant_id,
            session_id=session_id,
            clarification_strategy=clarification_strategy
        )
        
        # Response timing analysis
        response_times = [
            e.response_time for e in session_events 
            if e.response_time and e.response_time > 0
        ]
        
        if response_times:
            metrics.avg_response_time = statistics.mean(response_times)
            metrics.median_response_time = statistics.median(response_times)
            metrics.response_time_variance = statistics.variance(response_times) if len(response_times) > 1 else 0.0
        
        # Error and retry analysis
        retry_events = [e for e in session_events if e.event_type == BehaviorType.RETRY_ATTEMPT]
        error_events = [e for e in session_events if e.event_type == BehaviorType.ERROR_CORRECTION]
        
        metrics.total_retries = len(retry_events)
        metrics.total_errors = len(error_events)
        
        # Calculate retry speed (time between error and retry)
        retry_speeds = []
        for i, error_event in enumerate(error_events):
            # Find next retry after this error
            next_retry = next((r for r in retry_events if r.timestamp > error_event.timestamp), None)
            if next_retry:
                retry_speed = next_retry.timestamp - error_event.timestamp
                retry_speeds.append(retry_speed)
        
        if retry_speeds:
            metrics.avg_retry_speed = statistics.mean(retry_speeds)
        
        # Error recovery success rate
        task_events = [e for e in session_events if e.event_type == BehaviorType.TASK_COMPLETION]
        successful_tasks = [e for e in task_events if e.metadata.get('success', False)]
        
        if task_events:
            metrics.error_recovery_success_rate = len(successful_tasks) / len(task_events)
            metrics.task_completion_rate = len(successful_tasks) / len(task_events)
            
            task_durations = [e.duration for e in task_events if e.duration > 0]
            if task_durations:
                metrics.avg_task_duration = statistics.mean(task_durations)
        
        # Interaction pattern analysis
        clarification_events = [e for e in session_events if e.event_type == BehaviorType.CLARIFICATION_RECEIVED]
        response_events = [e for e in session_events if e.event_type == BehaviorType.USER_RESPONSE]
        
        if clarification_events:
            confirmations = [
                r for r in response_events 
                if r.metadata.get('response_type') == 'confirmation'
            ]
            overrides = [
                r for r in response_events 
                if r.metadata.get('response_type') == 'override'
            ]
            
            metrics.confirmation_rate = len(confirmations) / len(clarification_events)
            metrics.override_rate = len(overrides) / len(clarification_events)
        
        # Hesitation frequency (per minute)
        hesitation_events = [e for e in session_events if e.event_type == BehaviorType.HESITATION_PAUSE]
        if session_events:
            session_duration = (session_events[-1].timestamp - session_events[0].timestamp) / 60.0
            if session_duration > 0:
                metrics.hesitation_frequency = len(hesitation_events) / session_duration
        
        # Efficiency score (tasks per minute)
        if task_events and session_events:
            session_duration_minutes = (session_events[-1].timestamp - session_events[0].timestamp) / 60.0
            if session_duration_minutes > 0:
                metrics.efficiency_score = len(task_events) / session_duration_minutes
        
        self.metrics_history.append(metrics)
        
        self.logger.info(f"📊 Behavioral metrics calculated for {participant_id}")
        self.logger.info(f"   Response time: {metrics.avg_response_time:.2f}s")
        self.logger.info(f"   Success rate: {metrics.task_completion_rate:.2f}")
        self.logger.info(f"   Retry rate: {metrics.total_retries}")
        
        return metrics

    def compare_strategies(self, strategy1: str, strategy2: str) -> Dict[str, Any]:
        """
        Compare behavioral metrics between two clarification strategies.
        
        Parameters
        ----------
        strategy1 : str
            First strategy to compare
        strategy2 : str
            Second strategy to compare
            
        Returns
        -------
        Dict[str, Any]
            Comparison results
        """
        
        # Get metrics for each strategy
        metrics1 = [m for m in self.metrics_history if m.clarification_strategy == strategy1]
        metrics2 = [m for m in self.metrics_history if m.clarification_strategy == strategy2]
        
        if not metrics1 or not metrics2:
            return {'error': 'Insufficient data for comparison'}
        
        comparison = {
            'strategy1': strategy1,
            'strategy2': strategy2,
            'sample_sizes': {
                strategy1: len(metrics1),
                strategy2: len(metrics2)
            },
            'differences': {}
        }
        
        # Compare key metrics
        metrics_to_compare = [
            'avg_response_time', 'total_retries', 'task_completion_rate',
            'confirmation_rate', 'hesitation_frequency', 'efficiency_score'
        ]
        
        for metric_name in metrics_to_compare:
            values1 = [getattr(m, metric_name) for m in metrics1]
            values2 = [getattr(m, metric_name) for m in metrics2]
            
            if values1 and values2:
                mean1 = statistics.mean(values1)
                mean2 = statistics.mean(values2)
                
                comparison['differences'][metric_name] = {
                    f'{strategy1}_mean': mean1,
                    f'{strategy2}_mean': mean2,
                    'difference': mean2 - mean1,
                    'percent_change': ((mean2 - mean1) / mean1 * 100) if mean1 != 0 else 0
                }
        
        return comparison

    def _start_analysis_thread(self):
        """Start background thread for real-time analysis"""
        
        self.stop_analysis = False
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        self.logger.info("🔄 Real-time behavioral analysis started")

    def _analysis_loop(self):
        """Background analysis loop"""
        
        while not self.stop_analysis:
            try:
                # Clean up old timers
                current_time = time.time()
                expired_timers = [
                    timer_id for timer_id, start_time in self.response_timers.items()
                    if current_time - start_time > 30.0  # 30 second timeout
                ]
                
                for timer_id in expired_timers:
                    del self.response_timers[timer_id]
                
                # Sleep before next analysis cycle
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")

    def export_behavioral_data(self, filepath: str, participant_ids: Optional[List[str]] = None):
        """Export behavioral data for statistical analysis"""
        
        # Filter events and metrics
        events_to_export = self.behavioral_events
        metrics_to_export = self.metrics_history
        
        if participant_ids:
            events_to_export = [e for e in self.behavioral_events if e.participant_id in participant_ids]
            metrics_to_export = [m for m in self.metrics_history if m.participant_id in participant_ids]
        
        # Prepare export data
        export_data = {
            'behavioral_info': {
                'event_types': [e.value for e in BehaviorType],
                'response_types': [r.value for r in ResponseType],
                'hesitation_threshold': self.hesitation_threshold,
                'export_timestamp': time.time()
            },
            'events': [],
            'metrics': [],
            'interaction_sequences': []
        }
        
        # Export events
        for event in events_to_export:
            export_data['events'].append({
                'event_id': event.event_id,
                'participant_id': event.participant_id,
                'session_id': event.session_id,
                'event_type': event.event_type.value,
                'timestamp': event.timestamp,
                'duration': event.duration,
                'response_time': event.response_time,
                'error_count': event.error_count,
                'retry_count': event.retry_count,
                'confidence_level': event.confidence_level,
                'clarification_strategy': event.clarification_strategy,
                'user_expertise': event.user_expertise,
                'metadata': event.metadata
            })
        
        # Export calculated metrics
        for metrics in metrics_to_export:
            export_data['metrics'].append({
                'participant_id': metrics.participant_id,
                'session_id': metrics.session_id,
                'clarification_strategy': metrics.clarification_strategy,
                'avg_response_time': metrics.avg_response_time,
                'median_response_time': metrics.median_response_time,
                'response_time_variance': metrics.response_time_variance,
                'total_errors': metrics.total_errors,
                'total_retries': metrics.total_retries,
                'avg_retry_speed': metrics.avg_retry_speed,
                'error_recovery_success_rate': metrics.error_recovery_success_rate,
                'hesitation_frequency': metrics.hesitation_frequency,
                'confirmation_rate': metrics.confirmation_rate,
                'override_rate': metrics.override_rate,
                'task_completion_rate': metrics.task_completion_rate,
                'avg_task_duration': metrics.avg_task_duration,
                'efficiency_score': metrics.efficiency_score,
                'timestamp': metrics.timestamp
            })
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"📊 Behavioral data exported to {filepath}")
        self.logger.info(f"   Events: {len(export_data['events'])}")
        self.logger.info(f"   Metrics: {len(export_data['metrics'])}")

    def record_error_recovery(self, participant_id: str, session_id: str,
                            recovery_strategy: str, recovery_time: float,
                            success: bool = True) -> str:
        """
        Record error recovery behavior and time.
        
        Parameters
        ----------
        participant_id : str
            Participant identifier
        session_id : str
            Session identifier
        recovery_strategy : str
            Strategy used for error recovery
        recovery_time : float
            Time taken to recover from error
        success : bool, optional
            Whether recovery was successful
            
        Returns
        -------
        str
            Event ID for tracking
        """
        
        event_id = f"recovery_{int(time.time() * 1000)}"
        
        event = BehavioralEvent(
            event_id=event_id,
            participant_id=participant_id,
            session_id=session_id,
            event_type=BehaviorType.TASK_COMPLETION,  # Reuse existing type
            timestamp=time.time(),
            duration=recovery_time,
            metadata={
                'recovery_strategy': recovery_strategy,
                'recovery_success': success,
                'error_recovery': True
            }
        )
        
        self._add_event(event)
        
        if self.real_time_analysis:
            self.logger.info(f"🔄 Error recovery: {recovery_strategy} in {recovery_time:.1f}s")
        
        return event_id

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall behavioral statistics"""
        
        if not self.behavioral_events:
            return {"message": "No behavioral data recorded yet"}
        
        stats = {
            'total_events': len(self.behavioral_events),
            'unique_participants': len(set(e.participant_id for e in self.behavioral_events)),
            'unique_sessions': len(set(e.session_id for e in self.behavioral_events)),
            'event_type_counts': {},
            'metrics_calculated': len(self.metrics_history),
            'avg_response_times_by_strategy': {}
        }
        
        # Event type breakdown
        for event in self.behavioral_events:
            event_type = event.event_type.value
            if event_type not in stats['event_type_counts']:
                stats['event_type_counts'][event_type] = 0
            stats['event_type_counts'][event_type] += 1
        
        # Response times by strategy
        strategy_responses = defaultdict(list)
        for event in self.behavioral_events:
            if event.response_time and event.clarification_strategy:
                strategy_responses[event.clarification_strategy].append(event.response_time)
        
        for strategy, times in strategy_responses.items():
            if times:
                stats['avg_response_times_by_strategy'][strategy] = {
                    'mean': statistics.mean(times),
                    'count': len(times)
                }
        
        return stats

    def cleanup(self):
        """Clean up resources"""
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.stop_analysis = True
            self.analysis_thread.join(timeout=2.0)
        
        self.logger.info("📊 Behavioral Metrics cleaned up")