#!/usr/bin/env python3
"""
Trust Questionnaire for Construction HRI Research.

This module implements post-trial trust assessment questionnaires for measuring
trust formation in construction robotics scenarios. Based on established trust
measurement scales adapted for construction domain expertise inversion.

Provides:
- Multi-dimensional trust measurement (competence, benevolence, integrity)
- Construction-specific trust scenarios
- Pre/post interaction trust comparison
- Statistical analysis and reporting
- Integration with experimental controller
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrustDimension(Enum):
    """Trust measurement dimensions based on Mayer et al. (1995)"""
    COMPETENCE = "competence"        # Robot's ability to perform tasks effectively
    BENEVOLENCE = "benevolence"      # Robot's intention to do good for the user
    INTEGRITY = "integrity"          # Robot's adherence to principles the user finds acceptable
    OVERALL = "overall"              # General trust in the robot

class QuestionType(Enum):
    """Types of trust questions"""
    LIKERT_SCALE = "likert_scale"    # 1-7 scale questions
    SCENARIO_BASED = "scenario_based" # Construction scenario responses
    BEHAVIORAL_INTENT = "behavioral_intent" # Willingness to rely on robot
    COMPARATIVE = "comparative"       # Comparison to human workers

@dataclass
class TrustQuestion:
    """Individual trust assessment question"""
    id: str
    text: str
    dimension: TrustDimension
    question_type: QuestionType
    scale_min: int = 1
    scale_max: int = 7
    reverse_scored: bool = False
    construction_context: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrustResponse:
    """Response to a trust question"""
    question_id: str
    response_value: int
    response_text: Optional[str] = None
    response_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class TrustAssessment:
    """Complete trust assessment results"""
    participant_id: str
    session_id: str
    assessment_type: str  # "pre", "post", "follow_up"
    responses: List[TrustResponse]
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    completion_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

class ConstructionTrustQuestionnaire:
    """
    Trust questionnaire system for construction HRI research.
    
    Implements validated trust measurement scales adapted for construction
    robotics scenarios with expertise inversion considerations.
    
    Parameters
    ----------
    include_scenarios : bool, optional
        Include construction scenario questions, by default True
    expertise_adaptive : bool, optional
        Adapt questions based on user expertise, by default True
    randomize_questions : bool, optional
        Randomize question order, by default True
        
    Attributes
    ----------
    questions : List[TrustQuestion]
        All available trust questions
    construction_scenarios : List[Dict]
        Construction-specific trust scenarios
    assessment_history : List[TrustAssessment]
        History of completed assessments
    """
    
    def __init__(self, 
                 include_scenarios: bool = True,
                 expertise_adaptive: bool = True,
                 randomize_questions: bool = True):
        
        self.logger = logging.getLogger(__name__)
        self.include_scenarios = include_scenarios
        self.expertise_adaptive = expertise_adaptive
        self.randomize_questions = randomize_questions
        
        # Initialize question database
        self.questions = self._create_trust_questions()
        self.construction_scenarios = self._create_construction_scenarios()
        self.assessment_history = []
        
        self.logger.info("✅ Construction Trust Questionnaire initialized")
        self.logger.info(f"   Questions: {len(self.questions)}")
        self.logger.info(f"   Scenarios: {len(self.construction_scenarios)}")
        self.logger.info(f"   Expertise adaptive: {expertise_adaptive}")

    def _create_trust_questions(self) -> List[TrustQuestion]:
        """Create comprehensive trust assessment questions"""
        
        questions = []
        
        # COMPETENCE dimension questions
        competence_questions = [
            TrustQuestion(
                id="comp_01",
                text="The robot is capable of accurately identifying construction tools.",
                dimension=TrustDimension.COMPETENCE,
                question_type=QuestionType.LIKERT_SCALE
            ),
            TrustQuestion(
                id="comp_02", 
                text="I can rely on the robot to distinguish between similar tools (e.g., different types of hammers).",
                dimension=TrustDimension.COMPETENCE,
                question_type=QuestionType.LIKERT_SCALE
            ),
            TrustQuestion(
                id="comp_03",
                text="The robot understands construction terminology as well as an experienced worker.",
                dimension=TrustDimension.COMPETENCE,
                question_type=QuestionType.LIKERT_SCALE
            ),
            TrustQuestion(
                id="comp_04",
                text="The robot makes frequent mistakes when identifying tools.",
                dimension=TrustDimension.COMPETENCE,
                question_type=QuestionType.LIKERT_SCALE,
                reverse_scored=True
            ),
            TrustQuestion(
                id="comp_05",
                text="I trust the robot's ability to work safely around construction equipment.",
                dimension=TrustDimension.COMPETENCE,
                question_type=QuestionType.LIKERT_SCALE
            )
        ]
        
        # BENEVOLENCE dimension questions
        benevolence_questions = [
            TrustQuestion(
                id="bene_01",
                text="The robot genuinely cares about helping me complete my work efficiently.",
                dimension=TrustDimension.BENEVOLENCE,
                question_type=QuestionType.LIKERT_SCALE
            ),
            TrustQuestion(
                id="bene_02",
                text="When the robot asks for clarification, it's trying to be helpful, not questioning my expertise.",
                dimension=TrustDimension.BENEVOLENCE,
                question_type=QuestionType.LIKERT_SCALE
            ),
            TrustQuestion(
                id="bene_03",
                text="The robot considers my professional experience when making suggestions.",
                dimension=TrustDimension.BENEVOLENCE,
                question_type=QuestionType.LIKERT_SCALE
            ),
            TrustQuestion(
                id="bene_04",
                text="I believe the robot prioritizes job site safety over task completion speed.",
                dimension=TrustDimension.BENEVOLENCE,
                question_type=QuestionType.LIKERT_SCALE
            ),
            TrustQuestion(
                id="bene_05",
                text="The robot seems indifferent to whether I succeed or fail at my tasks.",
                dimension=TrustDimension.BENEVOLENCE,
                question_type=QuestionType.LIKERT_SCALE,
                reverse_scored=True
            )
        ]
        
        # INTEGRITY dimension questions
        integrity_questions = [
            TrustQuestion(
                id="integ_01",
                text="The robot is honest about its confidence level when identifying tools.",
                dimension=TrustDimension.INTEGRITY,
                question_type=QuestionType.LIKERT_SCALE
            ),
            TrustQuestion(
                id="integ_02",
                text="The robot admits when it doesn't know something rather than guessing.",
                dimension=TrustDimension.INTEGRITY,
                question_type=QuestionType.LIKERT_SCALE
            ),
            TrustQuestion(
                id="integ_03",
                text="I can depend on the robot to be consistent in its responses.",
                dimension=TrustDimension.INTEGRITY,
                question_type=QuestionType.LIKERT_SCALE
            ),
            TrustQuestion(
                id="integ_04",
                text="The robot follows construction safety protocols even when it might slow down work.",
                dimension=TrustDimension.INTEGRITY,
                question_type=QuestionType.LIKERT_SCALE
            ),
            TrustQuestion(
                id="integ_05",
                text="The robot sometimes gives conflicting information about the same tool.",
                dimension=TrustDimension.INTEGRITY,
                question_type=QuestionType.LIKERT_SCALE,
                reverse_scored=True
            )
        ]
        
        # OVERALL trust questions
        overall_questions = [
            TrustQuestion(
                id="overall_01",
                text="Overall, I trust this robot as a construction assistant.",
                dimension=TrustDimension.OVERALL,
                question_type=QuestionType.LIKERT_SCALE
            ),
            TrustQuestion(
                id="overall_02",
                text="I would be comfortable working alongside this robot on a construction site.",
                dimension=TrustDimension.OVERALL,
                question_type=QuestionType.LIKERT_SCALE
            ),
            TrustQuestion(
                id="overall_03",
                text="I would recommend this robot to other construction workers.",
                dimension=TrustDimension.OVERALL,
                question_type=QuestionType.LIKERT_SCALE
            )
        ]
        
        # BEHAVIORAL INTENT questions
        behavioral_questions = [
            TrustQuestion(
                id="behav_01",
                text="I would rely on this robot to hand me the correct tool even for critical tasks.",
                dimension=TrustDimension.OVERALL,
                question_type=QuestionType.BEHAVIORAL_INTENT
            ),
            TrustQuestion(
                id="behav_02",
                text="I would let this robot work unsupervised in my tool area.",
                dimension=TrustDimension.OVERALL,
                question_type=QuestionType.BEHAVIORAL_INTENT
            ),
            TrustQuestion(
                id="behav_03",
                text="I would trust this robot to identify safety hazards on the job site.",
                dimension=TrustDimension.OVERALL,
                question_type=QuestionType.BEHAVIORAL_INTENT
            )
        ]
        
        questions.extend(competence_questions)
        questions.extend(benevolence_questions)
        questions.extend(integrity_questions)
        questions.extend(overall_questions)
        questions.extend(behavioral_questions)
        
        return questions

    def _create_construction_scenarios(self) -> List[Dict[str, Any]]:
        """Create construction-specific trust scenarios"""
        
        scenarios = [
            {
                "id": "scenario_01",
                "title": "Tool Selection Under Pressure",
                "description": (
                    "You're working on a time-critical repair and need a specific wrench quickly. "
                    "The robot identifies several wrenches with varying confidence levels. "
                    "How much would you trust the robot's recommendation?"
                ),
                "options": [
                    "Completely trust the robot's choice",
                    "Ask the robot to double-check", 
                    "Look for the tool myself",
                    "Get a second opinion from a coworker"
                ],
                "dimension": TrustDimension.COMPETENCE
            },
            {
                "id": "scenario_02", 
                "title": "Safety Equipment Verification",
                "description": (
                    "The robot reports that you're missing required safety equipment before "
                    "starting a task. You're confident you have everything needed. "
                    "How would you respond?"
                ),
                "options": [
                    "Double-check my equipment as the robot suggested",
                    "Ask the robot to explain what it thinks is missing",
                    "Proceed with the task as I know I'm properly equipped",
                    "Call a supervisor to resolve the disagreement"
                ],
                "dimension": TrustDimension.BENEVOLENCE
            },
            {
                "id": "scenario_03",
                "title": "Expertise Contradiction", 
                "description": (
                    "The robot suggests using a different tool than what your 15 years of "
                    "experience tells you is correct. The robot says it's 85% confident. "
                    "What would you do?"
                ),
                "options": [
                    "Follow the robot's suggestion",
                    "Use my preferred tool but ask why the robot disagreed",
                    "Use my preferred tool without discussion",
                    "Try both approaches to see which works better"
                ],
                "dimension": TrustDimension.INTEGRITY
            },
            {
                "id": "scenario_04",
                "title": "Quality Control Check",
                "description": (
                    "After completing a task, the robot offers to inspect your work for "
                    "quality control. You're confident the work meets standards. "
                    "How do you respond?"
                ),
                "options": [
                    "Welcome the robot's quality check",
                    "Allow it but specify what to check",
                    "Politely decline the inspection",
                    "Feel insulted by the suggestion"
                ],
                "dimension": TrustDimension.OVERALL
            }
        ]
        
        return scenarios

    def conduct_assessment(self, 
                         participant_id: str,
                         session_id: str,
                         assessment_type: str = "post",
                         user_expertise: Optional[str] = None) -> TrustAssessment:
        """
        Conduct a trust assessment session.
        
        Parameters
        ----------
        participant_id : str
            Unique participant identifier
        session_id : str
            Experimental session identifier  
        assessment_type : str
            Type of assessment ("pre", "post", "follow_up")
        user_expertise : str, optional
            User expertise level for adaptive questions
            
        Returns
        -------
        TrustAssessment
            Completed trust assessment with scores
        """
        
        start_time = time.time()
        responses = []
        
        # Select questions based on assessment type and expertise
        selected_questions = self._select_questions(assessment_type, user_expertise)
        
        if self.randomize_questions:
            import random
            random.shuffle(selected_questions)
        
        self.logger.info(f"🔍 Starting {assessment_type} trust assessment for {participant_id}")
        self.logger.info(f"   Questions: {len(selected_questions)}")
        
        # In a real implementation, this would present questions to the user
        # For now, we'll simulate responses for testing
        for question in selected_questions:
            response = self._simulate_user_response(question)
            responses.append(response)
        
        # Include scenario questions if enabled
        if self.include_scenarios and assessment_type in ["post", "follow_up"]:
            for scenario in self.construction_scenarios:
                scenario_response = self._simulate_scenario_response(scenario)
                responses.append(scenario_response)
        
        completion_time = time.time() - start_time
        
        # Calculate scores
        assessment = TrustAssessment(
            participant_id=participant_id,
            session_id=session_id,
            assessment_type=assessment_type,
            responses=responses,
            completion_time=completion_time
        )
        
        self._calculate_trust_scores(assessment)
        self.assessment_history.append(assessment)
        
        self.logger.info(f"✅ Assessment completed in {completion_time:.1f}s")
        self.logger.info(f"   Overall trust: {assessment.overall_score:.2f}/7")
        
        return assessment

    def _select_questions(self, assessment_type: str, user_expertise: Optional[str]) -> List[TrustQuestion]:
        """Select appropriate questions based on context"""
        
        # Base question set
        selected = self.questions.copy()
        
        # Filter based on assessment type
        if assessment_type == "pre":
            # Pre-interaction: focus on general trust propensity
            selected = [q for q in selected if q.id.startswith(('overall', 'behav'))]
        elif assessment_type == "post":
            # Post-interaction: all questions
            pass  # Use all questions
        elif assessment_type == "follow_up":
            # Follow-up: focus on behavioral intent and overall trust
            selected = [q for q in selected if q.question_type in [
                QuestionType.BEHAVIORAL_INTENT, QuestionType.LIKERT_SCALE
            ]]
        
        # Expertise-based adaptation
        if self.expertise_adaptive and user_expertise:
            if user_expertise.lower() == "apprentice":
                # More detailed questions for apprentices
                pass  # Use all selected questions
            elif user_expertise.lower() == "master":
                # Shorter, focused questions for masters
                selected = [q for q in selected if not q.text.startswith("I can rely")]
        
        return selected

    def _simulate_user_response(self, question: TrustQuestion) -> TrustResponse:
        """Simulate user response for testing purposes"""
        
        # In real implementation, this would collect actual user input
        import random
        
        # Simulate realistic response patterns
        if question.dimension == TrustDimension.COMPETENCE:
            # Slightly lower trust in competence due to expertise inversion
            response_value = random.randint(4, 6)
        elif question.dimension == TrustDimension.BENEVOLENCE:
            # Moderate trust in benevolence
            response_value = random.randint(5, 7)
        elif question.dimension == TrustDimension.INTEGRITY:
            # Higher trust in integrity
            response_value = random.randint(5, 7)
        else:  # Overall
            response_value = random.randint(4, 6)
        
        # Apply reverse scoring
        if question.reverse_scored:
            response_value = question.scale_max + question.scale_min - response_value
        
        return TrustResponse(
            question_id=question.id,
            response_value=response_value,
            response_time=random.uniform(2.0, 8.0)  # Simulate response time
        )

    def _simulate_scenario_response(self, scenario: Dict[str, Any]) -> TrustResponse:
        """Simulate scenario-based response"""
        
        import random
        
        # Simulate choosing one of the scenario options
        response_value = random.randint(1, len(scenario['options']))
        
        return TrustResponse(
            question_id=scenario['id'],
            response_value=response_value,
            response_text=scenario['options'][response_value - 1],
            response_time=random.uniform(5.0, 15.0)
        )

    def _calculate_trust_scores(self, assessment: TrustAssessment):
        """Calculate dimension scores and overall trust"""
        
        # Group responses by dimension
        dimension_responses = {}
        for response in assessment.responses:
            # Find the question for this response
            question = next((q for q in self.questions if q.id == response.question_id), None)
            if question and question.question_type == QuestionType.LIKERT_SCALE:
                dim = question.dimension.value
                if dim not in dimension_responses:
                    dimension_responses[dim] = []
                dimension_responses[dim].append(response.response_value)
        
        # Calculate dimension means
        for dimension, values in dimension_responses.items():
            if values:
                assessment.dimension_scores[dimension] = statistics.mean(values)
        
        # Calculate overall trust score
        if assessment.dimension_scores:
            assessment.overall_score = statistics.mean(assessment.dimension_scores.values())

    def get_trust_change(self, participant_id: str, session_id: str) -> Optional[Dict[str, float]]:
        """
        Calculate trust changes between pre and post assessments.
        
        Parameters
        ----------
        participant_id : str
            Participant identifier
        session_id : str
            Session identifier
            
        Returns
        -------
        Optional[Dict[str, float]]
            Trust change scores by dimension, or None if pre/post not found
        """
        
        # Find pre and post assessments
        pre_assessment = None
        post_assessment = None
        
        for assessment in self.assessment_history:
            if (assessment.participant_id == participant_id and 
                assessment.session_id == session_id):
                if assessment.assessment_type == "pre":
                    pre_assessment = assessment
                elif assessment.assessment_type == "post":
                    post_assessment = assessment
        
        if not (pre_assessment and post_assessment):
            return None
        
        # Calculate changes
        changes = {}
        
        # Overall trust change
        changes['overall'] = post_assessment.overall_score - pre_assessment.overall_score
        
        # Dimension-specific changes
        for dimension in TrustDimension:
            dim_key = dimension.value
            pre_score = pre_assessment.dimension_scores.get(dim_key, 0)
            post_score = post_assessment.dimension_scores.get(dim_key, 0)
            if pre_score > 0 and post_score > 0:
                changes[dim_key] = post_score - pre_score
        
        return changes

    def get_assessment_summary(self, assessment: TrustAssessment) -> Dict[str, Any]:
        """Generate summary statistics for an assessment"""
        
        return {
            'participant_id': assessment.participant_id,
            'session_id': assessment.session_id,
            'assessment_type': assessment.assessment_type,
            'overall_trust': assessment.overall_score,
            'dimension_scores': assessment.dimension_scores,
            'response_count': len(assessment.responses),
            'completion_time': assessment.completion_time,
            'avg_response_time': statistics.mean([r.response_time for r in assessment.responses if r.response_time > 0]),
            'timestamp': assessment.timestamp
        }

    def export_assessment_data(self, filepath: str, participant_ids: Optional[List[str]] = None):
        """Export assessment data for statistical analysis"""
        
        # Filter assessments if participant list provided
        assessments_to_export = self.assessment_history
        if participant_ids:
            assessments_to_export = [
                a for a in self.assessment_history 
                if a.participant_id in participant_ids
            ]
        
        # Prepare export data
        export_data = {
            'questionnaire_info': {
                'questions_count': len(self.questions),
                'scenarios_count': len(self.construction_scenarios),
                'dimensions': [d.value for d in TrustDimension],
                'export_timestamp': time.time()
            },
            'assessments': [self.get_assessment_summary(a) for a in assessments_to_export],
            'detailed_responses': []
        }
        
        # Add detailed response data
        for assessment in assessments_to_export:
            for response in assessment.responses:
                export_data['detailed_responses'].append({
                    'participant_id': assessment.participant_id,
                    'session_id': assessment.session_id,
                    'assessment_type': assessment.assessment_type,
                    'question_id': response.question_id,
                    'response_value': response.response_value,
                    'response_text': response.response_text,
                    'response_time': response.response_time,
                    'timestamp': response.timestamp
                })
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"📊 Trust data exported to {filepath}")
        self.logger.info(f"   Assessments: {len(assessments_to_export)}")
        self.logger.info(f"   Responses: {len(export_data['detailed_responses'])}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall questionnaire statistics"""
        
        if not self.assessment_history:
            return {"message": "No assessments completed yet"}
        
        # Basic statistics
        stats = {
            'total_assessments': len(self.assessment_history),
            'unique_participants': len(set(a.participant_id for a in self.assessment_history)),
            'assessment_types': {},
            'average_scores': {},
            'completion_times': []
        }
        
        # Assessment type breakdown
        for assessment in self.assessment_history:
            assessment_type = assessment.assessment_type
            if assessment_type not in stats['assessment_types']:
                stats['assessment_types'][assessment_type] = 0
            stats['assessment_types'][assessment_type] += 1
            
            stats['completion_times'].append(assessment.completion_time)
        
        # Average scores by dimension
        dimension_scores = {d.value: [] for d in TrustDimension}
        dimension_scores['overall'] = []
        
        for assessment in self.assessment_history:
            if assessment.overall_score > 0:
                dimension_scores['overall'].append(assessment.overall_score)
            
            for dimension, score in assessment.dimension_scores.items():
                if score > 0:
                    dimension_scores[dimension].append(score)
        
        for dimension, scores in dimension_scores.items():
            if scores:
                stats['average_scores'][dimension] = {
                    'mean': statistics.mean(scores),
                    'median': statistics.median(scores),
                    'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0,
                    'count': len(scores)
                }
        
        # Completion time statistics
        if stats['completion_times']:
            stats['avg_completion_time'] = statistics.mean(stats['completion_times'])
            stats['median_completion_time'] = statistics.median(stats['completion_times'])
        
        return stats