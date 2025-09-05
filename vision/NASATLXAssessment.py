#!/usr/bin/env python3
"""
NASA Task Load Index (TLX) Assessment for Construction HRI Research.

This module implements the NASA-TLX cognitive workload assessment adapted 
for construction robotics scenarios. Measures subjective workload across
six dimensions to evaluate the cognitive impact of different clarification
strategies and expertise inversion scenarios.

Based on Hart & Staveland (1988) NASA Task Load Index with construction-specific
adaptations for human-robot interaction research.

Provides:
- Six-dimensional workload measurement (Mental, Physical, Temporal, Performance, Effort, Frustration)
- Construction task-specific adaptations
- Paired comparison weighting procedure
- Statistical analysis and comparison
- Integration with experimental controller
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import itertools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TLXDimension(Enum):
    """NASA-TLX workload dimensions"""
    MENTAL_DEMAND = "mental_demand"
    PHYSICAL_DEMAND = "physical_demand"
    TEMPORAL_DEMAND = "temporal_demand"
    PERFORMANCE = "performance"
    EFFORT = "effort"
    FRUSTRATION = "frustration"

@dataclass
class TLXQuestion:
    """NASA-TLX dimension question"""
    dimension: TLXDimension
    question: str
    description: str
    low_anchor: str
    high_anchor: str
    construction_adaptation: str = ""

@dataclass
class TLXResponse:
    """Response to TLX dimension rating"""
    dimension: TLXDimension
    raw_score: int  # 0-100 scale
    weighted_score: float = 0.0
    response_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class PairwiseComparison:
    """Pairwise comparison for dimension weighting"""
    dimension1: TLXDimension
    dimension2: TLXDimension
    selected_dimension: TLXDimension
    response_time: float = 0.0

@dataclass
class TLXAssessment:
    """Complete NASA-TLX assessment"""
    participant_id: str
    session_id: str
    task_description: str
    responses: List[TLXResponse]
    pairwise_comparisons: List[PairwiseComparison] = field(default_factory=list)
    dimension_weights: Dict[TLXDimension, float] = field(default_factory=dict)
    overall_workload: float = 0.0
    completion_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ConstructionNASATLX:
    """
    NASA-TLX assessment system adapted for construction HRI research.
    
    Implements the standard NASA-TLX procedure with construction-specific
    task contexts and clarification strategy evaluation focus.
    
    Parameters
    ----------
    use_pairwise_weighting : bool, optional
        Use pairwise comparison weighting, by default True
    construction_context : bool, optional
        Use construction-specific question adaptations, by default True
    randomize_questions : bool, optional
        Randomize question order, by default True
        
    Attributes
    ----------
    dimensions : List[TLXQuestion]
        Six NASA-TLX dimension questions
    pairwise_combinations : List[Tuple]
        All possible dimension pairs for weighting
    assessment_history : List[TLXAssessment]
        History of completed assessments
    """
    
    def __init__(self,
                 use_pairwise_weighting: bool = True,
                 construction_context: bool = True,
                 randomize_questions: bool = True):
        
        self.logger = logging.getLogger(__name__)
        self.use_pairwise_weighting = use_pairwise_weighting
        self.construction_context = construction_context
        self.randomize_questions = randomize_questions
        
        # Initialize TLX dimensions
        self.dimensions = self._create_tlx_dimensions()
        
        # Generate all pairwise combinations for weighting
        self.pairwise_combinations = list(itertools.combinations(TLXDimension, 2))
        
        self.assessment_history = []
        
        self.logger.info("✅ NASA-TLX Assessment initialized for construction HRI")
        self.logger.info(f"   Pairwise weighting: {use_pairwise_weighting}")
        self.logger.info(f"   Construction context: {construction_context}")
        self.logger.info(f"   Dimensions: {len(self.dimensions)}")

    def _create_tlx_dimensions(self) -> List[TLXQuestion]:
        """Create NASA-TLX dimension questions with construction adaptations"""
        
        dimensions = [
            TLXQuestion(
                dimension=TLXDimension.MENTAL_DEMAND,
                question="How mentally demanding was the task?",
                description="How much mental and perceptual activity was required?",
                low_anchor="Very Low - Simple, easy thinking required",
                high_anchor="Very High - Complex, difficult thinking required",
                construction_adaptation=(
                    "Consider the mental effort needed to process robot clarifications, "
                    "make tool identification decisions, and coordinate with the robot assistant."
                )
            ),
            
            TLXQuestion(
                dimension=TLXDimension.PHYSICAL_DEMAND,
                question="How physically demanding was the task?",
                description="How much physical activity was required?",
                low_anchor="Very Low - Slow, slack, restful pace",
                high_anchor="Very High - Fast, strenuous, vigorous pace",
                construction_adaptation=(
                    "Consider physical effort for tool handling, positioning yourself "
                    "for robot interaction, and overall construction task execution."
                )
            ),
            
            TLXQuestion(
                dimension=TLXDimension.TEMPORAL_DEMAND,
                question="How hurried or rushed was the pace of the task?",
                description="How much time pressure did you feel?",
                low_anchor="Very Low - Leisurely, plenty of time",
                high_anchor="Very High - Rushed, insufficient time", 
                construction_adaptation=(
                    "Consider time pressure from robot clarification delays, "
                    "communication overhead, and construction deadline pressure."
                )
            ),
            
            TLXQuestion(
                dimension=TLXDimension.PERFORMANCE,
                question="How successful were you in accomplishing the task?",
                description="How satisfied were you with your performance?",
                low_anchor="Perfect - Completely satisfied with performance",
                high_anchor="Failure - Completely dissatisfied with performance",
                construction_adaptation=(
                    "Consider success in getting correct tools, effective robot coordination, "
                    "and overall construction task completion quality."
                )
            ),
            
            TLXQuestion(
                dimension=TLXDimension.EFFORT,
                question="How hard did you have to work to accomplish your level of performance?",
                description="How much effort was required (mental and physical)?",
                low_anchor="Very Low - Minimal effort required",
                high_anchor="Very High - Maximum effort required",
                construction_adaptation=(
                    "Consider effort needed to understand robot responses, communicate "
                    "effectively, resolve clarification dialogues, and maintain work flow."
                )
            ),
            
            TLXQuestion(
                dimension=TLXDimension.FRUSTRATION,
                question="How insecure, discouraged, irritated, stressed, or annoyed were you?",
                description="How much frustration did you experience?",
                low_anchor="Very Low - Relaxed, content, satisfied",
                high_anchor="Very High - Stressed, annoyed, frustrated",
                construction_adaptation=(
                    "Consider frustration from robot misunderstandings, clarification delays, "
                    "expertise conflicts, and disruption of normal construction workflow."
                )
            )
        ]
        
        return dimensions

    def conduct_assessment(self,
                          participant_id: str,
                          session_id: str,
                          task_description: str,
                          clarification_strategy: Optional[str] = None,
                          user_expertise: Optional[str] = None) -> TLXAssessment:
        """
        Conduct NASA-TLX workload assessment.
        
        Parameters
        ----------
        participant_id : str
            Unique participant identifier
        session_id : str
            Experimental session identifier
        task_description : str
            Description of the construction task performed
        clarification_strategy : str, optional
            Which clarification strategy was used
        user_expertise : str, optional
            User expertise level
            
        Returns
        -------
        TLXAssessment
            Complete TLX assessment with workload scores
        """
        
        start_time = time.time()
        responses = []
        pairwise_comparisons = []
        
        self.logger.info(f"📋 Starting NASA-TLX assessment for {participant_id}")
        self.logger.info(f"   Task: {task_description}")
        if clarification_strategy:
            self.logger.info(f"   Strategy: {clarification_strategy}")
        
        # Step 1: Collect dimension ratings (0-100 scale)
        dimensions_to_rate = self.dimensions.copy()
        if self.randomize_questions:
            import random
            random.shuffle(dimensions_to_rate)
        
        for dimension_q in dimensions_to_rate:
            # In real implementation, this would present the question to the user
            response = self._simulate_dimension_rating(dimension_q, clarification_strategy)
            responses.append(response)
        
        # Step 2: Collect pairwise comparisons for weighting (if enabled)
        dimension_weights = {}
        if self.use_pairwise_weighting:
            comparisons = self.pairwise_combinations.copy()
            if self.randomize_questions:
                import random
                random.shuffle(comparisons)
            
            for dim1, dim2 in comparisons:
                comparison = self._simulate_pairwise_comparison(dim1, dim2, clarification_strategy)
                pairwise_comparisons.append(comparison)
            
            # Calculate weights from pairwise comparisons
            dimension_weights = self._calculate_weights(pairwise_comparisons)
        else:
            # Equal weighting
            dimension_weights = {dim: 1.0 for dim in TLXDimension}
        
        # Step 3: Calculate weighted scores
        total_weight = sum(dimension_weights.values())
        for response in responses:
            weight = dimension_weights[response.dimension]
            normalized_weight = weight / total_weight
            response.weighted_score = response.raw_score * normalized_weight
        
        completion_time = time.time() - start_time
        
        # Create assessment
        assessment = TLXAssessment(
            participant_id=participant_id,
            session_id=session_id,
            task_description=task_description,
            responses=responses,
            pairwise_comparisons=pairwise_comparisons,
            dimension_weights=dimension_weights,
            overall_workload=sum(r.weighted_score for r in responses),
            completion_time=completion_time,
            metadata={
                'clarification_strategy': clarification_strategy,
                'user_expertise': user_expertise,
                'use_pairwise_weighting': self.use_pairwise_weighting,
                'construction_context': self.construction_context
            }
        )
        
        self.assessment_history.append(assessment)
        
        self.logger.info(f"✅ TLX assessment completed in {completion_time:.1f}s")
        self.logger.info(f"   Overall workload: {assessment.overall_workload:.1f}")
        self.logger.info(f"   Highest demand: {self._get_highest_dimension(assessment)}")
        
        return assessment

    def _simulate_dimension_rating(self, dimension_q: TLXQuestion, 
                                 clarification_strategy: Optional[str] = None) -> TLXResponse:
        """Simulate user rating for a TLX dimension"""
        
        import random
        
        # Simulate realistic ratings based on dimension and strategy
        base_ratings = {
            TLXDimension.MENTAL_DEMAND: 45,
            TLXDimension.PHYSICAL_DEMAND: 25,  
            TLXDimension.TEMPORAL_DEMAND: 35,
            TLXDimension.PERFORMANCE: 25,  # Lower = better performance
            TLXDimension.EFFORT: 40,
            TLXDimension.FRUSTRATION: 30
        }
        
        base_rating = base_ratings.get(dimension_q.dimension, 40)
        
        # Adjust based on clarification strategy
        if clarification_strategy:
            if clarification_strategy == "direct":
                # Direct strategy: lower mental demand, higher temporal pressure
                if dimension_q.dimension == TLXDimension.MENTAL_DEMAND:
                    base_rating -= 10
                elif dimension_q.dimension == TLXDimension.TEMPORAL_DEMAND:
                    base_rating += 5
            elif clarification_strategy == "confidence_based":
                # Confidence-based: higher mental demand, lower frustration
                if dimension_q.dimension == TLXDimension.MENTAL_DEMAND:
                    base_rating += 15
                elif dimension_q.dimension == TLXDimension.FRUSTRATION:
                    base_rating -= 10
            elif clarification_strategy == "expertise_adaptive":
                # Expertise adaptive: varies by user level, generally lower frustration
                if dimension_q.dimension == TLXDimension.FRUSTRATION:
                    base_rating -= 15
                elif dimension_q.dimension == TLXDimension.EFFORT:
                    base_rating -= 5
        
        # Add random variation
        rating = max(0, min(100, base_rating + random.randint(-15, 15)))
        
        return TLXResponse(
            dimension=dimension_q.dimension,
            raw_score=rating,
            response_time=random.uniform(3.0, 12.0)
        )

    def _simulate_pairwise_comparison(self, dim1: TLXDimension, dim2: TLXDimension,
                                    clarification_strategy: Optional[str] = None) -> PairwiseComparison:
        """Simulate pairwise comparison for dimension weighting"""
        
        import random
        
        # Simulate preference based on construction context and strategy
        preference_weights = {
            TLXDimension.MENTAL_DEMAND: 0.3,
            TLXDimension.PHYSICAL_DEMAND: 0.1,
            TLXDimension.TEMPORAL_DEMAND: 0.25,
            TLXDimension.PERFORMANCE: 0.2,
            TLXDimension.EFFORT: 0.15,
            TLXDimension.FRUSTRATION: 0.35  # High importance in HRI
        }
        
        # Adjust weights based on strategy
        if clarification_strategy == "confidence_based":
            preference_weights[TLXDimension.MENTAL_DEMAND] += 0.1
        elif clarification_strategy == "direct":
            preference_weights[TLXDimension.TEMPORAL_DEMAND] += 0.1
        elif clarification_strategy == "expertise_adaptive":
            preference_weights[TLXDimension.FRUSTRATION] += 0.1
        
        weight1 = preference_weights.get(dim1, 0.2)
        weight2 = preference_weights.get(dim2, 0.2)
        
        # Choose dimension based on weights (with some randomness)
        if random.random() < (weight1 / (weight1 + weight2)):
            selected = dim1
        else:
            selected = dim2
        
        return PairwiseComparison(
            dimension1=dim1,
            dimension2=dim2,
            selected_dimension=selected,
            response_time=random.uniform(2.0, 8.0)
        )

    def _calculate_weights(self, comparisons: List[PairwiseComparison]) -> Dict[TLXDimension, float]:
        """Calculate dimension weights from pairwise comparisons"""
        
        # Count how many times each dimension was selected
        selection_counts = {dim: 0 for dim in TLXDimension}
        
        for comparison in comparisons:
            selection_counts[comparison.selected_dimension] += 1
        
        # Convert to weights (number of times selected out of 5 possible)
        weights = {}
        for dimension, count in selection_counts.items():
            weights[dimension] = count / 5.0  # Each dimension appears in 5 comparisons
        
        return weights

    def _get_highest_dimension(self, assessment: TLXAssessment) -> str:
        """Get the dimension with highest workload"""
        
        highest_score = 0
        highest_dimension = None
        
        for response in assessment.responses:
            if response.raw_score > highest_score:
                highest_score = response.raw_score
                highest_dimension = response.dimension
        
        return highest_dimension.value if highest_dimension else "unknown"

    def compare_workload(self, assessment1: TLXAssessment, assessment2: TLXAssessment) -> Dict[str, Any]:
        """
        Compare workload between two assessments.
        
        Parameters
        ----------
        assessment1 : TLXAssessment
            First assessment (e.g., baseline condition)
        assessment2 : TLXAssessment
            Second assessment (e.g., treatment condition)
            
        Returns
        -------
        Dict[str, Any]
            Workload comparison results
        """
        
        comparison = {
            'overall_difference': assessment2.overall_workload - assessment1.overall_workload,
            'dimension_differences': {},
            'percentage_change': ((assessment2.overall_workload - assessment1.overall_workload) / 
                                assessment1.overall_workload * 100) if assessment1.overall_workload > 0 else 0,
            'assessment1_strategy': assessment1.metadata.get('clarification_strategy', 'unknown'),
            'assessment2_strategy': assessment2.metadata.get('clarification_strategy', 'unknown')
        }
        
        # Compare each dimension
        responses1 = {r.dimension: r for r in assessment1.responses}
        responses2 = {r.dimension: r for r in assessment2.responses}
        
        for dimension in TLXDimension:
            if dimension in responses1 and dimension in responses2:
                diff = responses2[dimension].raw_score - responses1[dimension].raw_score
                comparison['dimension_differences'][dimension.value] = diff
        
        return comparison

    def get_strategy_workload_summary(self, clarification_strategy: str) -> Dict[str, Any]:
        """Get workload summary for a specific clarification strategy"""
        
        strategy_assessments = [
            a for a in self.assessment_history 
            if a.metadata.get('clarification_strategy') == clarification_strategy
        ]
        
        if not strategy_assessments:
            return {'message': f'No assessments found for strategy: {clarification_strategy}'}
        
        # Calculate statistics
        overall_scores = [a.overall_workload for a in strategy_assessments]
        dimension_scores = {dim.value: [] for dim in TLXDimension}
        
        for assessment in strategy_assessments:
            for response in assessment.responses:
                dimension_scores[response.dimension.value].append(response.raw_score)
        
        summary = {
            'strategy': clarification_strategy,
            'assessment_count': len(strategy_assessments),
            'overall_workload': {
                'mean': statistics.mean(overall_scores),
                'median': statistics.median(overall_scores),
                'std_dev': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
                'min': min(overall_scores),
                'max': max(overall_scores)
            },
            'dimension_workload': {}
        }
        
        for dimension, scores in dimension_scores.items():
            if scores:
                summary['dimension_workload'][dimension] = {
                    'mean': statistics.mean(scores),
                    'median': statistics.median(scores),
                    'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0
                }
        
        return summary

    def get_assessment_summary(self, assessment: TLXAssessment) -> Dict[str, Any]:
        """Generate summary for a single assessment"""
        
        dimension_scores = {r.dimension.value: r.raw_score for r in assessment.responses}
        weighted_scores = {r.dimension.value: r.weighted_score for r in assessment.responses}
        
        return {
            'participant_id': assessment.participant_id,
            'session_id': assessment.session_id,
            'task_description': assessment.task_description,
            'overall_workload': assessment.overall_workload,
            'dimension_scores': dimension_scores,
            'weighted_scores': weighted_scores,
            'dimension_weights': {d.value: w for d, w in assessment.dimension_weights.items()},
            'completion_time': assessment.completion_time,
            'clarification_strategy': assessment.metadata.get('clarification_strategy'),
            'user_expertise': assessment.metadata.get('user_expertise'),
            'highest_workload_dimension': self._get_highest_dimension(assessment),
            'timestamp': assessment.timestamp
        }

    def export_assessment_data(self, filepath: str, participant_ids: Optional[List[str]] = None):
        """Export TLX assessment data for statistical analysis"""
        
        # Filter assessments
        assessments_to_export = self.assessment_history
        if participant_ids:
            assessments_to_export = [
                a for a in self.assessment_history 
                if a.participant_id in participant_ids
            ]
        
        # Prepare export data
        export_data = {
            'tlx_info': {
                'dimensions': [d.value for d in TLXDimension],
                'use_pairwise_weighting': self.use_pairwise_weighting,
                'construction_context': self.construction_context,
                'export_timestamp': time.time()
            },
            'assessments': [self.get_assessment_summary(a) for a in assessments_to_export],
            'detailed_responses': [],
            'pairwise_comparisons': []
        }
        
        # Add detailed response data
        for assessment in assessments_to_export:
            for response in assessment.responses:
                export_data['detailed_responses'].append({
                    'participant_id': assessment.participant_id,
                    'session_id': assessment.session_id,
                    'dimension': response.dimension.value,
                    'raw_score': response.raw_score,
                    'weighted_score': response.weighted_score,
                    'response_time': response.response_time,
                    'timestamp': response.timestamp
                })
            
            # Add pairwise comparison data
            for comparison in assessment.pairwise_comparisons:
                export_data['pairwise_comparisons'].append({
                    'participant_id': assessment.participant_id,
                    'session_id': assessment.session_id,
                    'dimension1': comparison.dimension1.value,
                    'dimension2': comparison.dimension2.value,
                    'selected': comparison.selected_dimension.value,
                    'response_time': comparison.response_time
                })
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"📊 NASA-TLX data exported to {filepath}")
        self.logger.info(f"   Assessments: {len(assessments_to_export)}")
        self.logger.info(f"   Responses: {len(export_data['detailed_responses'])}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall NASA-TLX statistics"""
        
        if not self.assessment_history:
            return {"message": "No assessments completed yet"}
        
        # Overall statistics
        overall_scores = [a.overall_workload for a in self.assessment_history]
        strategies = [a.metadata.get('clarification_strategy') for a in self.assessment_history]
        
        stats = {
            'total_assessments': len(self.assessment_history),
            'unique_participants': len(set(a.participant_id for a in self.assessment_history)),
            'overall_workload': {
                'mean': statistics.mean(overall_scores),
                'median': statistics.median(overall_scores),
                'std_dev': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
                'range': [min(overall_scores), max(overall_scores)]
            },
            'strategies_tested': len(set(filter(None, strategies))),
            'dimension_averages': {}
        }
        
        # Dimension-level statistics
        for dimension in TLXDimension:
            dimension_scores = []
            for assessment in self.assessment_history:
                for response in assessment.responses:
                    if response.dimension == dimension:
                        dimension_scores.append(response.raw_score)
            
            if dimension_scores:
                stats['dimension_averages'][dimension.value] = {
                    'mean': statistics.mean(dimension_scores),
                    'std_dev': statistics.stdev(dimension_scores) if len(dimension_scores) > 1 else 0
                }
        
        return stats