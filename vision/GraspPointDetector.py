"""
Grasp point detection for robotic manipulation.
Analyzes object geometry and depth to find optimal grasp points.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from scipy import ndimage
from skimage import morphology, measure

class GraspPointDetector:
    """
    Grasp point detector for robotic manipulation.
    
    This class analyzes object geometry and depth information to
    determine optimal grasp points and approach vectors for robotic gripping.
    
    Parameters
    ----------
    gripper_width : float, optional
        Maximum gripper opening width in meters, by default 0.085
    gripper_finger_width : float, optional
        Width of gripper fingers in meters, by default 0.02
    """
    
    def __init__(self, gripper_width: float = 0.085, gripper_finger_width: float = 0.02):
        self.logger = logging.getLogger(__name__)
        
        # Gripper parameters
        self.gripper_width = gripper_width
        self.gripper_finger_width = gripper_finger_width
        
        # Detection parameters
        self.min_grasp_width = 0.02  # Minimum object width for grasping (meters)
        self.max_grasp_width = 0.15  # Maximum object width for grasping (meters)
        self.min_depth_points = 100  # Minimum points needed for reliable depth
        
        self.logger.info("✅ Grasp point detector initialized")

    def _calculate_grasp_quality(self, contour: np.ndarray, angle: float) -> float:
        """
        Calculate grasp quality score.
        
        Parameters
        ----------
        contour : np.ndarray
            Object contour points
        angle : float
            Grasp approach angle in radians
            
        Returns
        -------
        float
            Quality score between 0 and 1
        """
        # Factors that contribute to grasp quality:
        # 1. Symmetry around grasp axis
        # 2. Distance from center of mass
        # 3. Contour curvature at grasp points
        
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return 0.0
            
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        symmetry_score = self._calculate_symmetry(contour, (cx, cy), angle)
        
        curvature_score = self._calculate_curvature(contour)
        
        # Combine scores (weights can be adjusted)
        total_score = 0.7 * symmetry_score + 0.3 * curvature_score
        
        return float(np.clip(total_score, 0.0, 1.0))

    def _calculate_symmetry(self, contour: np.ndarray, center: Tuple[int, int], 
                          angle: float) -> float:
        """
        Calculate symmetry score around grasp axis.
        
        Parameters
        ----------
        contour : np.ndarray
            Object contour points
        center : Tuple[int, int]
            Center point coordinates
        angle : float
            Grasp approach angle in radians
            
        Returns
        -------
        float
            Symmetry score between 0 and 1
        """
        # Project points onto grasp axis
        axis_vector = np.array([np.cos(angle), np.sin(angle)])
        points = contour.reshape(-1, 2)
        
        # Center the points
        centered_points = points - np.array(center)
        
        # Project onto axis
        projections = np.dot(centered_points, axis_vector)
        
        left_points = projections[projections < 0]
        right_points = projections[projections > 0]
        
        if len(left_points) == 0 or len(right_points) == 0:
            return 0.0
            
        # Compare mean distances
        left_mean = np.mean(np.abs(left_points))
        right_mean = np.mean(np.abs(right_points))
        
        symmetry = 1.0 - np.abs(left_mean - right_mean) / max(left_mean, right_mean)
        
        return float(symmetry)

    def _calculate_curvature(self, contour: np.ndarray) -> float:
        """
        Calculate contour curvature score.
        
        Parameters
        ----------
        contour : np.ndarray
            Object contour points
            
        Returns
        -------
        float
            Curvature score between 0 and 1
        """
        # Simplified curvature calculation
        # Higher score for more rectangular/regular shapes
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
            
        # Circularity/compactness measure
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        return float(circularity)

    def _create_object_mask(self, roi: np.ndarray) -> np.ndarray:
        """Create object mask using color segmentation and morphology"""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        
        # Combine multiple segmentation approaches
        # Method 1: HSV-based
        mask_hsv = cv2.inRange(hsv, (0, 30, 30), (180, 255, 255))
        
        # Method 2: Edge-based
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        mask_edge = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
        
        # Method 3: Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask_hsv, mask_edge)
        combined_mask = cv2.bitwise_or(combined_mask, adaptive)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8) * 255
        
        return mask
    
    def _find_width_based_grasps(self, mask: np.ndarray, roi: np.ndarray) -> List[Dict]:
        """Find grasp points by analyzing object width at different heights"""
        grasps = []
        h, w = mask.shape
        
        # Sample at different heights
        for y in range(h // 4, 3 * h // 4, max(1, h // 10)):
            row = mask[y, :]
            
            # Find object boundaries in this row
            white_pixels = np.where(row > 0)[0]
            if len(white_pixels) < 2:
                continue
                
            left_edge = white_pixels[0]
            right_edge = white_pixels[-1]
            object_width_pixels = right_edge - left_edge
            
            # Convert to real-world coordinates (approximate)
            # This would need proper depth information for accuracy
            object_width_meters = object_width_pixels * 0.001  # Rough estimate
            
            if self.min_grasp_width <= object_width_meters <= self.max_grasp_width:
                grasp_x = (left_edge + right_edge) // 2
                
                quality = self._calculate_width_grasp_quality(mask, grasp_x, y, 
                                                            left_edge, right_edge)
                
                grasps.append({
                    'grasp_point': (grasp_x, y),
                    'grasp_width': object_width_meters,
                    'approach_angle': 0,  # Horizontal approach
                    'quality': quality,
                    'type': 'width_grasp'
                })
        
        return grasps
    
    def _find_edge_based_grasps(self, mask: np.ndarray, roi: np.ndarray) -> List[Dict]:
        """Find grasp points along object edges (good for flat objects)"""
        grasps = []
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return grasps
            
        # Use largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Find convex hull and defects
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        
        if len(hull) > 3:
            defects = cv2.convexityDefects(largest_contour, hull)
            
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    
                    start = tuple(largest_contour[s][0])
                    end = tuple(largest_contour[e][0])
                    far = tuple(largest_contour[f][0])
                    
                    if d > 1000:  # Sufficient depth
                        grasp_point = ((start[0] + end[0]) // 2, 
                                     (start[1] + end[1]) // 2)
                        
                        angle = np.arctan2(end[1] - start[1], end[0] - start[0])
                        
                        quality = self._calculate_edge_grasp_quality(mask, grasp_point, angle)
                        
                        grasps.append({
                            'grasp_point': grasp_point,
                            'grasp_width': d / 1000.0,  # Convert to meters (rough)
                            'approach_angle': angle,
                            'quality': quality,
                            'type': 'edge_grasp'
                        })
        
        return grasps
    
    def _find_corner_grasps(self, mask: np.ndarray, roi: np.ndarray) -> List[Dict]:
        """Find corner-based grasp points (good for rectangular objects)"""
        grasps = []
        
        # Find corners using Harris corner detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=10, qualityLevel=0.01, 
                                        minDistance=20, mask=mask)
        
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel().astype(int)
                
                if self._is_graspable_corner(mask, x, y):
                    quality = self._calculate_corner_grasp_quality(mask, x, y)
                    
                    grasps.append({
                        'grasp_point': (x, y),
                        'grasp_width': 0.02,  # Assume small corner grasp
                        'approach_angle': self._get_corner_approach_angle(mask, x, y),
                        'quality': quality,
                        'type': 'corner_grasp'
                    })
        
        return grasps
    
    def _calculate_width_grasp_quality(self, mask: np.ndarray, x: int, y: int, 
                                     left: int, right: int) -> float:
        """Calculate quality score for width-based grasp"""
        h, w = mask.shape
        
        # Stability: prefer grasps closer to object center of mass
        moments = cv2.moments(mask)
        if moments['m00'] > 0:
            cm_x = int(moments['m10'] / moments['m00'])
            cm_y = int(moments['m01'] / moments['m00'])
            
            stability_score = 1.0 - min(1.0, abs(y - cm_y) / (h / 2))
        else:
            stability_score = 0.5
        
        # Symmetry: prefer symmetric grasps
        left_width = x - left
        right_width = right - x
        total_width = right - left
        symmetry_score = 1.0 - abs(left_width - right_width) / max(1, total_width)
        
        # Clearance: ensure enough space around grasp point
        clearance_score = self._check_grasp_clearance(mask, x, y)
        
        return 0.4 * stability_score + 0.3 * symmetry_score + 0.3 * clearance_score
    
    def _calculate_edge_grasp_quality(self, mask: np.ndarray, point: Tuple[int, int], 
                                    angle: float) -> float:
        """Calculate quality score for edge-based grasp"""
        x, y = point
        
        if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
            return 0.0
            
        if mask[y, x] == 0:
            return 0.0
        
        # Distance from edge
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        edge_distance = dist_transform[y, x] / np.max(dist_transform)
        
        # Angle suitability (prefer horizontal/vertical approaches)
        angle_score = abs(np.cos(2 * angle))  # Prefer 0, 90, 180, 270 degrees
        
        return 0.6 * edge_distance + 0.4 * angle_score
    
    def _calculate_corner_grasp_quality(self, mask: np.ndarray, x: int, y: int) -> float:
        """Calculate quality score for corner-based grasp"""
        # Corner strength (how well-defined the corner is)
        gray = mask.astype(np.float32)
        corner_response = cv2.cornerHarris(gray, 2, 3, 0.04)
        
        if corner_response[y, x] > 0:
            corner_strength = corner_response[y, x] / np.max(corner_response)
        else:
            corner_strength = 0.0
        
        # Accessibility (distance from object center)
        h, w = mask.shape
        center_distance = np.sqrt((x - w/2)**2 + (y - h/2)**2) / (np.sqrt(w**2 + h**2)/2)
        
        return 0.7 * corner_strength + 0.3 * center_distance
    
    def _check_grasp_clearance(self, mask: np.ndarray, x: int, y: int, 
                              clearance_radius: int = 10) -> float:
        """Check clearance around grasp point"""
        h, w = mask.shape
        
        # Define clearance region
        y1 = max(0, y - clearance_radius)
        y2 = min(h, y + clearance_radius)
        x1 = max(0, x - clearance_radius)
        x2 = min(w, x + clearance_radius)
        
        clearance_region = mask[y1:y2, x1:x2]
        
        clear_pixels = np.sum(clearance_region == 0)
        total_pixels = clearance_region.size
        
        return clear_pixels / max(1, total_pixels)
    
    def _is_graspable_corner(self, mask: np.ndarray, x: int, y: int) -> bool:
        """Check if a corner is suitable for grasping"""
        h, w = mask.shape
        
        # Must be within object
        if mask[y, x] == 0:
            return False
        
        # Must be near object boundary
        boundary_distance = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)[y, x]
        
        return boundary_distance < 5  # Within 5 pixels of boundary
    
    def _get_corner_approach_angle(self, mask: np.ndarray, x: int, y: int) -> float:
        """Calculate optimal approach angle for corner grasp"""
        # Simple approach: use gradient direction
        sobel_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_angle = np.arctan2(sobel_y[y, x], sobel_x[y, x])
        
        # Approach perpendicular to gradient
        return gradient_angle + np.pi/2
    
    def _select_best_grasp(self, candidates: List[Dict], mask: np.ndarray, workspace_validator: Optional[WorkspaceValidator] = None) -> Dict:
        """Select best grasp from candidates considering quality and workspace constraints"""
        best_grasp = None
        best_score = -1
        
        for grasp in candidates:
            # Base score from grasp quality
            score = grasp['quality']
            
            # Add workspace safety score if validator provided
            if workspace_validator is not None:
                x, y, z = grasp.get('grasp_point_3d', (0, 0, 0))
                if not workspace_validator.is_reachable(x, y, z, safe_mode=True):
                    continue
                safety_score = workspace_validator.get_safety_score(x, y, z)
                score = 0.7 * score + 0.3 * safety_score
            
            # Check gripper constraints
            if not self._validate_gripper_constraints(grasp, mask):
                continue
                
            if score > best_score:
                best_score = score
                best_grasp = grasp
        
        if best_grasp is None:
            return self._fallback_center_grasp(mask.shape)
            
        return best_grasp
    
    def _validate_gripper_constraints(self, grasp: Dict, mask: np.ndarray, depth_frame: Optional[np.ndarray] = None) -> bool:
        """Validate grasp against gripper physical constraints"""
        if not (self.min_grasp_width <= grasp['grasp_width'] <= self.max_grasp_width):
            return False
        
        # Check approach clearance
        x, y = grasp['grasp_point']
        angle = grasp['approach_angle']
        
        # If depth information available, check approach path
        if depth_frame is not None:
            # Sample points along approach vector
            approach_length = 50  # pixels
            num_samples = 10
            for i in range(num_samples):
                sample_x = int(x + (i/num_samples) * approach_length * np.cos(angle + np.pi))
                sample_y = int(y + (i/num_samples) * approach_length * np.sin(angle + np.pi))
                
                # Check if point is within frame
                h, w = depth_frame.shape
                if 0 <= sample_x < w and 0 <= sample_y < h:
                    if depth_frame[sample_y, sample_x] < grasp.get('depth', float('inf')):
                        return False
        
        return True
    
    def _fallback_center_grasp(self, shape) -> Dict:
        """Fallback to center grasp if no other options"""
        h, w = shape[:2] if len(shape) > 2 else shape
        center_x = w // 2
        center_y = h // 2
        
        return {
            'grasp_point': (center_x, center_y),
            'grasp_width': min(w, h) * 0.001,  # Rough estimate
            'approach_angle': -np.pi/2,  # Top-down approach
            'quality': 0.3,  # Low quality fallback
            'type': 'center_fallback'
        }
    
    def visualize_grasp(self, frame: np.ndarray, grasp_info: Dict) -> np.ndarray:
        """Visualize grasp point and approach on frame"""
        if not grasp_info:
            return frame
        
        x, y = grasp_info['grasp_point']
        angle = grasp_info['approach_angle']
        
        # Draw grasp point
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        
        # Draw approach vector
        approach_length = 30
        end_x = int(x + approach_length * np.cos(angle))
        end_y = int(y + approach_length * np.sin(angle))
        cv2.arrowedLine(frame, (int(x), int(y)), (end_x, end_y), (255, 0, 0), 2)
        
        # Draw gripper simulation
        gripper_width_pixels = 20  # Approximate
        perpendicular_angle = angle + np.pi/2
        
        # Gripper jaw 1
        jaw1_x1 = int(x + (gripper_width_pixels/2) * np.cos(perpendicular_angle))
        jaw1_y1 = int(y + (gripper_width_pixels/2) * np.sin(perpendicular_angle))
        jaw1_x2 = int(jaw1_x1 + 15 * np.cos(angle))
        jaw1_y2 = int(jaw1_y1 + 15 * np.sin(angle))
        cv2.line(frame, (jaw1_x1, jaw1_y1), (jaw1_x2, jaw1_y2), (0, 0, 255), 3)
        
        # Gripper jaw 2
        jaw2_x1 = int(x - (gripper_width_pixels/2) * np.cos(perpendicular_angle))
        jaw2_y1 = int(y - (gripper_width_pixels/2) * np.sin(perpendicular_angle))
        jaw2_x2 = int(jaw2_x1 + 15 * np.cos(angle))
        jaw2_y2 = int(jaw2_y1 + 15 * np.sin(angle))
        cv2.line(frame, (jaw2_x1, jaw2_y1), (jaw2_x2, jaw2_y2), (0, 0, 255), 3)
        
        info_text = f"{grasp_info['type']}: Q={grasp_info['quality']:.2f}"
        cv2.putText(frame, info_text, (int(x-50), int(y-20)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    def detect_grasp_point(self, roi: np.ndarray) -> Dict:
        """
        Detect optimal grasp point from ROI.
        
        Parameters
        ----------
        roi : np.ndarray
            Region of interest containing object
            
        Returns
        -------
        Dict
            Grasp information including:
            - x, y: Normalized coordinates (0-1) within ROI
            - approach_angle: Approach angle in radians
            - width: Estimated grasp width
            - quality: Grasp quality score
            
        Notes
        -----
        Approach angles are defined relative to camera frame:
        - For eye-in-hand (camera looking down): 0 = approach from positive X
        - Angles increase counter-clockwise when viewed from above
        """
        try:
            mask = self._create_object_mask(roi)
            
            # Find different types of grasp candidates
            width_grasps = self._find_width_based_grasps(mask, roi)
            edge_grasps = self._find_edge_based_grasps(mask, roi)
            corner_grasps = self._find_corner_grasps(mask, roi)
            
            # Combine all candidates
            all_grasps = width_grasps + edge_grasps + corner_grasps
            
            # Select best grasp
            best_grasp = self._select_best_grasp(all_grasps, mask)
            
            # Convert to normalized coordinates
            h, w = roi.shape[:2]
            normalized_grasp = {
                'x': best_grasp['grasp_point'][0] / w,
                'y': best_grasp['grasp_point'][1] / h,
                'approach_angle': best_grasp['approach_angle'],
                'width': best_grasp['grasp_width'],
                'quality': best_grasp['quality']
            }
            
            return normalized_grasp
            
        except Exception as e:
            self.logger.error(f"Error detecting grasp point: {e}")
            # Fallback to center grasp
            return {
                'x': 0.5,
                'y': 0.5,
                'approach_angle': 0.0,  # Default horizontal approach
                'width': 0.04,
                'quality': 0.1
            }