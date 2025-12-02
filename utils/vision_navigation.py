"""
CARLA Vision Navigation System using Qwen2.5-VL

A production-ready class for vision-language navigation in CARLA autonomous driving.
Uses Qwen2.5-VL for visual grounding to detect navigation targets with pixel coordinates.

Author: CARLA Navigation Team
Date: 2025
"""

import torch
import numpy as np
from PIL import Image
import cv2
import re
import json
from typing import Dict, Tuple, Optional, Union
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class VisionNavigationSystem:
    """
    Vision-Language Navigation System for CARLA.
    
    Uses Qwen2.5-VL for visual grounding to detect navigation targets.
    Provides pixel coordinates for steering autonomous vehicles.
    
    Example:
        >>> navigator = VisionNavigationSystem(device='cuda')
        >>> image = cv2.imread('carla_screenshot.jpg')
        >>> image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        >>> result = navigator.navigate(image, "Navigate to the traffic light")
        >>> print(f"Target: ({result['x']}, {result['y']})")
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "auto",
        torch_dtype: str = "auto"
    ):
        """
        Initialize the Vision Navigation System.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cuda', 'cpu', or 'auto')
            torch_dtype: Torch dtype ('auto', 'float16', 'bfloat16')
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🤖 Loading {model_name}...")
        print(f"   Device: {self.device}")
        
        # Load model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        print("✅ Model loaded successfully!")
    
    def navigate(
        self,
        image: Union[np.ndarray, Image.Image, str],
        prompt: str,
        debug: bool = False
    ) -> Dict[str, any]:
        """
        Process navigation request and return target coordinates.
        
        Args:
            image: Input image (numpy array H,W,3 or PIL Image or file path)
            prompt: Navigation instruction (e.g., "Navigate to the traffic light")
            debug: Print debug information
        
        Returns:
            dict with keys:
                - x: Target x coordinate
                - y: Target y coordinate
                - confidence: Confidence score (0-1)
                - reasoning: Brief explanation
                - raw_response: Full model response
        """
        # Load image if it's a file path
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image.astype('uint8')).convert('RGB')
        else:
            image_pil = image
        
        width, height = image_pil.size
        
        # Extract target from prompt
        target = self._extract_target(prompt)
        
        # Create grounding prompt
        grounding_prompt = f"Find the {target} and return its location with bounding box coordinates."
        
        if debug:
            print(f"\n🔍 DEBUG:")
            print(f"  Target: {target}")
            print(f"  Grounding prompt: {grounding_prompt}")
            print(f"  Image size: {width}x{height}")
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": grounding_prompt}
                ]
            }
        ]
        
        # Process with model
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        if debug:
            print(f"\n📝 Raw response:")
            print(f"  {response[:150]}...")
        
        # Parse coordinates
        result = self._parse_coordinates(response, width, height, debug)
        
        return result
    
    def _extract_target(self, prompt: str) -> str:
        """Extract target object from navigation prompt."""
        target = prompt.lower()
        for phrase in ['navigate to the', 'go to the', 'head to the', 'navigate to']:
            target = target.replace(phrase, '')
        return target.strip()
    
    def _parse_coordinates(
        self,
        response: str,
        width: int,
        height: int,
        debug: bool = False
    ) -> Dict[str, any]:
        """
        Parse coordinates from model response using multiple strategies.
        
        Args:
            response: Model's text response
            width: Image width
            height: Image height
            debug: Print debug info
        
        Returns:
            dict with x, y, confidence, reasoning
        """
        x, y, confidence = None, None, 0.5
        
        # Strategy 1: Look for bounding box [x1, y1, x2, y2]
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        bbox_match = re.search(bbox_pattern, response)
        
        if bbox_match:
            x1, y1, x2, y2 = map(int, bbox_match.groups())
            x = (x1 + x2) // 2
            y = (y1 + y2) // 2
            confidence = 0.85
            if debug:
                print(f"✅ Found bbox: [{x1}, {y1}, {x2}, {y2}]")
                print(f"✅ Center: ({x}, {y})")
        
        # Strategy 2: Look for coordinate pairs
        if x is None:
            coord_pattern = r'(\d+),\s*(\d+)'
            matches = re.findall(coord_pattern, response)
            if matches:
                x, y = map(int, matches[0])
                confidence = 0.7
                if debug:
                    print(f"✅ Found coordinates: ({x}, {y})")
        
        # Strategy 3: Look for any numbers
        if x is None:
            numbers = re.findall(r'\b(\d+)\b', response)
            if len(numbers) >= 2:
                x, y = int(numbers[0]), int(numbers[1])
                confidence = 0.5
                if debug:
                    print(f"⚠️  Using first two numbers: ({x}, {y})")
        
        # Fallback: center of image
        if x is None:
            x, y = width // 2, height // 2
            confidence = 0.0
            if debug:
                print(f"❌ Fallback to center: ({x}, {y})")
        
        # Clip to bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        
        return {
            'x': x,
            'y': y,
            'confidence': confidence,
            'reasoning': response[:150] if len(response) > 150 else response,
            'raw_response': response
        }
    
    def visualize_result(
        self,
        image: Union[np.ndarray, str],
        result: Dict[str, any],
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Visualize navigation result with crosshair at target.
        
        Args:
            image: Input image (numpy array or file path)
            result: Result dict from navigate()
            show_confidence: Show confidence in visualization
        
        Returns:
            Visualization image with crosshair and info
        """
        # Load image if it's a file path
        if isinstance(image, str):
            vis = cv2.imread(image)
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        else:
            vis = image.copy()
        
        x, y = result['x'], result['y']
        conf = result['confidence']
        
        # Color based on confidence
        if conf > 0.7:
            color = (0, 255, 0)  # Green - high confidence
        elif conf > 0.4:
            color = (255, 165, 0)  # Orange - medium confidence
        else:
            color = (255, 0, 0)  # Red - low/fallback
        
        # Draw crosshair
        cv2.line(vis, (x-30, y), (x+30, y), color, 3)
        cv2.line(vis, (x, y-30), (x, y+30), color, 3)
        cv2.circle(vis, (x, y), 15, color, 3)
        
        # Draw info text
        if show_confidence:
            text = f"Target: ({x}, {y}) | Conf: {conf:.2f}"
            cv2.putText(vis, text, (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(vis, text, (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        return vis


class RoadAwareNavigator(VisionNavigationSystem):
    """
    Enhanced navigation system with road surface detection.
    
    Ensures all waypoints are on drivable road surfaces for safe navigation.
    
    Example:
        >>> navigator = RoadAwareNavigator(device='cuda')
        >>> result = navigator.navigate('carla_image.jpg', "Navigate to traffic light", 
        ...                              road_aware=True)
        >>> print(f"Safe waypoint on road: ({result['x']}, {result['y']})")
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("✅ Road-aware navigation enabled")
    
    def navigate(
        self,
        image: Union[np.ndarray, Image.Image, str],
        prompt: str,
        road_aware: bool = True,
        road_method: str = 'region',
        debug: bool = False
    ) -> Dict[str, any]:
        """
        Navigate with optional road awareness.
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
            prompt: Navigation instruction
            road_aware: Enforce waypoint on road surface
            road_method: 'region' or 'color' for road detection
            debug: Print debug info
        
        Returns:
            dict with x, y, confidence, on_road status
        """
        # Get basic navigation result
        result = super().navigate(image, prompt, debug=debug)
        
        if not road_aware:
            return result
        
        # Load and convert image if needed
        if isinstance(image, str):
            image_np = cv2.imread(image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Detect road
        road_mask = self.detect_road(image_np, method=road_method)
        
        # Check if target is on road
        target_x, target_y = result['x'], result['y']
        on_road = road_mask[target_y, target_x]
        
        if debug:
            print(f"\n🛣️  Road Detection:")
            print(f"  Method: {road_method}")
            print(f"  Road pixels: {np.sum(road_mask):,}")
            print(f"  Target on road: {on_road}")
        
        # If not on road, find closest road point
        if not on_road:
            safe_x, safe_y = self.find_closest_road_point(
                road_mask, target_x, target_y
            )
            
            if debug:
                offset = np.sqrt((safe_x - target_x)**2 + (safe_y - target_y)**2)
                print(f"  ⚠️  Target off-road!")
                print(f"  🚗 Closest road point: ({safe_x}, {safe_y})")
                print(f"  📏 Offset: {offset:.0f}px")
            
            result['x'] = safe_x
            result['y'] = safe_y
            result['reasoning'] += ' (adjusted to road)'
        
        result['on_road'] = True
        result['road_mask'] = road_mask
        
        return result
    
    def detect_road(
        self,
        image: np.ndarray,
        method: str = 'region'
    ) -> np.ndarray:
        """
        Detect road surface in image.
        
        Args:
            image: Input image (H, W, 3)
            method: 'region' (simple) or 'color' (accurate)
        
        Returns:
            Binary mask where True = road
        """
        height, width = image.shape[:2]
        
        if method == 'region':
            # Simple: assume bottom 60% is road
            road_mask = np.zeros((height, width), dtype=bool)
            road_mask[int(height * 0.4):, :] = True
            return road_mask
        
        elif method == 'color':
            # Color-based detection for gray roads
            lower_gray = np.array([50, 50, 50])
            upper_gray = np.array([90, 90, 90])
            
            road_mask = cv2.inRange(image, lower_gray, upper_gray)
            
            # Clear top portion (no roads in sky!)
            road_mask[:int(height * 0.35), :] = 0
            
            # Clean up noise
            kernel = np.ones((7, 7), np.uint8)
            road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
            road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
            
            return road_mask.astype(bool)
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'region' or 'color'")
    
    def find_closest_road_point(
        self,
        road_mask: np.ndarray,
        target_x: int,
        target_y: int
    ) -> Tuple[int, int]:
        """
        Find closest point on road to target.
        
        Args:
            road_mask: Boolean road mask
            target_x, target_y: Target coordinates
        
        Returns:
            (x, y) of closest road point
        """
        # Get all road pixels
        road_points = np.argwhere(road_mask)  # Returns (y, x) pairs
        
        if len(road_points) == 0:
            # Fallback: bottom center
            height, width = road_mask.shape
            return (width // 2, int(height * 0.8))
        
        # Calculate distances
        distances = np.sqrt(
            (road_points[:, 1] - target_x)**2 +
            (road_points[:, 0] - target_y)**2
        )
        
        # Find closest
        closest_idx = np.argmin(distances)
        closest_point = road_points[closest_idx]
        
        return (int(closest_point[1]), int(closest_point[0]))
    
    def visualize_result(
        self,
        image: Union[np.ndarray, str],
        result: Dict[str, any],
        show_road_overlay: bool = True
    ) -> np.ndarray:
        """
        Visualize with road overlay.
        
        Args:
            image: Input image (numpy array or file path)
            result: Navigation result
            show_road_overlay: Show green road overlay
        
        Returns:
            Visualization with road and waypoint
        """
        # Load image if it's a file path
        if isinstance(image, str):
            vis = cv2.imread(image)
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        else:
            vis = image.copy()
        
        # Draw road overlay
        if show_road_overlay and 'road_mask' in result:
            road_overlay = np.zeros_like(vis)
            road_overlay[result['road_mask']] = [0, 200, 0]  # Green
            vis = cv2.addWeighted(vis, 0.7, road_overlay, 0.3, 0)
        
        # Draw target and waypoint
        vis = super().visualize_result(vis, result, show_confidence=True)
        
        # Add "ON ROAD" indicator
        if result.get('on_road', False):
            cv2.putText(vis, "ON ROAD", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return vis


# Example usage
if __name__ == "__main__":
    import sys
    
    print("🚗 CARLA Vision Navigation System")
    print("=" * 60)
    
    # Check arguments
    if len(sys.argv) < 3:
        print("\nUsage: python vision_navigation.py <image_path> <instruction>")
        print("\nExample:")
        print("  python vision_navigation.py carla_screenshot.jpg \"Navigate to the traffic light\"")
        print("  python vision_navigation.py scene.png \"Go to the building\"")
        sys.exit(1)
    
    image_path = sys.argv[1]
    instruction = sys.argv[2]
    
    print(f"\n📸 Image: {image_path}")
    print(f"🎯 Instruction: {instruction}")
    
    # Initialize navigator
    print("\n🤖 Initializing navigation system...")
    navigator = RoadAwareNavigator(device='auto')
    
    # Navigate
    print("\n🚀 Processing navigation request...")
    result = navigator.navigate(
        image_path,
        instruction,
        road_aware=True,
        debug=True
    )
    
    # Show results
    print("\n" + "=" * 60)
    print("📊 RESULTS:")
    print("=" * 60)
    print(f"✅ Waypoint: ({result['x']}, {result['y']})")
    print(f"✅ Confidence: {result['confidence']:.2f}")
    print(f"✅ On road: {result['on_road']}")
    print(f"✅ Reasoning: {result['reasoning'][:80]}...")
    
    # Save visualization
    print("\n📊 Creating visualization...")
    vis = navigator.visualize_result(image_path, result)
    output_file = 'navigation_result.png'
    cv2.imwrite(output_file, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    print(f"✅ Saved visualization: {output_file}")
    print("\n" + "=" * 60)
    print("✅ Navigation complete!")