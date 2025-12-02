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
        >>> print(f"Target: ({result['u']}, {result['v']})")
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "auto", torch_dtype: str = "auto"):
        
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🤖 Loading {model_name}...")
        print(f"   Device: {self.device}")
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        print("Model loaded successfully!")
    
    def navigate(self, image: Union[np.ndarray, Image.Image, str], prompt: str, debug: bool = False) -> Dict[str, any]:

        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image.astype('uint8')).convert('RGB')
        else:
            image_pil = image
        
        width, height = image_pil.size
        target = self._extract_target(prompt)
        grounding_prompt = f"Find the {target} and return its location with bounding box coordinates."
        
        if debug:
            print(f"\n🔍 DEBUG:")
            print(f"  Target: {target}")
            print(f"  Grounding prompt: {grounding_prompt}")
            print(f"  Image size: {width}x{height}")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": grounding_prompt}
                ]
            }
        ]
        
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
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        
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
        
        result = self._parse_coordinates(response, width, height, debug)
        return result
    
    def _extract_target(self, prompt: str) -> str:
        target = prompt.lower()
        for phrase in ['navigate to the', 'go to the', 'head to the', 'navigate to']:
            target = target.replace(phrase, '')
        return target.strip()
    
    def _parse_coordinates(self, response: str, width: int, height: int, debug: bool = False) -> Dict[str, any]:
        """
        Parse coordinates from model response using multiple strategies.
        Returns 'action': [u,v] or 'action': "STOP"
        """
        u, v = None, None
        confidence = 0.5

        # Detect goal reached
        if re.search(r"GOAL\s+REACHED", response, re.IGNORECASE):
            return {
                "action": "STOP",
                "confidence": confidence,
                "reasoning": response[:150] if len(response) > 150 else response,
                "raw_response": response
            }

        # Strategy 1: bounding box [u1,v1,u2,v2]
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        bbox_match = re.search(bbox_pattern, response)
        if bbox_match:
            u1, v1, u2, v2 = map(int, bbox_match.groups())
            u = (u1 + u2) // 2
            v = (v1 + v2) // 2
            confidence = 0.85
            if debug:
                print(f"Found bbox center: ({u}, {v})")

        # Strategy 2: coordinate pair
        if u is None:
            coord_pattern = r'(\d+),\s*(\d+)'
            matches = re.findall(coord_pattern, response)
            if matches:
                u, v = map(int, matches[0])
                confidence = 0.7
                if debug:
                    print(f"Found coordinates: ({u}, {v})")

        # Strategy 3: first two numbers
        if u is None:
            nums = re.findall(r'\b(\d+)\b', response)
            if len(nums) >= 2:
                u, v = int(nums[0]), int(nums[1])
                confidence = 0.5
                if debug:
                    print(f"Using fallback numbers: ({u}, {v})")

        # Final fallback
        if u is None:
            u, v = width // 2, height // 2
            confidence = 0.0
            if debug:
                print(f"Final fallback center: ({u}, {v})")

        # Clip to bounds
        u = max(0, min(u, width - 1))
        v = max(0, min(v, height - 1))

        return {
            "action": [u, v],
            "confidence": confidence,
            "reasoning": response[:150] if len(response) > 150 else response,
            "raw_response": response
        }
    
    def visualize_result(self, image: Union[np.ndarray, str], result: Dict[str, any], show_confidence: bool = True) -> np.ndarray:
        
        if isinstance(image, str):
            vis = cv2.imread(image)
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        else:
            vis = image.copy()

        conf = result['confidence']

        if result['action'] == "STOP":
            cv2.putText(vis, "STOP", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            return vis

        u, v = result['action']

        # Color based on confidence
        if conf > 0.7:
            color = (0, 255, 0)
        elif conf > 0.4:
            color = (255, 165, 0)
        else:
            color = (255, 0, 0)

        # Draw crosshair
        cv2.line(vis, (u-30, v), (u+30, v), color, 3)
        cv2.line(vis, (u, v-30), (u, v+30), color, 3)
        cv2.circle(vis, (u, v), 15, color, 3)

        if show_confidence:
            text = f"Target: ({u}, {v}) | Conf: {conf:.2f}"
            cv2.putText(vis, text, (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(vis, text, (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        return vis


class RoadAwareNavigator(VisionNavigationSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Road-aware navigation enabled")
    
    def navigate(self, image: Union[np.ndarray, Image.Image, str], prompt: str, road_aware: bool = True, road_method: str = 'region', debug: bool = False) -> Dict[str, any]:
        result = super().navigate(image, prompt, debug=debug)
        if result['action'] == "STOP" or not road_aware:
            return result

        if isinstance(image, str):
            image_np = cv2.imread(image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        road_mask = self.detect_road(image_np, method=road_method)
        target_u, target_v = result['action'][0], result['action'][1]
        on_road = road_mask[target_v, target_u]

        if debug:
            print(f"\nRoad Detection:")
            print(f"  Method: {road_method}")
            print(f"  Road pixels: {np.sum(road_mask):,}")
            print(f"  Target on road: {on_road}")

        if not on_road:
            safe_u, safe_v = self.find_closest_road_point(
                road_mask, target_u, target_v
            )
            if debug:
                offset = np.sqrt((safe_u - target_u)**2 + (safe_v - target_v)**2)
                print(f"  Target off-road!")
                print(f"  Closest road point: ({safe_u}, {safe_v})")
                print(f"  Offset: {offset:.0f}px")
            result['action'] = [safe_u, safe_v]
            result['reasoning'] += ' (adjusted to road)'

        result['on_road'] = True
        result['road_mask'] = road_mask
        return result
    
    def detect_road(self, image: np.ndarray, method: str = 'region') -> np.ndarray:
        height, width = image.shape[:2]
        if method == 'region':
            road_mask = np.zeros((height, width), dtype=bool)
            road_mask[int(height * 0.4):, :] = True
            return road_mask
        elif method == 'color':
            lower_gray = np.array([50, 50, 50])
            upper_gray = np.array([90, 90, 90])
            road_mask = cv2.inRange(image, lower_gray, upper_gray)
            road_mask[:int(height * 0.35), :] = 0
            kernel = np.ones((7, 7), np.uint8)
            road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
            road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
            return road_mask.astype(bool)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'region' or 'color'")

    def find_closest_road_point(self, road_mask: np.ndarray, target_u: int, target_v: int) -> Tuple[int, int]:
        road_points = np.argwhere(road_mask)
        if len(road_points) == 0:
            height, width = road_mask.shape
            return (width // 2, int(height * 0.8))
        distances = np.sqrt((road_points[:, 1] - target_u)**2 + (road_points[:, 0] - target_v)**2)
        closest_idx = np.argmin(distances)
        closest_point = road_points[closest_idx]
        return (int(closest_point[1]), int(closest_point[0]))

    def visualize_result(self, image: Union[np.ndarray, str], result: Dict[str, any], show_road_overlay: bool = True) -> np.ndarray:
        if isinstance(image, str):
            vis = cv2.imread(image)
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        else:
            vis = image.copy()
        
        if show_road_overlay and 'road_mask' in result:
            road_overlay = np.zeros_like(vis)
            road_overlay[result['road_mask']] = [0, 200, 0]
            vis = cv2.addWeighted(vis, 0.7, road_overlay, 0.3, 0)

        vis = super().visualize_result(vis, result, show_confidence=True)
        if result.get('on_road', False):
            cv2.putText(vis, "ON ROAD", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return vis


# Example usage
if __name__ == "__main__":
    import sys
    
    print("CARLA Vision Navigation System")
    print("=" * 60)
    
    if len(sys.argv) < 3:
        print("\nUsage: python vision_navigation.py <image_path> <instruction>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    instruction = sys.argv[2]
    
    print(f"\nImage: {image_path}")
    print(f"Instruction: {instruction}")
    
    print("\nInitializing navigation system...")
    navigator = RoadAwareNavigator(device='auto')
    
    print("\nProcessing navigation request...")
    result = navigator.navigate(image_path, instruction, road_aware=True, debug=True)
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    if result['action'] == "STOP":
        print(f"Action: STOP")
    else:
        print(f"Waypoint: ({result['action'][0]}, {result['action'][1]})")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"On road: {result.get('on_road', False)}")
    print(f"Reasoning: {result['reasoning'][:80]}...")
    
    print("\nCreating visualization...")
    vis = navigator.visualize_result(image_path, result)
    output_file = 'navigation_result.png'
    cv2.imwrite(output_file, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization: {output_file}")
    print("\n" + "=" * 60)
    print("Navigation complete!")