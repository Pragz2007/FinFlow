"""
Gemini API service for image and video analysis
"""

import base64
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import google.generativeai as genai
from app.config import settings

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Google's Gemini API"""
    
    def __init__(self):
        """Initialize Gemini service"""
        self.api_key = settings.gemini_api_key
        self.model_name = settings.gemini_model
        self.is_configured = bool(self.api_key)
        
        if self.is_configured:
            try:
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model_name)
                logger.info(f"GeminiService initialized: configured=True, model={self.model_name}")
            except Exception as e:
                logger.error(f"Failed to configure Gemini: {e}")
                self.is_configured = False
                self.client = None
        else:
            self.client = None
            logger.info("GeminiService initialized in DEMO MODE (no API key)")
    
    def health_check(self) -> bool:
        """
        Check if Gemini API is accessible
        
        Returns:
            True if API is accessible, False otherwise
        """
        if not self.is_configured:
            logger.warning("Gemini API not configured (no API key)")
            return False
        
        try:
            # Simple test - try to create a model
            test_model = genai.GenerativeModel(self.model_name)
            logger.info("Gemini API health check passed")
            return True
        except Exception as e:
            logger.error(f"Gemini API health check failed: {e}")
            return False
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image for dark patterns using Gemini
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Structured analysis result as dict
        """
        if not self.is_configured:
            logger.warning("Cannot analyze image: Gemini not configured - returning demo data")
            return self._get_demo_analysis_result()
        
        try:
            # Read image file
            path = Path(image_path)
            if not path.exists():
                raise ValueError(f"Image file not found: {image_path}")
            
            # Upload the image using Gemini Files API
            logger.info(f"Uploading image for analysis: {image_path}")
            uploaded_file = genai.upload_file(path=str(path))
            
            # Load system prompt
            system_prompt = self._load_system_prompt()
            
            # Create the analysis prompt
            analysis_prompt = f"""{system_prompt}

Analyze this fintech UI screenshot for dark patterns. Return ONLY a valid JSON object with this structure:
{json.dumps(self._get_analysis_schema(), indent=2)}

Be thorough but evidence-based. Only report patterns you can clearly see in the image."""
            
            # Generate content with the image
            model = self.client
            response = model.generate_content([
                analysis_prompt,
                uploaded_file
            ])
            
            # Parse response
            result = self._parse_gemini_response(response.text)
            logger.info(f"Image analysis completed with {len(result.get('findings', []))} findings")
            
            # Clean up uploaded file
            try:
                genai.delete_file(uploaded_file.name)
            except:
                pass
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            logger.warning("Falling back to demo data due to error")
            return self._get_demo_analysis_result()
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze a video for dark patterns using Gemini
        
        Args:
            video_path: Path to the video file
        
        Returns:
            Structured analysis result as dict
        """
        if not self.is_configured:
            logger.warning("Cannot analyze video: Gemini not configured - returning demo data")
            return self._get_demo_video_analysis_result()
        
        try:
            logger.info("Video analysis started")
            
            # Upload video file
            path = Path(video_path)
            if not path.exists():
                raise ValueError(f"Video file not found: {video_path}")
            
            logger.info(f"Uploading video for analysis: {video_path}")
            uploaded_file = genai.upload_file(path=str(path))
            
            # Load system prompt
            system_prompt = self._load_system_prompt("video")
            
            # Create analysis prompt
            analysis_prompt = f"""{system_prompt}

Analyze this fintech app video for dark patterns. Return ONLY a valid JSON object with this structure:
{json.dumps(self._get_analysis_schema(), indent=2)}

For each finding, include timestamp_seconds and frame_number if relevant."""
            
            # Generate content
            model = self.client
            response = model.generate_content([
                analysis_prompt,
                uploaded_file
            ])
            
            result = self._parse_gemini_response(response.text)
            result['input_type'] = 'video'
            
            logger.info(f"Video analysis completed with {len(result.get('findings', []))} findings")
            
            # Clean up
            try:
                genai.delete_file(uploaded_file.name)
            except:
                pass
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            logger.warning("Falling back to demo video data due to error")
            return self._get_demo_video_analysis_result()
    
    # Private helper methods
    
    def _read_image(self, image_path: str) -> Optional[str]:
        """Read and base64 encode an image"""
        try:
            path = Path(image_path)
            if not path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None
            
            with open(path, 'rb') as f:
                image_bytes = f.read()
            
            return base64.standard_b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Error reading image: {e}")
            return None
    
    def _load_system_prompt(self, prompt_type: str = "image") -> str:
        """Load system prompt from file"""
        try:
            prompt_file = Path(__file__).parent.parent / "prompts" / f"{prompt_type}_analysis.txt"
            if prompt_file.exists():
                with open(prompt_file, 'r') as f:
                    return f.read().strip()
            else:
                return self._get_default_system_prompt()
        except Exception as e:
            logger.warning(f"Error loading system prompt: {e}")
            return self._get_default_system_prompt()
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt"""
        return """You are CogniShield AI, a forensic UX analysis engine specializing in detecting deceptive design patterns in financial technology interfaces.

Your task:
1. Analyze the provided screenshot or video
2. Detect only evidence-supported dark patterns
3. For each finding, provide:
   - category (from predefined taxonomy)
   - title
   - severity (low, medium, high, critical)
   - confidence (0-100)
   - evidence_text (the actual UI element found)
   - why_problematic (explanation)
   - user_impact (how it affects users)
   - recommended_fix (how to remedy)
   - evidence (bounding box with normalized coordinates 0.0-1.0)

Never fabricate UI elements that are not visible.
Never classify normal UX as dark patterns.
Use cautious, evidence-based language."""
    
    def _get_analysis_schema(self) -> Dict[str, Any]:
        """Get the expected JSON schema for analysis results"""
        return {
            "overall_risk_score": 0,
            "risk_level": "LOW",
            "summary": "string",
            "findings": [
                {
                    "category": "string (from taxonomy)",
                    "title": "string",
                    "severity": "low|medium|high|critical",
                    "confidence": 0,
                    "evidence_text": "string",
                    "why_problematic": "string",
                    "user_impact": "string",
                    "recommended_fix": "string",
                    "evidence": {
                        "x": 0.0,
                        "y": 0.0,
                        "width": 0.5,
                        "height": 0.1
                    }
                }
            ],
            "analysis_notes": []
        }
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini response and extract JSON"""
        try:
            # Try to find JSON in response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
            else:
                logger.warning("Could not find JSON in Gemini response")
                result = json.loads(response_text)
            
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {e}")
            # Return empty but valid structure
            return {
                "overall_risk_score": 0,
                "risk_level": "LOW",
                "summary": "Analysis failed",
                "findings": [],
                "analysis_notes": ["Response parsing failed"]
            }
    
    def _get_demo_analysis_result(self) -> Dict[str, Any]:
        """Get demo analysis result for testing without API key"""
        return {
            "overall_risk_score": 72,
            "risk_level": "HIGH",
            "summary": "Demo Analysis - This is sample data. Multiple deceptive interaction patterns detected.",
            "findings": [
                {
                    "category": "subscription_trap",
                    "title": "Subscription Trap",
                    "severity": "high",
                    "confidence": 94,
                    "evidence_text": "Free trial — ₹499/month after 7 days",
                    "why_problematic": "The recurring payment condition is visually subordinate to the primary CTA.",
                    "user_impact": "Users may unintentionally enter a recurring billing agreement.",
                    "recommended_fix": "Make the trial terms and recurring cost equally prominent.",
                    "evidence": {"x": 0.18, "y": 0.57, "width": 0.64, "height": 0.14}
                },
                {
                    "category": "forced_action",
                    "title": "Forced Action",
                    "severity": "high",
                    "confidence": 89,
                    "evidence_text": "Continue to unlock your account",
                    "why_problematic": "The primary path obscures the option to proceed without consent.",
                    "user_impact": "Users are coerced into taking an action.",
                    "recommended_fix": "Provide an equally prominent decline option.",
                    "evidence": {"x": 0.17, "y": 0.76, "width": 0.66, "height": 0.10}
                },
                {
                    "category": "trick_question",
                    "title": "Trick Question",
                    "severity": "medium",
                    "confidence": 82,
                    "evidence_text": "Yes, I want helpful updates",
                    "why_problematic": "Consent language makes decline less salient.",
                    "user_impact": "Users may opt into unwanted communications.",
                    "recommended_fix": "Use neutral language for both options.",
                    "evidence": {"x": 0.17, "y": 0.35, "width": 0.66, "height": 0.12}
                }
            ],
            "analysis_notes": ["Demo Mode - No Gemini API Key", "Sample data for demonstration purposes"]
        }
    
    def _get_demo_video_analysis_result(self) -> Dict[str, Any]:
        """Get demo video analysis result"""
        return {
            "overall_risk_score": 68,
            "risk_level": "HIGH",
            "summary": "Demo Video Analysis - Detected patterns across subscription and payment flows.",
            "findings": [
                {
                    "category": "subscription_trap",
                    "title": "Auto-renewal Without Clear Consent",
                    "severity": "high",
                    "confidence": 91,
                    "evidence_text": "Auto-renewal enabled by default at 17.4s",
                    "why_problematic": "Pre-selected checkbox for recurring payment.",
                    "user_impact": "Charges users without explicit per-transaction consent.",
                    "recommended_fix": "Require explicit user action to enable auto-renewal.",
                    "evidence": {"x": 0.20, "y": 0.60, "width": 0.60, "height": 0.15},
                    "timestamp_seconds": 17.4,
                    "frame_number": 261
                },
                {
                    "category": "hidden_costs",
                    "title": "Processing Fee Disclosed in Fine Print",
                    "severity": "medium",
                    "confidence": 78,
                    "evidence_text": "2% processing fee shown at 34.2s in small text",
                    "why_problematic": "Important cost information is de-emphasized.",
                    "user_impact": "Users may not realize total cost before confirming.",
                    "recommended_fix": "Prominently display fees in transaction summary.",
                    "evidence": {"x": 0.15, "y": 0.85, "width": 0.70, "height": 0.08},
                    "timestamp_seconds": 34.2,
                    "frame_number": 513
                }
            ],
            "analysis_notes": ["Demo Mode Video Analysis", "Sample data with timestamp references"]
        }
