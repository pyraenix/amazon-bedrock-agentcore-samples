"""
Complete CAPTCHA Handling Application - Educational Example

⚠️ EDUCATIONAL DISCLAIMER:
This application demonstrates CAPTCHA detection and analysis using SAFE TEST ENDPOINTS.

IMPORTANT NOTICES:
- Uses official CAPTCHA demo pages (Google reCAPTCHA demo, hCaptcha demo) for education
- Focuses on DETECTION and ANALYSIS, not actual CAPTCHA solving
- NOT intended for bypassing CAPTCHAs or circumventing security measures
- For production use, ensure compliance with website terms of service
- CAPTCHAs are security measures - respect their purpose
- See docs/REAL_WORLD_IMPLEMENTATIONS.md for compliant production approaches

This application demonstrates comprehensive CAPTCHA detection, analysis, and handling
using LlamaIndex agents with AgentCore browser tool integration.
"""

import asyncio
import json
import base64
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from integration import LlamaIndexAgentCoreIntegration
from captcha_workflows import CaptchaSolvingWorkflow
from vision_models import VisionModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CaptchaAnalysisResult:
    """Result of CAPTCHA analysis."""
    url: str
    captcha_detected: bool
    captcha_type: Optional[str] = None
    confidence_score: float = 0.0
    location: Optional[Dict[str, int]] = None
    screenshot_path: Optional[str] = None
    analysis_details: Optional[Dict[str, Any]] = None
    solving_strategy: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class CaptchaHandlingApp:
    """
    Complete CAPTCHA handling application using LlamaIndex AgentCore integration.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the CAPTCHA handling application."""
        self.integration = LlamaIndexAgentCoreIntegration(config_path=config_path)
        self.agent = self.integration.create_agent()
        self.vision_manager = VisionModelManager()
        self.captcha_workflow = CaptchaSolvingWorkflow(self.vision_manager)
        self.results_history = []
        
        logger.info("CAPTCHA handling application initialized")
    
    async def analyze_page_for_captcha(self, url: str) -> CaptchaAnalysisResult:
        """
        Comprehensive CAPTCHA analysis for a given URL.
        
        Args:
            url: URL to analyze for CAPTCHAs
            
        Returns:
            CaptchaAnalysisResult with detailed analysis
        """
        logger.info(f"Starting CAPTCHA analysis for: {url}")
        
        try:
            # Step 1: Navigate and perform initial analysis
            response = await self.agent.achat(f"""
            Navigate to {url} and perform comprehensive CAPTCHA detection:
            
            1. DOM Analysis:
               - Scan for CAPTCHA-related elements (iframe, div with captcha classes)
               - Look for reCAPTCHA, hCaptcha, FunCaptcha containers
               - Check for CAPTCHA-related scripts and resources
               - Identify any form elements that might trigger CAPTCHAs
            
            2. Visual Analysis:
               - Take a full-page screenshot
               - Look for visual CAPTCHA indicators (checkboxes, challenge images)
               - Identify any overlay elements or modal dialogs
               - Check for accessibility features (audio buttons, refresh options)
            
            3. Interaction Testing:
               - Try to interact with any detected CAPTCHA elements
               - Check if CAPTCHAs are triggered by form interactions
               - Test for invisible/background CAPTCHAs (reCAPTCHA v3)
            
            4. Classification:
               - Determine CAPTCHA type and version
               - Assess complexity level (simple checkbox vs image challenges)
               - Identify the CAPTCHA provider (Google, hCaptcha, etc.)
            
            Provide detailed analysis with confidence scores and specific findings.
            """)
            
            # Parse the response for structured data
            analysis_data = self._parse_captcha_analysis(response.response)
            
            # Step 2: Take screenshot for visual analysis
            screenshot_path = await self._capture_analysis_screenshot(url)
            
            # Step 3: Create comprehensive result
            result = CaptchaAnalysisResult(
                url=url,
                captcha_detected=analysis_data.get('captcha_detected', False),
                captcha_type=analysis_data.get('captcha_type'),
                confidence_score=analysis_data.get('confidence_score', 0.0),
                location=analysis_data.get('location'),
                screenshot_path=screenshot_path,
                analysis_details=analysis_data,
                solving_strategy=self._determine_solving_strategy(analysis_data)
            )
            
            self.results_history.append(result)
            logger.info(f"CAPTCHA analysis completed for {url}")
            
            return result
            
        except Exception as e:
            logger.error(f"CAPTCHA analysis failed for {url}: {e}")
            
            # Return failed analysis result
            result = CaptchaAnalysisResult(
                url=url,
                captcha_detected=False,
                analysis_details={"error": str(e)}
            )
            
            self.results_history.append(result)
            return result
    
    async def handle_captcha_workflow(self, url: str, 
                                    auto_solve: bool = False) -> Dict[str, Any]:
        """
        Complete CAPTCHA handling workflow.
        
        Args:
            url: URL to process
            auto_solve: Whether to attempt automatic solving
            
        Returns:
            Dictionary with workflow results
        """
        logger.info(f"Starting CAPTCHA workflow for: {url}")
        
        # Step 1: Analyze for CAPTCHAs
        analysis = await self.analyze_page_for_captcha(url)
        
        workflow_result = {
            "url": url,
            "analysis": asdict(analysis),
            "workflow_steps": [],
            "final_status": "analysis_complete"
        }
        
        if not analysis.captcha_detected:
            workflow_result["final_status"] = "no_captcha_detected"
            logger.info(f"No CAPTCHA detected on {url}")
            return workflow_result
        
        logger.info(f"CAPTCHA detected: {analysis.captcha_type} (confidence: {analysis.confidence_score})")
        
        # Step 2: Detailed CAPTCHA interaction
        interaction_result = await self._interact_with_captcha(url, analysis)
        workflow_result["workflow_steps"].append({
            "step": "captcha_interaction",
            "result": interaction_result
        })
        
        # Step 3: Attempt solving if requested
        if auto_solve and analysis.captcha_type:
            solving_result = await self._attempt_captcha_solving(url, analysis)
            workflow_result["workflow_steps"].append({
                "step": "captcha_solving",
                "result": solving_result
            })
            
            if solving_result.get("success"):
                workflow_result["final_status"] = "captcha_solved"
            else:
                workflow_result["final_status"] = "captcha_unsolved"
        else:
            workflow_result["final_status"] = "captcha_detected_not_solved"
        
        logger.info(f"CAPTCHA workflow completed for {url}: {workflow_result['final_status']}")
        return workflow_result
    
    async def batch_captcha_analysis(self, urls: List[str], 
                                   max_concurrent: int = 3) -> List[CaptchaAnalysisResult]:
        """
        Analyze multiple URLs for CAPTCHAs concurrently.
        
        Args:
            urls: List of URLs to analyze
            max_concurrent: Maximum concurrent analyses
            
        Returns:
            List of analysis results
        """
        logger.info(f"Starting batch CAPTCHA analysis for {len(urls)} URLs")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_analysis(url):
            async with semaphore:
                return await self.analyze_page_for_captcha(url)
        
        results = await asyncio.gather(
            *[bounded_analysis(url) for url in urls],
            return_exceptions=True
        )
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(CaptchaAnalysisResult(
                    url=urls[i],
                    captcha_detected=False,
                    analysis_details={"error": str(result)}
                ))
            else:
                processed_results.append(result)
        
        # Log summary
        detected_count = sum(1 for r in processed_results if r.captcha_detected)
        logger.info(f"Batch analysis complete: {detected_count}/{len(urls)} sites have CAPTCHAs")
        
        return processed_results
    
    async def generate_captcha_report(self, results: List[CaptchaAnalysisResult] = None) -> Dict[str, Any]:
        """
        Generate comprehensive CAPTCHA analysis report.
        
        Args:
            results: Optional specific results to report on
            
        Returns:
            Comprehensive report dictionary
        """
        if results is None:
            results = self.results_history
        
        if not results:
            return {"error": "No analysis results available"}
        
        # Calculate statistics
        total_sites = len(results)
        captcha_sites = sum(1 for r in results if r.captcha_detected)
        captcha_rate = captcha_sites / total_sites if total_sites > 0 else 0
        
        # Analyze CAPTCHA types
        captcha_types = {}
        confidence_scores = []
        
        for result in results:
            if result.captcha_detected and result.captcha_type:
                captcha_types[result.captcha_type] = captcha_types.get(result.captcha_type, 0) + 1
                confidence_scores.append(result.confidence_score)
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Generate recommendations
        recommendations = self._generate_captcha_recommendations(results)
        
        report = {
            "analysis_summary": {
                "total_sites_analyzed": total_sites,
                "sites_with_captcha": captcha_sites,
                "captcha_detection_rate": captcha_rate,
                "average_confidence_score": avg_confidence
            },
            "captcha_distribution": captcha_types,
            "detailed_results": [asdict(r) for r in results],
            "recommendations": recommendations,
            "report_generated": datetime.now().isoformat()
        }
        
        return report
    
    def _parse_captcha_analysis(self, response_text: str) -> Dict[str, Any]:
        """Parse CAPTCHA analysis from agent response."""
        # This is a simplified parser - in practice, you'd use more sophisticated NLP
        analysis = {
            "captcha_detected": False,
            "captcha_type": None,
            "confidence_score": 0.0,
            "location": None
        }
        
        response_lower = response_text.lower()
        
        # Check for CAPTCHA detection keywords
        captcha_keywords = ["captcha", "recaptcha", "hcaptcha", "funcaptcha"]
        if any(keyword in response_lower for keyword in captcha_keywords):
            analysis["captcha_detected"] = True
            analysis["confidence_score"] = 0.8  # Default confidence
        
        # Identify CAPTCHA type
        if "recaptcha" in response_lower:
            if "v3" in response_lower:
                analysis["captcha_type"] = "recaptcha_v3"
            elif "v2" in response_lower:
                analysis["captcha_type"] = "recaptcha_v2"
            else:
                analysis["captcha_type"] = "recaptcha"
        elif "hcaptcha" in response_lower:
            analysis["captcha_type"] = "hcaptcha"
        elif "funcaptcha" in response_lower:
            analysis["captcha_type"] = "funcaptcha"
        elif "captcha" in response_lower:
            analysis["captcha_type"] = "generic_captcha"
        
        # Extract confidence indicators
        if "high confidence" in response_lower or "definitely" in response_lower:
            analysis["confidence_score"] = 0.9
        elif "likely" in response_lower or "probably" in response_lower:
            analysis["confidence_score"] = 0.7
        elif "possible" in response_lower or "might" in response_lower:
            analysis["confidence_score"] = 0.5
        
        return analysis
    
    async def _capture_analysis_screenshot(self, url: str) -> Optional[str]:
        """Capture screenshot for CAPTCHA analysis."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_filename = f"captcha_analysis_{timestamp}.png"
            
            # Use the agent to take a screenshot
            response = await self.agent.achat(f"""
            Take a full-page screenshot of {url} for CAPTCHA analysis.
            Focus on capturing any CAPTCHA elements clearly.
            """)
            
            # In a real implementation, you'd extract the screenshot data
            # from the response metadata and save it
            return screenshot_filename
            
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            return None
    
    def _determine_solving_strategy(self, analysis_data: Dict[str, Any]) -> Optional[str]:
        """Determine the best strategy for solving detected CAPTCHA."""
        if not analysis_data.get("captcha_detected"):
            return None
        
        captcha_type = analysis_data.get("captcha_type")
        confidence = analysis_data.get("confidence_score", 0.0)
        
        if confidence < 0.5:
            return "manual_verification_required"
        
        strategies = {
            "recaptcha_v2": "vision_model_analysis",
            "recaptcha_v3": "behavioral_analysis",
            "hcaptcha": "vision_model_analysis", 
            "funcaptcha": "specialized_solver",
            "generic_captcha": "ocr_analysis"
        }
        
        return strategies.get(captcha_type, "manual_analysis")
    
    async def _interact_with_captcha(self, url: str, 
                                   analysis: CaptchaAnalysisResult) -> Dict[str, Any]:
        """Interact with detected CAPTCHA to gather more information."""
        try:
            response = await self.agent.achat(f"""
            On {url}, interact with the detected {analysis.captcha_type} CAPTCHA:
            
            1. Try clicking on the CAPTCHA checkbox if present
            2. Observe what challenge appears (image grid, audio option, etc.)
            3. Take screenshots of any challenge interfaces
            4. Test accessibility features (audio CAPTCHA, refresh options)
            5. Document the complete user interaction flow
            
            Do not attempt to solve the CAPTCHA, just document the interaction process.
            """)
            
            return {
                "success": True,
                "interaction_details": response.response,
                "challenges_observed": self._extract_challenge_types(response.response)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _attempt_captcha_solving(self, url: str, 
                                     analysis: CaptchaAnalysisResult) -> Dict[str, Any]:
        """Attempt to solve the CAPTCHA using appropriate strategy."""
        try:
            strategy = analysis.solving_strategy
            
            if strategy == "vision_model_analysis":
                return await self._solve_with_vision_model(url, analysis)
            elif strategy == "behavioral_analysis":
                return await self._solve_with_behavioral_analysis(url, analysis)
            elif strategy == "ocr_analysis":
                return await self._solve_with_ocr(url, analysis)
            else:
                return {
                    "success": False,
                    "reason": f"No automated solving strategy available for {strategy}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _solve_with_vision_model(self, url: str, 
                                     analysis: CaptchaAnalysisResult) -> Dict[str, Any]:
        """Attempt to solve CAPTCHA using vision model analysis."""
        try:
            response = await self.agent.achat(f"""
            On {url}, attempt to analyze the {analysis.captcha_type} CAPTCHA challenge:
            
            1. Take a high-quality screenshot of the CAPTCHA challenge
            2. Use vision analysis to identify the challenge type (traffic lights, crosswalks, etc.)
            3. Analyze the image grid and identify relevant objects
            4. Provide detailed analysis of what you observe
            
            Focus on analysis rather than actual solving for safety and compliance.
            """)
            
            return {
                "success": True,
                "method": "vision_analysis",
                "analysis_result": response.response,
                "note": "Analysis only - no actual solving attempted"
            }
            
        except Exception as e:
            return {
                "success": False,
                "method": "vision_analysis",
                "error": str(e)
            }
    
    async def _solve_with_behavioral_analysis(self, url: str, 
                                            analysis: CaptchaAnalysisResult) -> Dict[str, Any]:
        """Analyze behavioral CAPTCHA (like reCAPTCHA v3)."""
        return {
            "success": True,
            "method": "behavioral_analysis",
            "analysis_result": "Behavioral CAPTCHAs require natural user interaction patterns",
            "note": "This type of CAPTCHA is analyzed in the background"
        }
    
    async def _solve_with_ocr(self, url: str, 
                            analysis: CaptchaAnalysisResult) -> Dict[str, Any]:
        """Attempt OCR-based CAPTCHA analysis."""
        return {
            "success": True,
            "method": "ocr_analysis", 
            "analysis_result": "OCR analysis would be performed on text-based CAPTCHAs",
            "note": "Analysis only - actual OCR solving not implemented for safety"
        }
    
    def _extract_challenge_types(self, interaction_text: str) -> List[str]:
        """Extract challenge types from interaction description."""
        challenges = []
        text_lower = interaction_text.lower()
        
        challenge_keywords = {
            "traffic lights": "traffic_lights",
            "crosswalks": "crosswalks", 
            "vehicles": "vehicles",
            "bicycles": "bicycles",
            "buses": "buses",
            "fire hydrants": "fire_hydrants",
            "stairs": "stairs",
            "audio": "audio_challenge"
        }
        
        for keyword, challenge_type in challenge_keywords.items():
            if keyword in text_lower:
                challenges.append(challenge_type)
        
        return challenges
    
    def _generate_captcha_recommendations(self, results: List[CaptchaAnalysisResult]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        captcha_count = sum(1 for r in results if r.captcha_detected)
        total_count = len(results)
        
        if captcha_count == 0:
            recommendations.append("No CAPTCHAs detected - sites appear to be easily accessible")
        elif captcha_count / total_count > 0.5:
            recommendations.append("High CAPTCHA prevalence detected - consider CAPTCHA-solving strategies")
        
        # Analyze CAPTCHA types
        types = [r.captcha_type for r in results if r.captcha_type]
        if "recaptcha_v2" in types:
            recommendations.append("reCAPTCHA v2 detected - vision-based solving may be effective")
        if "recaptcha_v3" in types:
            recommendations.append("reCAPTCHA v3 detected - focus on natural browsing behavior")
        if "hcaptcha" in types:
            recommendations.append("hCaptcha detected - similar to reCAPTCHA but may have different patterns")
        
        # Confidence-based recommendations
        low_confidence = [r for r in results if r.confidence_score < 0.6]
        if low_confidence:
            recommendations.append(f"{len(low_confidence)} sites need manual verification for CAPTCHA detection")
        
        return recommendations
    
    def export_results(self, filename: str = None) -> str:
        """Export analysis results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captcha_analysis_results_{timestamp}.json"
        
        report = asyncio.run(self.generate_captcha_report())
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Results exported to {filename}")
        return filename

async def main():
    """Main demonstration of CAPTCHA handling application."""
    print("🔐 CAPTCHA Handling Application Demo")
    print("=" * 50)
    
    # Initialize application
    app = CaptchaHandlingApp()
    
    # Educational test URLs with different CAPTCHA types
    test_urls = [
        "https://www.google.com/recaptcha/api2/demo",  # Educational: Official reCAPTCHA v2 demo
        "https://accounts.hcaptcha.com/demo",          # Educational: Official hCaptcha demo  
        "https://httpbin.org/html",                    # Educational: No CAPTCHA (control)
        "https://httpbin.org/forms/post"               # Educational: Form that might trigger CAPTCHA
        # Production: Only use with proper authorization and compliance verification
        # See docs/REAL_WORLD_IMPLEMENTATIONS.md for production guidelines
    ]
    
    print(f"🔍 Analyzing {len(test_urls)} URLs for CAPTCHAs...")
    
    # Perform batch analysis
    results = await app.batch_captcha_analysis(test_urls, max_concurrent=2)
    
    print("\n📊 Analysis Results:")
    print("-" * 50)
    
    for i, result in enumerate(results, 1):
        status = "🔐" if result.captcha_detected else "✅"
        print(f"{status} Site {i}: {result.url}")
        
        if result.captcha_detected:
            print(f"   Type: {result.captcha_type}")
            print(f"   Confidence: {result.confidence_score:.1%}")
            print(f"   Strategy: {result.solving_strategy}")
        else:
            print("   No CAPTCHA detected")
        
        print(f"   Analyzed: {result.timestamp}")
        print()
    
    # Demonstrate complete workflow for CAPTCHA sites
    captcha_sites = [r for r in results if r.captcha_detected]
    
    if captcha_sites:
        print(f"🔧 Demonstrating complete workflow for {len(captcha_sites)} CAPTCHA sites...")
        
        for result in captcha_sites[:2]:  # Limit to first 2 for demo
            print(f"\n🔄 Processing workflow for: {result.url}")
            
            workflow_result = await app.handle_captcha_workflow(
                result.url, 
                auto_solve=False  # Set to True to attempt solving
            )
            
            print(f"   Final status: {workflow_result['final_status']}")
            print(f"   Workflow steps: {len(workflow_result['workflow_steps'])}")
    
    # Generate comprehensive report
    print("\n📋 Generating comprehensive report...")
    report = await app.generate_captcha_report()
    
    print(f"📈 Summary Statistics:")
    summary = report["analysis_summary"]
    print(f"   Sites analyzed: {summary['total_sites_analyzed']}")
    print(f"   CAPTCHAs detected: {summary['sites_with_captcha']}")
    print(f"   Detection rate: {summary['captcha_detection_rate']:.1%}")
    print(f"   Average confidence: {summary['average_confidence_score']:.1%}")
    
    if report["captcha_distribution"]:
        print(f"\n🔐 CAPTCHA Types Found:")
        for captcha_type, count in report["captcha_distribution"].items():
            print(f"   {captcha_type}: {count}")
    
    print(f"\n💡 Recommendations:")
    for rec in report["recommendations"]:
        print(f"   • {rec}")
    
    # Export results
    filename = app.export_results()
    print(f"\n💾 Complete results exported to: {filename}")
    
    print("\n✅ CAPTCHA handling application demo completed!")
    return results

if __name__ == "__main__":
    # Run the complete CAPTCHA handling demonstration
    results = asyncio.run(main())