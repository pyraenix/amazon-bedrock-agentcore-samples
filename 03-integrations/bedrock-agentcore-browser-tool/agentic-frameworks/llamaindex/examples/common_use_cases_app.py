"""
Common Use Cases Application - Educational Examples

⚠️ EDUCATIONAL DISCLAIMER:
This application demonstrates common use cases using SAFE TEST ENDPOINTS for educational purposes.

IMPORTANT NOTICES:
- Uses httpbin.org, example.com, and demo sites for safe learning
- NOT intended for production use with real websites without compliance verification
- For production implementations, see docs/REAL_WORLD_IMPLEMENTATIONS.md
- Always check robots.txt, terms of service, and applicable laws before automating real websites
- Implement proper rate limiting, error handling, and data protection measures
- Prefer official APIs over web scraping when available

This application demonstrates common real-world use cases for the LlamaIndex
AgentCore Browser Integration, including web scraping, form automation,
content monitoring, and competitive analysis.
"""

import asyncio
import json
import csv
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from integration import LlamaIndexAgentCoreIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class UseCase:
    """Represents a use case with its configuration."""
    name: str
    description: str
    urls: List[str]
    instructions: str
    expected_output: str
    metadata: Dict[str, Any] = None

@dataclass
class UseCaseResult:
    """Result of executing a use case."""
    use_case: UseCase
    success: bool
    results: List[Dict[str, Any]]
    execution_time: float
    error_message: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class CommonUseCasesApp:
    """
    Application demonstrating common use cases for web automation.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the common use cases application."""
        self.integration = LlamaIndexAgentCoreIntegration(config_path=config_path)
        self.agent = self.integration.create_agent()
        self.results_history = []
        
        # Define common use cases
        self.use_cases = self._define_use_cases()
        
        logger.info("Common use cases application initialized")
    
    def _define_use_cases(self) -> Dict[str, UseCase]:
        """Define common use cases for web automation."""
        return {
            "news_monitoring": UseCase(
                name="News Article Monitoring",
                description="Monitor news websites for articles on specific topics",
                urls=[
                    "https://httpbin.org/html",  # Educational: Safe test URL
                    "https://example.com"        # Educational: Generic example domain
                    # Production: Replace with compliant news sources after verification
                    # See docs/REAL_WORLD_IMPLEMENTATIONS.md for production examples
                ],
                instructions="""
                Monitor this news website for articles related to technology and AI:
                
                1. Navigate to the website
                2. Identify the main news articles on the homepage
                3. For each article, extract:
                   - Headline
                   - Publication date (if available)
                   - Brief summary or excerpt
                   - Author (if available)
                   - Article URL/link
                4. Filter for articles containing keywords: AI, artificial intelligence, technology, machine learning
                5. Return structured data with article information
                
                Focus on recent articles and provide clean, structured output.
                """,
                expected_output="List of news articles with metadata",
                metadata={"keywords": ["AI", "technology", "machine learning"], "category": "news"}
            ),
            
            "ecommerce_price_tracking": UseCase(
                name="E-commerce Price Tracking",
                description="Track product prices across e-commerce websites",
                urls=[
                    "https://demo.opencart.com/",      # Educational: Demo e-commerce site
                    "https://httpbin.org/json"         # Educational: Mock product data
                    # Production: Use official APIs (Amazon Product API, etc.) or compliant sources
                    # See docs/REAL_WORLD_IMPLEMENTATIONS.md for production examples
                ],
                instructions="""
                Track product prices on this e-commerce website:
                
                1. Navigate to the product catalog or homepage
                2. Identify featured or popular products
                3. For each product, extract:
                   - Product name
                   - Current price
                   - Original price (if on sale)
                   - Discount percentage (if applicable)
                   - Product availability status
                   - Product image URL
                   - Product page URL
                4. Handle any popups or cookie banners
                5. Return structured pricing data
                
                Focus on extracting accurate pricing information and product details.
                """,
                expected_output="Product pricing data with availability status",
                metadata={"tracking_type": "price", "category": "ecommerce"}
            ),
            
            "job_listings_aggregation": UseCase(
                name="Job Listings Aggregation",
                description="Aggregate job listings from multiple job boards",
                urls=[
                    "https://httpbin.org/html",  # Mock job board
                    "https://example.com"
                ],
                instructions="""
                Aggregate job listings from this job board:
                
                1. Navigate to the job listings page
                2. Search for or browse relevant job categories
                3. For each job listing, extract:
                   - Job title
                   - Company name
                   - Location (remote/on-site/hybrid)
                   - Salary range (if available)
                   - Job description summary
                   - Required skills/qualifications
                   - Application deadline (if available)
                   - Job posting date
                4. Filter for technology-related positions
                5. Return structured job data
                
                Focus on extracting comprehensive job information for analysis.
                """,
                expected_output="Structured job listings data",
                metadata={"job_categories": ["technology", "software"], "location_preference": "remote"}
            ),
            
            "social_media_monitoring": UseCase(
                name="Social Media Content Monitoring",
                description="Monitor social media platforms for brand mentions and trends",
                urls=[
                    "https://httpbin.org/html",  # Mock social media page
                ],
                instructions="""
                Monitor this social media platform for relevant content:
                
                1. Navigate to the platform
                2. Search for or browse content related to specific topics/hashtags
                3. For each relevant post, extract:
                   - Post content/text
                   - Author/username
                   - Post timestamp
                   - Engagement metrics (likes, shares, comments if visible)
                   - Hashtags used
                   - Media attachments (images/videos if any)
                4. Identify trending topics or popular hashtags
                5. Analyze sentiment and engagement patterns
                
                Focus on public content only and respect platform terms of service.
                """,
                expected_output="Social media content analysis with engagement metrics",
                metadata={"platform": "generic", "content_type": "public_posts"}
            ),
            
            "real_estate_monitoring": UseCase(
                name="Real Estate Listings Monitoring",
                description="Monitor real estate websites for property listings",
                urls=[
                    "https://httpbin.org/json",  # Mock real estate data
                    "https://example.com"
                ],
                instructions="""
                Monitor real estate listings on this website:
                
                1. Navigate to the property listings section
                2. Apply relevant filters (location, price range, property type)
                3. For each property listing, extract:
                   - Property address
                   - Listing price
                   - Property type (house, apartment, condo, etc.)
                   - Number of bedrooms and bathrooms
                   - Square footage/area
                   - Property features and amenities
                   - Listing agent contact information
                   - Property images (URLs)
                   - Days on market
                4. Identify new listings or price changes
                5. Return structured property data
                
                Focus on accurate property details and pricing information.
                """,
                expected_output="Real estate listings with property details",
                metadata={"property_types": ["house", "apartment"], "price_range": "market_rate"}
            ),
            
            "competitor_analysis": UseCase(
                name="Competitive Website Analysis",
                description="Analyze competitor websites for features and content",
                urls=[
                    "https://example.com",
                    "https://httpbin.org/html"
                ],
                instructions="""
                Perform competitive analysis on this website:
                
                1. Navigate to the website homepage
                2. Analyze the overall site structure and navigation
                3. Identify key features and services offered
                4. Extract information about:
                   - Main value propositions
                   - Product/service offerings
                   - Pricing information (if available)
                   - Contact information and locations
                   - Technology stack indicators
                   - User experience elements
                   - Content strategy and messaging
                5. Assess site performance and loading speed
                6. Identify unique selling points and differentiators
                
                Provide comprehensive competitive intelligence for strategic analysis.
                """,
                expected_output="Competitive analysis report with strategic insights",
                metadata={"analysis_type": "competitive_intelligence", "focus": "features_and_positioning"}
            ),
            
            "content_change_detection": UseCase(
                name="Website Content Change Detection",
                description="Monitor websites for content changes and updates",
                urls=[
                    "https://httpbin.org/uuid",  # Changes on each request
                    "https://httpbin.org/json"
                ],
                instructions="""
                Monitor this website for content changes:
                
                1. Navigate to the target page
                2. Extract the current content structure and key elements
                3. Identify:
                   - Main content sections
                   - Important text blocks
                   - Navigation elements
                   - Dynamic content areas
                   - Last updated timestamps (if available)
                4. Create a content fingerprint for change detection
                5. Compare with previous versions (if available)
                6. Highlight any changes or updates found
                
                Focus on detecting meaningful content changes, not minor formatting updates.
                """,
                expected_output="Content change detection report",
                metadata={"monitoring_frequency": "daily", "change_sensitivity": "medium"}
            ),
            
            "form_automation_testing": UseCase(
                name="Automated Form Testing",
                description="Test web forms with various input scenarios",
                urls=[
                    "https://httpbin.org/forms/post"
                ],
                instructions="""
                Test this web form with automated input scenarios:
                
                1. Navigate to the form page
                2. Identify all form fields and their types
                3. Test multiple scenarios:
                   - Valid input data
                   - Invalid input data (boundary testing)
                   - Required field validation
                   - Input format validation
                4. For each test scenario:
                   - Fill out the form with test data
                   - Submit the form
                   - Capture the response/result
                   - Document any validation errors
                5. Test form accessibility features
                6. Measure form completion time and usability
                
                Provide comprehensive form testing results and usability assessment.
                """,
                expected_output="Form testing results with validation analysis",
                metadata={"testing_type": "functional_and_usability", "scenarios": "multiple"}
            )
        }
    
    async def execute_use_case(self, use_case_name: str) -> UseCaseResult:
        """
        Execute a specific use case.
        
        Args:
            use_case_name: Name of the use case to execute
            
        Returns:
            UseCaseResult with execution details
        """
        if use_case_name not in self.use_cases:
            raise ValueError(f"Unknown use case: {use_case_name}")
        
        use_case = self.use_cases[use_case_name]
        logger.info(f"Executing use case: {use_case.name}")
        
        start_time = asyncio.get_event_loop().time()
        results = []
        
        try:
            # Execute the use case for each URL
            for url in use_case.urls:
                logger.info(f"Processing URL: {url}")
                
                # Create specific instructions for this URL
                url_instructions = f"""
                URL to process: {url}
                
                {use_case.instructions}
                
                Additional context:
                - Use case: {use_case.name}
                - Expected output: {use_case.expected_output}
                - Metadata: {json.dumps(use_case.metadata)}
                
                Provide detailed, structured results that match the expected output format.
                """
                
                # Execute with the agent
                response = await self.agent.achat(url_instructions)
                
                # Process the response
                url_result = {
                    "url": url,
                    "success": True,
                    "content": response.response,
                    "metadata": getattr(response, 'metadata', {}),
                    "processing_time": asyncio.get_event_loop().time() - start_time
                }
                
                results.append(url_result)
                logger.info(f"Successfully processed {url}")
        
        except Exception as e:
            logger.error(f"Use case execution failed: {e}")
            
            execution_time = asyncio.get_event_loop().time() - start_time
            result = UseCaseResult(
                use_case=use_case,
                success=False,
                results=results,
                execution_time=execution_time,
                error_message=str(e)
            )
            
            self.results_history.append(result)
            return result
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        result = UseCaseResult(
            use_case=use_case,
            success=True,
            results=results,
            execution_time=execution_time
        )
        
        self.results_history.append(result)
        logger.info(f"Use case '{use_case.name}' completed successfully in {execution_time:.2f}s")
        
        return result
    
    async def execute_all_use_cases(self, max_concurrent: int = 3) -> List[UseCaseResult]:
        """
        Execute all use cases concurrently.
        
        Args:
            max_concurrent: Maximum concurrent executions
            
        Returns:
            List of use case results
        """
        logger.info(f"Executing all {len(self.use_cases)} use cases with max concurrency {max_concurrent}")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_execute(use_case_name):
            async with semaphore:
                return await self.execute_use_case(use_case_name)
        
        # Execute all use cases concurrently
        results = await asyncio.gather(
            *[bounded_execute(name) for name in self.use_cases.keys()],
            return_exceptions=True
        )
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                use_case_name = list(self.use_cases.keys())[i]
                use_case = self.use_cases[use_case_name]
                
                error_result = UseCaseResult(
                    use_case=use_case,
                    success=False,
                    results=[],
                    execution_time=0.0,
                    error_message=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        successful = sum(1 for r in processed_results if r.success)
        logger.info(f"All use cases completed: {successful}/{len(processed_results)} successful")
        
        return processed_results
    
    def generate_use_case_report(self, results: List[UseCaseResult] = None) -> Dict[str, Any]:
        """
        Generate comprehensive report of use case executions.
        
        Args:
            results: Optional specific results to report on
            
        Returns:
            Comprehensive report dictionary
        """
        if results is None:
            results = self.results_history
        
        if not results:
            return {"error": "No execution results available"}
        
        # Calculate summary statistics
        total_executions = len(results)
        successful_executions = sum(1 for r in results if r.success)
        failed_executions = total_executions - successful_executions
        
        execution_times = [r.execution_time for r in results if r.success]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # Analyze use case performance
        use_case_performance = {}
        for result in results:
            use_case_name = result.use_case.name
            if use_case_name not in use_case_performance:
                use_case_performance[use_case_name] = {
                    "executions": 0,
                    "successes": 0,
                    "failures": 0,
                    "avg_execution_time": 0,
                    "total_urls_processed": 0
                }
            
            perf = use_case_performance[use_case_name]
            perf["executions"] += 1
            
            if result.success:
                perf["successes"] += 1
                perf["avg_execution_time"] = (
                    (perf["avg_execution_time"] * (perf["successes"] - 1) + result.execution_time) / 
                    perf["successes"]
                )
                perf["total_urls_processed"] += len(result.results)
            else:
                perf["failures"] += 1
        
        # Generate insights and recommendations
        insights = self._generate_insights(results)
        
        report = {
            "execution_summary": {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": failed_executions,
                "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
                "average_execution_time": avg_execution_time
            },
            "use_case_performance": use_case_performance,
            "detailed_results": [asdict(r) for r in results],
            "insights_and_recommendations": insights,
            "report_generated": datetime.now().isoformat()
        }
        
        return report
    
    def _generate_insights(self, results: List[UseCaseResult]) -> Dict[str, Any]:
        """Generate insights from execution results."""
        insights = {
            "performance_insights": [],
            "reliability_insights": [],
            "recommendations": []
        }
        
        # Performance insights
        execution_times = [r.execution_time for r in results if r.success]
        if execution_times:
            fastest = min(execution_times)
            slowest = max(execution_times)
            
            insights["performance_insights"].append(
                f"Execution times range from {fastest:.2f}s to {slowest:.2f}s"
            )
            
            if slowest > fastest * 3:  # Significant variation
                insights["recommendations"].append(
                    "Consider optimizing slower use cases or implementing caching"
                )
        
        # Reliability insights
        failed_results = [r for r in results if not r.success]
        if failed_results:
            common_errors = {}
            for result in failed_results:
                error_type = type(result.error_message).__name__ if result.error_message else "Unknown"
                common_errors[error_type] = common_errors.get(error_type, 0) + 1
            
            insights["reliability_insights"].append(
                f"Most common error types: {dict(sorted(common_errors.items(), key=lambda x: x[1], reverse=True))}"
            )
            
            if len(failed_results) > len(results) * 0.2:  # More than 20% failure rate
                insights["recommendations"].append(
                    "High failure rate detected - review error handling and retry mechanisms"
                )
        
        # Use case specific insights
        use_case_results = {}
        for result in results:
            use_case_name = result.use_case.name
            if use_case_name not in use_case_results:
                use_case_results[use_case_name] = []
            use_case_results[use_case_name].append(result)
        
        for use_case_name, case_results in use_case_results.items():
            success_rate = sum(1 for r in case_results if r.success) / len(case_results)
            if success_rate < 0.8:  # Less than 80% success rate
                insights["recommendations"].append(
                    f"Use case '{use_case_name}' has low success rate ({success_rate:.1%}) - needs optimization"
                )
        
        return insights
    
    def export_results(self, format: str = "json", filename: str = None) -> str:
        """
        Export use case results to file.
        
        Args:
            format: Export format ('json' or 'csv')
            filename: Optional filename
            
        Returns:
            Filename of exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"use_case_results_{timestamp}.{format}"
        
        if format == "json":
            report = self.generate_use_case_report()
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        elif format == "csv":
            # Flatten results for CSV export
            rows = []
            for result in self.results_history:
                base_row = {
                    "use_case_name": result.use_case.name,
                    "use_case_description": result.use_case.description,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "error_message": result.error_message,
                    "timestamp": result.timestamp,
                    "total_urls": len(result.use_case.urls)
                }
                
                if result.results:
                    for i, url_result in enumerate(result.results):
                        row = base_row.copy()
                        row.update({
                            "url_index": i,
                            "url": url_result.get("url"),
                            "url_success": url_result.get("success"),
                            "url_processing_time": url_result.get("processing_time")
                        })
                        rows.append(row)
                else:
                    rows.append(base_row)
            
            with open(filename, 'w', newline='') as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
        
        logger.info(f"Results exported to {filename}")
        return filename
    
    def list_available_use_cases(self) -> Dict[str, str]:
        """List all available use cases with descriptions."""
        return {name: case.description for name, case in self.use_cases.items()}

async def main():
    """Main demonstration of common use cases application."""
    print("🌐 Common Use Cases Application Demo")
    print("=" * 60)
    
    # Initialize application
    app = CommonUseCasesApp()
    
    # List available use cases
    print("📋 Available Use Cases:")
    use_cases = app.list_available_use_cases()
    for i, (name, description) in enumerate(use_cases.items(), 1):
        print(f"   {i}. {name}")
        print(f"      {description}")
        print()
    
    # Interactive mode
    print("Choose execution mode:")
    print("1. Execute all use cases")
    print("2. Execute specific use case")
    print("3. Execute selected use cases")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\n🚀 Executing all use cases...")
            results = await app.execute_all_use_cases(max_concurrent=2)
            
        elif choice == "2":
            print("\nAvailable use cases:")
            use_case_names = list(app.use_cases.keys())
            for i, name in enumerate(use_case_names, 1):
                print(f"   {i}. {name}")
            
            try:
                selection = int(input("\nSelect use case number: ")) - 1
                if 0 <= selection < len(use_case_names):
                    use_case_name = use_case_names[selection]
                    print(f"\n🔄 Executing use case: {use_case_name}")
                    result = await app.execute_use_case(use_case_name)
                    results = [result]
                else:
                    print("❌ Invalid selection")
                    return
            except ValueError:
                print("❌ Invalid input")
                return
                
        elif choice == "3":
            print("\nSelect multiple use cases (comma-separated numbers):")
            use_case_names = list(app.use_cases.keys())
            for i, name in enumerate(use_case_names, 1):
                print(f"   {i}. {name}")
            
            try:
                selections = input("\nEnter numbers (e.g., 1,3,5): ").strip().split(',')
                selected_names = []
                
                for sel in selections:
                    idx = int(sel.strip()) - 1
                    if 0 <= idx < len(use_case_names):
                        selected_names.append(use_case_names[idx])
                
                if selected_names:
                    print(f"\n🔄 Executing {len(selected_names)} selected use cases...")
                    results = []
                    for name in selected_names:
                        result = await app.execute_use_case(name)
                        results.append(result)
                else:
                    print("❌ No valid selections")
                    return
            except ValueError:
                print("❌ Invalid input format")
                return
        else:
            print("❌ Invalid choice")
            return
    
    except KeyboardInterrupt:
        print("\n⚠️ Execution interrupted by user")
        return
    
    # Display results summary
    print("\n📊 Execution Results Summary:")
    print("-" * 50)
    
    successful = sum(1 for r in results if r.success)
    total = len(results)
    
    print(f"Total use cases executed: {total}")
    print(f"Successful executions: {successful}")
    print(f"Failed executions: {total - successful}")
    print(f"Success rate: {successful/total:.1%}")
    
    # Show individual results
    print(f"\n📋 Individual Results:")
    for i, result in enumerate(results, 1):
        status = "✅" if result.success else "❌"
        print(f"{status} {i}. {result.use_case.name}")
        print(f"   Execution time: {result.execution_time:.2f}s")
        print(f"   URLs processed: {len(result.results)}")
        
        if result.success:
            print(f"   Results preview: {str(result.results[0].get('content', ''))[:100]}...")
        else:
            print(f"   Error: {result.error_message}")
        print()
    
    # Generate comprehensive report
    print("📈 Generating comprehensive report...")
    report = app.generate_use_case_report(results)
    
    # Show insights
    if "insights_and_recommendations" in report:
        insights = report["insights_and_recommendations"]
        
        if insights.get("performance_insights"):
            print("⚡ Performance Insights:")
            for insight in insights["performance_insights"]:
                print(f"   • {insight}")
        
        if insights.get("reliability_insights"):
            print("\n🛡️ Reliability Insights:")
            for insight in insights["reliability_insights"]:
                print(f"   • {insight}")
        
        if insights.get("recommendations"):
            print("\n💡 Recommendations:")
            for rec in insights["recommendations"]:
                print(f"   • {rec}")
    
    # Export results
    print(f"\n💾 Exporting results...")
    json_file = app.export_results("json")
    csv_file = app.export_results("csv")
    
    print(f"   JSON report: {json_file}")
    print(f"   CSV data: {csv_file}")
    
    print("\n✅ Common use cases application demo completed!")
    return results

if __name__ == "__main__":
    # Run the common use cases demonstration
    results = asyncio.run(main())