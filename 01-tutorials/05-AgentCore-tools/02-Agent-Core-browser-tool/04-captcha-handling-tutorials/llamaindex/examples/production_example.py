"""
Production-Ready LlamaIndex CAPTCHA Handling Example

This example demonstrates production-ready patterns for CAPTCHA handling
with proper error handling, monitoring, and enterprise features.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import time

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llamaindex_captcha_handler import LlamaIndexCaptchaHandler, CaptchaHandlingResult


@dataclass
class ProductionMetrics:
    """Production metrics for CAPTCHA handling"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    captcha_types_encountered: Dict[str, int] = None
    error_types: Dict[str, int] = None
    
    def __post_init__(self):
        if self.captcha_types_encountered is None:
            self.captcha_types_encountered = {}
        if self.error_types is None:
            self.error_types = {}


class ProductionCaptchaService:
    """Production-ready CAPTCHA handling service"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize production CAPTCHA service
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.logger = self._setup_logging()
        
        # Initialize CAPTCHA handler
        self.captcha_handler = LlamaIndexCaptchaHandler(
            aws_region=self.config.get("aws_region", "us-east-1"),
            bedrock_model=self.config.get("bedrock_model", "anthropic.claude-3-sonnet-20240229-v1:0")
        )
        
        # Production metrics
        self.metrics = ProductionMetrics()
        self.request_history: List[Dict[str, Any]] = []
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            max_requests=self.config.get("max_requests_per_minute", 30),
            time_window=60
        )
        
        print("✅ Production CAPTCHA service initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default production configuration"""
        return {
            "aws_region": "us-east-1",
            "bedrock_model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "max_requests_per_minute": 30,
            "max_retry_attempts": 3,
            "retry_delay_seconds": 2,
            "timeout_seconds": 30,
            "enable_monitoring": True,
            "log_level": "INFO"
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup production logging"""
        logger = logging.getLogger("production_captcha_service")
        logger.setLevel(getattr(logging, self.config.get("log_level", "INFO")))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    async def handle_captcha_with_monitoring(self, page_url: str, request_id: str = None) -> Dict[str, Any]:
        """
        Handle CAPTCHA with full production monitoring
        
        Args:
            page_url: URL to process
            request_id: Optional request identifier
            
        Returns:
            Enhanced result with monitoring data
        """
        request_id = request_id or f"req_{int(time.time())}"
        start_time = datetime.now()
        
        self.logger.info(f"Processing CAPTCHA request {request_id} for {page_url}")
        
        # Check rate limiting
        if not self.rate_limiter.allow_request():
            self.logger.warning(f"Rate limit exceeded for request {request_id}")
            return {
                "success": False,
                "error": "Rate limit exceeded",
                "request_id": request_id,
                "timestamp": start_time.isoformat()
            }
        
        try:
            # Execute CAPTCHA handling with retry logic
            result = await self._handle_with_retry(page_url, request_id)
            
            # Update metrics
            self._update_metrics(result, start_time)
            
            # Log request
            self._log_request(request_id, page_url, result, start_time)
            
            # Enhanced result with production metadata
            enhanced_result = {
                **asdict(result),
                "request_id": request_id,
                "timestamp": start_time.isoformat(),
                "service_version": "1.0.0",
                "processing_node": "production-node-1"
            }
            
            self.logger.info(f"Request {request_id} completed successfully: {result.success}")
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Request {request_id} failed: {str(e)}")
            
            # Update error metrics
            self.metrics.failed_requests += 1
            self.metrics.error_types[type(e).__name__] = self.metrics.error_types.get(type(e).__name__, 0) + 1
            
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "request_id": request_id,
                "timestamp": start_time.isoformat()
            }
    
    async def _handle_with_retry(self, page_url: str, request_id: str) -> CaptchaHandlingResult:
        """Handle CAPTCHA with retry logic"""
        max_attempts = self.config.get("max_retry_attempts", 3)
        retry_delay = self.config.get("retry_delay_seconds", 2)
        
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                self.logger.debug(f"Attempt {attempt + 1}/{max_attempts} for request {request_id}")
                
                # Add timeout
                result = await asyncio.wait_for(
                    self.captcha_handler.handle_captcha_on_page(page_url),
                    timeout=self.config.get("timeout_seconds", 30)
                )
                
                if result.success:
                    if attempt > 0:
                        self.logger.info(f"Request {request_id} succeeded on attempt {attempt + 1}")
                    return result
                
                # If not successful but no exception, treat as retriable
                last_error = Exception(result.error_message or "CAPTCHA handling failed")
                
            except asyncio.TimeoutError:
                last_error = Exception("Request timeout")
                self.logger.warning(f"Timeout on attempt {attempt + 1} for request {request_id}")
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Error on attempt {attempt + 1} for request {request_id}: {str(e)}")
            
            # Wait before retry (except on last attempt)
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
        
        # All attempts failed
        raise last_error or Exception("All retry attempts failed")
    
    def _update_metrics(self, result: CaptchaHandlingResult, start_time: datetime) -> None:
        """Update production metrics"""
        self.metrics.total_requests += 1
        
        if result.success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
        
        # Update response time
        processing_time = result.processing_time
        self.metrics.avg_response_time = (
            (self.metrics.avg_response_time * (self.metrics.total_requests - 1) + processing_time) /
            self.metrics.total_requests
        )
        
        # Update CAPTCHA type counts
        if result.captcha_type:
            self.metrics.captcha_types_encountered[result.captcha_type] = (
                self.metrics.captcha_types_encountered.get(result.captcha_type, 0) + 1
            )
    
    def _log_request(self, request_id: str, page_url: str, result: CaptchaHandlingResult, start_time: datetime) -> None:
        """Log request for audit and monitoring"""
        log_entry = {
            "request_id": request_id,
            "page_url": page_url,
            "timestamp": start_time.isoformat(),
            "success": result.success,
            "captcha_type": result.captcha_type,
            "processing_time": result.processing_time,
            "confidence_score": result.confidence_score
        }
        
        self.request_history.append(log_entry)
        
        # Keep only recent history (last 1000 requests)
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        success_rate = (
            (self.metrics.successful_requests / self.metrics.total_requests * 100)
            if self.metrics.total_requests > 0 else 0
        )
        
        return {
            "status": "healthy" if success_rate > 80 else "degraded" if success_rate > 50 else "unhealthy",
            "success_rate": success_rate,
            "total_requests": self.metrics.total_requests,
            "avg_response_time": self.metrics.avg_response_time,
            "uptime": "production",  # Would calculate actual uptime
            "rate_limit_status": self.rate_limiter.get_status(),
            "last_updated": datetime.now().isoformat()
        }
    
    def get_metrics_report(self) -> Dict[str, Any]:
        """Get comprehensive metrics report"""
        return {
            "summary": asdict(self.metrics),
            "health": self.get_health_status(),
            "recent_requests": self.request_history[-10:],  # Last 10 requests
            "configuration": {
                "max_requests_per_minute": self.config.get("max_requests_per_minute"),
                "max_retry_attempts": self.config.get("max_retry_attempts"),
                "timeout_seconds": self.config.get("timeout_seconds")
            }
        }
    
    async def run_production_simulation(self) -> None:
        """Run production simulation with multiple concurrent requests"""
        print("🏭 Running Production CAPTCHA Service Simulation")
        print("=" * 55)
        
        # Simulate production load
        test_urls = [
            "https://example.com/login",
            "https://example.com/register", 
            "https://example.com/contact",
            "https://example.com/checkout",
            "https://example.com/support"
        ]
        
        # Create concurrent requests
        tasks = []
        for i in range(15):  # Simulate 15 concurrent requests
            url = test_urls[i % len(test_urls)]
            request_id = f"prod_req_{i+1:03d}"
            
            task = self.handle_captcha_with_monitoring(url, request_id)
            tasks.append(task)
        
        print(f"🚀 Executing {len(tasks)} concurrent CAPTCHA requests...")
        
        # Execute all requests concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        execution_time = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed_results = [r for r in results if isinstance(r, dict) and not r.get("success")]
        exceptions = [r for r in results if isinstance(r, Exception)]
        
        print(f"\n📊 Production Simulation Results")
        print("=" * 40)
        print(f"Total Requests: {len(tasks)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")
        print(f"Exceptions: {len(exceptions)}")
        print(f"Success Rate: {(len(successful_results)/len(tasks))*100:.1f}%")
        print(f"Total Execution Time: {execution_time:.2f}s")
        print(f"Requests per Second: {len(tasks)/execution_time:.2f}")
        
        # Show health status
        health = self.get_health_status()
        print(f"\n🏥 Service Health: {health['status'].upper()}")
        print(f"Average Response Time: {health['avg_response_time']:.2f}s")
        
        # Show metrics report
        print(f"\n📈 Detailed Metrics Report")
        metrics_report = self.get_metrics_report()
        print(json.dumps(metrics_report, indent=2, default=str))


class RateLimiter:
    """Simple rate limiter for production use"""
    
    def __init__(self, max_requests: int, time_window: int):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def allow_request(self) -> bool:
        """Check if request is allowed under rate limit"""
        now = time.time()
        
        # Remove old requests outside time window
        self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
        
        # Check if under limit
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get rate limiter status"""
        now = time.time()
        recent_requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
        
        return {
            "current_requests": len(recent_requests),
            "max_requests": self.max_requests,
            "time_window": self.time_window,
            "requests_remaining": max(0, self.max_requests - len(recent_requests))
        }


async def main():
    """Main function to run production example"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create production service
        config = {
            "max_requests_per_minute": 20,
            "max_retry_attempts": 2,
            "timeout_seconds": 15,
            "log_level": "INFO"
        }
        
        service = ProductionCaptchaService(config)
        
        # Run production simulation
        await service.run_production_simulation()
        
        print("\n✅ Production example completed successfully!")
        
    except Exception as e:
        print(f"❌ Production example failed: {str(e)}")
        logging.error(f"Production example execution failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())