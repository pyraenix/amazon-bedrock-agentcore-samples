# LlamaIndex AgentCore Browser Integration Tutorial

## ⚠️ Educational Content Notice

**This tutorial uses safe test endpoints for educational purposes.**

- **Learning Environment**: Examples use `httpbin.org`, `example.com`, and demo sites
- **Production Guidance**: See [Real-World Implementations Guide](REAL_WORLD_IMPLEMENTATIONS.md) for production examples
- **Legal Compliance**: Always check robots.txt, terms of service, and applicable laws before automating real websites
- **Best Practices**: Implement rate limiting, respect website resources, and prefer official APIs

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Basic Setup](#basic-setup)
4. [Your First Web Automation](#your-first-web-automation)
5. [Understanding the Components](#understanding-the-components)
6. [Building a Complete Application](#building-a-complete-application)
7. [Advanced Features](#advanced-features)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Next Steps](#next-steps)

## Prerequisites

Before starting this tutorial, ensure you have:

- **Python 3.12** installed
- **AWS Account** with appropriate permissions
- **Basic knowledge** of Python async/await
- **Understanding** of web technologies (HTML, CSS selectors)
- **Familiarity** with LlamaIndex concepts (optional but helpful)

### Required AWS Permissions

Your AWS credentials need the following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock-agentcore:CreateBrowserSession",
                "bedrock-agentcore:InvokeBrowserTool",
                "bedrock-agentcore:CloseBrowserSession",
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "*"
        }
    ]
}
```

## Installation

### Step 1: Create a Virtual Environment

```bash
# Create a new directory for your project
mkdir llamaindex-browser-tutorial
cd llamaindex-browser-tutorial

# Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Install the integration package and dependencies
pip install llama-index-core
pip install llama-index-llms-bedrock
pip install llama-index-multi-modal-llms-bedrock
pip install boto3
pip install aiohttp
pip install pydantic
pip install pyyaml

# Install additional dependencies for examples
pip install pillow  # For image processing
pip install pandas  # For data manipulation
```

### Step 3: Download the Integration Code

```bash
# Clone or download the integration code
# (In a real scenario, this would be a pip package)
git clone <repository-url>
cd llamaindex-agentcore-browser-integration
```

## Basic Setup

### Step 1: Configure AWS Credentials

Choose one of these methods:

**Option A: AWS CLI Configuration**
```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and region
```

**Option B: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

**Option C: IAM Roles (for EC2/Lambda)**
```python
# No additional configuration needed - uses instance role
```

### Step 2: Create Configuration File

Create a `config.yaml` file:

```yaml
# config.yaml
aws:
  region: "us-east-1"
  # Credentials will be loaded from environment or AWS CLI

agentcore:
  browser_tool_endpoint: "https://agentcore.amazonaws.com"
  session_timeout: 300
  max_concurrent_sessions: 3

browser:
  headless: true
  viewport_width: 1920
  viewport_height: 1080
  user_agent: "LlamaIndex-AgentCore/1.0"
  enable_javascript: true
  enable_images: true
  page_load_timeout: 30

llamaindex:
  llm_model: "anthropic.claude-3-sonnet-20240229-v1:0"
  vision_model: "anthropic.claude-3-sonnet-20240229-v1:0"
  temperature: 0.1
  max_tokens: 4096

security:
  enable_pii_scrubbing: true
  log_sensitive_data: false

monitoring:
  enable_metrics: true
  log_level: "INFO"
```

### Step 3: Verify Setup

Create a test script `test_setup.py`:

```python
"""
Test script to verify the integration setup.
"""
import asyncio
import sys
from pathlib import Path

# Add the integration to Python path
sys.path.append(str(Path(__file__).parent))

from integration import LlamaIndexAgentCoreIntegration

async def test_setup():
    try:
        print("🔧 Testing LlamaIndex AgentCore Browser Integration setup...")
        
        # Initialize integration
        integration = LlamaIndexAgentCoreIntegration(config_path="config.yaml")
        print("✅ Integration initialized successfully")
        
        # Test configuration
        config = integration.config_manager.get_config()
        print(f"✅ Configuration loaded: AWS region = {config.aws.region}")
        
        # Test agent creation
        agent = integration.create_agent()
        print("✅ Agent created successfully")
        
        # Test basic functionality
        response = await agent.achat("Hello! Can you confirm that you have browser tools available?")
        print(f"✅ Agent response: {response.response[:100]}...")
        
        print("\n🎉 Setup verification complete! You're ready to start building.")
        return True
        
    except Exception as e:
        print(f"❌ Setup verification failed: {e}")
        print("\n🔍 Troubleshooting tips:")
        print("1. Check your AWS credentials")
        print("2. Verify your configuration file")
        print("3. Ensure you have the required permissions")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_setup())
    sys.exit(0 if success else 1)
```

Run the test:
```bash
python test_setup.py
```

## Your First Web Automation

Let's create your first web automation script that demonstrates the core capabilities.

### Step 1: Simple Web Scraping

Create `first_automation.py`:

```python
"""
Your first web automation with LlamaIndex AgentCore.
"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from integration import LlamaIndexAgentCoreIntegration

async def first_automation():
    print("🚀 Starting your first web automation...")
    
    # Initialize the integration
    integration = LlamaIndexAgentCoreIntegration(config_path="config.yaml")
    agent = integration.create_agent()
    
    # Simple web scraping task
    print("📄 Extracting content from a web page...")
    response = await agent.achat("""
    Navigate to https://httpbin.org/html and extract:
    1. The page title
    2. All heading text (h1, h2, etc.)
    3. The main paragraph content
    4. Any links found on the page
    
    Present the information in a clear, organized format.
    """)
    
    print("📊 Results:")
    print("-" * 50)
    print(response.response)
    print("-" * 50)
    
    return response.response

if __name__ == "__main__":
    result = asyncio.run(first_automation())
    print("✅ First automation complete!")
```

### Step 2: Interactive Form Filling

Create `form_automation.py`:

```python
"""
Demonstrate form interaction capabilities.
"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from integration import LlamaIndexAgentCoreIntegration

async def form_automation():
    print("📝 Demonstrating form automation...")
    
    integration = LlamaIndexAgentCoreIntegration(config_path="config.yaml")
    agent = integration.create_agent()
    
    # Form filling and submission
    response = await agent.achat("""
    Navigate to https://httpbin.org/forms/post and complete the following tasks:
    
    1. Fill out the pizza order form with these details:
       - Customer name: John Doe
       - Telephone: +1-555-123-4567
       - Email: john.doe@example.com
       - Size: Large
       - Toppings: Pepperoni, Mushrooms, Extra Cheese
       - Delivery time: ASAP
       - Comments: Please ring the doorbell twice
    
    2. Before submitting, take a screenshot of the filled form
    
    3. Submit the form
    
    4. Analyze the response and confirm successful submission
    
    Provide detailed feedback on each step.
    """)
    
    print("📋 Form Automation Results:")
    print("-" * 50)
    print(response.response)
    print("-" * 50)
    
    return response.response

if __name__ == "__main__":
    result = asyncio.run(form_automation())
    print("✅ Form automation complete!")
```

### Step 3: Screenshot and Visual Analysis

Create `visual_analysis.py`:

```python
"""
Demonstrate screenshot capture and visual analysis.
"""
import asyncio
import base64
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from integration import LlamaIndexAgentCoreIntegration

async def visual_analysis():
    print("📸 Demonstrating visual analysis capabilities...")
    
    integration = LlamaIndexAgentCoreIntegration(config_path="config.yaml")
    agent = integration.create_agent()
    
    # Visual analysis task
    response = await agent.achat("""
    Navigate to https://example.com and perform visual analysis:
    
    1. Take a screenshot of the entire page
    2. Analyze the visual layout and design
    3. Identify key visual elements (headers, buttons, images, etc.)
    4. Describe the color scheme and overall aesthetic
    5. Comment on the user experience and accessibility
    
    Provide both technical details and subjective observations about the design.
    """)
    
    print("🎨 Visual Analysis Results:")
    print("-" * 50)
    print(response.response)
    print("-" * 50)
    
    # Save screenshot if available
    if hasattr(response, 'metadata') and 'screenshot' in response.metadata:
        try:
            screenshot_data = response.metadata['screenshot']
            with open('example_screenshot.png', 'wb') as f:
                f.write(base64.b64decode(screenshot_data))
            print("💾 Screenshot saved as 'example_screenshot.png'")
        except Exception as e:
            print(f"⚠️ Could not save screenshot: {e}")
    
    return response.response

if __name__ == "__main__":
    result = asyncio.run(visual_analysis())
    print("✅ Visual analysis complete!")
```

## Understanding the Components

### Architecture Overview

The integration consists of several key components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                         │
├─────────────────────────────────────────────────────────────┤
│                LlamaIndex Agent Layer                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   ReActAgent    │  │  Browser Tools  │  │ Vision LLM  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                Integration Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Config Manager  │  │ Error Handler   │  │ Response    │ │
│  │                 │  │                 │  │ Parser      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                AgentCore Client Layer                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Browser Client  │  │ Session Manager │  │ Auth Handler│ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                AWS AgentCore Service                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Browser Tool    │  │ VM Isolation    │  │ Security    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Key Components Explained

#### 1. LlamaIndex Agent
- **Purpose**: Provides intelligent orchestration and decision-making
- **Capabilities**: Natural language understanding, tool selection, workflow management
- **Models**: Uses Bedrock LLMs for reasoning and vision models for image analysis

#### 2. Browser Tools
- **BrowserNavigationTool**: Handles page navigation and URL management
- **TextExtractionTool**: Extracts and processes text content
- **ScreenshotCaptureTool**: Captures visual content for analysis
- **ElementClickTool**: Interacts with clickable elements
- **FormInteractionTool**: Handles form filling and submission
- **CaptchaDetectionTool**: Detects and analyzes CAPTCHAs

#### 3. AgentCore Browser Client
- **Session Management**: Creates and manages browser sessions
- **API Communication**: Handles communication with AgentCore service
- **Error Handling**: Manages retries and error recovery
- **Response Processing**: Parses and formats responses

### Data Flow Example

```python
# 1. User provides natural language instruction
instruction = "Navigate to example.com and extract the main heading"

# 2. LlamaIndex agent processes instruction
agent_response = await agent.achat(instruction)

# 3. Agent selects appropriate tools
# - BrowserNavigationTool for navigation
# - TextExtractionTool for content extraction

# 4. Tools make API calls to AgentCore
browser_response = await client.navigate("https://example.com")
text_response = await client.extract_text(selector="h1")

# 5. AgentCore executes in VM-isolated browser
# - Creates secure browser session
# - Performs requested operations
# - Returns structured responses

# 6. Integration processes responses
parsed_data = response_parser.parse(browser_response)

# 7. Agent synthesizes final response
final_response = agent.synthesize_response(parsed_data)
```

## Building a Complete Application

Let's build a comprehensive web scraping application that demonstrates all major features.

### Step 1: Project Structure

Create the following directory structure:

```
web_scraper_app/
├── config.yaml
├── main.py
├── scraper/
│   ├── __init__.py
│   ├── core.py
│   ├── processors.py
│   └── exporters.py
├── examples/
│   ├── news_scraper.py
│   ├── ecommerce_scraper.py
│   └── social_media_analyzer.py
└── tests/
    ├── test_scraper.py
    └── test_integration.py
```

### Step 2: Core Scraper Implementation

Create `scraper/core.py`:

```python
"""
Core web scraper implementation using LlamaIndex AgentCore.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from integration import LlamaIndexAgentCoreIntegration

logger = logging.getLogger(__name__)

@dataclass
class ScrapingTask:
    """Represents a web scraping task."""
    url: str
    task_type: str  # 'content', 'form', 'analysis', 'monitoring'
    instructions: str
    metadata: Dict[str, Any] = None
    priority: int = 1

@dataclass
class ScrapingResult:
    """Represents the result of a scraping task."""
    task: ScrapingTask
    success: bool
    data: Any
    error: Optional[str] = None
    timestamp: str = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class WebScraper:
    """Advanced web scraper using LlamaIndex AgentCore integration."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.integration = LlamaIndexAgentCoreIntegration(config_path=config_path)
        self.agent = self.integration.create_agent()
        self.results_history = []
    
    async def scrape_single(self, task: ScrapingTask) -> ScrapingResult:
        """Execute a single scraping task."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Starting task: {task.task_type} for {task.url}")
            
            # Execute the scraping task
            response = await self.agent.achat(
                f"URL: {task.url}\n"
                f"Task Type: {task.task_type}\n"
                f"Instructions: {task.instructions}\n"
                f"Additional Context: {json.dumps(task.metadata or {})}"
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            result = ScrapingResult(
                task=task,
                success=True,
                data=response.response,
                processing_time=processing_time
            )
            
            # Store additional metadata if available
            if hasattr(response, 'metadata'):
                result.data = {
                    "content": response.response,
                    "metadata": response.metadata
                }
            
            logger.info(f"Task completed successfully in {processing_time:.2f}s")
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Task failed: {e}")
            
            result = ScrapingResult(
                task=task,
                success=False,
                data=None,
                error=str(e),
                processing_time=processing_time
            )
        
        self.results_history.append(result)
        return result
    
    async def scrape_batch(self, tasks: List[ScrapingTask], 
                          max_concurrent: int = 3) -> List[ScrapingResult]:
        """Execute multiple scraping tasks concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_scrape(task):
            async with semaphore:
                return await self.scrape_single(task)
        
        logger.info(f"Starting batch scraping of {len(tasks)} tasks")
        
        # Execute tasks concurrently
        results = await asyncio.gather(
            *[bounded_scrape(task) for task in tasks],
            return_exceptions=True
        )
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ScrapingResult(
                    task=tasks[i],
                    success=False,
                    data=None,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        successful = sum(1 for r in processed_results if r.success)
        logger.info(f"Batch completed: {successful}/{len(tasks)} successful")
        
        return processed_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scraping statistics."""
        if not self.results_history:
            return {"total_tasks": 0}
        
        total_tasks = len(self.results_history)
        successful_tasks = sum(1 for r in self.results_history if r.success)
        failed_tasks = total_tasks - successful_tasks
        
        processing_times = [r.processing_time for r in self.results_history if r.success]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        task_types = {}
        for result in self.results_history:
            task_type = result.task.task_type
            task_types[task_type] = task_types.get(task_type, 0) + 1
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "average_processing_time": avg_processing_time,
            "task_types": task_types
        }
    
    def export_results(self, format: str = "json", filename: str = None) -> str:
        """Export results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scraping_results_{timestamp}.{format}"
        
        if format == "json":
            data = [asdict(result) for result in self.results_history]
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        elif format == "csv":
            import pandas as pd
            
            # Flatten results for CSV
            rows = []
            for result in self.results_history:
                row = {
                    "url": result.task.url,
                    "task_type": result.task.task_type,
                    "success": result.success,
                    "error": result.error,
                    "timestamp": result.timestamp,
                    "processing_time": result.processing_time
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(filename, index=False)
        
        logger.info(f"Results exported to {filename}")
        return filename
```

### Step 3: Specialized Processors

Create `scraper/processors.py`:

```python
"""
Specialized processors for different types of content.
"""
import re
import json
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class ProcessedContent:
    """Represents processed content with metadata."""
    original_content: str
    processed_content: Any
    content_type: str
    metadata: Dict[str, Any]

class ContentProcessor:
    """Base class for content processors."""
    
    def process(self, content: str, metadata: Dict[str, Any] = None) -> ProcessedContent:
        raise NotImplementedError

class NewsArticleProcessor(ContentProcessor):
    """Processor for news articles."""
    
    def process(self, content: str, metadata: Dict[str, Any] = None) -> ProcessedContent:
        # Extract article components
        lines = content.split('\n')
        
        # Simple heuristics for article structure
        headline = None
        author = None
        date = None
        body_paragraphs = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for headline (usually first significant line)
            if headline is None and len(line) > 20:
                headline = line
            
            # Look for author patterns
            elif re.search(r'by\s+[\w\s]+', line, re.IGNORECASE):
                author = line
            
            # Look for date patterns
            elif re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},\s+\d{4}', line):
                date = line
            
            # Collect body paragraphs
            elif len(line) > 50:  # Assume substantial lines are body content
                body_paragraphs.append(line)
        
        processed = {
            "headline": headline,
            "author": author,
            "date": date,
            "body": '\n'.join(body_paragraphs),
            "word_count": len(' '.join(body_paragraphs).split()),
            "paragraph_count": len(body_paragraphs)
        }
        
        return ProcessedContent(
            original_content=content,
            processed_content=processed,
            content_type="news_article",
            metadata=metadata or {}
        )

class ProductListProcessor(ContentProcessor):
    """Processor for e-commerce product listings."""
    
    def process(self, content: str, metadata: Dict[str, Any] = None) -> ProcessedContent:
        # Extract product information using patterns
        products = []
        
        # Simple pattern matching for product data
        # In a real implementation, this would be more sophisticated
        lines = content.split('\n')
        current_product = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_product:
                    products.append(current_product)
                    current_product = {}
                continue
            
            # Look for price patterns
            price_match = re.search(r'\$[\d,]+\.?\d*', line)
            if price_match:
                current_product['price'] = price_match.group()
            
            # Look for product names (heuristic: lines with certain characteristics)
            elif len(line) > 10 and len(line) < 100 and not price_match:
                if 'name' not in current_product:
                    current_product['name'] = line
                else:
                    current_product['description'] = line
        
        # Add the last product if exists
        if current_product:
            products.append(current_product)
        
        processed = {
            "products": products,
            "product_count": len(products),
            "average_price": self._calculate_average_price(products)
        }
        
        return ProcessedContent(
            original_content=content,
            processed_content=processed,
            content_type="product_list",
            metadata=metadata or {}
        )
    
    def _calculate_average_price(self, products: List[Dict]) -> float:
        """Calculate average price from products."""
        prices = []
        for product in products:
            if 'price' in product:
                # Extract numeric value from price string
                price_str = re.sub(r'[^\d.]', '', product['price'])
                try:
                    prices.append(float(price_str))
                except ValueError:
                    continue
        
        return sum(prices) / len(prices) if prices else 0.0

class SocialMediaProcessor(ContentProcessor):
    """Processor for social media content."""
    
    def process(self, content: str, metadata: Dict[str, Any] = None) -> ProcessedContent:
        # Extract social media metrics and content
        posts = []
        hashtags = set()
        mentions = set()
        
        # Find hashtags
        hashtag_pattern = r'#\w+'
        hashtags.update(re.findall(hashtag_pattern, content, re.IGNORECASE))
        
        # Find mentions
        mention_pattern = r'@\w+'
        mentions.update(re.findall(mention_pattern, content, re.IGNORECASE))
        
        # Extract engagement metrics (likes, shares, comments)
        engagement_patterns = {
            'likes': r'(\d+)\s*likes?',
            'shares': r'(\d+)\s*shares?',
            'comments': r'(\d+)\s*comments?'
        }
        
        engagement = {}
        for metric, pattern in engagement_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                engagement[metric] = sum(int(match) for match in matches)
        
        processed = {
            "hashtags": list(hashtags),
            "mentions": list(mentions),
            "engagement": engagement,
            "hashtag_count": len(hashtags),
            "mention_count": len(mentions),
            "total_engagement": sum(engagement.values())
        }
        
        return ProcessedContent(
            original_content=content,
            processed_content=processed,
            content_type="social_media",
            metadata=metadata or {}
        )

# Processor factory
PROCESSORS = {
    "news": NewsArticleProcessor(),
    "products": ProductListProcessor(),
    "social": SocialMediaProcessor()
}

def get_processor(content_type: str) -> ContentProcessor:
    """Get appropriate processor for content type."""
    return PROCESSORS.get(content_type, ContentProcessor())
```

### Step 4: Example Applications

Create `examples/news_scraper.py`:

```python
"""
News scraper example application.
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scraper.core import WebScraper, ScrapingTask
from scraper.processors import get_processor

async def news_scraper_example():
    """Demonstrate news scraping capabilities."""
    print("📰 News Scraper Example")
    print("=" * 50)
    
    scraper = WebScraper()
    
    # Define news scraping tasks
    news_tasks = [
        ScrapingTask(
            url="https://example-news-site.com/article1",
            task_type="news_analysis",
            instructions="""
            Extract comprehensive information from this news article:
            1. Headline and subheadline
            2. Author name and publication date
            3. Full article text (main content only)
            4. Any quotes or key statements
            5. Related topics or tags
            6. Article length and reading time estimate
            
            Focus on factual content and ignore ads, navigation, or sidebar content.
            """,
            metadata={"source": "example-news", "category": "technology"}
        ),
        
        ScrapingTask(
            url="https://httpbin.org/html",  # Using httpbin as a safe example
            task_type="content_analysis",
            instructions="""
            Analyze this page as if it were a news article:
            1. Extract the main heading
            2. Identify any structured content
            3. Count paragraphs and estimate reading time
            4. Analyze the overall content structure
            """,
            metadata={"source": "httpbin", "category": "example"}
        )
    ]
    
    # Execute scraping tasks
    print(f"🔄 Processing {len(news_tasks)} news articles...")
    results = await scraper.scrape_batch(news_tasks, max_concurrent=2)
    
    # Process results with news processor
    processor = get_processor("news")
    
    print("\n📊 Results:")
    print("-" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"\nArticle {i}: {result.task.url}")
        
        if result.success:
            # Process the content
            processed = processor.process(result.data)
            
            print(f"✅ Success (processed in {result.processing_time:.2f}s)")
            
            if isinstance(processed.processed_content, dict):
                content = processed.processed_content
                print(f"   Headline: {content.get('headline', 'N/A')}")
                print(f"   Author: {content.get('author', 'N/A')}")
                print(f"   Word Count: {content.get('word_count', 'N/A')}")
                print(f"   Paragraphs: {content.get('paragraph_count', 'N/A')}")
            else:
                print(f"   Content: {str(processed.processed_content)[:100]}...")
        else:
            print(f"❌ Failed: {result.error}")
    
    # Show statistics
    stats = scraper.get_statistics()
    print(f"\n📈 Statistics:")
    print(f"   Total articles processed: {stats['total_tasks']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Average processing time: {stats['average_processing_time']:.2f}s")
    
    # Export results
    filename = scraper.export_results("json", "news_scraping_results.json")
    print(f"💾 Results exported to: {filename}")
    
    return results

if __name__ == "__main__":
    results = asyncio.run(news_scraper_example())
    print("\n✅ News scraper example completed!")
```

Create `examples/ecommerce_scraper.py`:

```python
"""
E-commerce scraper example application.
"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from scraper.core import WebScraper, ScrapingTask
from scraper.processors import get_processor

async def ecommerce_scraper_example():
    """Demonstrate e-commerce scraping capabilities."""
    print("🛒 E-commerce Scraper Example")
    print("=" * 50)
    
    scraper = WebScraper()
    
    # Define e-commerce scraping tasks
    ecommerce_tasks = [
        ScrapingTask(
            url="https://demo.opencart.com/",
            task_type="product_catalog",
            instructions="""
            Navigate to this e-commerce demo site and extract product information:
            
            1. Browse the featured products on the homepage
            2. For each product, extract:
               - Product name
               - Price (original and sale price if applicable)
               - Product image URL
               - Brief description
               - Availability status
               - Customer rating if visible
            
            3. Handle any popups or cookie banners
            4. Focus on the main product listings, ignore navigation and ads
            5. Return data in a structured format suitable for analysis
            """,
            metadata={"site": "opencart-demo", "category": "general"}
        ),
        
        ScrapingTask(
            url="https://httpbin.org/json",  # Mock product data
            task_type="product_data",
            instructions="""
            Analyze this JSON data as if it were product information:
            1. Extract any structured data
            2. Identify potential product attributes
            3. Format the information as product listings
            """,
            metadata={"site": "httpbin", "category": "mock-data"}
        )
    ]
    
    print(f"🔄 Processing {len(ecommerce_tasks)} e-commerce sites...")
    results = await scraper.scrape_batch(ecommerce_tasks, max_concurrent=2)
    
    # Process results with product processor
    processor = get_processor("products")
    
    print("\n📊 Results:")
    print("-" * 50)
    
    total_products = 0
    
    for i, result in enumerate(results, 1):
        print(f"\nSite {i}: {result.task.url}")
        
        if result.success:
            # Process the content
            processed = processor.process(result.data)
            
            print(f"✅ Success (processed in {result.processing_time:.2f}s)")
            
            if isinstance(processed.processed_content, dict):
                content = processed.processed_content
                products = content.get('products', [])
                product_count = content.get('product_count', 0)
                avg_price = content.get('average_price', 0)
                
                print(f"   Products found: {product_count}")
                print(f"   Average price: ${avg_price:.2f}")
                
                # Show first few products
                for j, product in enumerate(products[:3], 1):
                    print(f"   Product {j}: {product.get('name', 'N/A')} - {product.get('price', 'N/A')}")
                
                total_products += product_count
            else:
                print(f"   Raw content: {str(processed.processed_content)[:100]}...")
        else:
            print(f"❌ Failed: {result.error}")
    
    # Show statistics
    stats = scraper.get_statistics()
    print(f"\n📈 Statistics:")
    print(f"   Sites processed: {stats['total_tasks']}")
    print(f"   Total products found: {total_products}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Average processing time: {stats['average_processing_time']:.2f}s")
    
    # Export results
    filename = scraper.export_results("json", "ecommerce_scraping_results.json")
    print(f"💾 Results exported to: {filename}")
    
    return results

if __name__ == "__main__":
    results = asyncio.run(ecommerce_scraper_example())
    print("\n✅ E-commerce scraper example completed!")
```

### Step 5: Main Application

Create `main.py`:

```python
"""
Main application demonstrating the complete web scraper.
"""
import asyncio
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from scraper.core import WebScraper, ScrapingTask
from examples.news_scraper import news_scraper_example
from examples.ecommerce_scraper import ecommerce_scraper_example

async def interactive_demo():
    """Interactive demonstration of the web scraper."""
    print("🌐 LlamaIndex AgentCore Web Scraper")
    print("=" * 50)
    
    scraper = WebScraper()
    
    while True:
        print("\nChoose an option:")
        print("1. Quick URL scraping")
        print("2. News scraper demo")
        print("3. E-commerce scraper demo")
        print("4. Custom scraping task")
        print("5. View statistics")
        print("6. Export results")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        
        elif choice == "1":
            url = input("Enter URL to scrape: ").strip()
            if url:
                task = ScrapingTask(
                    url=url,
                    task_type="quick_scrape",
                    instructions="Extract the main content, title, and key information from this page."
                )
                
                print(f"🔄 Scraping {url}...")
                result = await scraper.scrape_single(task)
                
                if result.success:
                    print(f"✅ Success! Content extracted:")
                    print("-" * 30)
                    print(result.data[:500] + "..." if len(str(result.data)) > 500 else result.data)
                else:
                    print(f"❌ Failed: {result.error}")
        
        elif choice == "2":
            print("\n📰 Running news scraper demo...")
            await news_scraper_example()
        
        elif choice == "3":
            print("\n🛒 Running e-commerce scraper demo...")
            await ecommerce_scraper_example()
        
        elif choice == "4":
            print("\n🔧 Custom scraping task")
            url = input("Enter URL: ").strip()
            task_type = input("Enter task type (e.g., 'analysis', 'extraction'): ").strip()
            instructions = input("Enter detailed instructions: ").strip()
            
            if url and task_type and instructions:
                task = ScrapingTask(
                    url=url,
                    task_type=task_type,
                    instructions=instructions
                )
                
                print(f"🔄 Processing custom task...")
                result = await scraper.scrape_single(task)
                
                if result.success:
                    print(f"✅ Success!")
                    print("-" * 30)
                    print(result.data)
                else:
                    print(f"❌ Failed: {result.error}")
        
        elif choice == "5":
            stats = scraper.get_statistics()
            print("\n📈 Scraping Statistics:")
            print("-" * 30)
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.2f}")
                elif isinstance(value, dict):
                    print(f"{key}:")
                    for subkey, subvalue in value.items():
                        print(f"  {subkey}: {subvalue}")
                else:
                    print(f"{key}: {value}")
        
        elif choice == "6":
            format_choice = input("Export format (json/csv): ").strip().lower()
            if format_choice in ["json", "csv"]:
                filename = scraper.export_results(format_choice)
                print(f"💾 Results exported to: {filename}")
            else:
                print("❌ Invalid format. Choose 'json' or 'csv'.")
        
        else:
            print("❌ Invalid choice. Please try again.")

async def batch_demo():
    """Demonstrate batch processing capabilities."""
    print("🚀 Batch Processing Demo")
    print("=" * 50)
    
    scraper = WebScraper()
    
    # Create a variety of tasks
    tasks = [
        ScrapingTask(
            url="https://httpbin.org/html",
            task_type="content_analysis",
            instructions="Extract and analyze the page structure and content."
        ),
        ScrapingTask(
            url="https://httpbin.org/json",
            task_type="data_analysis",
            instructions="Analyze the JSON data structure and content."
        ),
        ScrapingTask(
            url="https://httpbin.org/xml",
            task_type="xml_analysis",
            instructions="Parse and analyze the XML structure and data."
        ),
        ScrapingTask(
            url="https://httpbin.org/robots.txt",
            task_type="robots_analysis",
            instructions="Analyze the robots.txt file and extract crawling rules."
        )
    ]
    
    print(f"🔄 Processing {len(tasks)} tasks concurrently...")
    results = await scraper.scrape_batch(tasks, max_concurrent=3)
    
    print("\n📊 Batch Results:")
    print("-" * 50)
    
    for i, result in enumerate(results, 1):
        status = "✅" if result.success else "❌"
        print(f"{status} Task {i}: {result.task.task_type} - {result.task.url}")
        if result.success:
            print(f"   Processing time: {result.processing_time:.2f}s")
            print(f"   Content preview: {str(result.data)[:100]}...")
        else:
            print(f"   Error: {result.error}")
        print()
    
    # Show final statistics
    stats = scraper.get_statistics()
    print("📈 Final Statistics:")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Average processing time: {stats['average_processing_time']:.2f}s")
    
    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LlamaIndex AgentCore Web Scraper")
    parser.add_argument("--mode", choices=["interactive", "batch", "news", "ecommerce"], 
                       default="interactive", help="Demo mode to run")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        asyncio.run(interactive_demo())
    elif args.mode == "batch":
        asyncio.run(batch_demo())
    elif args.mode == "news":
        asyncio.run(news_scraper_example())
    elif args.mode == "ecommerce":
        asyncio.run(ecommerce_scraper_example())

if __name__ == "__main__":
    main()
```

## Advanced Features

### CAPTCHA Handling

Create `examples/captcha_demo.py`:

```python
"""
Demonstrate CAPTCHA detection and handling capabilities.
"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from scraper.core import WebScraper, ScrapingTask

async def captcha_handling_demo():
    """Demonstrate CAPTCHA handling workflow."""
    print("🔐 CAPTCHA Handling Demo")
    print("=" * 50)
    
    scraper = WebScraper()
    
    # CAPTCHA detection and handling task
    captcha_task = ScrapingTask(
        url="https://www.google.com/recaptcha/api2/demo",
        task_type="captcha_handling",
        instructions="""
        Navigate to this reCAPTCHA demo page and perform comprehensive CAPTCHA analysis:
        
        1. CAPTCHA Detection:
           - Scan the page for any CAPTCHA elements
           - Identify the type of CAPTCHA (reCAPTCHA v2, v3, hCaptcha, etc.)
           - Take a screenshot for visual analysis
           - Analyze DOM structure for CAPTCHA containers
        
        2. CAPTCHA Analysis:
           - Determine if the CAPTCHA is solvable
           - Assess the complexity level
           - Identify any accessibility features
           - Check for audio alternatives
        
        3. Interaction Attempt:
           - Try to interact with the CAPTCHA checkbox if present
           - Document any challenges or image grids that appear
           - Analyze the challenge type (traffic lights, crosswalks, etc.)
        
        4. Reporting:
           - Provide detailed analysis of the CAPTCHA system
           - Report on detection accuracy and confidence
           - Suggest strategies for handling this type of CAPTCHA
        
        Focus on analysis and detection rather than actual solving.
        """,
        metadata={"captcha_type": "recaptcha_v2", "purpose": "demo"}
    )
    
    print("🔄 Analyzing CAPTCHA system...")
    result = await scraper.scrape_single(captcha_task)
    
    print("\n🔐 CAPTCHA Analysis Results:")
    print("-" * 50)
    
    if result.success:
        print("✅ CAPTCHA analysis completed successfully")
        print(f"⏱️  Processing time: {result.processing_time:.2f}s")
        print("\n📋 Analysis Report:")
        print(result.data)
        
        # Check for structured data in metadata
        if hasattr(result, 'metadata') and result.metadata:
            print("\n📊 Additional Metadata:")
            for key, value in result.metadata.items():
                print(f"   {key}: {value}")
    else:
        print(f"❌ CAPTCHA analysis failed: {result.error}")
    
    return result

if __name__ == "__main__":
    result = asyncio.run(captcha_handling_demo())
    print("\n✅ CAPTCHA handling demo completed!")
```

### Performance Optimization

Create `examples/performance_demo.py`:

```python
"""
Demonstrate performance optimization techniques.
"""
import asyncio
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from scraper.core import WebScraper, ScrapingTask

async def performance_comparison():
    """Compare sequential vs concurrent processing."""
    print("⚡ Performance Optimization Demo")
    print("=" * 50)
    
    scraper = WebScraper()
    
    # Create test tasks
    urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2", 
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/delay/1"
    ]
    
    tasks = [
        ScrapingTask(
            url=url,
            task_type="performance_test",
            instructions="Extract the response data and analyze the delay."
        )
        for url in urls
    ]
    
    print(f"🔄 Testing with {len(tasks)} tasks...")
    
    # Sequential processing
    print("\n1️⃣ Sequential Processing:")
    start_time = time.time()
    
    sequential_results = []
    for task in tasks:
        result = await scraper.scrape_single(task)
        sequential_results.append(result)
    
    sequential_time = time.time() - start_time
    sequential_success = sum(1 for r in sequential_results if r.success)
    
    print(f"   ⏱️  Total time: {sequential_time:.2f}s")
    print(f"   ✅ Success rate: {sequential_success}/{len(tasks)}")
    
    # Concurrent processing
    print("\n2️⃣ Concurrent Processing (max 3):")
    start_time = time.time()
    
    concurrent_results = await scraper.scrape_batch(tasks, max_concurrent=3)
    
    concurrent_time = time.time() - start_time
    concurrent_success = sum(1 for r in concurrent_results if r.success)
    
    print(f"   ⏱️  Total time: {concurrent_time:.2f}s")
    print(f"   ✅ Success rate: {concurrent_success}/{len(tasks)}")
    
    # Performance analysis
    print("\n📊 Performance Analysis:")
    print(f"   🚀 Speed improvement: {sequential_time/concurrent_time:.1f}x faster")
    print(f"   💾 Memory efficiency: Concurrent processing")
    print(f"   🔄 Resource utilization: Better with concurrency")
    
    # Optimal concurrency test
    print("\n3️⃣ Finding Optimal Concurrency:")
    
    concurrency_levels = [1, 2, 3, 5]
    results = {}
    
    for level in concurrency_levels:
        print(f"   Testing concurrency level {level}...")
        start_time = time.time()
        
        test_results = await scraper.scrape_batch(tasks, max_concurrent=level)
        test_time = time.time() - start_time
        test_success = sum(1 for r in test_results if r.success)
        
        results[level] = {
            "time": test_time,
            "success_rate": test_success / len(tasks)
        }
        
        print(f"      Time: {test_time:.2f}s, Success: {test_success}/{len(tasks)}")
    
    # Find optimal level
    optimal_level = min(results.keys(), key=lambda k: results[k]["time"])
    print(f"\n🎯 Optimal concurrency level: {optimal_level}")
    print(f"   Best time: {results[optimal_level]['time']:.2f}s")
    
    return results

if __name__ == "__main__":
    results = asyncio.run(performance_comparison())
    print("\n✅ Performance optimization demo completed!")
```

## Best Practices

### Configuration Management

Create `config/production.yaml`:

```yaml
# Production configuration
aws:
  region: "us-east-1"

agentcore:
  browser_tool_endpoint: "https://agentcore.amazonaws.com"
  session_timeout: 600  # 10 minutes for production
  max_concurrent_sessions: 10  # Higher limit for production

browser:
  headless: true
  viewport_width: 1920
  viewport_height: 1080
  user_agent: "ProductionScraper/1.0"
  enable_javascript: true
  enable_images: false  # Disable for better performance
  page_load_timeout: 45

llamaindex:
  llm_model: "anthropic.claude-3-sonnet-20240229-v1:0"
  vision_model: "anthropic.claude-3-sonnet-20240229-v1:0"
  temperature: 0.0  # More deterministic for production
  max_tokens: 8192

security:
  enable_pii_scrubbing: true
  log_sensitive_data: false
  session_encryption: true

monitoring:
  enable_metrics: true
  log_level: "WARNING"  # Less verbose for production
  performance_tracking: true
  alert_on_failures: true
```

### Error Handling Best Practices

Create `examples/error_handling_demo.py`:

```python
"""
Demonstrate comprehensive error handling strategies.
"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from scraper.core import WebScraper, ScrapingTask

async def error_handling_demo():
    """Demonstrate robust error handling."""
    print("🛡️ Error Handling Demo")
    print("=" * 50)
    
    scraper = WebScraper()
    
    # Create tasks that will likely fail for demonstration
    error_tasks = [
        ScrapingTask(
            url="https://httpbin.org/status/404",
            task_type="error_test",
            instructions="Try to extract content from this 404 page."
        ),
        ScrapingTask(
            url="https://httpbin.org/status/500",
            task_type="error_test", 
            instructions="Handle this server error gracefully."
        ),
        ScrapingTask(
            url="https://invalid-domain-that-does-not-exist.com",
            task_type="error_test",
            instructions="Handle DNS resolution failure."
        ),
        ScrapingTask(
            url="https://httpbin.org/delay/60",  # Will timeout
            task_type="timeout_test",
            instructions="This should timeout - handle gracefully."
        ),
        ScrapingTask(
            url="https://httpbin.org/html",  # This should succeed
            task_type="success_test",
            instructions="This should work normally."
        )
    ]
    
    print(f"🔄 Testing error handling with {len(error_tasks)} tasks...")
    results = await scraper.scrape_batch(error_tasks, max_concurrent=2)
    
    print("\n📊 Error Handling Results:")
    print("-" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"\nTask {i}: {result.task.url}")
        
        if result.success:
            print(f"✅ Success (unexpected for some tests)")
            print(f"   Content preview: {str(result.data)[:100]}...")
        else:
            print(f"❌ Failed (as expected for error tests)")
            print(f"   Error type: {type(result.error).__name__ if hasattr(result, 'error') else 'Unknown'}")
            print(f"   Error message: {result.error}")
            print(f"   Processing time: {result.processing_time:.2f}s")
    
    # Analyze error patterns
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    print(f"\n📈 Error Analysis:")
    print(f"   Total tasks: {len(results)}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Success rate: {successful/len(results):.1%}")
    
    # Show recovery strategies
    print(f"\n🔧 Recovery Strategies Demonstrated:")
    print("   ✓ Graceful failure handling")
    print("   ✓ Detailed error reporting")
    print("   ✓ Timeout management")
    print("   ✓ Network error handling")
    print("   ✓ Continued processing despite failures")
    
    return results

if __name__ == "__main__":
    results = asyncio.run(error_handling_demo())
    print("\n✅ Error handling demo completed!")
```

## Troubleshooting

### Common Issues and Solutions

1. **Authentication Errors**
   ```bash
   # Verify AWS credentials
   aws sts get-caller-identity
   
   # Check permissions
   aws iam simulate-principal-policy --policy-source-arn arn:aws:iam::ACCOUNT:user/USERNAME --action-names bedrock:InvokeModel
   ```

2. **Configuration Issues**
   ```python
   # Test configuration loading
   from integration import LlamaIndexAgentCoreIntegration
   
   try:
       integration = LlamaIndexAgentCoreIntegration(config_path="config.yaml")
       print("✅ Configuration loaded successfully")
   except Exception as e:
       print(f"❌ Configuration error: {e}")
   ```

3. **Performance Issues**
   ```python
   # Monitor resource usage
   import psutil
   import asyncio
   
   async def monitor_performance():
       process = psutil.Process()
       print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
       print(f"CPU usage: {process.cpu_percent():.1f}%")
   ```

## Next Steps

### Extending the Integration

1. **Custom Tools**: Create specialized tools for your use cases
2. **Advanced Processors**: Build domain-specific content processors  
3. **Integration Plugins**: Connect with other systems and databases
4. **Monitoring**: Add comprehensive monitoring and alerting
5. **Scaling**: Implement distributed processing capabilities

### Production Deployment

1. **Containerization**: Package the application in Docker containers
2. **Orchestration**: Deploy using Kubernetes or AWS ECS
3. **Monitoring**: Set up CloudWatch monitoring and alerting
4. **Security**: Implement proper security controls and audit logging
5. **CI/CD**: Create automated deployment pipelines

### Learning Resources

- **LlamaIndex Documentation**: Learn more about LlamaIndex capabilities
- **AWS AgentCore Docs**: Understand AgentCore service features
- **Python Async Programming**: Master async/await patterns
- **Web Scraping Ethics**: Learn about responsible scraping practices

Congratulations! You've completed the comprehensive tutorial for the LlamaIndex AgentCore Browser Integration. You now have the knowledge and tools to build sophisticated, AI-powered web automation applications.