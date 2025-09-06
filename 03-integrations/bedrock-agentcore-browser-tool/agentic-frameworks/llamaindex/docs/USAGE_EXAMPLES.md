# Usage Examples

## ⚠️ Important Disclaimer

**These examples are for educational purposes and use safe test endpoints.** 

- **Educational Examples**: This document uses `httpbin.org`, `example.com`, and demo sites for safe learning
- **Production Use**: For real-world implementations, see [Real-World Implementations Guide](REAL_WORLD_IMPLEMENTATIONS.md)
- **Legal Compliance**: Always verify compliance with robots.txt, terms of service, and applicable laws before automating real websites
- **Ethical Guidelines**: Respect website resources, implement rate limiting, and consider using official APIs when available

## Basic Examples

### Simple Web Scraping (Educational)

```python
"""
Educational web scraping example using safe test endpoint.

⚠️ EDUCATIONAL ONLY: This example uses httpbin.org, a safe testing service.
For production use with real websites, see REAL_WORLD_IMPLEMENTATIONS.md
"""
import asyncio
from integration import LlamaIndexAgentCoreIntegration

async def basic_scraping():
    # Initialize integration
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    # Simple content extraction
    response = await agent.achat(
        "Navigate to https://httpbin.org/html and extract all the text content from the page"
    )
    
    print("Extracted content:")
    print(response.response)
    
    return response.response

# Run the example
if __name__ == "__main__":
    content = asyncio.run(basic_scraping())

# Real-world alternative (see REAL_WORLD_IMPLEMENTATIONS.md for full example)
async def production_news_scraping():
    """
    Production example with compliance checks.
    See REAL_WORLD_IMPLEMENTATIONS.md for complete implementation.
    """
    # 1. Check robots.txt compliance
    # 2. Verify terms of service
    # 3. Implement rate limiting
    # 4. Use official APIs when available
    # 5. Handle data responsibly
    pass
```

### Form Automation

```python
"""
Automated form filling and submission.
"""
import asyncio
from integration import LlamaIndexAgentCoreIntegration

async def form_automation():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    # Fill and submit a contact form
    response = await agent.achat("""
    Navigate to https://httpbin.org/forms/post and fill out the form with:
    - Customer name: John Doe
    - Telephone: +1-555-123-4567
    - Email: john.doe@example.com
    - Size: Medium
    - Pizza toppings: Pepperoni and Mushrooms
    - Delivery time: Now
    - Comments: Please ring the doorbell
    
    Then submit the form and tell me if it was successful.
    """)
    
    print("Form submission result:")
    print(response.response)
    
    return response.response

if __name__ == "__main__":
    result = asyncio.run(form_automation())
```

### Screenshot Capture

```python
"""
Capture screenshots of web pages or specific elements.
"""
import asyncio
import base64
from integration import LlamaIndexAgentCoreIntegration

async def screenshot_example():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    # Navigate and capture screenshot
    response = await agent.achat(
        "Navigate to https://example.com and take a screenshot of the entire page"
    )
    
    # Extract screenshot data if available in metadata
    if "screenshot" in response.metadata:
        screenshot_data = response.metadata["screenshot"]
        
        # Save screenshot
        with open("example_screenshot.png", "wb") as f:
            f.write(base64.b64decode(screenshot_data))
        
        print("Screenshot saved as example_screenshot.png")
    
    print("Page analysis:")
    print(response.response)

if __name__ == "__main__":
    asyncio.run(screenshot_example())
```

## Intermediate Examples

### Multi-Page Data Collection

```python
"""
Collect data from multiple pages with structured output.
"""
import asyncio
import json
from integration import LlamaIndexAgentCoreIntegration

async def multi_page_scraping():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    urls = [
        "https://httpbin.org/html",
        "https://httpbin.org/json",
        "https://httpbin.org/xml"
    ]
    
    results = []
    
    for url in urls:
        response = await agent.achat(f"""
        Navigate to {url} and extract:
        1. Page title
        2. Main content type (HTML, JSON, XML, etc.)
        3. Key information or data structure
        4. Page load time if possible
        
        Format the response as structured data.
        """)
        
        results.append({
            "url": url,
            "analysis": response.response,
            "timestamp": response.metadata.get("timestamp")
        })
    
    # Save results
    with open("multi_page_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(results)} pages")
    return results

if __name__ == "__main__":
    results = asyncio.run(multi_page_scraping())
```

### E-commerce Product Scraping

```python
"""
Scrape product information from an e-commerce site.
"""
import asyncio
import json
from integration import LlamaIndexAgentCoreIntegration

async def ecommerce_scraping():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    # Example with a demo e-commerce site
    response = await agent.achat("""
    Navigate to https://demo.opencart.com/ and:
    
    1. Browse to the "Laptops & Notebooks" category
    2. Extract information for the first 5 products including:
       - Product name
       - Price
       - Brief description
       - Availability status
       - Product image URL if visible
    
    3. Return the data in a structured JSON format
    
    Handle any popups or cookie banners that might appear.
    """)
    
    print("Product scraping results:")
    print(response.response)
    
    # Try to extract structured data from response
    try:
        if "products" in response.metadata:
            products = response.metadata["products"]
            with open("products.json", "w") as f:
                json.dump(products, f, indent=2)
            print(f"Saved {len(products)} products to products.json")
    except Exception as e:
        print(f"Could not save structured data: {e}")
    
    return response.response

if __name__ == "__main__":
    result = asyncio.run(ecommerce_scraping())
```

### News Article Extraction

```python
"""
Extract and summarize news articles.
"""
import asyncio
from integration import LlamaIndexAgentCoreIntegration

async def news_extraction():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    # Extract article content and metadata
    response = await agent.achat("""
    Navigate to https://example-news-site.com/article and extract:
    
    1. Article headline
    2. Author name and publication date
    3. Full article text (main content only, no ads or sidebars)
    4. Any tags or categories
    5. Related article links if present
    
    Then provide a brief 2-3 sentence summary of the article.
    
    Focus on the main article content and ignore navigation, ads, and other page elements.
    """)
    
    print("Article extraction:")
    print(response.response)
    
    return response.response

if __name__ == "__main__":
    article = asyncio.run(news_extraction())
```

## Advanced Examples

### CAPTCHA Handling Workflow

```python
"""
Demonstrate CAPTCHA detection and handling.
"""
import asyncio
from integration import LlamaIndexAgentCoreIntegration
from captcha_workflows import CaptchaSolvingWorkflow

async def captcha_handling_example():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    # Navigate to a page that might have CAPTCHAs
    response = await agent.achat("""
    Navigate to https://www.google.com/recaptcha/api2/demo and:
    
    1. Detect if there are any CAPTCHAs on the page
    2. If CAPTCHAs are found, analyze what type they are
    3. Take a screenshot for analysis
    4. Attempt to solve simple CAPTCHAs if possible
    5. Report on the CAPTCHA detection and solving process
    
    Be detailed about what you observe and any challenges encountered.
    """)
    
    print("CAPTCHA handling results:")
    print(response.response)
    
    return response.response

if __name__ == "__main__":
    result = asyncio.run(captcha_handling_example())
```

### Dynamic Content Monitoring

```python
"""
Monitor a page for content changes over time.
"""
import asyncio
import time
from integration import LlamaIndexAgentCoreIntegration

async def content_monitoring():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    url = "https://httpbin.org/uuid"  # Returns different UUID each time
    previous_content = None
    
    for i in range(5):  # Monitor 5 times
        response = await agent.achat(f"""
        Navigate to {url} and extract the main content.
        Compare it with the previous content if this is not the first visit.
        Report any changes detected.
        """)
        
        current_content = response.response
        
        print(f"Check #{i+1}:")
        print(f"Content: {current_content}")
        
        if previous_content and previous_content != current_content:
            print("🔄 Content changed!")
        else:
            print("📄 Content unchanged")
        
        previous_content = current_content
        
        if i < 4:  # Don't wait after the last check
            await asyncio.sleep(10)  # Wait 10 seconds between checks
    
    return "Monitoring complete"

if __name__ == "__main__":
    result = asyncio.run(content_monitoring())
```

### Batch Processing with Concurrency

```python
"""
Process multiple URLs concurrently for better performance.
"""
import asyncio
import time
from integration import LlamaIndexAgentCoreIntegration

async def process_single_url(url, integration):
    """Process a single URL."""
    agent = integration.create_agent()
    
    try:
        response = await agent.achat(f"""
        Navigate to {url} and extract:
        1. Page title
        2. Meta description
        3. Main heading (H1)
        4. Word count of main content
        5. Any error messages if the page fails to load
        """)
        
        return {
            "url": url,
            "success": True,
            "data": response.response,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "url": url,
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }

async def batch_processing():
    integration = LlamaIndexAgentCoreIntegration()
    
    urls = [
        "https://httpbin.org/html",
        "https://httpbin.org/json",
        "https://httpbin.org/xml",
        "https://httpbin.org/robots.txt",
        "https://httpbin.org/status/200"
    ]
    
    print(f"Processing {len(urls)} URLs concurrently...")
    start_time = time.time()
    
    # Process URLs concurrently
    tasks = [process_single_url(url, integration) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    
    # Analyze results
    successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    failed = len(results) - successful
    
    print(f"\nProcessing complete in {end_time - start_time:.2f} seconds")
    print(f"Successful: {successful}, Failed: {failed}")
    
    for result in results:
        if isinstance(result, dict):
            status = "✅" if result.get("success") else "❌"
            print(f"{status} {result['url']}")
            if not result.get("success"):
                print(f"   Error: {result.get('error')}")
    
    return results

if __name__ == "__main__":
    results = asyncio.run(batch_processing())
```

### Complex Workflow Automation

```python
"""
Complex multi-step workflow with conditional logic.
"""
import asyncio
from integration import LlamaIndexAgentCoreIntegration

async def complex_workflow():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    # Multi-step workflow with decision points
    response = await agent.achat("""
    Perform this complex workflow:
    
    1. Navigate to https://httpbin.org/forms/post
    
    2. Fill out the form with test data:
       - Customer name: Test User
       - Email: test@example.com
       - Size: Large
       - Toppings: Cheese, Pepperoni
       - Comments: This is a test order
    
    3. Before submitting, take a screenshot of the filled form
    
    4. Submit the form
    
    5. Analyze the response page:
       - Check if submission was successful
       - Extract any confirmation details
       - Look for error messages
    
    6. If successful, navigate back and try submitting an empty form to test validation
    
    7. Compare the two submission attempts and report the differences
    
    Provide detailed feedback on each step and any issues encountered.
    """)
    
    print("Complex workflow results:")
    print(response.response)
    
    return response.response

if __name__ == "__main__":
    result = asyncio.run(complex_workflow())
```

## Specialized Use Cases

### API Documentation Scraping

```python
"""
Scrape API documentation and extract endpoint information.
"""
import asyncio
import json
from integration import LlamaIndexAgentCoreIntegration

async def api_docs_scraping():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    response = await agent.achat("""
    Navigate to https://httpbin.org/ and extract information about the API endpoints:
    
    1. Find all available endpoints listed on the page
    2. For each endpoint, extract:
       - HTTP method (GET, POST, etc.)
       - Endpoint path
       - Brief description
       - Parameters if mentioned
    
    3. Organize this information in a structured format
    
    4. Also extract any general API information like:
       - Base URL
       - Authentication requirements
       - Rate limits
       - Response formats
    
    Focus on creating a comprehensive API reference from the documentation.
    """)
    
    print("API documentation extraction:")
    print(response.response)
    
    return response.response

if __name__ == "__main__":
    api_info = asyncio.run(api_docs_scraping())
```

### Social Media Content Analysis

```python
"""
Analyze social media content (respecting terms of service).
"""
import asyncio
from integration import LlamaIndexAgentCoreIntegration

async def social_media_analysis():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    # Note: This is a hypothetical example - always respect robots.txt and ToS
    response = await agent.achat("""
    Navigate to a public social media page or demo site and analyze:
    
    1. Content themes and topics
    2. Engagement patterns (likes, shares, comments if visible)
    3. Content format distribution (text, images, videos)
    4. Posting frequency patterns
    5. Popular hashtags or keywords
    
    Provide insights about the content strategy and audience engagement.
    
    Important: Only analyze publicly available content and respect platform policies.
    """)
    
    print("Social media analysis:")
    print(response.response)
    
    return response.response

if __name__ == "__main__":
    analysis = asyncio.run(social_media_analysis())
```

### Competitive Analysis

```python
"""
Perform competitive analysis by comparing multiple websites.
"""
import asyncio
from integration import LlamaIndexAgentCoreIntegration

async def competitive_analysis():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    competitors = [
        "https://example.com",
        "https://httpbin.org",
        "https://jsonplaceholder.typicode.com"
    ]
    
    analysis_results = []
    
    for competitor in competitors:
        response = await agent.achat(f"""
        Navigate to {competitor} and perform a competitive analysis:
        
        1. Overall site design and user experience
        2. Key features and functionality offered
        3. Content quality and organization
        4. Performance (page load speed, responsiveness)
        5. Unique selling propositions or differentiators
        6. Target audience indicators
        
        Provide a structured analysis that can be compared with other competitors.
        """)
        
        analysis_results.append({
            "competitor": competitor,
            "analysis": response.response
        })
    
    # Generate comparative summary
    summary_response = await agent.achat(f"""
    Based on the following competitive analyses, provide a comparative summary:
    
    {json.dumps(analysis_results, indent=2)}
    
    Identify:
    1. Common strengths across competitors
    2. Unique differentiators for each
    3. Market gaps or opportunities
    4. Best practices observed
    5. Recommendations for competitive positioning
    """)
    
    print("Competitive Analysis Summary:")
    print(summary_response.response)
    
    return {
        "individual_analyses": analysis_results,
        "comparative_summary": summary_response.response
    }

if __name__ == "__main__":
    results = asyncio.run(competitive_analysis())
```

## Integration with LlamaIndex Components

### Document Processing Pipeline

```python
"""
Integrate web scraping with LlamaIndex document processing.
"""
import asyncio
from integration import LlamaIndexAgentCoreIntegration
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SimpleNodeParser

async def web_to_knowledge_base():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    urls = [
        "https://httpbin.org/html",
        "https://httpbin.org/json",
        "https://httpbin.org/xml"
    ]
    
    documents = []
    
    for url in urls:
        response = await agent.achat(f"""
        Navigate to {url} and extract all meaningful content.
        Clean up the text and provide it in a format suitable for indexing.
        Also extract metadata like title, description, and content type.
        """)
        
        # Create LlamaIndex document
        doc = Document(
            text=response.response,
            metadata={
                "source_url": url,
                "extraction_method": "agentcore_browser",
                "content_type": "web_page"
            }
        )
        documents.append(doc)
    
    # Build vector index
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)
    
    # Create query engine
    query_engine = index.as_query_engine()
    
    # Test queries
    test_queries = [
        "What content was found on the HTML page?",
        "What data formats were encountered?",
        "Summarize the key information from all pages"
    ]
    
    for query in test_queries:
        response = query_engine.query(query)
        print(f"Query: {query}")
        print(f"Response: {response}")
        print("-" * 50)
    
    return index

if __name__ == "__main__":
    index = asyncio.run(web_to_knowledge_base())
```

### RAG with Web Content

```python
"""
Retrieval-Augmented Generation using web-scraped content.
"""
import asyncio
from integration import LlamaIndexAgentCoreIntegration
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.bedrock import Bedrock

async def web_rag_example():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    # Scrape content from multiple sources
    sources = [
        "https://httpbin.org/",
        "https://jsonplaceholder.typicode.com/",
    ]
    
    documents = []
    
    for source in sources:
        response = await agent.achat(f"""
        Navigate to {source} and extract comprehensive information about:
        1. What the service provides
        2. Available endpoints or features
        3. Usage examples
        4. Any documentation or help content
        
        Provide detailed, well-structured content suitable for a knowledge base.
        """)
        
        doc = Document(
            text=response.response,
            metadata={
                "source": source,
                "type": "api_documentation"
            }
        )
        documents.append(doc)
    
    # Build RAG system
    index = VectorStoreIndex.from_documents(documents)
    
    # Configure LLM
    llm = Bedrock(model="anthropic.claude-3-sonnet-20240229-v1:0")
    
    # Create chat engine with RAG
    chat_engine = index.as_chat_engine(llm=llm)
    
    # Interactive Q&A
    questions = [
        "What testing services are available?",
        "How can I test HTTP requests?",
        "What's the difference between the services?",
        "Give me examples of API endpoints I can use for testing"
    ]
    
    for question in questions:
        response = chat_engine.chat(question)
        print(f"Q: {question}")
        print(f"A: {response}")
        print("-" * 80)
    
    return chat_engine

if __name__ == "__main__":
    chat_engine = asyncio.run(web_rag_example())
```

## Performance Optimization Examples

### Session Pooling

```python
"""
Implement session pooling for better performance.
"""
import asyncio
from typing import List
from integration import LlamaIndexAgentCoreIntegration

class SessionPool:
    def __init__(self, pool_size: int = 3):
        self.pool_size = pool_size
        self.available_sessions = []
        self.in_use_sessions = set()
        self.integration = LlamaIndexAgentCoreIntegration()
    
    async def get_session(self):
        if self.available_sessions:
            session = self.available_sessions.pop()
            self.in_use_sessions.add(session)
            return session
        
        if len(self.in_use_sessions) < self.pool_size:
            session = self.integration.create_agent()
            self.in_use_sessions.add(session)
            return session
        
        # Wait for a session to become available
        while not self.available_sessions:
            await asyncio.sleep(0.1)
        
        session = self.available_sessions.pop()
        self.in_use_sessions.add(session)
        return session
    
    def return_session(self, session):
        if session in self.in_use_sessions:
            self.in_use_sessions.remove(session)
            self.available_sessions.append(session)

async def pooled_processing():
    pool = SessionPool(pool_size=3)
    
    urls = [
        "https://httpbin.org/html",
        "https://httpbin.org/json",
        "https://httpbin.org/xml",
        "https://httpbin.org/robots.txt",
        "https://httpbin.org/status/200"
    ]
    
    async def process_url(url):
        session = await pool.get_session()
        try:
            response = await session.achat(f"Navigate to {url} and extract the main content")
            return {"url": url, "content": response.response}
        finally:
            pool.return_session(session)
    
    # Process URLs with session pooling
    tasks = [process_url(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    print(f"Processed {len(results)} URLs using session pool")
    return results

if __name__ == "__main__":
    results = asyncio.run(pooled_processing())
```

### Caching Strategy

```python
"""
Implement caching for frequently accessed content.
"""
import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from integration import LlamaIndexAgentCoreIntegration

class ContentCache:
    def __init__(self, ttl_minutes: int = 60):
        self.cache = {}
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def _get_key(self, url: str, operation: str) -> str:
        return hashlib.md5(f"{url}:{operation}".encode()).hexdigest()
    
    def get(self, url: str, operation: str):
        key = self._get_key(url, operation)
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry["timestamp"] < self.ttl:
                return entry["data"]
            else:
                del self.cache[key]
        return None
    
    def set(self, url: str, operation: str, data):
        key = self._get_key(url, operation)
        self.cache[key] = {
            "data": data,
            "timestamp": datetime.now()
        }

async def cached_scraping():
    cache = ContentCache(ttl_minutes=30)
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    urls = [
        "https://httpbin.org/html",
        "https://httpbin.org/json"
    ]
    
    async def get_content(url, operation="extract_content"):
        # Check cache first
        cached_result = cache.get(url, operation)
        if cached_result:
            print(f"Cache hit for {url}")
            return cached_result
        
        print(f"Cache miss for {url} - fetching...")
        response = await agent.achat(f"Navigate to {url} and extract all content")
        
        result = {
            "url": url,
            "content": response.response,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the result
        cache.set(url, operation, result)
        return result
    
    # First run - cache misses
    print("First run:")
    for url in urls:
        result = await get_content(url)
        print(f"Processed: {result['url']}")
    
    print("\nSecond run (should hit cache):")
    for url in urls:
        result = await get_content(url)
        print(f"Processed: {result['url']}")
    
    return "Caching demonstration complete"

if __name__ == "__main__":
    result = asyncio.run(cached_scraping())
```

These examples demonstrate the versatility and power of the LlamaIndex AgentCore Browser Integration. From simple web scraping to complex workflows, CAPTCHA handling, and performance optimization, the integration provides a comprehensive solution for intelligent web automation tasks.