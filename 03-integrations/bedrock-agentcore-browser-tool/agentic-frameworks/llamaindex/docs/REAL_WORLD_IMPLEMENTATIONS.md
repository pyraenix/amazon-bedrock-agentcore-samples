# Real-World Implementation Guide

## ⚠️ Important Disclaimer

This guide provides examples for real-world implementations using actual websites and services. **Please ensure you comply with all applicable terms of service, robots.txt files, and legal requirements** when implementing these examples.

## 🔒 Ethical Web Automation Guidelines

### Before You Begin

1. **Check robots.txt**: Always review `https://website.com/robots.txt`
2. **Read Terms of Service**: Ensure your use case is permitted
3. **Respect Rate Limits**: Implement appropriate delays between requests
4. **Use APIs When Available**: Prefer official APIs over web scraping
5. **Monitor Your Impact**: Ensure your automation doesn't overload servers
6. **Handle Personal Data Responsibly**: Comply with GDPR, CCPA, and other privacy laws

### Legal Considerations

- **Public Data Only**: Only scrape publicly available information
- **Commercial Use**: Check if commercial use is permitted
- **Copyright**: Respect intellectual property rights
- **Data Protection**: Handle personal data according to applicable laws

## 🌐 Real-World Examples

### News Monitoring

#### Safe Learning Example (Mock)
```python
# Educational example using test endpoints
response = await agent.achat(
    "Navigate to https://httpbin.org/html and extract content"
)
```

#### Real-World Implementation
```python
"""
Real news monitoring implementation.

⚠️ DISCLAIMER: This example is for educational purposes. Before implementing:
1. Check the news site's robots.txt and terms of service
2. Consider using their official API if available
3. Implement appropriate rate limiting
4. Respect copyright and fair use guidelines
"""

async def monitor_news_sites():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    # Example with publicly accessible news sites
    # Always check robots.txt and ToS before using
    news_sources = [
        {
            "name": "BBC News Technology",
            "url": "https://www.bbc.com/news/technology",
            "robots_check": "https://www.bbc.com/robots.txt",
            "rate_limit": 2.0  # seconds between requests
        },
        {
            "name": "Reuters Technology",
            "url": "https://www.reuters.com/technology/",
            "robots_check": "https://www.reuters.com/robots.txt", 
            "rate_limit": 3.0
        }
    ]
    
    results = []
    
    for source in news_sources:
        # Implement rate limiting
        await asyncio.sleep(source["rate_limit"])
        
        try:
            response = await agent.achat(f"""
            Navigate to {source['url']} and extract recent technology news:
            
            IMPORTANT: Respect the website's structure and only extract:
            1. Article headlines (main stories only)
            2. Publication dates
            3. Brief summaries (first paragraph only)
            4. Article URLs for reference
            
            Do not extract:
            - Full article text (copyright protected)
            - Images or multimedia content
            - User comments or personal data
            - Advertisement content
            
            Limit to the top 5 most recent articles.
            """)
            
            results.append({
                "source": source["name"],
                "success": True,
                "data": response.response,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            results.append({
                "source": source["name"],
                "success": False,
                "error": str(e)
            })
    
    return results

# Usage with proper error handling and logging
async def responsible_news_monitoring():
    try:
        results = await monitor_news_sites()
        
        # Log activity for compliance
        logging.info(f"News monitoring completed: {len(results)} sources processed")
        
        # Store results responsibly
        for result in results:
            if result["success"]:
                # Process and store data according to your data retention policy
                process_news_data(result["data"])
                
    except Exception as e:
        logging.error(f"News monitoring failed: {e}")
```

### E-commerce Price Monitoring

#### Safe Learning Example (Mock)
```python
# Educational example using demo sites
response = await agent.achat(
    "Navigate to https://demo.opencart.com/ and extract product prices"
)
```

#### Real-World Implementation
```python
"""
Real e-commerce price monitoring implementation.

⚠️ DISCLAIMER: This example is for educational purposes. Before implementing:
1. Check the e-commerce site's robots.txt and terms of service
2. Many sites prohibit automated price monitoring
3. Consider using official APIs (Amazon Product Advertising API, etc.)
4. Implement proper rate limiting and user-agent identification
5. Respect intellectual property and pricing data rights
"""

async def monitor_product_prices():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    # Example products to monitor (replace with your actual products)
    products = [
        {
            "name": "Example Product",
            "urls": [
                "https://www.example-retailer1.com/product/123",
                "https://www.example-retailer2.com/product/456"
            ],
            "target_price": 99.99
        }
    ]
    
    results = []
    
    for product in products:
        product_results = []
        
        for url in product["urls"]:
            # Implement significant delays to be respectful
            await asyncio.sleep(5.0)
            
            try:
                response = await agent.achat(f"""
                Navigate to {url} and extract ONLY the following information:
                
                1. Current product price (numerical value only)
                2. Product availability status
                3. Any sale/discount indicators
                
                IMPORTANT RESTRICTIONS:
                - Do not extract product descriptions or reviews
                - Do not extract competitor pricing information
                - Do not extract customer personal information
                - Only extract publicly displayed pricing information
                - Respect any "no robots" or access restrictions
                
                If the page requires login or shows access restrictions, report this and do not proceed.
                """)
                
                product_results.append({
                    "url": url,
                    "success": True,
                    "data": response.response
                })
                
            except Exception as e:
                product_results.append({
                    "url": url,
                    "success": False,
                    "error": str(e)
                })
        
        results.append({
            "product": product["name"],
            "results": product_results,
            "timestamp": datetime.now().isoformat()
        })
    
    return results

# Responsible implementation with compliance checks
async def responsible_price_monitoring():
    # Check robots.txt compliance first
    compliance_check = await verify_robots_compliance([
        "https://www.example-retailer1.com/robots.txt",
        "https://www.example-retailer2.com/robots.txt"
    ])
    
    if not compliance_check["compliant"]:
        logging.warning("Robots.txt compliance check failed")
        return None
    
    # Implement with proper logging and error handling
    results = await monitor_product_prices()
    
    # Log activity for audit trail
    logging.info(f"Price monitoring completed: {len(results)} products checked")
    
    return results

async def verify_robots_compliance(robots_urls):
    """Verify compliance with robots.txt files."""
    # Implementation to check robots.txt compliance
    # This is a simplified example - use a proper robots.txt parser
    return {"compliant": True, "details": "Manual verification required"}
```

### Job Listings Aggregation

#### Safe Learning Example (Mock)
```python
# Educational example using test data
response = await agent.achat(
    "Navigate to https://httpbin.org/html and extract job-like content"
)
```

#### Real-World Implementation
```python
"""
Real job listings aggregation implementation.

⚠️ DISCLAIMER: This example is for educational purposes. Before implementing:
1. Most job boards have official APIs (Indeed API, LinkedIn API, etc.)
2. Check terms of service - many prohibit automated scraping
3. Consider partnerships or official data feeds
4. Respect personal data and privacy regulations
5. Implement proper attribution and data handling
"""

async def aggregate_job_listings():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    # Example job boards (check their APIs first!)
    job_sources = [
        {
            "name": "Example Job Board",
            "search_url": "https://www.example-jobs.com/search?q=python+developer",
            "api_available": True,  # Prefer API when available
            "api_url": "https://api.example-jobs.com/v1/jobs",
            "rate_limit": 10.0  # Generous rate limiting
        }
    ]
    
    results = []
    
    for source in job_sources:
        # Always prefer official APIs
        if source["api_available"]:
            logging.info(f"Using API for {source['name']} - recommended approach")
            # Implement API-based job fetching here
            continue
        
        # Only use web scraping if no API is available and ToS permits
        await asyncio.sleep(source["rate_limit"])
        
        try:
            response = await agent.achat(f"""
            Navigate to {source['search_url']} and extract job listings information:
            
            EXTRACT ONLY PUBLIC INFORMATION:
            1. Job titles
            2. Company names (publicly listed)
            3. Location information
            4. Job posting dates
            5. Public job description summaries
            
            DO NOT EXTRACT:
            - Applicant personal information
            - Internal company data
            - Salary information unless publicly posted
            - Contact details beyond public company information
            
            COMPLIANCE REQUIREMENTS:
            - Respect any login requirements (do not bypass)
            - Follow robots.txt directives
            - Limit to first page of results only
            - Include proper attribution in data usage
            """)
            
            results.append({
                "source": source["name"],
                "method": "web_scraping",
                "success": True,
                "data": response.response,
                "compliance_notes": "Manual ToS verification required"
            })
            
        except Exception as e:
            results.append({
                "source": source["name"],
                "success": False,
                "error": str(e)
            })
    
    return results

# Recommended approach using APIs
async def api_based_job_aggregation():
    """
    Recommended approach using official APIs.
    This is the preferred method for job data aggregation.
    """
    
    # Example API implementations
    api_clients = {
        "indeed": {
            "endpoint": "https://api.indeed.com/ads/apisearch",
            "key": "your_api_key",
            "documentation": "https://opensource.indeedeng.io/api-documentation/"
        },
        "github_jobs": {
            "endpoint": "https://jobs.github.com/positions.json",
            "key": None,  # No key required
            "documentation": "https://jobs.github.com/api"
        }
    }
    
    results = []
    
    for provider, config in api_clients.items():
        try:
            # Use proper API client libraries
            job_data = await fetch_jobs_via_api(config)
            results.append({
                "provider": provider,
                "method": "official_api",
                "success": True,
                "data": job_data
            })
        except Exception as e:
            results.append({
                "provider": provider,
                "success": False,
                "error": str(e)
            })
    
    return results

async def fetch_jobs_via_api(config):
    """Implement proper API client for job data."""
    # This would implement the actual API calls
    # using appropriate authentication and rate limiting
    pass
```

### Social Media Monitoring

#### Safe Learning Example (Mock)
```python
# Educational example using test content
response = await agent.achat(
    "Navigate to https://httpbin.org/html and analyze social-media-like content"
)
```

#### Real-World Implementation
```python
"""
Real social media monitoring implementation.

⚠️ CRITICAL DISCLAIMER: Social media monitoring has strict legal and ethical requirements:
1. Most platforms prohibit automated data collection in their ToS
2. Use official APIs (Twitter API, Facebook Graph API, LinkedIn API, etc.)
3. Respect user privacy and data protection laws
4. Only collect publicly available data with proper permissions
5. Implement proper data retention and deletion policies
6. Consider user consent and opt-out mechanisms
"""

async def monitor_social_media_mentions():
    """
    RECOMMENDED: Use official APIs for social media monitoring.
    This example shows the proper approach using official APIs.
    """
    
    # Official API implementations (recommended approach)
    api_clients = {
        "twitter": {
            "api": "Twitter API v2",
            "endpoint": "https://api.twitter.com/2/tweets/search/recent",
            "documentation": "https://developer.twitter.com/en/docs/twitter-api",
            "authentication": "Bearer Token or OAuth 2.0"
        },
        "reddit": {
            "api": "Reddit API",
            "endpoint": "https://www.reddit.com/r/subreddit/search.json",
            "documentation": "https://www.reddit.com/dev/api/",
            "authentication": "OAuth 2.0"
        }
    }
    
    results = []
    
    for platform, config in api_clients.items():
        try:
            # Use official API clients
            mentions = await fetch_mentions_via_api(platform, config, "your_brand_name")
            results.append({
                "platform": platform,
                "method": "official_api",
                "success": True,
                "data": mentions,
                "compliance": "API ToS compliant"
            })
        except Exception as e:
            results.append({
                "platform": platform,
                "success": False,
                "error": str(e)
            })
    
    return results

async def fetch_mentions_via_api(platform, config, search_term):
    """
    Implement proper API-based social media monitoring.
    This is the only recommended approach for social media data.
    """
    
    if platform == "twitter":
        # Use official Twitter API client
        # Example: tweepy, python-twitter, etc.
        pass
    elif platform == "reddit":
        # Use official Reddit API client
        # Example: praw (Python Reddit API Wrapper)
        pass
    
    # Return properly formatted and compliant data
    return {
        "mentions": [],
        "metadata": {
            "search_term": search_term,
            "timestamp": datetime.now().isoformat(),
            "api_version": config["api"],
            "rate_limit_remaining": "check_api_response"
        }
    }

# IMPORTANT: Web scraping social media is generally prohibited
async def why_not_web_scraping_social_media():
    """
    This function explains why web scraping social media is not recommended.
    """
    
    reasons = [
        "Terms of Service violations - most platforms explicitly prohibit it",
        "Legal risks - potential copyright and privacy law violations",
        "Technical barriers - anti-bot measures, CAPTCHAs, rate limiting",
        "Data quality issues - dynamic content, incomplete data",
        "Ethical concerns - user privacy and consent",
        "Account suspension - platforms actively detect and ban scrapers"
    ]
    
    recommended_alternatives = [
        "Use official APIs with proper authentication",
        "Partner with social media platforms for data access",
        "Use licensed social media data providers",
        "Implement user-generated content with proper consent",
        "Focus on owned media and direct customer feedback"
    ]
    
    return {
        "recommendation": "Use official APIs only",
        "reasons_against_scraping": reasons,
        "alternatives": recommended_alternatives
    }
```

### Real Estate Monitoring

#### Safe Learning Example (Mock)
```python
# Educational example using test data
response = await agent.achat(
    "Navigate to https://httpbin.org/json and extract real-estate-like data"
)
```

#### Real-World Implementation
```python
"""
Real estate monitoring implementation.

⚠️ DISCLAIMER: Real estate data monitoring considerations:
1. Many MLS systems have strict access controls and licensing requirements
2. Check with local real estate boards for data usage policies
3. Consider official APIs (Zillow API, Realtor.com API, etc.)
4. Respect intellectual property rights in listing data
5. Comply with fair housing and anti-discrimination laws
"""

async def monitor_real_estate_listings():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    # Example real estate sources (verify compliance first)
    real_estate_sources = [
        {
            "name": "Example Real Estate Site",
            "search_url": "https://www.example-realty.com/search?location=city&type=house",
            "api_available": True,
            "api_documentation": "https://api.example-realty.com/docs",
            "data_license": "Check licensing requirements",
            "rate_limit": 15.0  # Very conservative rate limiting
        }
    ]
    
    results = []
    
    for source in real_estate_sources:
        # Always check for official APIs first
        if source["api_available"]:
            logging.info(f"Recommend using API for {source['name']}")
            # Implement API-based approach
            continue
        
        # Only proceed with web scraping if legally compliant
        compliance_check = await verify_real_estate_compliance(source)
        if not compliance_check["approved"]:
            logging.warning(f"Compliance check failed for {source['name']}")
            continue
        
        await asyncio.sleep(source["rate_limit"])
        
        try:
            response = await agent.achat(f"""
            Navigate to {source['search_url']} and extract public real estate listing information:
            
            EXTRACT ONLY PUBLIC LISTING DATA:
            1. Property addresses (publicly listed)
            2. Listing prices (if publicly displayed)
            3. Basic property details (bedrooms, bathrooms, square footage)
            4. Listing dates
            5. Public listing agent contact information
            
            COMPLIANCE REQUIREMENTS:
            - Only extract data from public listings
            - Do not extract private or confidential information
            - Respect any access restrictions or login requirements
            - Include proper data source attribution
            - Limit to first page of results
            
            DO NOT EXTRACT:
            - Private seller information
            - Internal MLS data
            - Proprietary valuation algorithms
            - Personal financial information
            """)
            
            results.append({
                "source": source["name"],
                "method": "compliant_web_extraction",
                "success": True,
                "data": response.response,
                "compliance_verified": True,
                "data_license": source["data_license"]
            })
            
        except Exception as e:
            results.append({
                "source": source["name"],
                "success": False,
                "error": str(e)
            })
    
    return results

async def verify_real_estate_compliance(source):
    """
    Verify compliance with real estate data usage requirements.
    This is a critical step for real estate data collection.
    """
    
    compliance_checks = {
        "robots_txt": f"Check {source['search_url']}/robots.txt",
        "terms_of_service": "Manual review required",
        "mls_licensing": "Verify MLS data licensing requirements",
        "fair_housing": "Ensure compliance with fair housing laws",
        "data_attribution": "Implement proper data source attribution"
    }
    
    # In a real implementation, this would perform actual compliance verification
    return {
        "approved": False,  # Default to false - manual verification required
        "checks_required": compliance_checks,
        "recommendation": "Use official APIs or licensed data providers"
    }

# Recommended approach using official APIs
async def api_based_real_estate_monitoring():
    """
    Recommended approach using official real estate APIs.
    """
    
    api_providers = {
        "zillow": {
            "api": "Zillow API",
            "documentation": "https://www.zillow.com/howto/api/APIOverview.htm",
            "note": "Check current API availability and terms"
        },
        "realtor_com": {
            "api": "Realtor.com API",
            "documentation": "Contact Realtor.com for API access",
            "note": "Professional/commercial licensing may be required"
        }
    }
    
    results = []
    
    for provider, config in api_providers.items():
        try:
            # Implement proper API client with authentication
            listings = await fetch_listings_via_api(provider, config)
            results.append({
                "provider": provider,
                "method": "official_api",
                "success": True,
                "data": listings,
                "licensing": "API terms compliant"
            })
        except Exception as e:
            results.append({
                "provider": provider,
                "success": False,
                "error": str(e)
            })
    
    return results

async def fetch_listings_via_api(provider, config):
    """Implement proper API-based real estate data fetching."""
    # This would implement the actual API calls with proper authentication
    # and compliance with API terms of service
    pass
```

## 🛠️ Implementation Best Practices

### 1. Compliance Framework

```python
class ComplianceFramework:
    """Framework for ensuring compliant web automation."""
    
    def __init__(self):
        self.compliance_checks = [
            "robots_txt_verification",
            "terms_of_service_review",
            "rate_limit_implementation",
            "data_protection_compliance",
            "api_availability_check"
        ]
    
    async def verify_compliance(self, url: str) -> dict:
        """Verify compliance before automation."""
        results = {}
        
        # Check robots.txt
        results["robots_txt"] = await self.check_robots_txt(url)
        
        # Verify rate limiting
        results["rate_limiting"] = self.verify_rate_limiting()
        
        # Check for API alternatives
        results["api_available"] = await self.check_api_availability(url)
        
        return results
    
    async def check_robots_txt(self, url: str) -> dict:
        """Check robots.txt compliance."""
        # Implementation to parse and verify robots.txt
        pass
    
    def verify_rate_limiting(self) -> dict:
        """Verify appropriate rate limiting is implemented."""
        # Implementation to check rate limiting configuration
        pass
    
    async def check_api_availability(self, url: str) -> dict:
        """Check if official APIs are available."""
        # Implementation to check for official APIs
        pass
```

### 2. Responsible Data Handling

```python
class ResponsibleDataHandler:
    """Handle scraped data responsibly and compliantly."""
    
    def __init__(self):
        self.data_retention_policy = {
            "max_retention_days": 30,
            "anonymization_required": True,
            "deletion_on_request": True
        }
    
    def process_scraped_data(self, data: dict) -> dict:
        """Process scraped data according to compliance requirements."""
        
        # Remove or anonymize personal data
        cleaned_data = self.anonymize_personal_data(data)
        
        # Add data source attribution
        attributed_data = self.add_attribution(cleaned_data)
        
        # Set expiration date
        final_data = self.set_expiration(attributed_data)
        
        return final_data
    
    def anonymize_personal_data(self, data: dict) -> dict:
        """Remove or anonymize personal information."""
        # Implementation for PII removal/anonymization
        pass
    
    def add_attribution(self, data: dict) -> dict:
        """Add proper data source attribution."""
        # Implementation for data attribution
        pass
    
    def set_expiration(self, data: dict) -> dict:
        """Set data expiration according to retention policy."""
        # Implementation for data expiration
        pass
```

### 3. Error Handling and Logging

```python
class ComplianceLogger:
    """Logging system for compliance and audit purposes."""
    
    def __init__(self):
        self.audit_log = []
    
    def log_automation_activity(self, activity: dict):
        """Log automation activity for compliance auditing."""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "activity_type": activity["type"],
            "target_url": activity["url"],
            "compliance_verified": activity.get("compliance_verified", False),
            "data_extracted": activity.get("data_extracted", False),
            "rate_limit_respected": activity.get("rate_limit_respected", True),
            "user_agent": activity.get("user_agent"),
            "success": activity.get("success", False)
        }
        
        self.audit_log.append(log_entry)
        
        # Also log to external audit system
        self.external_audit_log(log_entry)
    
    def external_audit_log(self, entry: dict):
        """Log to external audit system for compliance."""
        # Implementation for external audit logging
        pass
```

## 📋 Pre-Implementation Checklist

Before implementing any real-world web automation:

### Legal and Compliance
- [ ] Review target website's Terms of Service
- [ ] Check robots.txt file for restrictions
- [ ] Verify compliance with applicable data protection laws (GDPR, CCPA, etc.)
- [ ] Ensure compliance with copyright and intellectual property laws
- [ ] Check for official APIs as alternatives to web scraping

### Technical Implementation
- [ ] Implement appropriate rate limiting (minimum 1-2 seconds between requests)
- [ ] Use proper User-Agent identification
- [ ] Implement robust error handling and retry logic
- [ ] Set up comprehensive logging for audit purposes
- [ ] Plan for data retention and deletion policies

### Ethical Considerations
- [ ] Ensure minimal impact on target website performance
- [ ] Respect user privacy and personal data
- [ ] Implement proper data anonymization where required
- [ ] Plan for handling opt-out requests
- [ ] Consider the broader impact of your automation

### Monitoring and Maintenance
- [ ] Set up monitoring for compliance violations
- [ ] Plan for regular compliance reviews
- [ ] Implement alerting for automation failures
- [ ] Establish procedures for handling legal requests
- [ ] Create documentation for audit purposes

## 🚨 When NOT to Use Web Automation

Avoid web automation in these scenarios:

1. **Terms of Service Prohibition**: If the website explicitly prohibits automated access
2. **Login Required**: If the content requires user authentication
3. **Personal Data**: If the automation would collect personal or sensitive information
4. **Commercial Restrictions**: If the website prohibits commercial use of data
5. **API Available**: If an official API provides the same data
6. **Legal Uncertainty**: If there's any doubt about the legality of the automation

## 📞 Getting Help

For complex compliance questions:

1. **Legal Counsel**: Consult with lawyers specializing in technology and data privacy
2. **Industry Associations**: Check with relevant industry groups for best practices
3. **Platform Support**: Contact website owners directly for permission
4. **Compliance Experts**: Work with data protection and compliance specialists

Remember: **When in doubt, don't automate**. It's always better to err on the side of caution and seek proper permissions or use official APIs.