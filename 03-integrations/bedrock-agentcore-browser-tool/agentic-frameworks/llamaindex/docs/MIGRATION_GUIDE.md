# Migration Guide

## Overview

This guide helps you migrate from traditional browser automation tools (Selenium, Playwright, Puppeteer) to the LlamaIndex AgentCore Browser Integration. The key difference is moving from imperative browser control to declarative, AI-driven web automation.

## Migration Philosophy

### Traditional Approach (Imperative)
```python
# You write explicit steps
driver.get("https://example.com")
element = driver.find_element(By.CSS_SELECTOR, ".login-form")
username_field = element.find_element(By.NAME, "username")
username_field.send_keys("user@example.com")
password_field = element.find_element(By.NAME, "password")
password_field.send_keys("password123")
submit_button = element.find_element(By.CSS_SELECTOR, "button[type='submit']")
submit_button.click()
```

### LlamaIndex AgentCore Approach (Declarative)
```python
# You describe what you want to achieve
response = await agent.achat(
    "Navigate to https://example.com and log in using username 'user@example.com' "
    "and password 'password123'"
)
```

## Migration from Selenium

### Basic Navigation

**Selenium:**
```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()
driver.get("https://example.com")
wait = WebDriverWait(driver, 10)
element = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "content")))
text = element.text
driver.quit()
```

**LlamaIndex AgentCore:**
```python
from integration import LlamaIndexAgentCoreIntegration

integration = LlamaIndexAgentCoreIntegration()
agent = integration.create_agent()

response = await agent.achat(
    "Navigate to https://example.com and extract text from the element with class 'content'"
)
text = response.response
```

### Form Interaction

**Selenium:**
```python
driver.get("https://example.com/form")

# Fill form fields
username = driver.find_element(By.ID, "username")
username.clear()
username.send_keys("testuser")

password = driver.find_element(By.ID, "password")
password.clear()
password.send_keys("testpass")

# Select dropdown
from selenium.webdriver.support.ui import Select
dropdown = Select(driver.find_element(By.ID, "country"))
dropdown.select_by_visible_text("United States")

# Submit form
submit_btn = driver.find_element(By.CSS_SELECTOR, "input[type='submit']")
submit_btn.click()
```

**LlamaIndex AgentCore:**
```python
response = await agent.achat(
    "Navigate to https://example.com/form and fill out the form with: "
    "username: 'testuser', password: 'testpass', country: 'United States'. "
    "Then submit the form."
)
```

### Element Interaction

**Selenium:**
```python
# Click button
button = driver.find_element(By.XPATH, "//button[contains(text(), 'Click Me')]")
button.click()

# Handle alerts
alert = driver.switch_to.alert
alert_text = alert.text
alert.accept()

# Switch to iframe
driver.switch_to.frame("iframe_name")
element_in_iframe = driver.find_element(By.ID, "element_id")
driver.switch_to.default_content()
```

**LlamaIndex AgentCore:**
```python
response = await agent.achat(
    "Click the button that says 'Click Me', handle any alerts that appear, "
    "and if there's an iframe, interact with elements inside it as needed."
)
```

### Screenshot and Debugging

**Selenium:**
```python
# Take screenshot
driver.save_screenshot("screenshot.png")

# Get page source
page_source = driver.page_source

# Get current URL
current_url = driver.current_url
```

**LlamaIndex AgentCore:**
```python
# Screenshots are automatically taken for debugging
# You can also explicitly request them
response = await agent.achat(
    "Take a screenshot of the current page and tell me what you see"
)

# Get page information
response = await agent.achat(
    "What is the current URL and page title?"
)
```

## Migration from Playwright

### Async Operations

**Playwright:**
```python
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("https://example.com")
        
        # Wait for element
        await page.wait_for_selector(".content")
        
        # Extract text
        text = await page.text_content(".content")
        
        # Take screenshot
        await page.screenshot(path="screenshot.png")
        
        await browser.close()
        return text
```

**LlamaIndex AgentCore:**
```python
async def run():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    response = await agent.achat(
        "Navigate to https://example.com, wait for content to load, "
        "extract text from .content element, and take a screenshot"
    )
    return response.response
```

### Multiple Pages

**Playwright:**
```python
async def scrape_multiple_pages():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        
        results = []
        for url in urls:
            page = await browser.new_page()
            await page.goto(url)
            title = await page.title()
            content = await page.text_content("body")
            results.append({"url": url, "title": title, "content": content})
            await page.close()
        
        await browser.close()
        return results
```

**LlamaIndex AgentCore:**
```python
async def scrape_multiple_pages():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    results = []
    for url in urls:
        response = await agent.achat(
            f"Navigate to {url} and extract the page title and main content"
        )
        results.append(response.response)
    
    return results
```

### Mobile Emulation

**Playwright:**
```python
async def mobile_scraping():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context(
            viewport={'width': 375, 'height': 667},
            user_agent='Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)'
        )
        page = await context.new_page()
        await page.goto("https://example.com")
```

**LlamaIndex AgentCore:**
```python
# Configure mobile viewport in config.yaml
browser:
  viewport_width: 375
  viewport_height: 667
  user_agent: "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"

# Then use normally
response = await agent.achat("Navigate to https://example.com")
```

## Migration from Puppeteer

### Basic Setup

**Puppeteer:**
```javascript
const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  await page.goto('https://example.com');
  
  const title = await page.title();
  const content = await page.$eval('.content', el => el.textContent);
  
  await browser.close();
})();
```

**LlamaIndex AgentCore:**
```python
integration = LlamaIndexAgentCoreIntegration()
agent = integration.create_agent()

response = await agent.achat(
    "Navigate to https://example.com and extract the page title and content from .content element"
)
```

### PDF Generation

**Puppeteer:**
```javascript
await page.goto('https://example.com');
await page.pdf({
  path: 'page.pdf',
  format: 'A4',
  printBackground: true
});
```

**LlamaIndex AgentCore:**
```python
# PDF generation not directly supported, but you can:
# 1. Take full-page screenshot
# 2. Extract content and generate PDF programmatically

response = await agent.achat(
    "Navigate to https://example.com and take a full-page screenshot"
)

# Convert screenshot to PDF using external library
from PIL import Image
import base64

image_data = base64.b64decode(response.metadata["screenshot"])
image = Image.open(io.BytesIO(image_data))
image.save("page.pdf", "PDF")
```

## Common Migration Patterns

### 1. Error Handling

**Traditional:**
```python
from selenium.common.exceptions import TimeoutException, NoSuchElementException

try:
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "myElement"))
    )
    element.click()
except TimeoutException:
    print("Element not found within timeout")
except NoSuchElementException:
    print("Element not found")
```

**LlamaIndex AgentCore:**
```python
try:
    response = await agent.achat(
        "Find and click the element with ID 'myElement'. "
        "If it's not found, tell me what elements are available instead."
    )
except AgentCoreBrowserError as e:
    print(f"Browser operation failed: {e}")
```

### 2. Wait Strategies

**Traditional:**
```python
# Explicit waits
wait = WebDriverWait(driver, 10)
element = wait.until(EC.element_to_be_clickable((By.ID, "submit")))

# Implicit waits
driver.implicitly_wait(10)
```

**LlamaIndex AgentCore:**
```python
# Waits are handled automatically by the AI agent
response = await agent.achat(
    "Wait for the submit button to become clickable, then click it"
)

# Or configure global timeouts
config = {
    "browser": {
        "page_load_timeout": 30,
        "element_wait_timeout": 10
    }
}
```

### 3. Data Extraction

**Traditional:**
```python
# Extract structured data
products = []
product_elements = driver.find_elements(By.CSS_SELECTOR, ".product")

for element in product_elements:
    name = element.find_element(By.CSS_SELECTOR, ".name").text
    price = element.find_element(By.CSS_SELECTOR, ".price").text
    products.append({"name": name, "price": price})
```

**LlamaIndex AgentCore:**
```python
response = await agent.achat(
    "Extract all products from this page. For each product, get the name and price. "
    "Return the data in a structured format."
)

# The AI will automatically structure the data
products = response.metadata.get("structured_data", [])
```

## Advanced Migration Scenarios

### 1. Complex Workflows

**Traditional Multi-Step Process:**
```python
def complex_workflow():
    driver.get("https://example.com")
    
    # Step 1: Login
    driver.find_element(By.ID, "username").send_keys("user")
    driver.find_element(By.ID, "password").send_keys("pass")
    driver.find_element(By.ID, "login").click()
    
    # Step 2: Navigate to dashboard
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "dashboard"))
    )
    driver.find_element(By.LINK_TEXT, "Dashboard").click()
    
    # Step 3: Extract data
    data_elements = driver.find_elements(By.CSS_SELECTOR, ".data-row")
    data = [elem.text for elem in data_elements]
    
    # Step 4: Logout
    driver.find_element(By.ID, "logout").click()
    
    return data
```

**LlamaIndex AgentCore Workflow:**
```python
async def complex_workflow():
    integration = LlamaIndexAgentCoreIntegration()
    agent = integration.create_agent()
    
    response = await agent.achat(
        "Please perform the following workflow:\n"
        "1. Navigate to https://example.com\n"
        "2. Log in with username 'user' and password 'pass'\n"
        "3. Go to the Dashboard page\n"
        "4. Extract all data from elements with class 'data-row'\n"
        "5. Log out\n"
        "Return the extracted data."
    )
    
    return response.response
```

### 2. Dynamic Content Handling

**Traditional:**
```python
# Handle dynamic content loading
def wait_for_dynamic_content():
    # Wait for AJAX to complete
    wait = WebDriverWait(driver, 20)
    wait.until(lambda driver: driver.execute_script("return jQuery.active == 0"))
    
    # Wait for specific content
    wait.until(EC.text_to_be_present_in_element((By.ID, "status"), "Complete"))
    
    # Extract content
    content = driver.find_element(By.ID, "dynamic-content").text
    return content
```

**LlamaIndex AgentCore:**
```python
response = await agent.achat(
    "Wait for all dynamic content to finish loading (including AJAX requests), "
    "then extract the content from the 'dynamic-content' element"
)
```

### 3. File Handling

**Traditional:**
```python
# File upload
file_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
file_input.send_keys("/path/to/file.pdf")

# File download
driver.get("https://example.com/download/file.pdf")
# Handle download dialog or wait for download
```

**LlamaIndex AgentCore:**
```python
# File operations require special handling
# Upload: Use form interaction tool
form_tool = FormInteractionTool(browser_client)
await form_tool.acall(
    form_data={"input[type='file']": "/path/to/file.pdf"}
)

# Download: Describe the action
response = await agent.achat(
    "Click the download link for file.pdf and confirm the download"
)
```

## Migration Checklist

### Pre-Migration Assessment

- [ ] **Identify current browser operations**
  - Navigation patterns
  - Element interactions
  - Data extraction methods
  - Error handling strategies

- [ ] **Catalog dependencies**
  - Browser drivers
  - Wait strategies
  - Custom utilities
  - Test frameworks

- [ ] **Document workflows**
  - Multi-step processes
  - Conditional logic
  - Error recovery
  - Data processing

### Migration Steps

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   aws configure
   ```

2. **Create Configuration**
   ```yaml
   # config.yaml
   aws:
     region: "us-east-1"
   browser:
     headless: true
     viewport_width: 1920
     viewport_height: 1080
   ```

3. **Convert Simple Operations First**
   - Start with basic navigation
   - Move to simple element interactions
   - Add data extraction

4. **Handle Complex Workflows**
   - Break down into declarative steps
   - Use AI agent orchestration
   - Implement error handling

5. **Test and Validate**
   - Compare results with original implementation
   - Performance testing
   - Error scenario testing

### Post-Migration Optimization

- [ ] **Performance Tuning**
  - Optimize browser settings
  - Implement session pooling
  - Use concurrent operations

- [ ] **Error Handling**
  - Implement retry mechanisms
  - Add monitoring and alerting
  - Create fallback strategies

- [ ] **Monitoring**
  - Set up logging
  - Track performance metrics
  - Monitor costs

## Migration Examples

### Complete Example: E-commerce Scraper

**Original Selenium Implementation:**
```python
class EcommerceScraper:
    def __init__(self):
        self.driver = webdriver.Chrome()
    
    def scrape_products(self, category_url):
        self.driver.get(category_url)
        
        # Handle cookie banner
        try:
            cookie_btn = self.driver.find_element(By.ID, "accept-cookies")
            cookie_btn.click()
        except NoSuchElementException:
            pass
        
        # Load all products (infinite scroll)
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        # Extract products
        products = []
        product_elements = self.driver.find_elements(By.CSS_SELECTOR, ".product-item")
        
        for element in product_elements:
            try:
                name = element.find_element(By.CSS_SELECTOR, ".product-name").text
                price = element.find_element(By.CSS_SELECTOR, ".price").text
                rating = element.find_element(By.CSS_SELECTOR, ".rating").get_attribute("data-rating")
                image_url = element.find_element(By.CSS_SELECTOR, "img").get_attribute("src")
                
                products.append({
                    "name": name,
                    "price": price,
                    "rating": rating,
                    "image_url": image_url
                })
            except NoSuchElementException:
                continue
        
        return products
    
    def close(self):
        self.driver.quit()
```

**Migrated LlamaIndex AgentCore Implementation:**
```python
class EcommerceScraper:
    def __init__(self):
        self.integration = LlamaIndexAgentCoreIntegration()
        self.agent = self.integration.create_agent()
    
    async def scrape_products(self, category_url):
        response = await self.agent.achat(f"""
        Navigate to {category_url} and scrape all products. Please:
        
        1. Accept any cookie banners if they appear
        2. Scroll down to load all products (handle infinite scroll)
        3. Extract the following information for each product:
           - Product name
           - Price
           - Rating (if available)
           - Image URL
        4. Return the data in a structured JSON format
        
        Make sure to get all products on the page, even if you need to scroll multiple times.
        """)
        
        # The agent returns structured data automatically
        return response.metadata.get("products", [])
    
    async def close(self):
        # Sessions are automatically managed
        pass
```

### Benefits of Migration

1. **Reduced Code Complexity**: 50+ lines → 20 lines
2. **Better Error Handling**: AI handles edge cases automatically
3. **Improved Maintainability**: Declarative approach is easier to understand
4. **Enhanced Reliability**: Managed infrastructure reduces failures
5. **Built-in Intelligence**: AI can adapt to page changes

### Migration Timeline

**Week 1-2: Assessment and Planning**
- Audit existing automation scripts
- Identify migration priorities
- Set up development environment

**Week 3-4: Basic Migration**
- Convert simple navigation and extraction
- Implement basic error handling
- Create test cases

**Week 5-6: Complex Workflows**
- Migrate multi-step processes
- Implement advanced features
- Performance optimization

**Week 7-8: Testing and Deployment**
- Comprehensive testing
- Performance validation
- Production deployment

## Getting Help

### Migration Support Resources

1. **Documentation**: Complete API reference and examples
2. **Migration Tools**: Automated conversion utilities (coming soon)
3. **Community**: Forums and discussion groups
4. **Professional Services**: Migration consulting available

### Common Migration Issues

1. **Timing Issues**: AI handles waits automatically
2. **Element Selection**: Describe elements naturally instead of selectors
3. **Complex Logic**: Break down into simple, declarative steps
4. **Performance**: Use concurrent operations and session pooling

### Success Metrics

Track these metrics to measure migration success:

- **Code Reduction**: Lines of code decreased
- **Reliability**: Error rate improvement
- **Maintenance**: Time spent on updates
- **Performance**: Execution time comparison
- **Scalability**: Concurrent operation capacity

The migration to LlamaIndex AgentCore Browser Integration represents a shift from imperative browser control to intelligent, declarative web automation. While the initial learning curve exists, the long-term benefits in maintainability, reliability, and scalability make it a worthwhile investment.