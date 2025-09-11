# Tutorial Validation Framework - Get Started

## ✅ Installation Complete!

The Tutorial Validation Framework is now installed and ready to use.

## 🚀 Quick Start

### 1. Set Up Python 3.12 Environment

**First time setup:**
```bash
# Run the setup script (creates Python 3.12 virtual environment)
python3 setup_python312_env.py
```

**Activate the environment:**
```bash
source tutorial_validation_framework/activate_tvf_312.sh
```

> 📖 **Detailed Setup Guide**: See `tutorial_validation_framework/SETUP_PYTHON312.md` for complete setup instructions and troubleshooting.

### 2. Basic Commands

```bash
# Show help
tvf --help

# List available frameworks
tvf --list-frameworks

# Run a dry-run to see what would happen
tvf --dry-run --verbose

# Test a specific framework
tvf --frameworks novaact

# Test all frameworks
tvf --all

# Run in interactive mode
tvf --interactive
```

### 3. Try the Demo
```bash
# Make sure environment is activated first
source tutorial_validation_framework/activate_tvf_312.sh

# Navigate to framework directory
cd tutorial_validation_framework

# Run the working demo (recommended)
python working_demo.py

# Or run the interactive demo
python demo.py

# Or run structure validation
python structure_validation.py
```

## 🎯 What This Framework Does

The Tutorial Validation Framework tests your AgentCore browser tool tutorials across multiple AI frameworks:

### **Frameworks Supported:**
- ✅ **NovaAct** - AI-powered browser automation
- ✅ **BrowserUse** - Browser interaction framework  
- ✅ **Strands** - Web automation toolkit
- ✅ **LlamaIndex** - Data framework with browser tools

### **What It Tests:**
1. **🔧 Basic Functionality** - Does the tutorial run without errors?
2. **🤖 CAPTCHA Handling** - Can it detect and solve CAPTCHAs?
3. **🔒 Security & Privacy** - Are passwords and sensitive data protected?
4. **⚡ Performance** - How fast and efficient is each framework?
5. **🔄 Cross-Framework Compatibility** - Do tutorials work consistently?

## 📊 Example Usage

### Test All Frameworks
```bash
tvf --all --verbose
```

**Output:**
```
🚀 Starting validation for frameworks: ['novaact', 'browseruse', 'strands', 'llamaindex']
📋 NovaAct: ✅ Basic (5.2s) ✅ CAPTCHA (85% success) ✅ Security
📋 BrowserUse: ✅ Basic (4.8s) ⚠️ CAPTCHA (72% success) ✅ Security  
📋 Strands: ❌ Setup failed (missing dependency)
📋 LlamaIndex: ✅ Basic (6.1s) ✅ CAPTCHA (91% success) ✅ Security

📄 Report generated: ./reports/validation_report_20250911.html
```

### Focus on Security Testing
```bash
tvf --frameworks all --security-tests --sensitive-info-tests --verbose
```

### CI/CD Integration
```bash
tvf --all --ci-mode --output-format junit --output-dir ./test-results
```

## 📁 Configuration

Your configuration is stored at: `~/.tutorial_validation_framework/config.yaml`

```yaml
# Example configuration
frameworks:
  - novaact
  - browseruse
  - strands
  - llamaindex

execution:
  parallel: false
  timeout_minutes: 10

testing:
  captcha_tests: true
  sensitive_info_tests: true

reporting:
  output_directory: "./reports"
  formats: ["html", "json"]
```

## 🔧 Troubleshooting

### Common Issues

**Issue**: `ImportError` when running commands
**Solution**: Make sure you've activated the Python 3.12 environment:
```bash
source tutorial_validation_framework/activate_tvf_312.sh
```

**Issue**: Framework not found
**Solution**: Check available frameworks:
```bash
tvf --list-frameworks
```

**Issue**: Tests fail with dependency errors
**Solution**: Install missing dependencies:
```bash
pip install pyyaml requests psutil
```

## 📚 Documentation

- **Quick Start**: `tutorial_validation_framework/QUICK_START_GUIDE.md`
- **Practical Examples**: `tutorial_validation_framework/PRACTICAL_EXAMPLE.md`
- **User Guide**: `tutorial_validation_framework/docs/USER_GUIDE.md`
- **API Reference**: `tutorial_validation_framework/docs/API_REFERENCE.md`
- **Troubleshooting**: `tutorial_validation_framework/docs/TROUBLESHOOTING.md`

## 🎉 You're Ready!

The Tutorial Validation Framework is now set up and ready to validate your AgentCore browser tool tutorials. Start with:

```bash
# Activate environment
source tutorial_validation_framework/activate_tvf_312.sh

# Navigate to framework directory
cd tutorial_validation_framework

# Run the working demo
python working_demo.py

# Test your tutorials
python cli.py --help
```

Happy validating! 🚀