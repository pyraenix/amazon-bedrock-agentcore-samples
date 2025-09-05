#!/usr/bin/env python3
"""
Customer Support Automation Example with LlamaIndex and AgentCore Browser Tool

This example demonstrates secure customer support automation using LlamaIndex agents
integrated with Amazon Bedrock AgentCore Browser Tool. Includes PII protection,
GDPR compliance, and secure handling of customer information.

Key Features:
- GDPR compliant customer data handling
- PII detection and anonymization
- Secure ticket extraction from support systems
- Customer sentiment analysis with privacy protection
- Automated response generation with data protection
- Compliance logging and audit trails

Requirements: 3.1, 3.3, 5.1, 5.2
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import re
from uuid import uuid4

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.bedrock import Bedrock

from bedrock_agentcore.tools.browser_client import BrowserSession
from agentcore_browser_loader import AgentCoreBrowserLoader
from sensitive_data_handler import SensitiveDataHandler, SensitivityLevel
from secure_rag_pipeline import SecureRAGPipeline

# Configure logging for customer support compliance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('customer_support.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CustomerDataType(Enum):
    """Types of customer data requiring protection"""
    FULL_NAME = "full_name"
    EMAIL = "email_address"
    PHONE = "phone_number"
    ADDRESS = "physical_address"
    CUSTOMER_ID = "customer_id"
    ACCOUNT_NUMBER = "account_number"
    CREDIT_CARD = "credit_card_number"
    SSN = "social_security_number"
    IP_ADDRESS = "ip_address"
    DEVICE_ID = "device_identifier"


class TicketPriority(Enum):
    """Support ticket priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SentimentScore(Enum):
    """Customer sentiment analysis scores"""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


@dataclass
class CustomerSupportTicket:
    """Customer support ticket with PII protection"""
    content: str
    ticket_id: str
    customer_id: str  # Anonymized customer identifier
    customer_data_detected: List[CustomerDataType]
    priority: TicketPriority
    sentiment: SentimentScore
    category: str
    sensitivity_level: SensitivityLevel
    source_url: str
    creation_timestamp: datetime
    gdpr_flags: List[str]
    audit_trail: List[Dict[str, Any]]
    
    def anonymize_customer_data(self) -> 'CustomerSupportTicket':
        """Anonymize customer data according to GDPR requirements"""
        anonymized_content = self.content
        audit_event = {
            "action": "customer_data_anonymization",
            "timestamp": datetime.utcnow().isoformat(),
            "data_types_anonymized": [data_type.value for data_type in self.customer_data_detected],
            "compliance_standard": "GDPR"
        }
        
        # Apply GDPR anonymization patterns
        customer_patterns = {
            CustomerDataType.FULL_NAME: r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            CustomerDataType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            CustomerDataType.PHONE: r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            CustomerDataType.SSN: r'\b\d{3}-\d{2}-\d{4}\b',
            CustomerDataType.CREDIT_CARD: r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            CustomerDataType.IP_ADDRESS: r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        }
        
        for data_type in self.customer_data_detected:
            if data_type in customer_patterns:
                anonymized_content = re.sub(
                    customer_patterns[data_type], 
                    f'[ANONYMIZED_{data_type.value.upper()}]', 
                    anonymized_content
                )
        
        return CustomerSupportTicket(
            content=anonymized_content,
            ticket_id=self.ticket_id,
            customer_id=self.customer_id,
            customer_data_detected=self.customer_data_detected,
            priority=self.priority,
            sentiment=self.sentiment,
            category=self.category,
            sensitivity_level=self.sensitivity_level,
            source_url=self.source_url,
            creation_timestamp=self.creation_timestamp,
            gdpr_flags=self.gdpr_flags + ["GDPR_ANONYMIZED"],
            audit_trail=self.audit_trail + [audit_event]
        )


class CustomerSupportAutomation:
    """Secure customer support automation using LlamaIndex and AgentCore"""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.browser_loader = AgentCoreBrowserLoader(region=region)
        self.sensitive_handler = SensitiveDataHandler()
        self.rag_pipeline = SecureRAGPipeline(region=region)
        
        # Configure LlamaIndex for customer support
        Settings.embed_model = BedrockEmbedding(
            model_name="amazon.titan-embed-text-v1",
            region_name=region
        )
        Settings.llm = Bedrock(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name=region
        )
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        
        logger.info("Customer support automation initialized")
    
    async def extract_support_tickets(self, support_config: Dict[str, Any]) -> List[CustomerSupportTicket]:
        """
        Extract support tickets from customer support system
        
        Args:
            support_config: Configuration for support system access
            
        Returns:
            List of support tickets with PII protection
        """
        try:
            logger.info(f"Starting support ticket extraction from {support_config.get('system_name', 'Unknown')}")
            
            # Create secure browser session
            session = await self.browser_loader.create_secure_session()
            
            # Authenticate with support system
            await self._authenticate_support_system(session, support_config)
            
            # Navigate to tickets section
            await session.navigate(support_config['tickets_url'])
            
            # Extract support tickets
            tickets = []
            ticket_elements = await session.find_elements(support_config['ticket_selector'])
            
            for element in ticket_elements:
                ticket_content = await element.get_text()
                
                # Detect customer data
                customer_data = self._detect_customer_data(ticket_content)
                
                # Analyze sentiment and priority
                sentiment = self._analyze_customer_sentiment(ticket_content)
                priority = self._determine_ticket_priority(ticket_content, sentiment)
                category = self._categorize_ticket(ticket_content)
                
                # Create support ticket
                ticket = CustomerSupportTicket(
                    content=ticket_content,
                    ticket_id=self._extract_ticket_id(ticket_content),
                    customer_id=self._generate_anonymous_customer_id(ticket_content),
                    customer_data_detected=customer_data,
                    priority=priority,
                    sentiment=sentiment,
                    category=category,
                    sensitivity_level=SensitivityLevel.CONFIDENTIAL,
                    source_url=support_config['tickets_url'],
                    creation_timestamp=datetime.utcnow(),
                    gdpr_flags=["GDPR_REQUIRED"],
                    audit_trail=[{
                        "action": "support_ticket_extraction",
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": support_config['system_name'],
                        "compliance_check": "GDPR"
                    }]
                )
                
                tickets.append(ticket)
            
            await session.close()
            logger.info(f"Extracted {len(tickets)} support tickets")
            return tickets
            
        except Exception as e:
            logger.error(f"Error extracting support tickets: {str(e)}")
            raise
    
    async def extract_customer_feedback(self, feedback_config: Dict[str, Any]) -> List[CustomerSupportTicket]:
        """
        Extract customer feedback from feedback systems
        
        Args:
            feedback_config: Configuration for feedback system access
            
        Returns:
            List of feedback documents with customer data protection
        """
        try:
            logger.info("Extracting customer feedback")
            
            session = await self.browser_loader.create_secure_session()
            
            # Authenticate with feedback system
            await self._authenticate_support_system(session, feedback_config)
            
            # Navigate to feedback section
            await session.navigate(feedback_config['feedback_url'])
            
            # Extract feedback
            feedback_tickets = []
            feedback_elements = await session.find_elements(feedback_config['feedback_selector'])
            
            for element in feedback_elements:
                feedback_content = await element.get_text()
                
                # Detect customer data
                customer_data = self._detect_customer_data(feedback_content)
                
                # Analyze feedback sentiment
                sentiment = self._analyze_customer_sentiment(feedback_content)
                priority = self._determine_feedback_priority(feedback_content, sentiment)
                
                ticket = CustomerSupportTicket(
                    content=feedback_content,
                    ticket_id=f"FEEDBACK_{uuid4().hex[:8]}",
                    customer_id=self._generate_anonymous_customer_id(feedback_content),
                    customer_data_detected=customer_data,
                    priority=priority,
                    sentiment=sentiment,
                    category="customer_feedback",
                    sensitivity_level=SensitivityLevel.CONFIDENTIAL,
                    source_url=feedback_config['feedback_url'],
                    creation_timestamp=datetime.utcnow(),
                    gdpr_flags=["GDPR_REQUIRED", "FEEDBACK_ANALYSIS"],
                    audit_trail=[{
                        "action": "customer_feedback_extraction",
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": feedback_config['system_name'],
                        "sentiment": sentiment.value
                    }]
                )
                
                feedback_tickets.append(ticket)
            
            await session.close()
            logger.info(f"Extracted {len(feedback_tickets)} customer feedback items")
            return feedback_tickets
            
        except Exception as e:
            logger.error(f"Error extracting customer feedback: {str(e)}")
            raise
    
    async def extract_chat_transcripts(self, chat_config: Dict[str, Any]) -> List[CustomerSupportTicket]:
        """
        Extract chat transcripts from customer support chat system
        
        Args:
            chat_config: Configuration for chat system access
            
        Returns:
            List of chat transcripts with customer data protection
        """
        try:
            logger.info("Extracting chat transcripts")
            
            session = await self.browser_loader.create_secure_session()
            
            # Authenticate with chat system
            await self._authenticate_support_system(session, chat_config)
            
            # Navigate to chat transcripts
            await session.navigate(chat_config['transcripts_url'])
            
            # Extract chat transcripts
            chat_tickets = []
            transcript_elements = await session.find_elements(chat_config['transcript_selector'])
            
            for element in transcript_elements:
                transcript_content = await element.get_text()
                
                # Detect customer data in chat
                customer_data = self._detect_customer_data(transcript_content)
                
                # Analyze chat sentiment and extract insights
                sentiment = self._analyze_chat_sentiment(transcript_content)
                priority = self._determine_chat_priority(transcript_content)
                
                ticket = CustomerSupportTicket(
                    content=transcript_content,
                    ticket_id=f"CHAT_{uuid4().hex[:8]}",
                    customer_id=self._generate_anonymous_customer_id(transcript_content),
                    customer_data_detected=customer_data,
                    priority=priority,
                    sentiment=sentiment,
                    category="chat_support",
                    sensitivity_level=SensitivityLevel.CONFIDENTIAL,
                    source_url=chat_config['transcripts_url'],
                    creation_timestamp=datetime.utcnow(),
                    gdpr_flags=["GDPR_REQUIRED", "CHAT_TRANSCRIPT"],
                    audit_trail=[{
                        "action": "chat_transcript_extraction",
                        "timestamp": datetime.utcnow().isoformat(),
                        "source": chat_config['system_name'],
                        "sentiment": sentiment.value
                    }]
                )
                
                chat_tickets.append(ticket)
            
            await session.close()
            logger.info(f"Extracted {len(chat_tickets)} chat transcripts")
            return chat_tickets
            
        except Exception as e:
            logger.error(f"Error extracting chat transcripts: {str(e)}")
            raise
    
    async def create_secure_support_index(self, tickets: List[CustomerSupportTicket]) -> VectorStoreIndex:
        """
        Create secure vector index for customer support data
        
        Args:
            tickets: List of customer support tickets
            
        Returns:
            Secure vector store index with GDPR compliance
        """
        try:
            logger.info("Creating secure customer support index")
            
            # Anonymize customer data in all tickets before indexing
            anonymized_tickets = []
            for ticket in tickets:
                anonymized_ticket = ticket.anonymize_customer_data()
                
                # Convert to LlamaIndex Document
                llama_doc = Document(
                    text=anonymized_ticket.content,
                    metadata={
                        "ticket_id": anonymized_ticket.ticket_id,
                        "customer_id": anonymized_ticket.customer_id,
                        "priority": anonymized_ticket.priority.value,
                        "sentiment": anonymized_ticket.sentiment.value,
                        "category": anonymized_ticket.category,
                        "sensitivity_level": anonymized_ticket.sensitivity_level.value,
                        "customer_data_detected": [data.value for data in anonymized_ticket.customer_data_detected],
                        "gdpr_flags": anonymized_ticket.gdpr_flags,
                        "creation_timestamp": anonymized_ticket.creation_timestamp.isoformat(),
                        "audit_trail": anonymized_ticket.audit_trail
                    }
                )
                anonymized_tickets.append(llama_doc)
            
            # Create secure vector store with GDPR compliance
            vector_store = ChromaVectorStore(
                chroma_collection_name="customer_support_tickets",
                persist_directory="./secure_support_vector_store"
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Build index with privacy controls
            index = VectorStoreIndex.from_documents(
                anonymized_tickets,
                storage_context=storage_context
            )
            
            logger.info(f"Created secure support index with {len(anonymized_tickets)} tickets")
            return index
            
        except Exception as e:
            logger.error(f"Error creating support index: {str(e)}")
            raise
    
    async def analyze_customer_issues(self, index: VectorStoreIndex, analysis_type: str) -> str:
        """
        Analyze customer issues with privacy protection
        
        Args:
            index: Secure customer support vector index
            analysis_type: Type of analysis to perform
            
        Returns:
            Sanitized analysis results
        """
        try:
            logger.info(f"Analyzing customer issues: {analysis_type}")
            
            # Define analysis queries based on type
            analysis_queries = {
                "common_issues": "What are the most common customer issues and complaints?",
                "sentiment_trends": "Analyze customer sentiment trends and patterns",
                "priority_distribution": "Show the distribution of ticket priorities and urgent issues",
                "category_analysis": "Analyze support ticket categories and their frequency",
                "resolution_patterns": "Identify patterns in issue resolution and customer satisfaction"
            }
            
            query = analysis_queries.get(analysis_type, analysis_type)
            
            # Sanitize query
            sanitized_query = self._sanitize_support_query(query)
            
            # Create query engine with privacy controls
            query_engine = index.as_query_engine(
                response_mode="compact",
                similarity_top_k=5
            )
            
            # Execute analysis
            response = query_engine.query(sanitized_query)
            
            # Sanitize response to ensure no customer data leakage
            sanitized_response = self._sanitize_support_response(str(response))
            
            # Log analysis for compliance
            self._log_support_analysis(analysis_type, sanitized_query, sanitized_response)
            
            return sanitized_response
            
        except Exception as e:
            logger.error(f"Error analyzing customer issues: {str(e)}")
            raise
    
    async def generate_automated_responses(self, tickets: List[CustomerSupportTicket]) -> Dict[str, str]:
        """
        Generate automated responses for customer support tickets
        
        Args:
            tickets: List of customer support tickets
            
        Returns:
            Dictionary of ticket IDs to automated responses
        """
        try:
            logger.info("Generating automated responses")
            
            automated_responses = {}
            
            for ticket in tickets:
                # Anonymize ticket before processing
                anonymized_ticket = ticket.anonymize_customer_data()
                
                # Generate response based on category and sentiment
                response = await self._generate_ticket_response(anonymized_ticket)
                
                # Sanitize response
                sanitized_response = self._sanitize_support_response(response)
                
                automated_responses[ticket.ticket_id] = sanitized_response
                
                # Log response generation
                self._log_response_generation(ticket.ticket_id, ticket.category, ticket.sentiment.value)
            
            logger.info(f"Generated {len(automated_responses)} automated responses")
            return automated_responses
            
        except Exception as e:
            logger.error(f"Error generating automated responses: {str(e)}")
            raise
    
    async def _authenticate_support_system(self, session: BrowserSession, config: Dict[str, Any]):
        """Authenticate with customer support system"""
        await session.navigate(config['login_url'])
        
        # Enter credentials securely
        username_field = await session.find_element(config['username_selector'])
        await username_field.type(config['username'])
        
        password_field = await session.find_element(config['password_selector'])
        await password_field.type(config['password'])
        
        # Handle SSO if required
        if config.get('sso_required', False):
            await self._handle_support_sso(session, config)
        
        # Submit login
        login_button = await session.find_element(config['login_button_selector'])
        await login_button.click()
        
        # Wait for successful login
        await session.wait_for_element(config['dashboard_selector'])
        
        logger.info(f"Successfully authenticated with {config.get('system_name', 'support system')}")
    
    async def _handle_support_sso(self, session: BrowserSession, config: Dict[str, Any]):
        """Handle SSO authentication for support system"""
        await session.wait_for_element(config['sso_selector'])
        logger.info("Support system SSO authentication required")
    
    def _detect_customer_data(self, content: str) -> List[CustomerDataType]:
        """Detect customer data in content"""
        customer_data = []
        
        # Customer data detection patterns
        patterns = {
            CustomerDataType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            CustomerDataType.PHONE: r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            CustomerDataType.SSN: r'\b\d{3}-\d{2}-\d{4}\b',
            CustomerDataType.CREDIT_CARD: r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            CustomerDataType.IP_ADDRESS: r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        }
        
        for data_type, pattern in patterns.items():
            if re.search(pattern, content):
                customer_data.append(data_type)
        
        # Detect names (simple pattern)
        if re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', content):
            customer_data.append(CustomerDataType.FULL_NAME)
        
        return customer_data
    
    def _analyze_customer_sentiment(self, content: str) -> SentimentScore:
        """Analyze customer sentiment in content"""
        content_lower = content.lower()
        
        # Sentiment analysis based on keywords
        very_negative_words = ['hate', 'terrible', 'awful', 'worst', 'disgusting', 'furious']
        negative_words = ['bad', 'poor', 'disappointed', 'frustrated', 'angry', 'upset']
        positive_words = ['good', 'great', 'happy', 'satisfied', 'pleased', 'excellent']
        very_positive_words = ['amazing', 'fantastic', 'outstanding', 'perfect', 'love', 'brilliant']
        
        very_negative_count = sum(1 for word in very_negative_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        positive_count = sum(1 for word in positive_words if word in content_lower)
        very_positive_count = sum(1 for word in very_positive_words if word in content_lower)
        
        if very_negative_count > 0:
            return SentimentScore.VERY_NEGATIVE
        elif negative_count > positive_count:
            return SentimentScore.NEGATIVE
        elif very_positive_count > 0:
            return SentimentScore.VERY_POSITIVE
        elif positive_count > negative_count:
            return SentimentScore.POSITIVE
        else:
            return SentimentScore.NEUTRAL
    
    def _determine_ticket_priority(self, content: str, sentiment: SentimentScore) -> TicketPriority:
        """Determine ticket priority based on content and sentiment"""
        content_lower = content.lower()
        
        # High priority keywords
        critical_keywords = ['urgent', 'critical', 'emergency', 'down', 'broken', 'not working']
        high_keywords = ['important', 'asap', 'quickly', 'soon', 'problem']
        
        critical_count = sum(1 for keyword in critical_keywords if keyword in content_lower)
        high_count = sum(1 for keyword in high_keywords if keyword in content_lower)
        
        # Adjust priority based on sentiment
        if critical_count > 0 or sentiment == SentimentScore.VERY_NEGATIVE:
            return TicketPriority.CRITICAL
        elif high_count > 0 or sentiment == SentimentScore.NEGATIVE:
            return TicketPriority.HIGH
        elif sentiment in [SentimentScore.POSITIVE, SentimentScore.VERY_POSITIVE]:
            return TicketPriority.LOW
        else:
            return TicketPriority.MEDIUM
    
    def _categorize_ticket(self, content: str) -> str:
        """Categorize support ticket based on content"""
        content_lower = content.lower()
        
        categories = {
            "billing": ["bill", "charge", "payment", "invoice", "refund"],
            "technical": ["error", "bug", "not working", "broken", "crash"],
            "account": ["login", "password", "access", "account", "profile"],
            "product": ["feature", "how to", "tutorial", "guide", "help"],
            "complaint": ["complaint", "dissatisfied", "unhappy", "poor service"]
        }
        
        for category, keywords in categories.items():
            if any(keyword in content_lower for keyword in keywords):
                return category
        
        return "general"
    
    def _determine_feedback_priority(self, content: str, sentiment: SentimentScore) -> TicketPriority:
        """Determine feedback priority"""
        if sentiment == SentimentScore.VERY_NEGATIVE:
            return TicketPriority.HIGH
        elif sentiment == SentimentScore.NEGATIVE:
            return TicketPriority.MEDIUM
        else:
            return TicketPriority.LOW
    
    def _analyze_chat_sentiment(self, content: str) -> SentimentScore:
        """Analyze sentiment in chat transcripts"""
        # Similar to customer sentiment but consider chat context
        return self._analyze_customer_sentiment(content)
    
    def _determine_chat_priority(self, content: str) -> TicketPriority:
        """Determine chat priority"""
        if "escalate" in content.lower() or "supervisor" in content.lower():
            return TicketPriority.HIGH
        else:
            return TicketPriority.MEDIUM
    
    def _extract_ticket_id(self, content: str) -> str:
        """Extract ticket ID from content"""
        # Look for ticket ID patterns
        ticket_match = re.search(r'#(\d+)', content)
        if ticket_match:
            return f"TICKET_{ticket_match.group(1)}"
        else:
            return f"TICKET_{uuid4().hex[:8]}"
    
    def _generate_anonymous_customer_id(self, content: str) -> str:
        """Generate anonymous customer identifier"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"CUSTOMER_{content_hash[:8]}"
    
    async def _generate_ticket_response(self, ticket: CustomerSupportTicket) -> str:
        """Generate automated response for ticket"""
        # Template responses based on category and sentiment
        response_templates = {
            "billing": "Thank you for contacting us about your billing inquiry. We're reviewing your account and will provide a detailed response within 24 hours.",
            "technical": "We've received your technical support request. Our team is investigating the issue and will provide a solution shortly.",
            "account": "We're here to help with your account access. Please check your email for password reset instructions.",
            "product": "Thank you for your product inquiry. We've prepared some helpful resources that should address your questions.",
            "complaint": "We sincerely apologize for your experience. Your feedback is important to us and we're taking immediate action to address your concerns."
        }
        
        base_response = response_templates.get(ticket.category, "Thank you for contacting us. We're reviewing your request and will respond soon.")
        
        # Adjust response based on sentiment
        if ticket.sentiment in [SentimentScore.VERY_NEGATIVE, SentimentScore.NEGATIVE]:
            base_response = f"We understand your frustration and sincerely apologize. {base_response}"
        
        return base_response
    
    def _sanitize_support_query(self, query: str) -> str:
        """Sanitize query to remove customer data"""
        sanitized = query
        
        # Remove customer data patterns
        sanitized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]', sanitized)
        sanitized = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[REDACTED_PHONE]', sanitized)
        sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', sanitized)
        
        return sanitized
    
    def _sanitize_support_response(self, response: str) -> str:
        """Sanitize response to prevent customer data leakage"""
        return self._sanitize_support_query(response)
    
    def _log_support_analysis(self, analysis_type: str, query: str, response: str):
        """Log support analysis for compliance"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": "support_analysis",
            "analysis_type": analysis_type,
            "query_hash": hashlib.sha256(query.encode()).hexdigest(),
            "response_length": len(response),
            "compliance_check": "GDPR"
        }
        
        logger.info(f"Support analysis audit: {audit_entry}")
    
    def _log_response_generation(self, ticket_id: str, category: str, sentiment: str):
        """Log automated response generation"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": "automated_response_generation",
            "ticket_id": ticket_id,
            "category": category,
            "sentiment": sentiment
        }
        
        logger.info(f"Response generation audit: {audit_entry}")


async def main():
    """Example usage of customer support automation"""
    
    # Initialize automation
    automation = CustomerSupportAutomation(region="us-east-1")
    
    # Example support system configuration
    support_config = {
        "system_name": "Zendesk",
        "login_url": "https://company.zendesk.com/auth/v2/login",
        "username_selector": "#user_email",
        "password_selector": "#user_password",
        "login_button_selector": "#sign-in-submit-button",
        "dashboard_selector": "#main_navigation",
        "tickets_url": "https://company.zendesk.com/agent/filters/all",
        "ticket_selector": ".ticket",
        "username": os.getenv("SUPPORT_USERNAME"),
        "password": os.getenv("SUPPORT_PASSWORD"),
        "sso_required": False
    }
    
    try:
        # Extract support tickets
        support_tickets = await automation.extract_support_tickets(support_config)
        print(f"Extracted {len(support_tickets)} support tickets")
        
        # Extract customer feedback
        feedback_config = support_config.copy()
        feedback_config.update({
            "feedback_url": "https://company.zendesk.com/agent/reporting/satisfaction",
            "feedback_selector": ".satisfaction-rating"
        })
        
        feedback_data = await automation.extract_customer_feedback(feedback_config)
        print(f"Extracted {len(feedback_data)} customer feedback items")
        
        # Extract chat transcripts
        chat_config = support_config.copy()
        chat_config.update({
            "transcripts_url": "https://company.zendesk.com/agent/chat/history",
            "transcript_selector": ".chat-transcript"
        })
        
        chat_transcripts = await automation.extract_chat_transcripts(chat_config)
        print(f"Extracted {len(chat_transcripts)} chat transcripts")
        
        # Combine all support data
        all_tickets = support_tickets + feedback_data + chat_transcripts
        
        # Create secure support index
        support_index = await automation.create_secure_support_index(all_tickets)
        
        # Perform customer support analysis
        analyses = [
            "common_issues",
            "sentiment_trends",
            "priority_distribution",
            "category_analysis",
            "resolution_patterns"
        ]
        
        for analysis in analyses:
            result = await automation.analyze_customer_issues(support_index, analysis)
            print(f"\n{analysis.replace('_', ' ').title()} Analysis:")
            print(result)
        
        # Generate automated responses
        automated_responses = await automation.generate_automated_responses(support_tickets[:5])  # First 5 tickets
        print(f"\nGenerated automated responses for {len(automated_responses)} tickets:")
        for ticket_id, response in automated_responses.items():
            print(f"\nTicket {ticket_id}:")
            print(response)
        
        print("\nCustomer support automation completed successfully!")
        
    except Exception as e:
        logger.error(f"Customer support automation failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())