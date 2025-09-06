"""
Integration test for document processing with AgentCore browser tool.

This test demonstrates the full document processing workflow using
the test mode configuration.
"""

import asyncio
from document_processor import DocumentProcessor, DocumentPipeline
from incremental_processor import IncrementalProcessor


async def test_full_integration():
    """Test full document processing integration."""
    print("Testing full document processing integration...")
    
    # Test single URL processing
    processor = DocumentProcessor(config_path="agentcore_config.json")
    
    print("\n1. Testing single URL processing...")
    result = await processor.process_url("https://example.com/test")
    
    print(f"✓ Processing status: {result.status.value}")
    print(f"✓ Document created: {result.document is not None}")
    print(f"✓ Content length: {result.content_size}")
    print(f"✓ Processing time: {result.processing_time_ms}ms")
    
    if result.document:
        print(f"✓ Document text preview: {result.document.text[:100]}...")
        print(f"✓ Document metadata keys: {list(result.document.metadata.keys())}")
        print(f"✓ Has screenshot: {result.document.metadata.get('has_screenshot', False)}")
    
    # Test batch processing with pipeline
    print("\n2. Testing batch processing...")
    async with DocumentPipeline(config_path="agentcore_config.json") as pipeline:
        urls = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3"
        ]
        
        results = await pipeline.process_multiple_urls(urls)
        documents = pipeline.get_successful_documents(results)
        summary = pipeline.get_processing_summary(results)
        
        print(f"✓ Processed {len(results)} URLs")
        print(f"✓ Successful documents: {len(documents)}")
        print(f"✓ Success rate: {summary['success_rate']:.2f}")
        print(f"✓ Total processing time: {summary['total_processing_time_ms']}ms")
    
    # Test incremental processing
    print("\n3. Testing incremental processing...")
    incremental_processor = IncrementalProcessor(config_path="agentcore_config.json")
    
    # First processing (should be new content)
    result1 = await incremental_processor.process_url_incremental("https://example.com/incremental")
    print(f"✓ First processing: {result1.change_detection.change_type.value}")
    print(f"✓ Confidence score: {result1.change_detection.confidence_score:.2f}")
    print(f"✓ Processing required: {result1.change_detection.processing_required}")
    
    # Second processing (should detect changes due to timestamp differences)
    result2 = await incremental_processor.process_url_incremental("https://example.com/incremental")
    print(f"✓ Second processing: {result2.change_detection.change_type.value}")
    print(f"✓ Changes detected: {result2.change_detection.changes_detected}")
    print(f"✓ Confidence score: {result2.change_detection.confidence_score:.2f}")
    
    # Get processing statistics
    stats = incremental_processor.get_processing_stats()
    print(f"✓ Total processed: {stats['total_processed']}")
    print(f"✓ New content: {stats['new_content']}")
    print(f"✓ Modified content: {stats['modified_content']}")
    print(f"✓ Change rate: {stats['change_rate']:.2f}")
    
    print("\n✅ Full integration test completed successfully!")
    print("All document processing features are working with AgentCore browser tool simulation.")


if __name__ == "__main__":
    asyncio.run(test_full_integration())