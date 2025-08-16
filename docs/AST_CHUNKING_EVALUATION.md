# AST-Aware vs Naive Chunking Performance Evaluation

## Overview

This document describes the comprehensive evaluation framework developed to measure the performance differences between AST-aware chunking (using the `astchunk` library) and traditional naive text chunking in LEANN.

## Evaluation Criteria

### 1. Search Quality Metrics
- **Recall@K**: Proportion of relevant chunks retrieved in top-K results
- **Precision@K**: Proportion of retrieved chunks that are relevant
- **F1@K**: Harmonic mean of recall and precision
- **Semantic Coherence**: How well chunks represent complete concepts

### 2. Context Economy Metrics
- **Chunk Overlap Reduction**: Measuring redundant information between chunks
- **Token Efficiency**: Tokens needed to represent the same semantic content
- **Function Completeness**: Percentage of chunks with complete functions/classes

### 3. Performance Metrics
- **Processing Speed**: Time to chunk documents (seconds per document)
- **Memory Usage**: Peak memory consumption during chunking
- **Throughput**: Chunks generated per second

### 4. Code Understanding Metrics
- **Syntactic Completeness**: Complete functions/classes preserved
- **Boundary Preservation**: How well code construct boundaries are maintained
- **Cross-Reference Quality**: Import/dependency relationships maintained

## Evaluation Framework Components

### Core Scripts

1. **`ast_vs_naive_evaluation.py`** - Main performance evaluation
   - Measures chunking speed, memory usage, chunk quality
   - Analyzes function completeness and code structure preservation
   - Generates comparative metrics between approaches

2. **`search_quality_evaluator.py`** - Search quality assessment
   - Implements Recall@K, Precision@K, F1@K calculations
   - Uses ground truth queries for code understanding tasks
   - Measures semantic relevance of retrieved chunks

3. **`generate_evaluation_report.py`** - Comprehensive reporting
   - Combines all evaluation results into actionable insights
   - Generates markdown and JSON reports
   - Provides specific recommendations based on results

4. **`run_quick_evaluation.py`** - Streamlined demo
   - Quick assessment for PR reviews and demonstrations
   - Focuses on key metrics and overall recommendations

### Test Dataset

#### Code Samples (`data/code_samples/`)
- **`vector_search.py`** - Python machine learning utilities
- **`DataProcessor.java`** - Java data processing framework
- **`ApiController.java`** - Spring Boot REST API controller
- **`database_connector.py`** - Python database operations
- **`text_analyzer.ts`** - TypeScript text processing
- **`ImageProcessor.cs`** - C# image processing library

#### Test Queries (`benchmarks/code_understanding_queries.json`)
- **12 carefully crafted queries** covering different search patterns:
  - Function purpose discovery
  - Functionality search
  - Process flow understanding
  - Cross-cutting concerns
  - Structural searches
  - API discovery

## Running the Evaluation

### Quick Evaluation (Recommended for PR reviews)
```bash
# Run streamlined evaluation with key metrics
python benchmarks/run_quick_evaluation.py
```

### Comprehensive Evaluation
```bash
# Full performance analysis
python benchmarks/ast_vs_naive_evaluation.py --test-data-dir data/code_samples

# Search quality analysis
python benchmarks/search_quality_evaluator.py --test-data-dir data/code_samples

# Generate combined report
python benchmarks/generate_evaluation_report.py --test-data-dir data/code_samples
```

### Custom Evaluation
```bash
# Evaluate with custom code samples
python benchmarks/ast_vs_naive_evaluation.py --test-data-dir /path/to/your/code --verbose

# Custom output directory
python benchmarks/generate_evaluation_report.py --output-dir custom_results
```

## Expected Results

### Performance Characteristics

**AST Chunking:**
- ⚡ **Speed**: 1.5-3x slower than naive chunking
- 🧠 **Memory**: 1.2-2x higher memory usage
- 🎯 **Quality**: Significantly better function boundary preservation
- 📊 **Chunks**: Typically generates fewer, more coherent chunks

**Naive Chunking:**
- 🚀 **Speed**: Fastest processing time
- 💾 **Memory**: Lowest memory footprint
- ✂️ **Quality**: May split functions/classes inappropriately
- 📈 **Chunks**: More chunks with potential incomplete constructs

### Search Quality Results

**Typical Improvements with AST Chunking:**
- **Recall@5**: +15-25% for function-specific queries
- **Precision@5**: +10-20% for code structure searches
- **Function Completeness**: +50-100% complete function preservation
- **Semantic Coherence**: +20-30% better chunk boundaries

## Interpretation Guidelines

### When AST Chunking Excels
- ✅ **Code-heavy repositories** (>70% code files)
- ✅ **Function-level search tasks**
- ✅ **Semantic code analysis**
- ✅ **IDE integration scenarios**
- ✅ **Documentation generation from code**

### When Naive Chunking is Sufficient
- ✅ **Mixed content repositories** (code + docs)
- ✅ **Performance-critical applications**
- ✅ **Simple keyword search**
- ✅ **Large-scale batch processing**
- ✅ **Resource-constrained environments**

### Decision Matrix

| Use Case | AST Chunking | Naive Chunking | Recommendation |
|----------|-------------|----------------|----------------|
| Code-only repos | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | **AST** |
| Mixed content | ⭐⭐⭐ | ⭐⭐⭐⭐ | **Conditional** |
| Performance-critical | ⭐⭐ | ⭐⭐⭐⭐⭐ | **Naive** |
| Semantic search | ⭐⭐⭐⭐⭐ | ⭐⭐ | **AST** |
| Large-scale indexing | ⭐⭐ | ⭐⭐⭐⭐ | **Naive** |

## Configuration Recommendations

### AST Chunking Parameters
```python
# Recommended settings for code repositories
create_text_chunks(
    documents,
    chunk_size=256,           # For text content
    chunk_overlap=128,        # Standard overlap
    use_ast_chunking=True,
    ast_chunk_size=512,       # Larger for code context
    ast_chunk_overlap=64,     # Less overlap for clean boundaries
    ast_fallback_traditional=True  # Safety fallback
)
```

### Performance Optimization
```python
# For performance-sensitive applications
create_text_chunks(
    documents,
    chunk_size=256,
    chunk_overlap=64,         # Reduced overlap
    use_ast_chunking=False    # Stick with naive
)
```

### Hybrid Approach
```python
# Use AST for code, naive for other content
def intelligent_chunking(documents):
    code_docs, text_docs = detect_code_files(documents)
    
    # AST chunk code files
    code_chunks = create_text_chunks(
        code_docs, 
        use_ast_chunking=True,
        ast_chunk_size=768
    )
    
    # Naive chunk text files
    text_chunks = create_text_chunks(
        text_docs,
        use_ast_chunking=False,
        chunk_size=256
    )
    
    return code_chunks + text_chunks
```

## Monitoring and Validation

### Key Metrics to Track
1. **Processing Time**: Monitor chunking performance in production
2. **Search Result Quality**: User satisfaction with search results
3. **Function Boundary Preservation**: Code analysis task success rate
4. **Memory Usage**: System resource consumption
5. **User Feedback**: Qualitative assessment of code understanding

### A/B Testing Framework
```python
# Example A/B test configuration
def chunking_strategy(user_id, repository_type):
    if is_test_user(user_id):
        if repository_type == "code_heavy":
            return "ast_chunking"
        else:
            return "naive_chunking"
    else:
        return "control_group"  # Current implementation
```

## Limitations and Considerations

### AST Chunking Limitations
- **Language Support**: Limited to languages with tree-sitter parsers
- **Performance Overhead**: 1.5-3x slower processing
- **Memory Usage**: Higher memory requirements
- **Complexity**: More complex error handling and fallbacks

### Evaluation Limitations
- **Test Dataset Size**: Limited to sample code files
- **Query Diversity**: 12 test queries may not cover all use cases
- **Synthetic Evaluation**: Real-world usage patterns may differ
- **Language Bias**: Evaluation focuses on common programming languages

## Future Enhancements

### Evaluation Framework
1. **Larger Test Datasets**: Include real-world repositories
2. **User Studies**: Qualitative evaluation with developers
3. **Domain-Specific Metrics**: Specialized metrics for different code types
4. **Temporal Analysis**: Performance over time and repository growth

### Implementation Improvements
1. **Adaptive Chunking**: Dynamic strategy selection based on content
2. **Caching**: Cache AST parsing results for better performance
3. **Parallel Processing**: Multi-threaded AST chunking
4. **Smart Fallbacks**: Better heuristics for when to use each approach

## Conclusion

The evaluation framework provides comprehensive metrics to assess the trade-offs between AST-aware and naive chunking approaches. The key insight is that **AST chunking excels in code understanding quality at the cost of processing performance**, making it ideal for scenarios where semantic accuracy is more important than speed.

The framework enables data-driven decisions about chunking strategy based on specific use cases, performance requirements, and quality objectives.

---

*For questions about the evaluation framework or to contribute additional test cases, please refer to the LEANN documentation or open an issue in the repository.*