#!/usr/bin/env python3
"""
Simple AST vs Naive Chunking Demonstration

This script provides a working demonstration of the differences between
AST-aware and naive chunking using the actual LEANN codebase.
"""

import sys
import time
from pathlib import Path

# Add apps directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

def load_sample_code_files():
    """Load sample code files for comparison."""
    code_samples = []
    
    # Sample Python code
    python_code = '''
def calculate_vector_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    import numpy as np
    
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

class VectorDatabase:
    def __init__(self, dimensions=768):
        self.dimensions = dimensions
        self.vectors = []
        self.metadata = []
    
    def add_vector(self, vector, metadata=None):
        if len(vector) != self.dimensions:
            raise ValueError(f"Vector must have {self.dimensions} dimensions")
        self.vectors.append(vector)
        self.metadata.append(metadata or {})
    
    def search_similar(self, query_vector, top_k=5):
        similarities = []
        for i, stored_vector in enumerate(self.vectors):
            similarity = calculate_vector_similarity(query_vector, stored_vector)
            similarities.append((similarity, i, self.metadata[i]))
        
        similarities.sort(reverse=True)
        return similarities[:top_k]
'''
    
    # Sample Java code
    java_code = '''
public class DataProcessor {
    private List<String> data;
    private boolean isProcessed;
    
    public DataProcessor() {
        this.data = new ArrayList<>();
        this.isProcessed = false;
    }
    
    public void addData(String item) {
        if (item == null || item.trim().isEmpty()) {
            throw new IllegalArgumentException("Data item cannot be null or empty");
        }
        this.data.add(item.trim());
    }
    
    public List<String> processData() {
        if (this.isProcessed) {
            return this.data;
        }
        
        List<String> processed = new ArrayList<>();
        for (String item : this.data) {
            String processedItem = item.toLowerCase()
                                     .replaceAll("[^a-z0-9\\s]", "")
                                     .trim();
            if (!processedItem.isEmpty()) {
                processed.add(processedItem);
            }
        }
        
        this.data = processed;
        this.isProcessed = true;
        return processed;
    }
    
    public int getDataCount() {
        return this.data.size();
    }
}
'''
    
    return [
        {"content": python_code, "filename": "vector_utils.py", "language": "python"},
        {"content": java_code, "filename": "DataProcessor.java", "language": "java"}
    ]

def create_mock_documents(code_samples):
    """Create mock document objects."""
    documents = []
    for sample in code_samples:
        doc = type('Document', (), {
            'text': sample['content'],
            'metadata': {
                'file_path': sample['filename'],
                'file_name': sample['filename']
            },
            'get_content': lambda self=sample: sample['content']
        })()
        documents.append(doc)
    return documents

def run_ast_chunking(documents):
    """Run AST-aware chunking."""
    try:
        from chunking import create_text_chunks
        
        start_time = time.time()
        chunks = create_text_chunks(
            documents,
            chunk_size=256,
            chunk_overlap=128,
            use_ast_chunking=True,
            ast_chunk_size=512,
            ast_chunk_overlap=64
        )
        processing_time = time.time() - start_time
        
        return {
            "chunks": chunks,
            "count": len(chunks),
            "processing_time": processing_time,
            "avg_length": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
            "success": True
        }
    except Exception as e:
        return {
            "chunks": [],
            "count": 0,
            "processing_time": 0,
            "avg_length": 0,
            "success": False,
            "error": str(e)
        }

def run_naive_chunking(documents):
    """Run naive text chunking."""
    try:
        from chunking import create_text_chunks
        
        start_time = time.time()
        chunks = create_text_chunks(
            documents,
            chunk_size=256,
            chunk_overlap=128,
            use_ast_chunking=False
        )
        processing_time = time.time() - start_time
        
        return {
            "chunks": chunks,
            "count": len(chunks),
            "processing_time": processing_time,
            "avg_length": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
            "success": True
        }
    except Exception as e:
        return {
            "chunks": [],
            "count": 0,
            "processing_time": 0,
            "avg_length": 0,
            "success": False,
            "error": str(e)
        }

def analyze_chunk_quality(chunks, approach_name):
    """Analyze the quality of chunks."""
    if not chunks:
        return {"error": f"No chunks generated for {approach_name}"}
    
    analysis = {
        "approach": approach_name,
        "total_chunks": len(chunks),
        "complete_functions": 0,
        "partial_functions": 0,
        "import_statements": 0,
        "class_definitions": 0,
        "avg_lines_per_chunk": 0
    }
    
    total_lines = 0
    
    for chunk in chunks:
        lines = chunk.split('\n')
        total_lines += len(lines)
        
        # Count complete functions (simple heuristic)
        chunk_lower = chunk.lower()
        
        # Function definitions
        if ('def ' in chunk and ':' in chunk) or ('function ' in chunk and '{' in chunk and '}' in chunk):
            if chunk.count('{') == chunk.count('}') or ('def ' in chunk and not chunk.strip().endswith('\\')):
                analysis["complete_functions"] += 1
            else:
                analysis["partial_functions"] += 1
        elif 'def ' in chunk or 'function ' in chunk or 'public ' in chunk:
            analysis["partial_functions"] += 1
            
        # Import statements
        if 'import ' in chunk_lower or 'from ' in chunk_lower or '#include' in chunk_lower:
            analysis["import_statements"] += 1
            
        # Class definitions
        if 'class ' in chunk_lower:
            analysis["class_definitions"] += 1
    
    analysis["avg_lines_per_chunk"] = total_lines / len(chunks) if chunks else 0
    analysis["function_completeness_ratio"] = (
        analysis["complete_functions"] / 
        max(analysis["complete_functions"] + analysis["partial_functions"], 1)
    )
    
    return analysis

def display_chunk_examples(chunks, approach_name, max_examples=2):
    """Display example chunks for comparison."""
    print(f"\n📝 {approach_name} Chunking Examples:")
    print("-" * 50)
    
    for i, chunk in enumerate(chunks[:max_examples]):
        print(f"\nChunk {i+1} ({len(chunk)} chars, {len(chunk.split())} lines):")
        print("```")
        print(chunk[:300] + ("..." if len(chunk) > 300 else ""))
        print("```")

def main():
    print("🔬 Simple AST vs Naive Chunking Demo")
    print("=" * 50)
    
    # Load sample code
    print("📁 Loading sample code files...")
    code_samples = load_sample_code_files()
    documents = create_mock_documents(code_samples)
    print(f"Loaded {len(documents)} code samples")
    
    # Run AST chunking
    print("\n🧠 Running AST-aware chunking...")
    ast_results = run_ast_chunking(documents)
    
    if ast_results["success"]:
        print(f"✅ Generated {ast_results['count']} chunks in {ast_results['processing_time']:.3f}s")
        ast_analysis = analyze_chunk_quality(ast_results["chunks"], "AST")
    else:
        print(f"❌ AST chunking failed: {ast_results['error']}")
        ast_analysis = {"error": ast_results["error"]}
    
    # Run naive chunking
    print("\n⚡ Running naive chunking...")
    naive_results = run_naive_chunking(documents)
    
    if naive_results["success"]:
        print(f"✅ Generated {naive_results['count']} chunks in {naive_results['processing_time']:.3f}s")
        naive_analysis = analyze_chunk_quality(naive_results["chunks"], "Naive")
    else:
        print(f"❌ Naive chunking failed: {naive_results['error']}")
        naive_analysis = {"error": naive_results["error"]}
    
    # Compare results
    print("\n📊 COMPARISON RESULTS")
    print("=" * 50)
    
    if ast_results["success"] and naive_results["success"]:
        print(f"\n📈 Performance Metrics:")
        print(f"  AST Chunks:     {ast_results['count']}")
        print(f"  Naive Chunks:   {naive_results['count']}")
        print(f"  AST Time:       {ast_results['processing_time']:.3f}s")
        print(f"  Naive Time:     {naive_results['processing_time']:.3f}s")
        print(f"  Speed Ratio:    {naive_results['processing_time'] / ast_results['processing_time']:.2f}x")
        
        print(f"\n🎯 Quality Metrics:")
        if "error" not in ast_analysis and "error" not in naive_analysis:
            print(f"  AST Complete Functions:     {ast_analysis['complete_functions']}")
            print(f"  Naive Complete Functions:   {naive_analysis['complete_functions']}")
            print(f"  AST Completeness Ratio:     {ast_analysis['function_completeness_ratio']:.2%}")
            print(f"  Naive Completeness Ratio:   {naive_analysis['function_completeness_ratio']:.2%}")
            print(f"  Completeness Improvement:   {ast_analysis['function_completeness_ratio'] - naive_analysis['function_completeness_ratio']:+.1%}")
            
            print(f"  AST Avg Lines/Chunk:        {ast_analysis['avg_lines_per_chunk']:.1f}")
            print(f"  Naive Avg Lines/Chunk:      {naive_analysis['avg_lines_per_chunk']:.1f}")
        
        # Show example chunks
        if ast_results["chunks"]:
            display_chunk_examples(ast_results["chunks"], "AST")
        if naive_results["chunks"]:
            display_chunk_examples(naive_results["chunks"], "Naive")
            
        # Recommendation
        print(f"\n🎯 RECOMMENDATION:")
        if ast_analysis.get('function_completeness_ratio', 0) > naive_analysis.get('function_completeness_ratio', 0) + 0.1:
            print("✅ **AST Chunking Recommended** - Better function preservation")
        elif ast_results['processing_time'] > naive_results['processing_time'] * 3:
            print("⚡ **Naive Chunking Recommended** - Much faster processing")
        else:
            print("⚖️ **Mixed Results** - Consider use case requirements")
    
    else:
        print("❌ Cannot compare - one or both approaches failed")
        if "error" not in ast_analysis:
            print(f"AST Analysis: {ast_analysis}")
        if "error" not in naive_analysis:
            print(f"Naive Analysis: {naive_analysis}")

if __name__ == "__main__":
    main()