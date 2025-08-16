#!/usr/bin/env python3
"""
Working AST vs Naive Chunking Demo

This script demonstrates the actual differences between AST-aware and naive chunking
using real code files from the LEANN project.
"""

import sys
import time
from pathlib import Path

# Add apps directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

def load_real_code_files():
    """Load actual code files from the data/code_samples directory."""
    data_dir = Path(__file__).parent.parent / "data" / "code_samples"
    code_files = []
    
    for file_path in data_dir.glob("*.py"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                code_files.append({
                    "content": content,
                    "filename": file_path.name,
                    "path": str(file_path)
                })
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
    
    return code_files

def run_chunking_comparison():
    """Run a direct comparison of chunking approaches."""
    
    # Load real code files
    print("📁 Loading real code files...")
    code_files = load_real_code_files()
    
    if not code_files:
        print("❌ No code files found. Using sample content.")
        # Fallback to sample content
        sample_python = '''
def search_similar_vectors(query_vector, vector_database, top_k=5):
    """Search for vectors similar to the query vector."""
    import numpy as np
    
    similarities = []
    for i, stored_vector in enumerate(vector_database.vectors):
        # Calculate cosine similarity
        dot_product = np.dot(query_vector, stored_vector)
        magnitude1 = np.linalg.norm(query_vector)
        magnitude2 = np.linalg.norm(stored_vector)
        
        if magnitude1 == 0 or magnitude2 == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (magnitude1 * magnitude2)
        
        similarities.append({
            'index': i,
            'similarity': similarity,
            'metadata': vector_database.metadata[i]
        })
    
    # Sort by similarity and return top_k
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_k]

class VectorIndex:
    def __init__(self, dimensions=768):
        self.dimensions = dimensions
        self.vectors = []
        self.metadata = []
    
    def add_vector(self, vector, metadata=None):
        if len(vector) != self.dimensions:
            raise ValueError(f"Vector must have {self.dimensions} dimensions")
        self.vectors.append(vector)
        self.metadata.append(metadata or {})
    
    def search(self, query_vector, top_k=5):
        return search_similar_vectors(query_vector, self, top_k)
'''
        code_files = [{"content": sample_python, "filename": "sample.py"}]
    
    print(f"Loaded {len(code_files)} code files")
    
    # Test chunking approaches directly
    results = {}
    
    for file_data in code_files:
        print(f"\n🔍 Analyzing {file_data['filename']}...")
        
        # AST-aware chunking
        print("  🧠 AST chunking...")
        ast_chunks = chunk_with_ast(file_data['content'])
        
        # Naive chunking  
        print("  ⚡ Naive chunking...")
        naive_chunks = chunk_naively(file_data['content'])
        
        results[file_data['filename']] = {
            'ast': ast_chunks,
            'naive': naive_chunks,
            'content_length': len(file_data['content'])
        }
    
    return results

def chunk_with_ast(content):
    """Chunk content using AST-aware approach."""
    try:
        # Import astchunk if available
        from astchunk import ASTChunkBuilder
        
        start_time = time.time()
        
        # Configure for Python
        configs = {
            "max_chunk_size": 512,
            "language": "python",
            "chunk_overlap": 64,
        }
        
        chunk_builder = ASTChunkBuilder(**configs)
        chunks = chunk_builder.chunkify(content)
        
        processing_time = time.time() - start_time
        
        # Extract text from chunks
        chunk_texts = []
        for chunk in chunks:
            if hasattr(chunk, 'text'):
                chunk_texts.append(chunk.text)
            elif isinstance(chunk, dict) and 'text' in chunk:
                chunk_texts.append(chunk['text'])
            else:
                chunk_texts.append(str(chunk))
        
        return {
            "chunks": chunk_texts,
            "count": len(chunk_texts),
            "processing_time": processing_time,
            "method": "AST",
            "success": True
        }
        
    except ImportError:
        print("    ⚠️  astchunk not available, simulating AST behavior...")
        return simulate_ast_chunking(content)
    except Exception as e:
        print(f"    ❌ AST chunking failed: {e}")
        return simulate_ast_chunking(content)

def simulate_ast_chunking(content):
    """Simulate AST chunking behavior by splitting on function boundaries."""
    start_time = time.time()
    
    lines = content.split('\n')
    chunks = []
    current_chunk = []
    current_indent = 0
    
    for line in lines:
        stripped = line.strip()
        
        # Detect function/class definitions
        if stripped.startswith(('def ', 'class ', 'async def ')):
            # Start new chunk if we have content
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
            
            current_indent = len(line) - len(line.lstrip())
            current_chunk.append(line)
            
        elif line.strip() == '' or len(line) - len(line.lstrip()) > current_indent or not line.strip():
            # Continue current chunk
            current_chunk.append(line)
            
        else:
            # End of function/class, start new chunk
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
            current_chunk.append(line)
    
    # Add remaining content
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    # Filter out very small chunks
    chunks = [chunk for chunk in chunks if len(chunk.strip()) > 20]
    
    processing_time = time.time() - start_time
    
    return {
        "chunks": chunks,
        "count": len(chunks),
        "processing_time": processing_time,
        "method": "Simulated AST",
        "success": True
    }

def chunk_naively(content, chunk_size=512, overlap=64):
    """Chunk content using naive text splitting."""
    start_time = time.time()
    
    chunks = []
    start = 0
    
    while start < len(content):
        end = start + chunk_size
        chunk = content[start:end]
        
        # Try to break at word boundaries
        if end < len(content):
            last_space = chunk.rfind(' ')
            if last_space > chunk_size * 0.7:  # Only break at word boundary if it's not too far back
                chunk = chunk[:last_space]
                end = start + last_space
        
        chunks.append(chunk)
        start = end - overlap
    
    processing_time = time.time() - start_time
    
    return {
        "chunks": chunks,
        "count": len(chunks),
        "processing_time": processing_time,
        "method": "Naive",
        "success": True
    }

def analyze_chunk_quality(chunks, content):
    """Analyze the quality of chunks."""
    analysis = {
        "complete_functions": 0,
        "partial_functions": 0,
        "function_starts": 0,
        "function_ends": 0,
        "avg_chunk_size": 0,
        "size_variance": 0
    }
    
    if not chunks:
        return analysis
    
    chunk_sizes = []
    
    for chunk in chunks:
        chunk_sizes.append(len(chunk))
        
        # Count function-related patterns
        if 'def ' in chunk:
            analysis["function_starts"] += chunk.count('def ')
            
        # Check for complete functions (very basic heuristic)
        lines = chunk.split('\n')
        in_function = False
        function_indent = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('def '):
                in_function = True
                function_indent = len(line) - len(line.lstrip())
            elif in_function and line.strip() and len(line) - len(line.lstrip()) <= function_indent and not line.startswith(' '):
                # Function appears to end
                analysis["complete_functions"] += 1
                in_function = False
    
    analysis["avg_chunk_size"] = sum(chunk_sizes) / len(chunk_sizes)
    analysis["size_variance"] = sum((size - analysis["avg_chunk_size"]) ** 2 for size in chunk_sizes) / len(chunk_sizes)
    
    return analysis

def display_results(results):
    """Display comparison results."""
    print("\n" + "=" * 60)
    print("📊 CHUNKING COMPARISON RESULTS")
    print("=" * 60)
    
    total_ast_chunks = 0
    total_naive_chunks = 0
    total_ast_time = 0
    total_naive_time = 0
    
    for filename, data in results.items():
        print(f"\n📄 {filename}:")
        
        ast_data = data['ast']
        naive_data = data['naive']
        
        print(f"  Content Length: {data['content_length']:,} characters")
        print(f"  AST Chunks:     {ast_data['count']} ({ast_data['method']})")
        print(f"  Naive Chunks:   {naive_data['count']}")
        print(f"  AST Time:       {ast_data['processing_time']:.3f}s")
        print(f"  Naive Time:     {naive_data['processing_time']:.3f}s")
        
        # Analyze quality
        ast_quality = analyze_chunk_quality(ast_data['chunks'], data['content_length'])
        naive_quality = analyze_chunk_quality(naive_data['chunks'], data['content_length'])
        
        print(f"  AST Avg Size:   {ast_quality['avg_chunk_size']:.0f} chars")
        print(f"  Naive Avg Size: {naive_quality['avg_chunk_size']:.0f} chars")
        print(f"  AST Functions:  {ast_quality['complete_functions']}")
        print(f"  Naive Functions: {naive_quality['complete_functions']}")
        
        total_ast_chunks += ast_data['count']
        total_naive_chunks += naive_data['count']
        total_ast_time += ast_data['processing_time']
        total_naive_time += naive_data['processing_time']
        
        # Show sample chunks
        if ast_data['chunks']:
            print(f"\n  📝 Sample AST Chunk (first 200 chars):")
            print(f"  {repr(ast_data['chunks'][0][:200] + '...' if len(ast_data['chunks'][0]) > 200 else ast_data['chunks'][0])}")
        
        if naive_data['chunks']:
            print(f"\n  📝 Sample Naive Chunk (first 200 chars):")
            print(f"  {repr(naive_data['chunks'][0][:200] + '...' if len(naive_data['chunks'][0]) > 200 else naive_data['chunks'][0])}")
    
    # Overall summary
    print(f"\n🎯 OVERALL SUMMARY:")
    print(f"  Total AST Chunks:     {total_ast_chunks}")
    print(f"  Total Naive Chunks:   {total_naive_chunks}")
    print(f"  Total AST Time:       {total_ast_time:.3f}s")
    print(f"  Total Naive Time:     {total_naive_time:.3f}s")
    
    if total_naive_time > 0:
        speed_ratio = total_ast_time / total_naive_time
        print(f"  Speed Ratio:          {speed_ratio:.2f}x ({'AST slower' if speed_ratio > 1 else 'AST faster'})")
    
    chunk_ratio = total_ast_chunks / total_naive_chunks if total_naive_chunks > 0 else 0
    print(f"  Chunk Count Ratio:    {chunk_ratio:.2f}x")
    
    # Recommendation
    print(f"\n🎯 RECOMMENDATION:")
    if chunk_ratio > 1.2:
        print("✅ **AST Chunking** generates more granular, potentially better-structured chunks")
    elif speed_ratio > 3:
        print("⚡ **Naive Chunking** for performance-critical applications")
    else:
        print("⚖️ **Mixed Results** - choice depends on specific requirements")

def main():
    print("🔬 Working AST vs Naive Chunking Demo")
    print("=" * 50)
    print("This demo uses real code files and shows actual chunking differences.")
    
    try:
        results = run_chunking_comparison()
        display_results(results)
        
        print(f"\n💡 Key Insight:")
        print(f"   AST-aware chunking tends to preserve function boundaries")
        print(f"   while naive chunking focuses on consistent character counts.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()