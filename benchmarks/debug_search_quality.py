#!/usr/bin/env python3
"""
Debug Search Quality Evaluation

This script debugs why the search quality evaluator is returning all zeros.
"""

import sys
from pathlib import Path

# Add apps directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

def debug_chunk_relevance():
    """Debug why relevance scoring is failing."""
    
    # Sample query from our test set
    sample_query = {
        "id": 1,
        "query": "How do you search for similar vectors?",
        "type": "function_purpose",
        "expected_files": ["vector_search.py"],
        "expected_functions": ["search_similar_vectors", "cosine_similarity"],
        "relevance_keywords": ["vector", "similarity", "search", "cosine", "numpy"],
        "difficulty": "easy"
    }
    
    # Sample chunks (from actual AST vs naive output)
    sample_chunks = [
        {
            "id": 0,
            "text": "def search_similar_vectors(query_vector, database, top_k=5):\n    \"\"\"Search for vectors similar to the query vector.\"\"\"\n    import numpy as np\n    similarities = []\n    for vector in database:\n        similarity = cosine_similarity(query_vector, vector)\n        similarities.append(similarity)\n    return sorted(similarities, reverse=True)[:top_k]",
            "approach": "AST",
            "source_file": "vector_search.py"
        },
        {
            "id": 1,
            "text": "vectors similar to the query vector.\"\"\"\n    import numpy as np\n    similarities = []\n    for vector in database:\n        similarity = cosine_similarity(query_vector, vector)\n        similarities.append(similarity)\n    return sorted(similarities, reverse=True)[:top_k]\n\ndef cosine_similarity(vec1, vec2):",
            "approach": "Naive",
            "source_file": "vector_search.py"
        }
    ]
    
    print("🔍 Debug Search Quality Evaluation")
    print("=" * 50)
    
    print(f"\n📋 Sample Query:")
    print(f"  Text: '{sample_query['query']}'")
    print(f"  Expected Functions: {sample_query['expected_functions']}")
    print(f"  Keywords: {sample_query['relevance_keywords']}")
    
    print(f"\n📦 Sample Chunks:")
    for i, chunk in enumerate(sample_chunks):
        print(f"\n  Chunk {i+1} ({chunk['approach']}):")
        print(f"    Length: {len(chunk['text'])} chars")
        print(f"    Content: {repr(chunk['text'][:100])}...")
        
        # Manual relevance scoring
        relevance_score = calculate_manual_relevance(chunk, sample_query)
        print(f"    Relevance Score: {relevance_score}")

def calculate_manual_relevance(chunk, query):
    """Manually calculate relevance score to debug the logic."""
    chunk_text = chunk["text"].lower()
    query_text = query["query"].lower()
    expected_functions = [f.lower() for f in query.get("expected_functions", [])]
    relevance_keywords = [k.lower() for k in query.get("relevance_keywords", [])]
    
    relevance_score = 0.0
    scoring_details = []
    
    # Keyword matching
    for keyword in relevance_keywords:
        if keyword in chunk_text:
            relevance_score += 1.0
            scoring_details.append(f"+1.0 for keyword '{keyword}'")
    
    # Function name matching  
    for func_name in expected_functions:
        if func_name in chunk_text:
            if f"def {func_name}" in chunk_text:
                relevance_score += 5.0
                scoring_details.append(f"+5.0 for function definition '{func_name}'")
            else:
                relevance_score += 2.0
                scoring_details.append(f"+2.0 for function reference '{func_name}'")
    
    # Query word overlap
    query_words = set(query_text.split())
    chunk_words = set(chunk_text.split())
    word_overlap = len(query_words & chunk_words)
    if word_overlap > 0:
        bonus = word_overlap * 0.1
        relevance_score += bonus
        scoring_details.append(f"+{bonus:.1f} for {word_overlap} overlapping words")
    
    print(f"    Scoring Details: {scoring_details}")
    return relevance_score

def test_actual_search_evaluator():
    """Test the actual search evaluator to see where it fails."""
    try:
        from search_quality_evaluator import SearchQualityEvaluator
        
        print(f"\n🧪 Testing Actual Search Evaluator...")
        
        # Try to create evaluator
        queries_file = Path(__file__).parent / "code_understanding_queries.json"
        evaluator = SearchQualityEvaluator("data/code_samples", str(queries_file))
        
        print(f"✅ Evaluator created successfully")
        print(f"📄 Documents loaded: {len(evaluator.documents)}")
        print(f"❓ Queries loaded: {len(evaluator.queries)}")
        
        # Test chunk generation
        ast_chunks = evaluator._generate_chunks(use_ast=True)
        naive_chunks = evaluator._generate_chunks(use_ast=False)
        
        print(f"🧠 AST chunks generated: {len(ast_chunks)}")
        print(f"⚡ Naive chunks generated: {len(naive_chunks)}")
        
        if ast_chunks:
            print(f"\n📝 Sample AST chunk structure:")
            sample_ast = ast_chunks[0]
            for key, value in sample_ast.items():
                if key == "text":
                    print(f"  {key}: {repr(value[:100])}...")
                else:
                    print(f"  {key}: {value}")
        
        # Test relevance finding
        sample_query = evaluator.queries[0]
        print(f"\n🔍 Testing relevance finding for query: '{sample_query['query']}'")
        
        relevant_ast = evaluator._find_relevant_chunks(ast_chunks, sample_query)
        relevant_naive = evaluator._find_relevant_chunks(naive_chunks, sample_query)
        
        print(f"🧠 AST relevant chunks found: {len(relevant_ast)}")
        print(f"⚡ Naive relevant chunks found: {len(relevant_naive)}")
        
        if relevant_ast:
            print(f"   Best AST relevance score: {relevant_ast[0].get('relevance_score', 'N/A')}")
        if relevant_naive:
            print(f"   Best Naive relevance score: {relevant_naive[0].get('relevance_score', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Search evaluator test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🐛 Debugging Search Quality Evaluation Issues")
    print("=" * 60)
    
    # Manual relevance calculation
    debug_chunk_relevance()
    
    # Test actual evaluator
    test_actual_search_evaluator()
    
    print(f"\n💡 Analysis:")
    print(f"   The search quality evaluator likely fails because:")
    print(f"   1. Chunk structure doesn't match expected format")
    print(f"   2. Relevance scoring logic has bugs")
    print(f"   3. Ground truth queries don't match actual chunk content")
    print(f"   4. Import/dependency issues cause silent failures")

if __name__ == "__main__":
    main()