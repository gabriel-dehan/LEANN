#!/usr/bin/env python3
"""
Search Quality Evaluation for AST vs Naive Chunking

This module evaluates search quality by measuring Recall@K, Precision@K, and other
relevance metrics when comparing AST-aware chunking against naive text chunking.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
import statistics
import re

# Add apps directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

from chunking import create_text_chunks
from llama_index.core import SimpleDirectoryReader, Document

logger = logging.getLogger(__name__)


class SearchQualityEvaluator:
    """Evaluates search quality using ground truth queries and relevance judgments."""
    
    def __init__(self, test_data_dir: str, queries_file: str):
        self.test_data_dir = Path(test_data_dir)
        self.queries_file = Path(queries_file)
        self.documents = None
        self.queries = None
        
        # Load test data
        self._load_test_data()
        
    def _load_test_data(self):
        """Load test documents and queries."""
        logger.info("Loading test documents and queries...")
        
        # Load documents
        self.documents = SimpleDirectoryReader(
            str(self.test_data_dir),
            recursive=True,
            required_exts=[".py", ".java", ".ts", ".tsx", ".js", ".jsx", ".cs"],
            filename_as_id=True
        ).load_data()
        
        logger.info(f"Loaded {len(self.documents)} documents")
        
        # Load queries
        with open(self.queries_file, 'r') as f:
            query_data = json.load(f)
            self.queries = query_data['queries']
            
        logger.info(f"Loaded {len(self.queries)} test queries")
        
    def evaluate_search_quality(self) -> Dict[str, Any]:
        """Evaluate search quality for both AST and naive chunking."""
        logger.info("Starting search quality evaluation...")
        
        results = {
            "ast_chunking": {},
            "naive_chunking": {},
            "comparison": {}
        }
        
        # Generate chunks for both approaches
        ast_chunks = self._generate_chunks(use_ast=True)
        naive_chunks = self._generate_chunks(use_ast=False)
        
        # Evaluate each approach
        results["ast_chunking"] = self._evaluate_chunking_approach(ast_chunks, "AST")
        results["naive_chunking"] = self._evaluate_chunking_approach(naive_chunks, "Naive")
        
        # Calculate comparison metrics
        results["comparison"] = self._calculate_search_comparison(
            results["ast_chunking"], 
            results["naive_chunking"]
        )
        
        return results
        
    def _generate_chunks(self, use_ast: bool) -> List[Dict[str, Any]]:
        """Generate chunks using specified approach and return with metadata."""
        approach = "AST" if use_ast else "Naive"
        logger.info(f"Generating chunks using {approach} approach...")
        
        start_time = time.time()
        chunks = create_text_chunks(
            self.documents,
            chunk_size=256,
            chunk_overlap=128,
            use_ast_chunking=use_ast,
            ast_chunk_size=512,
            ast_chunk_overlap=64
        )
        processing_time = time.time() - start_time
        
        # Enrich chunks with metadata
        enriched_chunks = []
        for i, chunk in enumerate(chunks):
            enriched_chunk = {
                "id": i,
                "text": chunk,
                "approach": approach,
                "length": len(chunk),
                "lines": len(chunk.split('\n')),
                "source_file": self._guess_source_file(chunk),
                "contains_functions": self._contains_function_definitions(chunk),
                "contains_classes": self._contains_class_definitions(chunk),
                "contains_imports": self._contains_imports(chunk),
                "is_complete_construct": self._is_complete_code_construct(chunk)
            }
            enriched_chunks.append(enriched_chunk)
            
        logger.info(f"{approach} chunking: {len(chunks)} chunks in {processing_time:.2f}s")
        return enriched_chunks
        
    def _evaluate_chunking_approach(self, chunks: List[Dict], approach: str) -> Dict[str, Any]:
        """Evaluate a single chunking approach against all queries."""
        logger.info(f"Evaluating {approach} chunking approach...")
        
        query_results = []
        
        for query in self.queries:
            query_result = self._evaluate_single_query(chunks, query, approach)
            query_results.append(query_result)
            
        # Aggregate results
        aggregated_results = self._aggregate_query_results(query_results, approach)
        aggregated_results["query_results"] = query_results
        
        return aggregated_results
        
    def _evaluate_single_query(self, chunks: List[Dict], query: Dict, approach: str) -> Dict[str, Any]:
        """Evaluate a single query against the chunks."""
        query_text = query["query"]
        query_id = query["id"]
        
        # Find relevant chunks using different strategies
        relevant_chunks = self._find_relevant_chunks(chunks, query)
        
        # Calculate metrics for different K values
        k_values = [1, 3, 5, 10]
        metrics = {}
        
        for k in k_values:
            # Get top-k chunks
            top_k_chunks = relevant_chunks[:k] if len(relevant_chunks) >= k else relevant_chunks
            
            # Calculate recall and precision
            recall = self._calculate_recall(top_k_chunks, query, k)
            precision = self._calculate_precision(top_k_chunks, query, k)
            f1 = self._calculate_f1(recall, precision)
            
            metrics[f"recall@{k}"] = recall
            metrics[f"precision@{k}"] = precision
            metrics[f"f1@{k}"] = f1
            
        # Additional metrics
        metrics["total_relevant_found"] = len(relevant_chunks)
        metrics["avg_relevance_score"] = statistics.mean(
            chunk.get("relevance_score", 0) for chunk in relevant_chunks
        ) if relevant_chunks else 0.0
        
        return {
            "query_id": query_id,
            "query_text": query_text,
            "query_type": query.get("type", "unknown"),
            "approach": approach,
            "metrics": metrics,
            "relevant_chunks": relevant_chunks[:10]  # Store top 10 for analysis
        }
        
    def _find_relevant_chunks(self, chunks: List[Dict], query: Dict) -> List[Dict]:
        """Find chunks relevant to the query using multiple strategies."""
        query_text = query["query"].lower()
        expected_functions = [f.lower() for f in query.get("expected_functions", [])]
        expected_files = [f.lower() for f in query.get("expected_files", [])]
        relevance_keywords = [k.lower() for k in query.get("relevance_keywords", [])]
        
        scored_chunks = []
        
        for chunk in chunks:
            chunk_text = chunk["text"].lower()
            source_file = chunk.get("source_file", "").lower()
            
            # Calculate relevance score
            relevance_score = 0.0
            
            # Keyword matching (basic relevance)
            for keyword in relevance_keywords:
                if keyword in chunk_text:
                    relevance_score += 1.0
                    
            # Function name matching (high relevance)
            for func_name in expected_functions:
                if func_name in chunk_text:
                    # Check if it's a function definition (higher score)
                    if self._is_function_definition(chunk_text, func_name):
                        relevance_score += 5.0
                    else:
                        relevance_score += 2.0
                        
            # File matching (medium relevance)
            for expected_file in expected_files:
                if expected_file in source_file:
                    relevance_score += 1.5
                    
            # Query text similarity (basic relevance)
            query_words = set(query_text.split())
            chunk_words = set(chunk_text.split())
            word_overlap = len(query_words & chunk_words)
            if word_overlap > 0:
                relevance_score += word_overlap * 0.1
                
            # Bonus for complete constructs
            if chunk.get("is_complete_construct", False):
                relevance_score *= 1.2
                
            # Store score in chunk
            chunk_copy = chunk.copy()
            chunk_copy["relevance_score"] = relevance_score
            
            if relevance_score > 0:
                scored_chunks.append(chunk_copy)
                
        # Sort by relevance score (descending)
        scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return scored_chunks
        
    def _calculate_recall(self, retrieved_chunks: List[Dict], query: Dict, k: int) -> float:
        """Calculate Recall@K for the query."""
        if not retrieved_chunks:
            return 0.0
            
        # Count highly relevant chunks in top-k
        highly_relevant = sum(
            1 for chunk in retrieved_chunks 
            if chunk.get("relevance_score", 0) >= 2.0  # Threshold for "relevant"
        )
        
        # Estimate total relevant chunks based on query difficulty
        difficulty = query.get("difficulty", "medium")
        if difficulty == "easy":
            expected_relevant = 2
        elif difficulty == "medium":
            expected_relevant = 4
        else:  # hard
            expected_relevant = 6
            
        return min(highly_relevant / expected_relevant, 1.0)
        
    def _calculate_precision(self, retrieved_chunks: List[Dict], query: Dict, k: int) -> float:
        """Calculate Precision@K for the query."""
        if not retrieved_chunks:
            return 0.0
            
        # Count relevant chunks in retrieved set
        relevant_count = sum(
            1 for chunk in retrieved_chunks 
            if chunk.get("relevance_score", 0) >= 1.0  # Lower threshold for precision
        )
        
        return relevant_count / len(retrieved_chunks)
        
    def _calculate_f1(self, recall: float, precision: float) -> float:
        """Calculate F1 score from recall and precision."""
        if recall + precision == 0:
            return 0.0
        return 2 * (recall * precision) / (recall + precision)
        
    def _aggregate_query_results(self, query_results: List[Dict], approach: str) -> Dict[str, Any]:
        """Aggregate results across all queries."""
        if not query_results:
            return {}
            
        # Aggregate metrics
        k_values = [1, 3, 5, 10]
        aggregated = {"approach": approach}
        
        for k in k_values:
            recalls = [qr["metrics"][f"recall@{k}"] for qr in query_results]
            precisions = [qr["metrics"][f"precision@{k}"] for qr in query_results]
            f1_scores = [qr["metrics"][f"f1@{k}"] for qr in query_results]
            
            aggregated[f"avg_recall@{k}"] = statistics.mean(recalls)
            aggregated[f"avg_precision@{k}"] = statistics.mean(precisions)
            aggregated[f"avg_f1@{k}"] = statistics.mean(f1_scores)
            
        # Overall metrics
        aggregated["avg_relevance_score"] = statistics.mean(
            qr["metrics"]["avg_relevance_score"] for qr in query_results
        )
        
        aggregated["total_relevant_found"] = sum(
            qr["metrics"]["total_relevant_found"] for qr in query_results
        )
        
        # Performance by query type
        by_type = {}
        for qr in query_results:
            query_type = qr["query_type"]
            if query_type not in by_type:
                by_type[query_type] = []
            by_type[query_type].append(qr)
            
        aggregated["performance_by_type"] = {}
        for query_type, type_results in by_type.items():
            type_metrics = {}
            for k in k_values:
                type_recalls = [tr["metrics"][f"recall@{k}"] for tr in type_results]
                type_metrics[f"avg_recall@{k}"] = statistics.mean(type_recalls)
                
            aggregated["performance_by_type"][query_type] = type_metrics
            
        return aggregated
        
    def _calculate_search_comparison(self, ast_results: Dict, naive_results: Dict) -> Dict[str, Any]:
        """Calculate comparison metrics between AST and naive search results."""
        comparison = {}
        
        k_values = [1, 3, 5, 10]
        
        for k in k_values:
            ast_recall = ast_results.get(f"avg_recall@{k}", 0)
            naive_recall = naive_results.get(f"avg_recall@{k}", 0)
            ast_precision = ast_results.get(f"avg_precision@{k}", 0)
            naive_precision = naive_results.get(f"avg_precision@{k}", 0)
            ast_f1 = ast_results.get(f"avg_f1@{k}", 0)
            naive_f1 = naive_results.get(f"avg_f1@{k}", 0)
            
            comparison[f"recall_improvement@{k}"] = ast_recall - naive_recall
            comparison[f"precision_improvement@{k}"] = ast_precision - naive_precision
            comparison[f"f1_improvement@{k}"] = ast_f1 - naive_f1
            
            comparison[f"recall_ratio@{k}"] = ast_recall / naive_recall if naive_recall > 0 else float('inf')
            comparison[f"precision_ratio@{k}"] = ast_precision / naive_precision if naive_precision > 0 else float('inf')
            
        # Overall improvements
        comparison["avg_relevance_improvement"] = (
            ast_results.get("avg_relevance_score", 0) - 
            naive_results.get("avg_relevance_score", 0)
        )
        
        comparison["total_relevant_ratio"] = (
            ast_results.get("total_relevant_found", 0) / 
            naive_results.get("total_relevant_found", 1)
        )
        
        return comparison
        
    # Helper methods for chunk analysis
    
    def _guess_source_file(self, chunk: str) -> str:
        """Guess the source file based on chunk content."""
        # Simple heuristics based on file-specific patterns
        if "class DataProcessor" in chunk or "package com.example.ml" in chunk:
            return "DataProcessor.java"
        elif "class ApiController" in chunk or "@RestController" in chunk:
            return "ApiController.java"
        elif "def search_similar_vectors" in chunk or "import numpy" in chunk:
            return "vector_search.py"
        elif "ConnectionPool" in chunk or "DatabaseConnector" in chunk:
            return "database_connector.py"
        elif "interface TextAnalyzer" in chunk or "preprocess(" in chunk:
            return "text_analyzer.ts"
        elif "class ImageProcessor" in chunk or "using System.Drawing" in chunk:
            return "ImageProcessor.cs"
        else:
            return "unknown"
            
    def _contains_function_definitions(self, chunk: str) -> bool:
        """Check if chunk contains function definitions."""
        patterns = [
            r'\bdef\s+\w+\s*\(',  # Python
            r'\bfunction\s+\w+\s*\(',  # JavaScript/TypeScript
            r'\b(public|private|protected)\s+\w+\s+\w+\s*\(',  # Java/C#
            r'\b\w+\s*\([^)]*\)\s*\{',  # General function pattern
        ]
        
        for pattern in patterns:
            if re.search(pattern, chunk):
                return True
        return False
        
    def _contains_class_definitions(self, chunk: str) -> bool:
        """Check if chunk contains class definitions."""
        patterns = [
            r'\bclass\s+\w+',  # Python, TypeScript, Java, C#
            r'\binterface\s+\w+',  # TypeScript, Java, C#
            r'\benum\s+\w+',  # Java, C#, TypeScript
        ]
        
        for pattern in patterns:
            if re.search(pattern, chunk):
                return True
        return False
        
    def _contains_imports(self, chunk: str) -> bool:
        """Check if chunk contains import statements."""
        patterns = [
            r'\bimport\s+',  # Python, Java, TypeScript
            r'\bfrom\s+\w+\s+import',  # Python
            r'\busing\s+',  # C#
            r'\brequire\s*\(',  # JavaScript/Node.js
        ]
        
        for pattern in patterns:
            if re.search(pattern, chunk):
                return True
        return False
        
    def _is_complete_code_construct(self, chunk: str) -> bool:
        """Check if chunk represents a complete code construct."""
        # Check for balanced braces/brackets
        open_braces = chunk.count('{')
        close_braces = chunk.count('}')
        open_brackets = chunk.count('[')
        close_brackets = chunk.count(']')
        open_parens = chunk.count('(')
        close_parens = chunk.count(')')
        
        # For brace languages, braces should be balanced
        if open_braces > 0 and open_braces == close_braces:
            return True
            
        # For Python, check indentation consistency
        if 'def ' in chunk or 'class ' in chunk:
            lines = chunk.strip().split('\n')
            if len(lines) > 1:
                # Very basic check for proper indentation
                return not chunk.strip().endswith('\\')
                
        return False
        
    def _is_function_definition(self, text: str, func_name: str) -> bool:
        """Check if the function name appears as a definition."""
        patterns = [
            rf'\bdef\s+{re.escape(func_name)}\s*\(',
            rf'\bfunction\s+{re.escape(func_name)}\s*\(',
            rf'\b(public|private|protected)\s+\w+\s+{re.escape(func_name)}\s*\(',
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate search quality of AST vs naive chunking")
    parser.add_argument(
        "--test-data-dir",
        type=str,
        default="data/code_samples",
        help="Directory containing test code files"
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        default="benchmarks/code_understanding_queries.json",
        help="JSON file containing test queries"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="search_quality_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        
    try:
        evaluator = SearchQualityEvaluator(args.test_data_dir, args.queries_file)
        results = evaluator.evaluate_search_quality()
        
        # Save results
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Print summary
        print("\n" + "="*60)
        print("SEARCH QUALITY EVALUATION SUMMARY")
        print("="*60)
        
        ast_results = results["ast_chunking"]
        naive_results = results["naive_chunking"]
        comparison = results["comparison"]
        
        print(f"\nRecall@5:")
        print(f"  AST Chunking:   {ast_results.get('avg_recall@5', 0):.3f}")
        print(f"  Naive Chunking: {naive_results.get('avg_recall@5', 0):.3f}")
        print(f"  Improvement:    {comparison.get('recall_improvement@5', 0):+.3f}")
        
        print(f"\nPrecision@5:")
        print(f"  AST Chunking:   {ast_results.get('avg_precision@5', 0):.3f}")
        print(f"  Naive Chunking: {naive_results.get('avg_precision@5', 0):.3f}")
        print(f"  Improvement:    {comparison.get('precision_improvement@5', 0):+.3f}")
        
        print(f"\nF1@5:")
        print(f"  AST Chunking:   {ast_results.get('avg_f1@5', 0):.3f}")
        print(f"  Naive Chunking: {naive_results.get('avg_f1@5', 0):.3f}")
        print(f"  Improvement:    {comparison.get('f1_improvement@5', 0):+.3f}")
        
        print(f"\nResults saved to: {args.output_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())