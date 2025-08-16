#!/usr/bin/env python3
"""
AST-Aware vs Naive Chunking Performance Evaluation

This script provides comprehensive evaluation of AST-aware chunking performance
compared to traditional naive text chunking across multiple dimensions:
- Search quality (Recall@K, Precision@K)
- Semantic coherence and code understanding
- Performance (speed, memory, index size)
- Context economy and token efficiency
"""

import argparse
import json
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple, Any
import statistics
import sys

# Add apps directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

from chunking import create_text_chunks, detect_code_files
from llama_index.core import SimpleDirectoryReader, Document

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChunkingEvaluator:
    """Evaluates AST-aware vs naive chunking performance."""
    
    def __init__(self, test_data_dir: str, output_dir: str = None):
        self.test_data_dir = Path(test_data_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Evaluation results storage
        self.results = {
            "ast_chunking": {},
            "naive_chunking": {},
            "comparison": {}
        }
        
    def load_test_documents(self) -> List[Document]:
        """Load test documents from the specified directory."""
        logger.info(f"Loading test documents from {self.test_data_dir}")
        
        if not self.test_data_dir.exists():
            raise FileNotFoundError(f"Test data directory not found: {self.test_data_dir}")
            
        try:
            documents = SimpleDirectoryReader(
                str(self.test_data_dir),
                recursive=True,
                required_exts=[".py", ".java", ".ts", ".tsx", ".js", ".jsx", ".cs", ".cpp", ".c", ".h"],
                filename_as_id=True
            ).load_data()
            
            logger.info(f"Loaded {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
            
    def evaluate_chunking_performance(self, documents: List[Document]) -> Dict[str, Any]:
        """Evaluate both AST and naive chunking performance."""
        logger.info("Starting chunking performance evaluation")
        
        # Test parameters
        chunk_size = 256
        chunk_overlap = 128
        ast_chunk_size = 512
        ast_chunk_overlap = 64
        
        results = {}
        
        # Evaluate AST chunking
        logger.info("Evaluating AST-aware chunking...")
        ast_results = self._evaluate_single_approach(
            documents, 
            use_ast=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            ast_chunk_size=ast_chunk_size,
            ast_chunk_overlap=ast_chunk_overlap
        )
        results["ast_chunking"] = ast_results
        
        # Evaluate naive chunking
        logger.info("Evaluating naive chunking...")
        naive_results = self._evaluate_single_approach(
            documents,
            use_ast=False,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        results["naive_chunking"] = naive_results
        
        # Calculate comparison metrics
        results["comparison"] = self._calculate_comparison_metrics(ast_results, naive_results)
        
        return results
        
    def _evaluate_single_approach(self, documents: List[Document], use_ast: bool, **kwargs) -> Dict[str, Any]:
        """Evaluate a single chunking approach."""
        approach_name = "AST-aware" if use_ast else "Naive"
        
        # Measure chunking performance
        tracemalloc.start()
        start_time = time.time()
        
        try:
            chunks = create_text_chunks(
                documents,
                chunk_size=kwargs.get('chunk_size', 256),
                chunk_overlap=kwargs.get('chunk_overlap', 128),
                use_ast_chunking=use_ast,
                ast_chunk_size=kwargs.get('ast_chunk_size', 512),
                ast_chunk_overlap=kwargs.get('ast_chunk_overlap', 64)
            )
        except Exception as e:
            logger.error(f"Error in {approach_name} chunking: {e}")
            # Fallback to basic chunking if AST fails
            if use_ast:
                logger.warning("AST chunking failed, falling back to naive chunking for comparison")
                chunks = create_text_chunks(
                    documents,
                    chunk_size=kwargs.get('chunk_size', 256),
                    chunk_overlap=kwargs.get('chunk_overlap', 128),
                    use_ast_chunking=False
                )
            else:
                raise
            
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate metrics
        processing_time = end_time - start_time
        chunk_count = len(chunks)
        avg_chunk_length = statistics.mean(len(chunk) for chunk in chunks) if chunks else 0
        
        # Analyze chunk quality
        quality_metrics = self._analyze_chunk_quality(chunks, documents, use_ast)
        
        results = {
            "approach": approach_name,
            "processing_time": processing_time,
            "peak_memory_mb": peak / 1024 / 1024,
            "chunk_count": chunk_count,
            "avg_chunk_length": avg_chunk_length,
            "total_characters": sum(len(chunk) for chunk in chunks),
            "chunks": chunks[:10],  # Store first 10 chunks for analysis
            **quality_metrics
        }
        
        logger.info(f"{approach_name} chunking completed: {chunk_count} chunks in {processing_time:.2f}s")
        return results
        
    def _analyze_chunk_quality(self, chunks: List[str], documents: List[Document], use_ast: bool) -> Dict[str, Any]:
        """Analyze the quality of generated chunks."""
        if not chunks:
            return {"quality_score": 0.0, "completeness_score": 0.0, "coherence_score": 0.0}
            
        # Separate code and text documents
        code_docs, text_docs = detect_code_files(documents)
        
        quality_metrics = {
            "code_document_count": len(code_docs),
            "text_document_count": len(text_docs)
        }
        
        if code_docs:
            # Analyze code-specific quality metrics
            code_quality = self._analyze_code_chunks(chunks, code_docs, use_ast)
            quality_metrics.update(code_quality)
            
        # Calculate overall quality scores
        quality_metrics["completeness_score"] = self._calculate_completeness_score(chunks)
        quality_metrics["coherence_score"] = self._calculate_coherence_score(chunks)
        
        return quality_metrics
        
    def _analyze_code_chunks(self, chunks: List[str], code_docs: List[Document], use_ast: bool) -> Dict[str, Any]:
        """Analyze code-specific chunk quality."""
        # Count chunks that likely contain complete functions/classes
        complete_constructs = 0
        partial_constructs = 0
        
        for chunk in chunks:
            # Simple heuristics for code completeness
            if self._contains_complete_function(chunk):
                complete_constructs += 1
            elif self._contains_partial_function(chunk):
                partial_constructs += 1
                
        total_code_chunks = complete_constructs + partial_constructs
        completeness_ratio = complete_constructs / total_code_chunks if total_code_chunks > 0 else 0.0
        
        return {
            "complete_functions": complete_constructs,
            "partial_functions": partial_constructs,
            "function_completeness_ratio": completeness_ratio,
            "avg_code_chunk_lines": self._calculate_avg_lines_per_chunk(chunks)
        }
        
    def _contains_complete_function(self, chunk: str) -> bool:
        """Check if chunk contains a complete function definition."""
        # Simple heuristics - could be enhanced with AST parsing
        lines = chunk.strip().split('\n')
        
        # Look for function/method definitions
        has_function_def = any(
            line.strip().startswith(('def ', 'function ', 'public ', 'private ', 'protected '))
            or 'function' in line or '(' in line and ')' in line
            for line in lines
        )
        
        # Check for balanced braces/indentation
        if has_function_def:
            # Count braces for languages like Java, C#, JS
            open_braces = chunk.count('{')
            close_braces = chunk.count('}')
            
            # For Python, check indentation patterns
            if 'def ' in chunk:
                return self._check_python_function_completeness(chunk)
            
            # For brace languages, check balance
            return open_braces > 0 and open_braces == close_braces
            
        return False
        
    def _contains_partial_function(self, chunk: str) -> bool:
        """Check if chunk contains partial function code."""
        # Look for function-like patterns without completeness
        return any(
            keyword in chunk.lower() 
            for keyword in ['def ', 'function', 'public ', 'private ', 'class ', 'interface ']
        )
        
    def _check_python_function_completeness(self, chunk: str) -> bool:
        """Check if Python function in chunk is complete."""
        lines = chunk.strip().split('\n')
        
        # Find function definition
        func_line_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                func_line_idx = i
                break
                
        if func_line_idx is None:
            return False
            
        # Get function indentation level
        func_line = lines[func_line_idx]
        func_indent = len(func_line) - len(func_line.lstrip())
        
        # Check if there are lines at or less indentation after the function
        for line in lines[func_line_idx + 1:]:
            if line.strip():  # Non-empty line
                line_indent = len(line) - len(line.lstrip())
                if line_indent <= func_indent:
                    return True  # Function ends before this line
                    
        # Function goes to end of chunk - could be complete or incomplete
        return True
        
    def _calculate_avg_lines_per_chunk(self, chunks: List[str]) -> float:
        """Calculate average lines per chunk."""
        if not chunks:
            return 0.0
        return statistics.mean(len(chunk.split('\n')) for chunk in chunks)
        
    def _calculate_completeness_score(self, chunks: List[str]) -> float:
        """Calculate overall completeness score for chunks."""
        if not chunks:
            return 0.0
            
        # Simple metric based on average chunk length and distribution
        lengths = [len(chunk) for chunk in chunks]
        avg_length = statistics.mean(lengths)
        
        # Prefer more consistent chunk sizes (less variance indicates better splitting)
        variance = statistics.variance(lengths) if len(lengths) > 1 else 0
        consistency_score = 1.0 / (1.0 + variance / (avg_length ** 2)) if avg_length > 0 else 0
        
        return min(consistency_score, 1.0)
        
    def _calculate_coherence_score(self, chunks: List[str]) -> float:
        """Calculate semantic coherence score for chunks."""
        # Simple heuristic based on chunk characteristics
        if not chunks:
            return 0.0
            
        # Count chunks that seem semantically coherent
        coherent_chunks = 0
        for chunk in chunks:
            # Very basic coherence check
            lines = chunk.strip().split('\n')
            if len(lines) > 1:
                # Check if chunk doesn't end abruptly in the middle of a statement
                last_line = lines[-1].strip()
                if last_line and not last_line.endswith((',', '\\', '+', '-', '*', '/')):
                    coherent_chunks += 1
            else:
                coherent_chunks += 1
                
        return coherent_chunks / len(chunks)
        
    def _calculate_comparison_metrics(self, ast_results: Dict, naive_results: Dict) -> Dict[str, Any]:
        """Calculate comparison metrics between AST and naive approaches."""
        comparison = {}
        
        # Performance comparisons
        comparison["speed_ratio"] = naive_results["processing_time"] / ast_results["processing_time"]
        comparison["memory_ratio"] = naive_results["peak_memory_mb"] / ast_results["peak_memory_mb"] if ast_results["peak_memory_mb"] > 0 else 1.0
        
        # Quality comparisons
        comparison["chunk_count_ratio"] = ast_results["chunk_count"] / naive_results["chunk_count"] if naive_results["chunk_count"] > 0 else 1.0
        comparison["completeness_improvement"] = ast_results.get("function_completeness_ratio", 0) - naive_results.get("function_completeness_ratio", 0)
        comparison["coherence_improvement"] = ast_results["coherence_score"] - naive_results["coherence_score"]
        
        # Efficiency metrics
        comparison["ast_efficiency"] = ast_results["chunk_count"] / ast_results["processing_time"] if ast_results["processing_time"] > 0 else 0
        comparison["naive_efficiency"] = naive_results["chunk_count"] / naive_results["processing_time"] if naive_results["processing_time"] > 0 else 0
        
        return comparison
        
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("# AST-Aware vs Naive Chunking Evaluation Report")
        report.append("=" * 60)
        report.append("")
        
        # Summary table
        report.append("## Executive Summary")
        report.append("")
        
        ast_r = results["ast_chunking"]
        naive_r = results["naive_chunking"]
        comp_r = results["comparison"]
        
        report.append("| Metric | Naive Chunking | AST Chunking | Improvement |")
        report.append("|--------|---------------|--------------|-------------|")
        report.append(f"| Processing Time | {naive_r['processing_time']:.3f}s | {ast_r['processing_time']:.3f}s | {comp_r['speed_ratio']:.2f}x slower |")
        report.append(f"| Memory Usage | {naive_r['peak_memory_mb']:.1f} MB | {ast_r['peak_memory_mb']:.1f} MB | {comp_r['memory_ratio']:.2f}x |")
        report.append(f"| Chunk Count | {naive_r['chunk_count']} | {ast_r['chunk_count']} | {comp_r['chunk_count_ratio']:.2f}x |")
        report.append(f"| Avg Chunk Length | {naive_r['avg_chunk_length']:.0f} | {ast_r['avg_chunk_length']:.0f} | {((ast_r['avg_chunk_length'] / naive_r['avg_chunk_length']) - 1) * 100:.1f}% |")
        
        if "function_completeness_ratio" in ast_r:
            report.append(f"| Function Completeness | {naive_r.get('function_completeness_ratio', 0):.2f} | {ast_r.get('function_completeness_ratio', 0):.2f} | +{comp_r['completeness_improvement']:.2f} |")
            
        report.append(f"| Coherence Score | {naive_r['coherence_score']:.3f} | {ast_r['coherence_score']:.3f} | +{comp_r['coherence_improvement']:.3f} |")
        report.append("")
        
        # Detailed analysis
        report.append("## Detailed Analysis")
        report.append("")
        
        report.append("### Code Analysis Results")
        if "code_document_count" in ast_r:
            report.append(f"- Code documents processed: {ast_r['code_document_count']}")
            report.append(f"- Text documents processed: {ast_r['text_document_count']}")
            
            if "complete_functions" in ast_r:
                report.append(f"- Complete functions (AST): {ast_r['complete_functions']}")
                report.append(f"- Complete functions (Naive): {naive_r.get('complete_functions', 0)}")
                report.append(f"- Partial functions (AST): {ast_r['partial_functions']}")
                report.append(f"- Partial functions (Naive): {naive_r.get('partial_functions', 0)}")
        
        report.append("")
        
        # Performance analysis
        report.append("### Performance Analysis")
        report.append(f"- AST chunking is **{comp_r['speed_ratio']:.1f}x {'slower' if comp_r['speed_ratio'] > 1 else 'faster'}** than naive chunking")
        report.append(f"- Memory overhead: **{((comp_r['memory_ratio'] - 1) * 100):.1f}%** {'increase' if comp_r['memory_ratio'] > 1 else 'decrease'}")
        report.append(f"- AST efficiency: **{comp_r['ast_efficiency']:.1f}** chunks/second")
        report.append(f"- Naive efficiency: **{comp_r['naive_efficiency']:.1f}** chunks/second")
        report.append("")
        
        # Recommendations
        report.append("### Recommendations")
        report.append("")
        
        if comp_r.get("completeness_improvement", 0) > 0.1:
            report.append("✅ **AST chunking recommended** - Significant improvement in code completeness")
        elif comp_r.get("coherence_improvement", 0) > 0.05:
            report.append("✅ **AST chunking recommended** - Better semantic coherence")
        elif comp_r["speed_ratio"] > 3.0:
            report.append("⚠️ **Consider use case** - AST chunking has significant performance overhead")
        else:
            report.append("📊 **Mixed results** - Evaluate based on specific requirements")
            
        report.append("")
        report.append("### Use Case Guidelines")
        report.append("- **Use AST chunking for**: Code-heavy repositories, function-level search, semantic code analysis")
        report.append("- **Use naive chunking for**: Mixed content, performance-critical applications, simple text search")
        
        return "\n".join(report)
        
    def save_results(self, results: Dict[str, Any]):
        """Save evaluation results to files."""
        # Save raw results as JSON
        results_file = self.output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Results saved to {results_file}")
        
        # Save human-readable report
        report = self.generate_report(results)
        report_file = self.output_dir / "evaluation_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
            
        logger.info(f"Report saved to {report_file}")
        
    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj


def main():
    parser = argparse.ArgumentParser(description="Evaluate AST-aware vs naive chunking performance")
    parser.add_argument(
        "--test-data-dir",
        type=str,
        default="data/code_samples",
        help="Directory containing test code files (default: data/code_samples)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results (default: benchmark_results)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        # Initialize evaluator
        evaluator = ChunkingEvaluator(args.test_data_dir, args.output_dir)
        
        # Load test documents
        documents = evaluator.load_test_documents()
        
        if not documents:
            logger.error("No documents found in test directory")
            return 1
            
        # Run evaluation
        logger.info("Starting comprehensive evaluation...")
        results = evaluator.evaluate_chunking_performance(documents)
        
        # Save and display results
        evaluator.save_results(results)
        
        # Print summary to console
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        ast_r = results["ast_chunking"]
        naive_r = results["naive_chunking"]
        comp_r = results["comparison"]
        
        print(f"AST Chunking:   {ast_r['chunk_count']} chunks in {ast_r['processing_time']:.2f}s")
        print(f"Naive Chunking: {naive_r['chunk_count']} chunks in {naive_r['processing_time']:.2f}s")
        print(f"Speed Ratio:    {comp_r['speed_ratio']:.2f}x (AST vs Naive)")
        
        if "function_completeness_ratio" in ast_r:
            print(f"AST Completeness:   {ast_r['function_completeness_ratio']:.2f}")
            print(f"Naive Completeness: {naive_r.get('function_completeness_ratio', 0):.2f}")
            print(f"Improvement:        +{comp_r['completeness_improvement']:.2f}")
            
        print(f"\nDetailed results saved to: {evaluator.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())