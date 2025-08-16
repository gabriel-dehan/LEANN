#!/usr/bin/env python3
"""
Large-Scale Codebase Evaluation Framework

This framework evaluates AST vs Naive chunking on enterprise-scale repositories
to measure performance gains that only become apparent at scale.
"""

import json
import logging
import sys
import time
import subprocess
import shutil
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import os

# Add apps directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

from chunking import create_text_chunks
from llama_index.core import SimpleDirectoryReader, Document

logger = logging.getLogger(__name__)

@dataclass
class RepositoryConfig:
    """Configuration for a test repository."""
    name: str
    url: str
    expected_size_mb: int
    primary_languages: List[str]
    complexity_score: float  # 1-10 scale
    description: str

@dataclass
class EvaluationConfig:
    """Configuration for the evaluation run."""
    chunk_sizes: List[int] = None
    chunk_overlaps: List[int] = None
    max_files_per_repo: int = 1000
    max_file_size_kb: int = 500
    timeout_minutes: int = 30
    parallel_workers: int = 4
    enable_memory_profiling: bool = True
    enable_cross_file_analysis: bool = True

class LargeScaleEvaluator:
    """Evaluates AST vs Naive chunking on large codebases."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.work_dir = Path("large_scale_evaluation")
        self.work_dir.mkdir(exist_ok=True)
        
        # Default configurations
        if self.config.chunk_sizes is None:
            self.config.chunk_sizes = [256, 512, 1024, 2048]
        if self.config.chunk_overlaps is None:
            self.config.chunk_overlaps = [64, 128, 256]
    
    def get_test_repositories(self) -> List[RepositoryConfig]:
        """Get list of repositories to test on."""
        return [
            RepositoryConfig(
                name="vscode",
                url="https://github.com/microsoft/vscode.git",
                expected_size_mb=150,
                primary_languages=["typescript", "javascript"],
                complexity_score=9.0,
                description="Large TypeScript codebase with complex architecture"
            ),
            RepositoryConfig(
                name="nodejs",
                url="https://github.com/nodejs/node.git", 
                expected_size_mb=100,
                primary_languages=["javascript", "c++"],
                complexity_score=8.5,
                description="High-performance JavaScript runtime"
            ),
            RepositoryConfig(
                name="django",
                url="https://github.com/django/django.git",
                expected_size_mb=40,
                primary_languages=["python"],
                complexity_score=8.0,
                description="Python web framework with extensive ORM"
            ),
            RepositoryConfig(
                name="kubernetes",
                url="https://github.com/kubernetes/kubernetes.git",
                expected_size_mb=200,
                primary_languages=["go"],
                complexity_score=9.5,
                description="Container orchestration system"
            ),
            RepositoryConfig(
                name="tensorflow",
                url="https://github.com/tensorflow/tensorflow.git",
                expected_size_mb=300,
                primary_languages=["python", "c++"],
                complexity_score=9.8,
                description="Machine learning framework"
            ),
            RepositoryConfig(
                name="react",
                url="https://github.com/facebook/react.git",
                expected_size_mb=30,
                primary_languages=["javascript"],
                complexity_score=7.5,
                description="JavaScript UI library"
            )
        ]
    
    def clone_repository(self, repo: RepositoryConfig) -> Optional[Path]:
        """Clone a repository for testing."""
        repo_path = self.work_dir / repo.name
        
        if repo_path.exists():
            logger.info(f"Repository {repo.name} already exists, updating...")
            try:
                subprocess.run(
                    ["git", "pull"], 
                    cwd=repo_path, 
                    check=True, 
                    capture_output=True,
                    timeout=300
                )
                return repo_path
            except subprocess.TimeoutExpired:
                logger.warning(f"Git pull timeout for {repo.name}, using existing version")
                return repo_path
            except subprocess.CalledProcessError as e:
                logger.warning(f"Git pull failed for {repo.name}: {e}")
                return repo_path
        
        logger.info(f"Cloning {repo.name} from {repo.url}...")
        try:
            # Clone with depth limit for faster download
            subprocess.run([
                "git", "clone", 
                "--depth", "1",  # Shallow clone for faster setup
                "--single-branch",
                repo.url, 
                str(repo_path)
            ], check=True, capture_output=True, timeout=600)
            
            logger.info(f"Successfully cloned {repo.name}")
            return repo_path
            
        except subprocess.TimeoutExpired:
            logger.error(f"Clone timeout for {repo.name}")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone {repo.name}: {e}")
            return None
    
    def analyze_repository_structure(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze repository structure and complexity."""
        analysis = {
            "total_files": 0,
            "code_files": 0,
            "total_size_mb": 0.0,
            "languages": {},
            "avg_file_size": 0,
            "max_file_size": 0,
            "directory_depth": 0,
            "large_files": [],
            "complexity_indicators": {}
        }
        
        logger.info(f"Analyzing repository structure: {repo_path.name}")
        
        # Extensions to consider as code
        code_extensions = {
            '.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.c', '.cpp', '.cc', 
            '.h', '.hpp', '.cs', '.go', '.rs', '.rb', '.php', '.kt', '.swift',
            '.scala', '.clj', '.cljs', '.hs', '.ml', '.f90', '.r', '.m'
        }
        
        file_sizes = []
        
        for file_path in repo_path.rglob('*'):
            if file_path.is_file():
                try:
                    file_size = file_path.stat().st_size
                    analysis["total_files"] += 1
                    file_sizes.append(file_size)
                    
                    if file_size > analysis["max_file_size"]:
                        analysis["max_file_size"] = file_size
                    
                    # Track large files
                    if file_size > 100 * 1024:  # > 100KB
                        analysis["large_files"].append({
                            "path": str(file_path.relative_to(repo_path)),
                            "size_kb": file_size / 1024
                        })
                    
                    # Language detection
                    suffix = file_path.suffix.lower()
                    if suffix in code_extensions:
                        analysis["code_files"] += 1
                        if suffix not in analysis["languages"]:
                            analysis["languages"][suffix] = 0
                        analysis["languages"][suffix] += 1
                    
                    # Directory depth
                    depth = len(file_path.relative_to(repo_path).parts) - 1
                    if depth > analysis["directory_depth"]:
                        analysis["directory_depth"] = depth
                        
                except (OSError, PermissionError):
                    continue
        
        analysis["total_size_mb"] = sum(file_sizes) / (1024 * 1024)
        analysis["avg_file_size"] = statistics.mean(file_sizes) if file_sizes else 0
        
        # Complexity indicators
        analysis["complexity_indicators"] = {
            "file_size_variance": statistics.variance(file_sizes) if len(file_sizes) > 1 else 0,
            "language_diversity": len(analysis["languages"]),
            "large_file_ratio": len(analysis["large_files"]) / max(analysis["total_files"], 1),
            "code_file_ratio": analysis["code_files"] / max(analysis["total_files"], 1)
        }
        
        return analysis
    
    def filter_files_for_evaluation(self, repo_path: Path) -> List[Path]:
        """Filter repository files for evaluation based on size and type."""
        logger.info(f"Filtering files for evaluation: {repo_path.name}")
        
        code_extensions = {
            '.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.c', '.cpp', '.cc', 
            '.h', '.hpp', '.cs', '.go', '.rs', '.rb', '.php', '.kt', '.swift'
        }
        
        files = []
        for file_path in repo_path.rglob('*'):
            if (file_path.is_file() and 
                file_path.suffix.lower() in code_extensions and
                file_path.stat().st_size <= self.config.max_file_size_kb * 1024):
                
                # Skip common directories that don't contain source code
                skip_dirs = {'node_modules', '.git', '__pycache__', 'build', 'dist', 'target', 'vendor'}
                if not any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                    files.append(file_path)
                    
                if len(files) >= self.config.max_files_per_repo:
                    break
        
        logger.info(f"Selected {len(files)} files for evaluation")
        return files
    
    def load_documents_from_files(self, file_paths: List[Path]) -> List[Document]:
        """Load documents from file paths."""
        documents = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Skip very small or empty files
                if len(content.strip()) < 100:
                    continue
                    
                doc = Document(
                    text=content,
                    metadata={
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "file_size": len(content),
                        "language": file_path.suffix.lower()
                    }
                )
                documents.append(doc)
                
            except (UnicodeDecodeError, PermissionError, OSError) as e:
                logger.warning(f"Could not read {file_path}: {e}")
                continue
        
        return documents
    
    def evaluate_chunking_performance(self, documents: List[Document], repo_name: str) -> Dict[str, Any]:
        """Evaluate chunking performance on a set of documents."""
        logger.info(f"Evaluating chunking performance for {repo_name} ({len(documents)} documents)")
        
        results = {
            "repository": repo_name,
            "document_count": len(documents),
            "total_size": sum(len(doc.text) for doc in documents),
            "ast_results": {},
            "naive_results": {},
            "comparison": {}
        }
        
        # Test different chunk size configurations
        for chunk_size in self.config.chunk_sizes:
            for overlap in self.config.chunk_overlaps:
                config_name = f"size_{chunk_size}_overlap_{overlap}"
                logger.info(f"Testing configuration: {config_name}")
                
                # AST Chunking
                ast_result = self._evaluate_chunking_approach(
                    documents, use_ast=True, chunk_size=chunk_size, 
                    overlap=overlap, config_name=config_name
                )
                results["ast_results"][config_name] = ast_result
                
                # Naive Chunking
                naive_result = self._evaluate_chunking_approach(
                    documents, use_ast=False, chunk_size=chunk_size, 
                    overlap=overlap, config_name=config_name
                )
                results["naive_results"][config_name] = naive_result
                
                # Comparison
                comparison = self._compare_chunking_results(ast_result, naive_result)
                results["comparison"][config_name] = comparison
        
        return results
    
    def _evaluate_chunking_approach(self, documents: List[Document], use_ast: bool, 
                                   chunk_size: int, overlap: int, config_name: str) -> Dict[str, Any]:
        """Evaluate a single chunking approach."""
        approach = "AST" if use_ast else "Naive"
        
        # Memory profiling
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        try:
            chunks = create_text_chunks(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                use_ast_chunking=use_ast,
                ast_chunk_size=chunk_size * 2 if use_ast else chunk_size,
                ast_chunk_overlap=overlap
            )
            
            processing_time = time.time() - start_time
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            # Analyze chunk quality
            chunk_analysis = self._analyze_chunk_quality(chunks, approach)
            
            return {
                "approach": approach,
                "config": config_name,
                "chunk_count": len(chunks),
                "processing_time": processing_time,
                "memory_used_mb": memory_used,
                "chunks_per_second": len(chunks) / processing_time if processing_time > 0 else 0,
                "success": True,
                "chunk_analysis": chunk_analysis
            }
            
        except Exception as e:
            logger.error(f"{approach} chunking failed for {config_name}: {e}")
            return {
                "approach": approach,
                "config": config_name,
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _analyze_chunk_quality(self, chunks: List[str], approach: str) -> Dict[str, Any]:
        """Analyze the quality of generated chunks."""
        if not chunks:
            return {"error": "No chunks to analyze"}
        
        # Extract text from chunks if they're not strings
        chunk_texts = []
        for chunk in chunks:
            if isinstance(chunk, str):
                chunk_texts.append(chunk)
            elif hasattr(chunk, 'text'):
                chunk_texts.append(chunk.text)
            elif isinstance(chunk, dict) and 'text' in chunk:
                chunk_texts.append(chunk['text'])
            else:
                chunk_texts.append(str(chunk))
        
        chunk_sizes = [len(chunk) for chunk in chunk_texts]
        
        analysis = {
            "avg_chunk_size": statistics.mean(chunk_sizes),
            "chunk_size_variance": statistics.variance(chunk_sizes) if len(chunk_sizes) > 1 else 0,
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_characters": sum(chunk_sizes),
            "function_preservation": self._calculate_function_preservation(chunk_texts),
            "class_preservation": self._calculate_class_preservation(chunk_texts),
            "import_coherence": self._calculate_import_coherence(chunk_texts),
            "semantic_coherence": self._calculate_semantic_coherence(chunk_texts)
        }
        
        return analysis
    
    def _calculate_function_preservation(self, chunks: List[str]) -> float:
        """Calculate how well function boundaries are preserved."""
        complete_functions = 0
        partial_functions = 0
        
        for chunk in chunks:
            lines = chunk.split('\n')
            function_starts = sum(1 for line in lines if line.strip().startswith(('def ', 'function ', 'func ', 'fn ', 'public ', 'private ', 'protected ')))
            
            # Simple heuristic: count braces/indentation to detect completeness
            if 'def ' in chunk or 'function ' in chunk:
                if self._is_likely_complete_function(chunk):
                    complete_functions += function_starts
                else:
                    partial_functions += function_starts
        
        total_functions = complete_functions + partial_functions
        return complete_functions / total_functions if total_functions > 0 else 0.0
    
    def _calculate_class_preservation(self, chunks: List[str]) -> float:
        """Calculate how well class boundaries are preserved."""
        complete_classes = 0
        partial_classes = 0
        
        for chunk in chunks:
            if 'class ' in chunk:
                if self._is_likely_complete_class(chunk):
                    complete_classes += chunk.count('class ')
                else:
                    partial_classes += chunk.count('class ')
        
        total_classes = complete_classes + partial_classes
        return complete_classes / total_classes if total_classes > 0 else 0.0
    
    def _calculate_import_coherence(self, chunks: List[str]) -> float:
        """Calculate how well import statements are grouped."""
        chunks_with_imports = 0
        chunks_with_mixed_content = 0
        
        for chunk in chunks:
            has_imports = any(line.strip().startswith(('import ', 'from ', 'using ', '#include')) 
                            for line in chunk.split('\n'))
            has_other_content = any(line.strip() and not line.strip().startswith(('import ', 'from ', 'using ', '#include', '//', '/*', '*', '#')) 
                                  for line in chunk.split('\n'))
            
            if has_imports:
                chunks_with_imports += 1
                if has_other_content:
                    chunks_with_mixed_content += 1
        
        return 1.0 - (chunks_with_mixed_content / chunks_with_imports) if chunks_with_imports > 0 else 1.0
    
    def _calculate_semantic_coherence(self, chunks: List[str]) -> float:
        """Calculate semantic coherence based on code structure."""
        coherent_chunks = 0
        
        for chunk in chunks:
            # Simple heuristics for semantic coherence
            lines = [line.strip() for line in chunk.split('\n') if line.strip()]
            if not lines:
                continue
                
            # Check for consistent indentation patterns
            indentation_levels = []
            for line in lines:
                if line and not line.startswith(('//','/*','*','#')):
                    indent = len(line) - len(line.lstrip())
                    indentation_levels.append(indent)
            
            if indentation_levels:
                # Coherent if indentation follows logical patterns
                indent_variance = statistics.variance(indentation_levels) if len(indentation_levels) > 1 else 0
                if indent_variance < 100:  # Reasonable indentation consistency
                    coherent_chunks += 1
        
        return coherent_chunks / len(chunks) if chunks else 0.0
    
    def _is_likely_complete_function(self, chunk: str) -> bool:
        """Heuristic to determine if a chunk contains a complete function."""
        lines = chunk.split('\n')
        brace_count = 0
        has_function_start = False
        
        for line in lines:
            if line.strip().startswith(('def ', 'function ', 'func ', 'fn ')):
                has_function_start = True
            brace_count += line.count('{') - line.count('}')
        
        return has_function_start and brace_count == 0
    
    def _is_likely_complete_class(self, chunk: str) -> bool:
        """Heuristic to determine if a chunk contains a complete class."""
        lines = chunk.split('\n')
        brace_count = 0
        has_class_start = False
        
        for line in lines:
            if line.strip().startswith('class '):
                has_class_start = True
            brace_count += line.count('{') - line.count('}')
        
        return has_class_start and brace_count == 0
    
    def _compare_chunking_results(self, ast_result: Dict, naive_result: Dict) -> Dict[str, Any]:
        """Compare AST vs Naive chunking results."""
        if not ast_result.get("success") or not naive_result.get("success"):
            return {"error": "One or both approaches failed"}
        
        comparison = {}
        
        # Performance comparison
        comparison["speed_ratio"] = ast_result["processing_time"] / naive_result["processing_time"]
        comparison["memory_ratio"] = ast_result["memory_used_mb"] / naive_result["memory_used_mb"] if naive_result["memory_used_mb"] > 0 else float('inf')
        comparison["chunk_count_ratio"] = ast_result["chunk_count"] / naive_result["chunk_count"] if naive_result["chunk_count"] > 0 else float('inf')
        
        # Quality comparison
        ast_analysis = ast_result.get("chunk_analysis", {})
        naive_analysis = naive_result.get("chunk_analysis", {})
        
        quality_metrics = ["function_preservation", "class_preservation", "import_coherence", "semantic_coherence"]
        
        for metric in quality_metrics:
            ast_value = ast_analysis.get(metric, 0)
            naive_value = naive_analysis.get(metric, 0)
            comparison[f"{metric}_improvement"] = ast_value - naive_value
            comparison[f"{metric}_ratio"] = ast_value / naive_value if naive_value > 0 else float('inf')
        
        return comparison
    
    def run_large_scale_evaluation(self, repositories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run the complete large-scale evaluation."""
        logger.info("Starting large-scale AST vs Naive chunking evaluation")
        
        all_repos = self.get_test_repositories()
        if repositories:
            all_repos = [repo for repo in all_repos if repo.name in repositories]
        
        evaluation_results = {
            "evaluation_config": {
                "chunk_sizes": self.config.chunk_sizes,
                "chunk_overlaps": self.config.chunk_overlaps,
                "max_files_per_repo": self.config.max_files_per_repo,
                "max_file_size_kb": self.config.max_file_size_kb
            },
            "repository_results": {},
            "aggregate_results": {},
            "conclusions": {}
        }
        
        for repo in all_repos:
            logger.info(f"Processing repository: {repo.name}")
            
            # Clone repository
            repo_path = self.clone_repository(repo)
            if not repo_path:
                logger.error(f"Failed to clone {repo.name}, skipping")
                continue
            
            # Analyze repository structure
            repo_analysis = self.analyze_repository_structure(repo_path)
            
            # Filter files for evaluation
            eval_files = self.filter_files_for_evaluation(repo_path)
            if not eval_files:
                logger.warning(f"No suitable files found in {repo.name}, skipping")
                continue
            
            # Load documents
            documents = self.load_documents_from_files(eval_files)
            if not documents:
                logger.warning(f"No documents loaded from {repo.name}, skipping")
                continue
            
            # Evaluate chunking performance
            chunking_results = self.evaluate_chunking_performance(documents, repo.name)
            
            evaluation_results["repository_results"][repo.name] = {
                "repository_info": repo.__dict__,
                "repository_analysis": repo_analysis,
                "file_count": len(eval_files),
                "document_count": len(documents),
                "chunking_results": chunking_results
            }
        
        # Calculate aggregate results
        evaluation_results["aggregate_results"] = self._calculate_aggregate_results(evaluation_results["repository_results"])
        
        # Generate conclusions
        evaluation_results["conclusions"] = self._generate_conclusions(evaluation_results)
        
        return evaluation_results
    
    def _calculate_aggregate_results(self, repo_results: Dict) -> Dict[str, Any]:
        """Calculate aggregate results across all repositories."""
        aggregate = {
            "total_repositories": len(repo_results),
            "total_documents": 0,
            "total_files": 0,
            "avg_performance_ratios": {},
            "avg_quality_improvements": {},
            "repository_size_correlations": {}
        }
        
        # Collect metrics across all repos and configurations
        all_speed_ratios = []
        all_memory_ratios = []
        all_function_improvements = []
        all_semantic_improvements = []
        
        for repo_name, repo_data in repo_results.items():
            chunking_results = repo_data.get("chunking_results", {})
            aggregate["total_documents"] += chunking_results.get("document_count", 0)
            aggregate["total_files"] += repo_data.get("file_count", 0)
            
            # Collect comparison metrics
            comparisons = chunking_results.get("comparison", {})
            for config, comparison in comparisons.items():
                if "speed_ratio" in comparison:
                    all_speed_ratios.append(comparison["speed_ratio"])
                if "memory_ratio" in comparison:
                    all_memory_ratios.append(comparison["memory_ratio"])
                if "function_preservation_improvement" in comparison:
                    all_function_improvements.append(comparison["function_preservation_improvement"])
                if "semantic_coherence_improvement" in comparison:
                    all_semantic_improvements.append(comparison["semantic_coherence_improvement"])
        
        # Calculate averages
        if all_speed_ratios:
            aggregate["avg_performance_ratios"]["speed"] = statistics.mean(all_speed_ratios)
        if all_memory_ratios:
            aggregate["avg_performance_ratios"]["memory"] = statistics.mean(all_memory_ratios)
        if all_function_improvements:
            aggregate["avg_quality_improvements"]["function_preservation"] = statistics.mean(all_function_improvements)
        if all_semantic_improvements:
            aggregate["avg_quality_improvements"]["semantic_coherence"] = statistics.mean(all_semantic_improvements)
        
        return aggregate
    
    def _generate_conclusions(self, evaluation_results: Dict) -> Dict[str, Any]:
        """Generate conclusions from the evaluation results."""
        aggregate = evaluation_results.get("aggregate_results", {})
        
        conclusions = {
            "overall_recommendation": "Conditional",
            "key_findings": [],
            "use_case_recommendations": {},
            "scale_effects": {},
            "configuration_recommendations": {}
        }
        
        # Analyze performance vs quality tradeoffs
        avg_speed_ratio = aggregate.get("avg_performance_ratios", {}).get("speed", 1.0)
        avg_function_improvement = aggregate.get("avg_quality_improvements", {}).get("function_preservation", 0.0)
        avg_semantic_improvement = aggregate.get("avg_quality_improvements", {}).get("semantic_coherence", 0.0)
        
        # Generate recommendations based on results
        if avg_function_improvement > 0.2 and avg_semantic_improvement > 0.15:
            conclusions["overall_recommendation"] = "AST Chunking"
            conclusions["key_findings"].append("Significant quality improvements justify performance cost")
        elif avg_speed_ratio > 5.0 and avg_function_improvement < 0.1:
            conclusions["overall_recommendation"] = "Naive Chunking"
            conclusions["key_findings"].append("Performance benefits outweigh minimal quality gains")
        else:
            conclusions["overall_recommendation"] = "Conditional"
            conclusions["key_findings"].append("Choice depends on specific requirements and constraints")
        
        return conclusions


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Large-scale AST vs Naive chunking evaluation")
    parser.add_argument("--repositories", nargs="+", help="Specific repositories to test")
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[512, 1024], help="Chunk sizes to test")
    parser.add_argument("--max-files", type=int, default=500, help="Maximum files per repository")
    parser.add_argument("--output", type=str, default="large_scale_results.json", help="Output file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Configure evaluation
    config = EvaluationConfig(
        chunk_sizes=args.chunk_sizes,
        max_files_per_repo=args.max_files,
        chunk_overlaps=[64, 128]
    )
    
    evaluator = LargeScaleEvaluator(config)
    
    try:
        results = evaluator.run_large_scale_evaluation(args.repositories)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("LARGE-SCALE EVALUATION SUMMARY")
        print("="*80)
        
        aggregate = results.get("aggregate_results", {})
        conclusions = results.get("conclusions", {})
        
        print(f"Repositories Evaluated: {aggregate.get('total_repositories', 0)}")
        print(f"Total Documents: {aggregate.get('total_documents', 0)}")
        print(f"Total Files: {aggregate.get('total_files', 0)}")
        
        performance = aggregate.get("avg_performance_ratios", {})
        quality = aggregate.get("avg_quality_improvements", {})
        
        if "speed" in performance:
            print(f"Average Speed Ratio (AST/Naive): {performance['speed']:.2f}x")
        if "function_preservation" in quality:
            print(f"Function Preservation Improvement: {quality['function_preservation']:.1%}")
        if "semantic_coherence" in quality:
            print(f"Semantic Coherence Improvement: {quality['semantic_coherence']:.1%}")
        
        print(f"\nOverall Recommendation: {conclusions.get('overall_recommendation', 'Unknown')}")
        
        for finding in conclusions.get('key_findings', []):
            print(f"• {finding}")
        
        print(f"\nDetailed results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())