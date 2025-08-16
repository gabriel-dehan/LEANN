#!/usr/bin/env python3
"""
Context Economy Analyzer

This module analyzes the "context economy" of different chunking approaches - 
how efficiently they use tokens to provide meaningful context for RAG applications.
This is crucial for evaluating real-world performance where token costs and 
context window limits matter.
"""

import json
import logging
import sys
import time
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import re

# Add apps directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

from chunking import create_text_chunks
from llama_index.core import SimpleDirectoryReader, Document

logger = logging.getLogger(__name__)

@dataclass
class TokenUsageStats:
    """Statistics about token usage for a set of chunks."""
    total_tokens: int
    avg_tokens_per_chunk: float
    token_efficiency_score: float
    redundancy_ratio: float
    information_density: float

@dataclass
class ContextEfficiency:
    """Measures how efficiently context is used."""
    relevant_token_ratio: float
    semantic_coherence_per_token: float
    cross_reference_density: float
    completeness_per_token: float

class ContextEconomyAnalyzer:
    """Analyzes context economy and token efficiency of chunking approaches."""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.token_cache = {}  # Cache token counts for performance
    
    def analyze_context_economy(self) -> Dict[str, Any]:
        """Perform comprehensive context economy analysis."""
        logger.info(f"Starting context economy analysis on {len(self.documents)} documents")
        
        results = {
            "analysis_metadata": {
                "document_count": len(self.documents),
                "total_content_size": sum(len(doc.text) for doc in self.documents),
                "analysis_timestamp": time.time()
            },
            "ast_analysis": {},
            "naive_analysis": {},
            "comparison": {},
            "token_efficiency": {},
            "redundancy_analysis": {},
            "retrieval_efficiency": {}
        }
        
        # Generate chunks for both approaches
        logger.info("Generating chunks for analysis...")
        ast_chunks = self._generate_chunks_with_metadata(use_ast=True)
        naive_chunks = self._generate_chunks_with_metadata(use_ast=False)
        
        # Analyze each approach
        results["ast_analysis"] = self._analyze_chunking_approach(ast_chunks, "AST")
        results["naive_analysis"] = self._analyze_chunking_approach(naive_chunks, "Naive")
        
        # Comparative analysis
        results["comparison"] = self._compare_approaches(
            results["ast_analysis"], 
            results["naive_analysis"]
        )
        
        results["token_efficiency"] = self._analyze_token_efficiency(ast_chunks, naive_chunks)
        results["redundancy_analysis"] = self._analyze_redundancy(ast_chunks, naive_chunks)
        results["retrieval_efficiency"] = self._analyze_retrieval_efficiency(ast_chunks, naive_chunks)
        
        return results
    
    def _generate_chunks_with_metadata(self, use_ast: bool) -> List[Dict[str, Any]]:
        """Generate chunks with comprehensive metadata for analysis."""
        approach = "AST" if use_ast else "Naive"
        logger.info(f"Generating {approach} chunks...")
        
        start_time = time.time()
        raw_chunks = create_text_chunks(
            self.documents,
            chunk_size=512,
            chunk_overlap=128,
            use_ast_chunking=use_ast,
            ast_chunk_size=1024 if use_ast else 512,
            ast_chunk_overlap=128
        )
        processing_time = time.time() - start_time
        
        # Enrich chunks with metadata
        enriched_chunks = []
        for i, raw_chunk in enumerate(raw_chunks):
            # Extract text content
            text_content = self._extract_text_from_chunk(raw_chunk)
            
            if len(text_content.strip()) < 20:
                continue
            
            # Calculate token count (approximation: 1 token ≈ 4 characters)
            token_count = len(text_content) // 4
            
            # Analyze chunk content
            metadata = self._analyze_chunk_for_economy(text_content, token_count)
            
            enriched_chunk = {
                "id": i,
                "text": text_content,
                "approach": approach,
                "token_count": token_count,
                "character_count": len(text_content),
                "line_count": len(text_content.split('\n')),
                "processing_time_contribution": processing_time / len(raw_chunks),
                **metadata
            }
            
            enriched_chunks.append(enriched_chunk)
        
        logger.info(f"{approach} generated {len(enriched_chunks)} chunks")
        return enriched_chunks
    
    def _extract_text_from_chunk(self, chunk) -> str:
        """Extract text content from various chunk formats."""
        if isinstance(chunk, str):
            return chunk
        elif hasattr(chunk, 'text'):
            return chunk.text
        elif isinstance(chunk, dict):
            if 'content' in chunk:
                return chunk['content']
            elif 'text' in chunk:
                return chunk['text']
        return str(chunk)
    
    def _analyze_chunk_for_economy(self, text: str, token_count: int) -> Dict[str, Any]:
        """Analyze a chunk for context economy factors."""
        metadata = {
            # Basic structure analysis
            "has_complete_functions": self._has_complete_functions(text),
            "has_complete_classes": self._has_complete_classes(text),
            "has_imports": self._has_imports(text),
            "has_exports": self._has_exports(text),
            "has_comments": self._has_comments(text),
            "has_docstrings": self._has_docstrings(text),
            
            # Information density metrics
            "information_density": self._calculate_information_density(text),
            "semantic_keywords_count": self._count_semantic_keywords(text),
            "unique_identifiers_count": self._count_unique_identifiers(text),
            "api_surface_count": self._count_api_surface_elements(text),
            
            # Context efficiency metrics
            "cross_reference_count": self._count_cross_references(text),
            "dependency_indicators": self._count_dependency_indicators(text),
            "architectural_patterns": self._identify_architectural_patterns(text),
            
            # Redundancy indicators
            "repeated_patterns": self._identify_repeated_patterns(text),
            "boilerplate_ratio": self._calculate_boilerplate_ratio(text),
            "noise_ratio": self._calculate_noise_ratio(text),
            
            # Retrieval value
            "search_value_score": self._calculate_search_value(text),
            "context_completeness": self._calculate_context_completeness(text),
            "standalone_value": self._calculate_standalone_value(text)
        }
        
        return metadata
    
    def _has_complete_functions(self, text: str) -> bool:
        """Check if chunk contains complete function definitions."""
        function_patterns = [
            r'\bdef\s+\w+\s*\([^)]*\)\s*:.*?(?=\ndef|\nclass|\n\n|\Z)',
            r'\bfunction\s+\w+\s*\([^)]*\)\s*\{.*?\}',
            r'\b\w+\s*\([^)]*\)\s*\{.*?\}',  # General function pattern
        ]
        
        for pattern in function_patterns:
            if re.search(pattern, text, re.DOTALL):
                return True
        return False
    
    def _has_complete_classes(self, text: str) -> bool:
        """Check if chunk contains complete class definitions."""
        class_patterns = [
            r'\bclass\s+\w+.*?:.*?(?=\nclass|\ndef|\n\n|\Z)',
            r'\bclass\s+\w+.*?\{.*?\}',
            r'\binterface\s+\w+.*?\{.*?\}',
        ]
        
        for pattern in class_patterns:
            if re.search(pattern, text, re.DOTALL):
                return True
        return False
    
    def _has_imports(self, text: str) -> bool:
        """Check for import statements."""
        import_patterns = [
            r'\bimport\s+\w+',
            r'\bfrom\s+\w+\s+import',
            r'\brequire\s*\([\'"].*?[\'"]\)',
            r'#include\s*<.*?>',
            r'\busing\s+\w+'
        ]
        
        return any(re.search(pattern, text) for pattern in import_patterns)
    
    def _has_exports(self, text: str) -> bool:
        """Check for export statements."""
        export_patterns = [
            r'\bexport\s+',
            r'\bmodule\.exports\s*=',
            r'\bexports\.',
            r'\bpublic\s+(class|interface|enum)',
            r'__all__\s*='
        ]
        
        return any(re.search(pattern, text) for pattern in export_patterns)
    
    def _has_comments(self, text: str) -> bool:
        """Check for comments."""
        comment_patterns = [
            r'//.*$',
            r'/\*.*?\*/',
            r'#.*$',
            r'<!--.*?-->',
        ]
        
        return any(re.search(pattern, text, re.MULTILINE | re.DOTALL) for pattern in comment_patterns)
    
    def _has_docstrings(self, text: str) -> bool:
        """Check for docstrings/documentation."""
        docstring_patterns = [
            r'""".*?"""',
            r"'''.*?'''",
            r'/\*\*.*?\*/',
            r'@doc\s*""".*?"""',
        ]
        
        return any(re.search(pattern, text, re.DOTALL) for pattern in docstring_patterns)
    
    def _calculate_information_density(self, text: str) -> float:
        """Calculate information density (meaningful content vs total content)."""
        lines = text.split('\n')
        meaningful_lines = 0
        total_lines = len(lines)
        
        for line in lines:
            stripped = line.strip()
            # Skip empty lines, pure whitespace, and simple braces
            if stripped and stripped not in ['{', '}', '(', ')', '[', ']']:
                # Skip pure comment lines
                if not (stripped.startswith('//') or stripped.startswith('#') or 
                       stripped.startswith('/*') or stripped.startswith('*')):
                    meaningful_lines += 1
        
        return meaningful_lines / max(total_lines, 1)
    
    def _count_semantic_keywords(self, text: str) -> int:
        """Count semantically meaningful keywords."""
        semantic_keywords = [
            # Control flow
            'if', 'else', 'elif', 'for', 'while', 'try', 'catch', 'except', 'finally',
            # Function/class keywords
            'def', 'function', 'class', 'interface', 'enum', 'struct',
            # Visibility modifiers
            'public', 'private', 'protected', 'static', 'final', 'abstract',
            # Data types
            'int', 'string', 'bool', 'float', 'double', 'char', 'byte',
            # Modern patterns
            'async', 'await', 'promise', 'observable', 'stream',
        ]
        
        text_lower = text.lower()
        count = 0
        
        for keyword in semantic_keywords:
            # Use word boundaries to avoid partial matches
            if re.search(rf'\b{re.escape(keyword)}\b', text_lower):
                count += 1
        
        return count
    
    def _count_unique_identifiers(self, text: str) -> int:
        """Count unique identifiers (variable names, function names, etc.)."""
        # Simple heuristic: find words that start with letter and contain alphanumeric/underscore
        identifier_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        identifiers = set(re.findall(identifier_pattern, text))
        
        # Filter out common keywords and short identifiers
        common_keywords = {
            'if', 'else', 'for', 'while', 'try', 'catch', 'return', 'true', 'false',
            'null', 'undefined', 'var', 'let', 'const', 'function', 'class', 'def',
            'import', 'from', 'as', 'and', 'or', 'not', 'in', 'is', 'with'
        }
        
        meaningful_identifiers = {
            ident for ident in identifiers 
            if len(ident) > 2 and ident.lower() not in common_keywords
        }
        
        return len(meaningful_identifiers)
    
    def _count_api_surface_elements(self, text: str) -> int:
        """Count API surface elements (public methods, exported functions, etc.)."""
        api_patterns = [
            r'\bpublic\s+\w+\s+\w+\s*\(',  # Public methods (Java/C#)
            r'\bexport\s+(function|class|const|let|var)\s+\w+',  # Exported elements (JS/TS)
            r'^\s*def\s+\w+\s*\(',  # Python functions (potentially public)
            r'^\s*class\s+\w+',  # Class definitions
            r'@\w+\s*\n\s*def',  # Decorated functions (often API endpoints)
        ]
        
        count = 0
        for pattern in api_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            count += len(matches)
        
        return count
    
    def _count_cross_references(self, text: str) -> int:
        """Count references to other modules/files."""
        cross_ref_patterns = [
            r'\bimport\s+\w+',
            r'\bfrom\s+\w+\s+import',
            r'\brequire\s*\([\'"].*?[\'"]\)',
            r'#include\s*[<"].*?[>"]',
            r'\busing\s+\w+',
            r'\w+\.\w+\(',  # Method calls on external objects
        ]
        
        count = 0
        for pattern in cross_ref_patterns:
            matches = re.findall(pattern, text)
            count += len(matches)
        
        return count
    
    def _count_dependency_indicators(self, text: str) -> int:
        """Count indicators of dependencies on external systems."""
        dependency_patterns = [
            r'\b(database|db|sql|redis|mongodb)\b',
            r'\b(http|api|rest|graphql|grpc)\b',
            r'\b(auth|authentication|authorization)\b',
            r'\b(cache|session|cookie)\b',
            r'\b(queue|event|message|kafka|rabbitmq)\b',
            r'\b(config|environment|env)\b',
        ]
        
        text_lower = text.lower()
        count = 0
        
        for pattern in dependency_patterns:
            if re.search(pattern, text_lower):
                count += 1
        
        return count
    
    def _identify_architectural_patterns(self, text: str) -> List[str]:
        """Identify architectural patterns in the text."""
        patterns = {
            'mvc': r'\b(model|view|controller)\b',
            'repository': r'\brepository\b',
            'service': r'\bservice\b',
            'factory': r'\bfactory\b',
            'singleton': r'\bsingleton\b',
            'observer': r'\b(observer|subscribe|notify)\b',
            'strategy': r'\bstrategy\b',
            'decorator': r'\bdecorator\b',
            'adapter': r'\badapter\b',
            'facade': r'\bfacade\b',
            'proxy': r'\bproxy\b',
            'middleware': r'\bmiddleware\b',
        }
        
        found_patterns = []
        text_lower = text.lower()
        
        for pattern_name, regex in patterns.items():
            if re.search(regex, text_lower):
                found_patterns.append(pattern_name)
        
        return found_patterns
    
    def _identify_repeated_patterns(self, text: str) -> Dict[str, int]:
        """Identify repeated patterns that might indicate redundancy."""
        # Find repeated lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        line_counts = Counter(lines)
        repeated_lines = {line: count for line, count in line_counts.items() if count > 1}
        
        # Find repeated import statements
        import_pattern = r'\b(import|from|require|using|#include)\s+[^\n]+'
        imports = re.findall(import_pattern, text)
        import_counts = Counter(imports)
        repeated_imports = {imp: count for imp, count in import_counts.items() if count > 1}
        
        return {
            "repeated_lines": len(repeated_lines),
            "repeated_imports": len(repeated_imports),
            "total_repetitions": sum(repeated_lines.values()) + sum(repeated_imports.values())
        }
    
    def _calculate_boilerplate_ratio(self, text: str) -> float:
        """Calculate ratio of boilerplate code to meaningful code."""
        lines = text.split('\n')
        boilerplate_lines = 0
        total_lines = len([line for line in lines if line.strip()])
        
        boilerplate_patterns = [
            r'^\s*{\s*$',  # Opening braces
            r'^\s*}\s*$',  # Closing braces
            r'^\s*\(\s*$',  # Opening parentheses
            r'^\s*\)\s*$',  # Closing parentheses
            r'^\s*/\*.*\*/\s*$',  # Single-line comments
            r'^\s*//.*$',  # Line comments
            r'^\s*#.*$',  # Hash comments
            r'^\s*import\s+',  # Import statements
            r'^\s*from\s+.*import',  # From imports
            r'^\s*using\s+',  # Using statements
        ]
        
        for line in lines:
            if line.strip():  # Only count non-empty lines
                for pattern in boilerplate_patterns:
                    if re.match(pattern, line):
                        boilerplate_lines += 1
                        break
        
        return boilerplate_lines / max(total_lines, 1)
    
    def _calculate_noise_ratio(self, text: str) -> float:
        """Calculate ratio of noise (whitespace, empty lines) to content."""
        total_chars = len(text)
        meaningful_chars = len(re.sub(r'\s+', ' ', text).strip())
        
        if total_chars == 0:
            return 1.0
        
        return 1.0 - (meaningful_chars / total_chars)
    
    def _calculate_search_value(self, text: str) -> float:
        """Calculate how valuable this chunk would be for search/retrieval."""
        score = 0.0
        
        # Higher value for complete constructs
        if self._has_complete_functions(text):
            score += 0.3
        if self._has_complete_classes(text):
            score += 0.3
        
        # Value for documentation
        if self._has_docstrings(text):
            score += 0.2
        if self._has_comments(text):
            score += 0.1
        
        # Value for API surface
        api_count = self._count_api_surface_elements(text)
        score += min(api_count * 0.1, 0.3)
        
        # Value for architectural patterns
        arch_patterns = self._identify_architectural_patterns(text)
        score += min(len(arch_patterns) * 0.05, 0.2)
        
        # Penalty for high boilerplate
        boilerplate_ratio = self._calculate_boilerplate_ratio(text)
        score *= (1.0 - boilerplate_ratio * 0.5)
        
        return min(score, 1.0)
    
    def _calculate_context_completeness(self, text: str) -> float:
        """Calculate how complete the context is for understanding."""
        score = 0.0
        
        # Complete functions provide better context
        if self._has_complete_functions(text):
            score += 0.4
        
        # Imports provide dependency context
        if self._has_imports(text):
            score += 0.2
        
        # Documentation provides understanding context
        if self._has_docstrings(text):
            score += 0.3
        
        # Cross-references provide broader context
        cross_refs = self._count_cross_references(text)
        score += min(cross_refs * 0.02, 0.1)
        
        return min(score, 1.0)
    
    def _calculate_standalone_value(self, text: str) -> float:
        """Calculate how valuable this chunk is as a standalone piece."""
        # High standalone value means the chunk is useful without additional context
        score = 0.0
        
        # Complete constructs have high standalone value
        if self._has_complete_functions(text):
            score += 0.4
        if self._has_complete_classes(text):
            score += 0.4
        
        # Documentation adds standalone value
        if self._has_docstrings(text):
            score += 0.2
        
        # Penalty for incomplete constructs (high cross-reference dependency)
        cross_refs = self._count_cross_references(text)
        dependency_penalty = min(cross_refs * 0.05, 0.3)
        score = max(0.0, score - dependency_penalty)
        
        return min(score, 1.0)
    
    def _analyze_chunking_approach(self, chunks: List[Dict], approach: str) -> Dict[str, Any]:
        """Analyze a single chunking approach for context economy."""
        if not chunks:
            return {"approach": approach, "error": "No chunks to analyze"}
        
        # Token usage statistics
        total_tokens = sum(chunk["token_count"] for chunk in chunks)
        avg_tokens = statistics.mean(chunk["token_count"] for chunk in chunks)
        token_variance = statistics.variance(chunk["token_count"] for chunk in chunks)
        
        # Information density metrics
        info_densities = [chunk["information_density"] for chunk in chunks]
        avg_info_density = statistics.mean(info_densities)
        
        # Context efficiency metrics
        search_values = [chunk["search_value_score"] for chunk in chunks]
        avg_search_value = statistics.mean(search_values)
        
        context_completeness = [chunk["context_completeness"] for chunk in chunks]
        avg_context_completeness = statistics.mean(context_completeness)
        
        standalone_values = [chunk["standalone_value"] for chunk in chunks]
        avg_standalone_value = statistics.mean(standalone_values)
        
        # Redundancy metrics
        boilerplate_ratios = [chunk["boilerplate_ratio"] for chunk in chunks]
        avg_boilerplate_ratio = statistics.mean(boilerplate_ratios)
        
        # Structural metrics
        complete_functions = sum(1 for chunk in chunks if chunk["has_complete_functions"])
        complete_classes = sum(1 for chunk in chunks if chunk["has_complete_classes"])
        
        # API surface metrics
        total_api_elements = sum(chunk["api_surface_count"] for chunk in chunks)
        
        # Cross-reference metrics
        total_cross_refs = sum(chunk["cross_reference_count"] for chunk in chunks)
        avg_cross_refs = statistics.mean(chunk["cross_reference_count"] for chunk in chunks)
        
        # Architectural pattern coverage
        all_patterns = []
        for chunk in chunks:
            all_patterns.extend(chunk["architectural_patterns"])
        unique_patterns = set(all_patterns)
        
        return {
            "approach": approach,
            "chunk_count": len(chunks),
            
            # Token efficiency
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": avg_tokens,
            "token_variance": token_variance,
            "tokens_per_complete_function": total_tokens / max(complete_functions, 1),
            "tokens_per_api_element": total_tokens / max(total_api_elements, 1),
            
            # Information density
            "avg_information_density": avg_info_density,
            "avg_search_value": avg_search_value,
            "avg_context_completeness": avg_context_completeness,
            "avg_standalone_value": avg_standalone_value,
            
            # Redundancy and efficiency
            "avg_boilerplate_ratio": avg_boilerplate_ratio,
            "efficiency_score": avg_info_density * (1.0 - avg_boilerplate_ratio),
            
            # Structural completeness
            "complete_functions_ratio": complete_functions / len(chunks),
            "complete_classes_ratio": complete_classes / len(chunks),
            
            # Cross-reference and context
            "avg_cross_references": avg_cross_refs,
            "cross_reference_density": total_cross_refs / total_tokens * 1000,  # per 1000 tokens
            
            # Architectural coverage
            "unique_architectural_patterns": len(unique_patterns),
            "architectural_pattern_density": len(all_patterns) / len(chunks),
            
            # Overall economy score
            "context_economy_score": self._calculate_context_economy_score(
                avg_info_density, avg_search_value, avg_context_completeness,
                avg_boilerplate_ratio, complete_functions / len(chunks)
            )
        }
    
    def _calculate_context_economy_score(self, info_density: float, search_value: float,
                                       context_completeness: float, boilerplate_ratio: float,
                                       completeness_ratio: float) -> float:
        """Calculate overall context economy score (0-1, higher is better)."""
        # Weighted combination of factors
        score = (
            info_density * 0.25 +
            search_value * 0.25 +
            context_completeness * 0.25 +
            completeness_ratio * 0.15 +
            (1.0 - boilerplate_ratio) * 0.10
        )
        
        return min(max(score, 0.0), 1.0)
    
    def _compare_approaches(self, ast_analysis: Dict, naive_analysis: Dict) -> Dict[str, Any]:
        """Compare AST vs Naive approaches on context economy metrics."""
        comparison = {}
        
        # Token efficiency comparison
        comparison["token_efficiency"] = {
            "tokens_per_function_ratio": (
                ast_analysis.get("tokens_per_complete_function", float('inf')) /
                naive_analysis.get("tokens_per_complete_function", 1)
            ),
            "tokens_per_api_ratio": (
                ast_analysis.get("tokens_per_api_element", float('inf')) /
                naive_analysis.get("tokens_per_api_element", 1)
            ),
            "avg_tokens_improvement": (
                naive_analysis.get("avg_tokens_per_chunk", 0) -
                ast_analysis.get("avg_tokens_per_chunk", 0)
            )
        }
        
        # Information density comparison
        comparison["information_quality"] = {
            "density_improvement": (
                ast_analysis.get("avg_information_density", 0) -
                naive_analysis.get("avg_information_density", 0)
            ),
            "search_value_improvement": (
                ast_analysis.get("avg_search_value", 0) -
                naive_analysis.get("avg_search_value", 0)
            ),
            "completeness_improvement": (
                ast_analysis.get("avg_context_completeness", 0) -
                naive_analysis.get("avg_context_completeness", 0)
            ),
            "standalone_improvement": (
                ast_analysis.get("avg_standalone_value", 0) -
                naive_analysis.get("avg_standalone_value", 0)
            )
        }
        
        # Structural comparison
        comparison["structural_quality"] = {
            "function_completeness_improvement": (
                ast_analysis.get("complete_functions_ratio", 0) -
                naive_analysis.get("complete_functions_ratio", 0)
            ),
            "class_completeness_improvement": (
                ast_analysis.get("complete_classes_ratio", 0) -
                naive_analysis.get("complete_classes_ratio", 0)
            ),
            "boilerplate_reduction": (
                naive_analysis.get("avg_boilerplate_ratio", 0) -
                ast_analysis.get("avg_boilerplate_ratio", 0)
            )
        }
        
        # Overall economy comparison
        comparison["overall_economy"] = {
            "economy_score_improvement": (
                ast_analysis.get("context_economy_score", 0) -
                naive_analysis.get("context_economy_score", 0)
            ),
            "efficiency_improvement": (
                ast_analysis.get("efficiency_score", 0) -
                naive_analysis.get("efficiency_score", 0)
            ),
            "cross_reference_density_improvement": (
                ast_analysis.get("cross_reference_density", 0) -
                naive_analysis.get("cross_reference_density", 0)
            )
        }
        
        return comparison
    
    def _analyze_token_efficiency(self, ast_chunks: List[Dict], naive_chunks: List[Dict]) -> Dict[str, Any]:
        """Analyze token efficiency in detail."""
        return {
            "ast_token_distribution": self._calculate_token_distribution(ast_chunks),
            "naive_token_distribution": self._calculate_token_distribution(naive_chunks),
            "efficiency_metrics": self._calculate_efficiency_metrics(ast_chunks, naive_chunks)
        }
    
    def _calculate_token_distribution(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Calculate token distribution statistics."""
        if not chunks:
            return {}
        
        token_counts = [chunk["token_count"] for chunk in chunks]
        
        return {
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "median_tokens": statistics.median(token_counts),
            "q25_tokens": statistics.quantiles(token_counts, n=4)[0],
            "q75_tokens": statistics.quantiles(token_counts, n=4)[2],
            "std_deviation": statistics.stdev(token_counts) if len(token_counts) > 1 else 0
        }
    
    def _calculate_efficiency_metrics(self, ast_chunks: List[Dict], naive_chunks: List[Dict]) -> Dict[str, Any]:
        """Calculate detailed efficiency metrics."""
        ast_tokens = sum(chunk["token_count"] for chunk in ast_chunks)
        naive_tokens = sum(chunk["token_count"] for chunk in naive_chunks)
        
        ast_value = sum(chunk["search_value_score"] for chunk in ast_chunks)
        naive_value = sum(chunk["search_value_score"] for chunk in naive_chunks)
        
        return {
            "total_token_ratio": ast_tokens / max(naive_tokens, 1),
            "value_per_token_ratio": (ast_value / max(ast_tokens, 1)) / max(naive_value / max(naive_tokens, 1), 0.001),
            "efficiency_improvement": (ast_value / max(ast_tokens, 1)) - (naive_value / max(naive_tokens, 1))
        }
    
    def _analyze_redundancy(self, ast_chunks: List[Dict], naive_chunks: List[Dict]) -> Dict[str, Any]:
        """Analyze redundancy between chunks."""
        return {
            "ast_redundancy": self._calculate_redundancy_metrics(ast_chunks),
            "naive_redundancy": self._calculate_redundancy_metrics(naive_chunks),
            "redundancy_comparison": self._compare_redundancy(ast_chunks, naive_chunks)
        }
    
    def _calculate_redundancy_metrics(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Calculate redundancy metrics for a set of chunks."""
        if not chunks:
            return {}
        
        # Text similarity analysis
        chunk_texts = [chunk["text"] for chunk in chunks]
        similarity_scores = []
        
        for i in range(len(chunk_texts)):
            for j in range(i + 1, len(chunk_texts)):
                similarity = self._calculate_text_similarity(chunk_texts[i], chunk_texts[j])
                similarity_scores.append(similarity)
        
        # Repeated pattern analysis
        all_patterns = []
        for chunk in chunks:
            patterns_data = chunk.get("repeated_patterns", {})
            if isinstance(patterns_data, dict):
                total_reps = patterns_data.get("total_repetitions", 0)
                if isinstance(total_reps, (list, tuple)):
                    all_patterns.extend(total_reps)
                else:
                    all_patterns.append(total_reps)
            else:
                all_patterns.append(0)
        
        return {
            "avg_similarity": statistics.mean(similarity_scores) if similarity_scores else 0,
            "max_similarity": max(similarity_scores) if similarity_scores else 0,
            "high_similarity_pairs": sum(1 for s in similarity_scores if s > 0.7),
            "total_repeated_patterns": sum(all_patterns) if all_patterns else 0,
            "avg_boilerplate_ratio": statistics.mean(chunk["boilerplate_ratio"] for chunk in chunks)
        }
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text chunks using simple word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _compare_redundancy(self, ast_chunks: List[Dict], naive_chunks: List[Dict]) -> Dict[str, Any]:
        """Compare redundancy between AST and Naive approaches."""
        ast_redundancy = self._calculate_redundancy_metrics(ast_chunks)
        naive_redundancy = self._calculate_redundancy_metrics(naive_chunks)
        
        return {
            "similarity_reduction": (
                naive_redundancy.get("avg_similarity", 0) -
                ast_redundancy.get("avg_similarity", 0)
            ),
            "boilerplate_reduction": (
                naive_redundancy.get("avg_boilerplate_ratio", 0) -
                ast_redundancy.get("avg_boilerplate_ratio", 0)
            ),
            "high_similarity_reduction": (
                naive_redundancy.get("high_similarity_pairs", 0) -
                ast_redundancy.get("high_similarity_pairs", 0)
            )
        }
    
    def _analyze_retrieval_efficiency(self, ast_chunks: List[Dict], naive_chunks: List[Dict]) -> Dict[str, Any]:
        """Analyze efficiency for retrieval scenarios."""
        return {
            "ast_retrieval_metrics": self._calculate_retrieval_metrics(ast_chunks),
            "naive_retrieval_metrics": self._calculate_retrieval_metrics(naive_chunks),
            "retrieval_comparison": self._compare_retrieval_efficiency(ast_chunks, naive_chunks)
        }
    
    def _calculate_retrieval_metrics(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics relevant for retrieval scenarios."""
        if not chunks:
            return {}
        
        # Calculate retrieval value metrics
        search_values = [chunk["search_value_score"] for chunk in chunks]
        standalone_values = [chunk["standalone_value"] for chunk in chunks]
        completeness_scores = [chunk["context_completeness"] for chunk in chunks]
        
        # High-value chunk analysis
        high_value_chunks = [chunk for chunk in chunks if chunk["search_value_score"] > 0.7]
        
        return {
            "avg_search_value": statistics.mean(search_values),
            "avg_standalone_value": statistics.mean(standalone_values),
            "avg_completeness": statistics.mean(completeness_scores),
            "high_value_chunk_ratio": len(high_value_chunks) / len(chunks),
            "retrieval_efficiency_score": statistics.mean([
                chunk["search_value_score"] / max(chunk["token_count"] / 100, 1)  # Value per 100 tokens
                for chunk in chunks
            ])
        }
    
    def _compare_retrieval_efficiency(self, ast_chunks: List[Dict], naive_chunks: List[Dict]) -> Dict[str, Any]:
        """Compare retrieval efficiency between approaches."""
        ast_metrics = self._calculate_retrieval_metrics(ast_chunks)
        naive_metrics = self._calculate_retrieval_metrics(naive_chunks)
        
        return {
            "search_value_improvement": (
                ast_metrics.get("avg_search_value", 0) -
                naive_metrics.get("avg_search_value", 0)
            ),
            "standalone_improvement": (
                ast_metrics.get("avg_standalone_value", 0) -
                naive_metrics.get("avg_standalone_value", 0)
            ),
            "completeness_improvement": (
                ast_metrics.get("avg_completeness", 0) -
                naive_metrics.get("avg_completeness", 0)
            ),
            "high_value_ratio_improvement": (
                ast_metrics.get("high_value_chunk_ratio", 0) -
                naive_metrics.get("high_value_chunk_ratio", 0)
            ),
            "efficiency_score_improvement": (
                ast_metrics.get("retrieval_efficiency_score", 0) -
                naive_metrics.get("retrieval_efficiency_score", 0)
            )
        }


def main():
    """Main function for context economy analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Context economy analysis for chunking approaches")
    parser.add_argument("--data-dir", type=str, default="data/code_samples",
                       help="Directory containing code files")
    parser.add_argument("--output", type=str, default="context_economy_results.json",
                       help="Output file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Load documents
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            logger.error(f"Data directory {data_dir} does not exist")
            return 1
        
        reader = SimpleDirectoryReader(
            str(data_dir),
            recursive=True,
            required_exts=[".py", ".js", ".ts", ".tsx", ".java", ".go", ".cpp", ".c", ".cs"],
            filename_as_id=True
        )
        documents = reader.load_data()
        
        if not documents:
            logger.error("No documents found to analyze")
            return 1
        
        logger.info(f"Loaded {len(documents)} documents for analysis")
        
        # Run analysis
        analyzer = ContextEconomyAnalyzer(documents)
        results = analyzer.analyze_context_economy()
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("CONTEXT ECONOMY ANALYSIS SUMMARY")
        print("="*80)
        
        metadata = results["analysis_metadata"]
        comparison = results["comparison"]
        
        print(f"Documents Analyzed: {metadata['document_count']}")
        print(f"Total Content Size: {metadata['total_content_size']:,} characters")
        
        # Token efficiency
        token_eff = comparison.get("token_efficiency", {})
        print(f"\n💰 Token Efficiency:")
        if "avg_tokens_improvement" in token_eff:
            print(f"  Avg Tokens per Chunk: {token_eff['avg_tokens_improvement']:+.1f}")
        if "tokens_per_function_ratio" in token_eff:
            print(f"  Tokens per Function:  {token_eff['tokens_per_function_ratio']:.2f}x")
        
        # Information quality
        info_qual = comparison.get("information_quality", {})
        print(f"\n📊 Information Quality:")
        if "density_improvement" in info_qual:
            print(f"  Density:       {info_qual['density_improvement']:+.3f}")
        if "search_value_improvement" in info_qual:
            print(f"  Search Value:  {info_qual['search_value_improvement']:+.3f}")
        if "completeness_improvement" in info_qual:
            print(f"  Completeness:  {info_qual['completeness_improvement']:+.3f}")
        
        # Overall economy
        overall = comparison.get("overall_economy", {})
        print(f"\n🎯 Overall Economy:")
        if "economy_score_improvement" in overall:
            print(f"  Economy Score: {overall['economy_score_improvement']:+.3f}")
        if "efficiency_improvement" in overall:
            print(f"  Efficiency:    {overall['efficiency_improvement']:+.3f}")
        
        print(f"\nDetailed results saved to: {args.output}")
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())