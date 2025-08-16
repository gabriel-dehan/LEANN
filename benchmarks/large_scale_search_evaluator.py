#!/usr/bin/env python3
"""
Large-Scale Search Quality Evaluator

This module evaluates search quality improvements of AST vs Naive chunking
specifically on large codebases where architectural patterns and cross-file
dependencies become more important.
"""

import json
import logging
import sys
import time
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import re

# Add apps directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

from chunking import create_text_chunks
from llama_index.core import SimpleDirectoryReader, Document

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result with relevance score."""
    chunk_id: int
    content: str
    relevance_score: float
    file_path: str
    chunk_metadata: Dict[str, Any]

class EnterpriseSearchEvaluator:
    """Evaluates search quality on enterprise-scale codebases."""
    
    def __init__(self, documents: List[Document], queries_file: str):
        self.documents = documents
        self.queries_file = Path(queries_file)
        self.queries = self._load_complex_queries()
        
        # Cache for expensive operations
        self._ast_chunks_cache = None
        self._naive_chunks_cache = None
        
    def _load_complex_queries(self) -> List[Dict[str, Any]]:
        """Load complex enterprise queries."""
        with open(self.queries_file, 'r') as f:
            data = json.load(f)
            return data.get('queries', [])
    
    def evaluate_large_scale_search_quality(self) -> Dict[str, Any]:
        """Evaluate search quality on large codebase with complex queries."""
        logger.info(f"Starting large-scale search evaluation with {len(self.documents)} documents")
        
        results = {
            "evaluation_metadata": {
                "document_count": len(self.documents),
                "query_count": len(self.queries),
                "total_content_size": sum(len(doc.text) for doc in self.documents),
                "evaluation_timestamp": time.time()
            },
            "ast_results": {},
            "naive_results": {},
            "comparison": {},
            "architectural_analysis": {},
            "scale_effects": {}
        }
        
        # Generate chunks for both approaches
        logger.info("Generating AST chunks...")
        ast_chunks = self._get_or_generate_ast_chunks()
        
        logger.info("Generating Naive chunks...")
        naive_chunks = self._get_or_generate_naive_chunks()
        
        # Evaluate both approaches
        results["ast_results"] = self._evaluate_search_approach(ast_chunks, "AST")
        results["naive_results"] = self._evaluate_search_approach(naive_chunks, "Naive")
        
        # Advanced comparisons
        results["comparison"] = self._calculate_advanced_comparison(
            results["ast_results"], 
            results["naive_results"]
        )
        
        results["architectural_analysis"] = self._analyze_architectural_patterns(
            ast_chunks, naive_chunks
        )
        
        results["scale_effects"] = self._analyze_scale_effects(
            ast_chunks, naive_chunks
        )
        
        return results
    
    def _get_or_generate_ast_chunks(self) -> List[Dict[str, Any]]:
        """Get AST chunks with caching."""
        if self._ast_chunks_cache is None:
            start_time = time.time()
            raw_chunks = create_text_chunks(
                self.documents,
                chunk_size=512,
                chunk_overlap=128,
                use_ast_chunking=True,
                ast_chunk_size=1024,
                ast_chunk_overlap=128
            )
            processing_time = time.time() - start_time
            
            self._ast_chunks_cache = self._enrich_chunks_with_metadata(
                raw_chunks, "AST", processing_time
            )
        
        return self._ast_chunks_cache
    
    def _get_or_generate_naive_chunks(self) -> List[Dict[str, Any]]:
        """Get Naive chunks with caching."""
        if self._naive_chunks_cache is None:
            start_time = time.time()
            raw_chunks = create_text_chunks(
                self.documents,
                chunk_size=512,
                chunk_overlap=128,
                use_ast_chunking=False
            )
            processing_time = time.time() - start_time
            
            self._naive_chunks_cache = self._enrich_chunks_with_metadata(
                raw_chunks, "Naive", processing_time
            )
        
        return self._naive_chunks_cache
    
    def _extract_text_from_chunk_robust(self, chunk) -> str:
        """Extract text content with robust parsing for stringified dictionaries."""
        import re
        import ast
        
        # Handle different chunk formats that might be returned
        if isinstance(chunk, str):
            # If it's a string that looks like a dictionary representation
            if chunk.startswith("{'content':") or chunk.startswith('{"content":'):
                try:
                    # Try to extract content from string representation
                    parsed = ast.literal_eval(chunk)
                    if isinstance(parsed, dict) and 'content' in parsed:
                        return parsed['content']
                except:
                    pass
                
                # Fallback: regex extraction for stringified dictionaries
                # Handle escaped quotes and multiline content
                content_match = re.search(r"'content':\s*'(.*?)',\s*'metadata'", chunk, re.DOTALL)
                if content_match:
                    # Unescape newlines and other escaped characters
                    content = content_match.group(1)
                    content = content.replace('\\n', '\n').replace('\\t', '\t').replace("\\'", "'")
                    return content
                    
                content_match = re.search(r'"content":\s*"(.*?)",\s*"metadata"', chunk, re.DOTALL)
                if content_match:
                    content = content_match.group(1)
                    content = content.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
                    return content
            
            # If it's a regular string, return as-is
            return chunk
            
        elif isinstance(chunk, dict):
            # If it's a dictionary, extract content
            if 'content' in chunk:
                return chunk['content']
            elif 'text' in chunk:
                return chunk['text']
            else:
                return str(chunk)
                
        elif hasattr(chunk, 'text'):
            # If it's an object with a text attribute
            return chunk.text
            
        elif hasattr(chunk, 'content'):
            # If it's an object with a content attribute
            return chunk.content
            
        else:
            # Fallback: convert to string
            return str(chunk)
    
    def _enrich_chunks_with_metadata(self, raw_chunks: List, approach: str, processing_time: float) -> List[Dict[str, Any]]:
        """Enrich chunks with detailed metadata for enterprise analysis."""
        enriched_chunks = []
        
        for i, raw_chunk in enumerate(raw_chunks):
            # Extract text content with robust parsing
            text_content = self._extract_text_from_chunk_robust(raw_chunk)
            
            # Skip very short chunks
            if len(text_content.strip()) < 50:
                continue
            
            # Analyze chunk content
            metadata = self._analyze_chunk_content(text_content, i)
            
            enriched_chunk = {
                "id": i,
                "text": text_content,
                "approach": approach,
                "size": len(text_content),
                "lines": len(text_content.split('\n')),
                "processing_time_contribution": processing_time / len(raw_chunks),
                **metadata
            }
            
            enriched_chunks.append(enriched_chunk)
        
        logger.info(f"{approach} chunking: {len(enriched_chunks)} enriched chunks")
        return enriched_chunks
    
    def _analyze_chunk_content(self, text: str, chunk_id: int) -> Dict[str, Any]:
        """Analyze chunk content for enterprise-relevant patterns."""
        metadata = {
            # File and language detection
            "inferred_language": self._infer_language(text),
            "inferred_file_type": self._infer_file_type(text),
            
            # Code structure analysis
            "has_functions": self._contains_functions(text),
            "has_classes": self._contains_classes(text),
            "has_interfaces": self._contains_interfaces(text),
            "has_imports": self._contains_imports(text),
            "has_exports": self._contains_exports(text),
            
            # Enterprise patterns
            "has_authentication": self._contains_auth_patterns(text),
            "has_database_access": self._contains_db_patterns(text),
            "has_api_endpoints": self._contains_api_patterns(text),
            "has_error_handling": self._contains_error_handling(text),
            "has_configuration": self._contains_config_patterns(text),
            "has_testing": self._contains_test_patterns(text),
            "has_logging": self._contains_logging_patterns(text),
            "has_caching": self._contains_cache_patterns(text),
            "has_messaging": self._contains_messaging_patterns(text),
            "has_security": self._contains_security_patterns(text),
            
            # Architectural indicators
            "architectural_keywords": self._extract_architectural_keywords(text),
            "complexity_score": self._calculate_complexity_score(text),
            "completeness_score": self._calculate_completeness_score(text),
            "cross_reference_count": self._count_cross_references(text)
        }
        
        return metadata
    
    def _infer_language(self, text: str) -> str:
        """Infer programming language from text content."""
        language_indicators = {
            "python": ["def ", "import ", "from ", "class ", "self.", "__init__"],
            "javascript": ["function ", "const ", "let ", "var ", "=>", "require("],
            "typescript": ["interface ", "type ", "implements ", "extends ", ": string", ": number"],
            "java": ["public class", "private ", "protected ", "package ", "import java"],
            "go": ["func ", "package ", "import ", "type ", "var ", "const "],
            "csharp": ["public class", "private ", "using ", "namespace ", "void "],
            "cpp": ["#include", "class ", "namespace ", "std::", "void "],
            "rust": ["fn ", "impl ", "struct ", "enum ", "use ", "pub "]
        }
        
        text_lower = text.lower()
        scores = {}
        
        for lang, indicators in language_indicators.items():
            score = sum(1 for indicator in indicators if indicator.lower() in text_lower)
            if score > 0:
                scores[lang] = score
        
        return max(scores.keys(), key=lambda k: scores[k]) if scores else "unknown"
    
    def _infer_file_type(self, text: str) -> str:
        """Infer file type from content patterns."""
        if "test" in text.lower() and any(word in text.lower() for word in ["describe", "it(", "expect", "assert"]):
            return "test"
        elif any(word in text for word in ["@RestController", "@RequestMapping", "router.", "app.get"]):
            return "controller"
        elif any(word in text.lower() for word in ["repository", "dao", "entity", "model"]):
            return "data_access"
        elif any(word in text.lower() for word in ["service", "business", "logic"]):
            return "service"
        elif any(word in text.lower() for word in ["config", "configuration", "settings"]):
            return "configuration"
        else:
            return "general"
    
    def _contains_functions(self, text: str) -> bool:
        """Check if text contains function definitions."""
        patterns = [
            r'\bdef\s+\w+\s*\(',  # Python
            r'\bfunction\s+\w+\s*\(',  # JavaScript
            r'\b\w+\s*\([^)]*\)\s*\{',  # Java/C#/Go
            r'\bfn\s+\w+\s*\(',  # Rust
        ]
        return any(re.search(pattern, text) for pattern in patterns)
    
    def _contains_classes(self, text: str) -> bool:
        """Check if text contains class definitions."""
        patterns = [
            r'\bclass\s+\w+',
            r'\bstruct\s+\w+',
            r'\binterface\s+\w+',
            r'\benum\s+\w+'
        ]
        return any(re.search(pattern, text) for pattern in patterns)
    
    def _contains_interfaces(self, text: str) -> bool:
        """Check if text contains interface definitions."""
        return bool(re.search(r'\binterface\s+\w+', text))
    
    def _contains_imports(self, text: str) -> bool:
        """Check if text contains import statements."""
        patterns = [
            r'\bimport\s+',
            r'\bfrom\s+\w+\s+import',
            r'\busing\s+',
            r'\brequire\s*\(',
            r'#include\s*<'
        ]
        return any(re.search(pattern, text) for pattern in patterns)
    
    def _contains_exports(self, text: str) -> bool:
        """Check if text contains export statements."""
        patterns = [
            r'\bexport\s+',
            r'\bmodule\.exports\s*=',
            r'\bexports\.',
            r'\bpublic\s+(class|interface|enum)'
        ]
        return any(re.search(pattern, text) for pattern in patterns)
    
    def _contains_auth_patterns(self, text: str) -> bool:
        """Check for authentication/authorization patterns."""
        keywords = ["auth", "login", "password", "token", "jwt", "oauth", "session", "cookie", "passport"]
        return any(keyword in text.lower() for keyword in keywords)
    
    def _contains_db_patterns(self, text: str) -> bool:
        """Check for database access patterns."""
        keywords = ["database", "db", "sql", "query", "repository", "entity", "model", "orm", "mongoose", "sequelize"]
        return any(keyword in text.lower() for keyword in keywords)
    
    def _contains_api_patterns(self, text: str) -> bool:
        """Check for API patterns."""
        keywords = ["@RestController", "@RequestMapping", "router", "app.get", "app.post", "endpoint", "api"]
        return any(keyword in text.lower() for keyword in keywords)
    
    def _contains_error_handling(self, text: str) -> bool:
        """Check for error handling patterns."""
        keywords = ["try", "catch", "except", "error", "exception", "throw", "raise"]
        return any(keyword in text.lower() for keyword in keywords)
    
    def _contains_config_patterns(self, text: str) -> bool:
        """Check for configuration patterns."""
        keywords = ["config", "configuration", "env", "environment", "settings", "properties"]
        return any(keyword in text.lower() for keyword in keywords)
    
    def _contains_test_patterns(self, text: str) -> bool:
        """Check for testing patterns."""
        keywords = ["test", "spec", "describe", "it(", "expect", "assert", "mock", "jest", "mocha"]
        return any(keyword in text.lower() for keyword in keywords)
    
    def _contains_logging_patterns(self, text: str) -> bool:
        """Check for logging patterns."""
        keywords = ["log", "logger", "console.", "winston", "bunyan", "log4j"]
        return any(keyword in text.lower() for keyword in keywords)
    
    def _contains_cache_patterns(self, text: str) -> bool:
        """Check for caching patterns."""
        keywords = ["cache", "redis", "memcached", "ttl", "expire", "invalidate"]
        return any(keyword in text.lower() for keyword in keywords)
    
    def _contains_messaging_patterns(self, text: str) -> bool:
        """Check for messaging patterns."""
        keywords = ["queue", "publish", "subscribe", "kafka", "rabbitmq", "event", "message"]
        return any(keyword in text.lower() for keyword in keywords)
    
    def _contains_security_patterns(self, text: str) -> bool:
        """Check for security patterns."""
        keywords = ["security", "encrypt", "decrypt", "hash", "salt", "csrf", "xss", "injection"]
        return any(keyword in text.lower() for keyword in keywords)
    
    def _extract_architectural_keywords(self, text: str) -> List[str]:
        """Extract architectural pattern keywords."""
        architectural_patterns = [
            "mvc", "mvp", "mvvm", "microservice", "monolith", "singleton", "factory", 
            "observer", "strategy", "decorator", "adapter", "facade", "proxy",
            "repository", "service", "controller", "middleware", "interceptor"
        ]
        
        found_patterns = []
        text_lower = text.lower()
        for pattern in architectural_patterns:
            if pattern in text_lower:
                found_patterns.append(pattern)
        
        return found_patterns
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate complexity score based on various factors."""
        lines = text.split('\n')
        
        # Base complexity factors
        line_count = len(lines)
        nesting_levels = max(len(line) - len(line.lstrip()) for line in lines if line.strip())
        
        # Pattern complexity
        complex_patterns = [
            r'\bif\s+.*\belse\b',  # Conditional complexity
            r'\bfor\s+.*\bin\b',  # Loop complexity
            r'\bwhile\s+',        # Loop complexity
            r'\btry\s+.*\bcatch\b',  # Exception handling
            r'\bswitch\s+.*\bcase\b'  # Switch complexity
        ]
        
        pattern_complexity = sum(len(re.findall(pattern, text)) for pattern in complex_patterns)
        
        # Normalize to 0-1 scale
        complexity = min(1.0, (line_count * 0.01 + nesting_levels * 0.1 + pattern_complexity * 0.2))
        return complexity
    
    def _calculate_completeness_score(self, text: str) -> float:
        """Calculate how complete/coherent the chunk appears."""
        # Check for balanced braces/brackets
        open_braces = text.count('{')
        close_braces = text.count('}')
        open_parens = text.count('(')
        close_parens = text.count(')')
        open_brackets = text.count('[')
        close_brackets = text.count(']')
        
        # Calculate balance scores
        brace_balance = 1.0 - abs(open_braces - close_braces) / max(open_braces + close_braces, 1)
        paren_balance = 1.0 - abs(open_parens - close_parens) / max(open_parens + close_parens, 1)
        bracket_balance = 1.0 - abs(open_brackets - close_brackets) / max(open_brackets + close_brackets, 1)
        
        # Check for complete constructs
        has_complete_function = bool(re.search(r'\bdef\s+\w+.*?:\s*\n.*?\n\s*\n', text, re.DOTALL))
        has_complete_class = bool(re.search(r'\bclass\s+\w+.*?:\s*\n.*?\n\s*\n', text, re.DOTALL))
        
        construct_completeness = (has_complete_function + has_complete_class) / 2
        
        return (brace_balance + paren_balance + bracket_balance + construct_completeness) / 4
    
    def _count_cross_references(self, text: str) -> int:
        """Count potential cross-references to other files/modules."""
        import_count = len(re.findall(r'\bimport\s+\w+', text))
        require_count = len(re.findall(r'\brequire\s*\([\'"].*?[\'"]\)', text))
        include_count = len(re.findall(r'#include\s*[<"].*?[>"]', text))
        
        return import_count + require_count + include_count
    
    def _evaluate_search_approach(self, chunks: List[Dict], approach: str) -> Dict[str, Any]:
        """Evaluate search quality for a chunking approach."""
        logger.info(f"Evaluating {approach} approach with {len(chunks)} chunks")
        
        query_results = []
        
        for query in self.queries:
            query_result = self._evaluate_query_on_chunks(chunks, query, approach)
            query_results.append(query_result)
        
        # Aggregate results
        aggregated = self._aggregate_search_results(query_results, approach)
        aggregated["query_results"] = query_results
        
        return aggregated
    
    def _evaluate_query_on_chunks(self, chunks: List[Dict], query: Dict, approach: str) -> Dict[str, Any]:
        """Evaluate a single query against chunks."""
        query_text = query["query"]
        query_id = query["id"]
        query_type = query.get("type", "unknown")
        
        # Find relevant chunks
        relevant_chunks = self._find_relevant_chunks_advanced(chunks, query)
        
        # Calculate enterprise-specific metrics
        metrics = {}
        k_values = [1, 3, 5, 10, 20]
        
        for k in k_values:
            top_k = relevant_chunks[:k]
            
            # Standard metrics
            metrics[f"recall@{k}"] = self._calculate_enterprise_recall(top_k, query, k)
            metrics[f"precision@{k}"] = self._calculate_enterprise_precision(top_k, query, k)
            metrics[f"f1@{k}"] = self._calculate_f1_score(
                metrics[f"recall@{k}"], 
                metrics[f"precision@{k}"]
            )
            
            # Enterprise-specific metrics
            metrics[f"architectural_relevance@{k}"] = self._calculate_architectural_relevance(top_k, query)
            metrics[f"cross_file_coherence@{k}"] = self._calculate_cross_file_coherence(top_k)
            metrics[f"completeness_score@{k}"] = self._calculate_result_completeness(top_k, query)
        
        return {
            "query_id": query_id,
            "query_text": query_text,
            "query_type": query_type,
            "enterprise_context": query.get("enterprise_context", ""),
            "approach": approach,
            "metrics": metrics,
            "relevant_chunks_count": len(relevant_chunks),
            "top_chunks": relevant_chunks[:5]  # Store top 5 for analysis
        }
    
    def _find_relevant_chunks_advanced(self, chunks: List[Dict], query: Dict) -> List[Dict]:
        """Advanced relevance finding for enterprise queries."""
        query_text = query["query"].lower()
        expected_functions = [f.lower() for f in query.get("expected_functions", [])]
        expected_files = [f.lower() for f in query.get("expected_files", [])]
        relevance_keywords = [k.lower() for k in query.get("relevance_keywords", [])]
        query_type = query.get("type", "")
        enterprise_context = query.get("enterprise_context", "").lower()
        
        scored_chunks = []
        
        for chunk in chunks:
            relevance_score = self._calculate_advanced_relevance_score(
                chunk, query_text, expected_functions, expected_files, 
                relevance_keywords, query_type, enterprise_context
            )
            
            if relevance_score > 0:
                chunk_copy = chunk.copy()
                chunk_copy["relevance_score"] = relevance_score
                scored_chunks.append(chunk_copy)
        
        # Sort by relevance score
        scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return scored_chunks
    
    def _calculate_advanced_relevance_score(self, chunk: Dict, query_text: str, 
                                          expected_functions: List[str], expected_files: List[str],
                                          relevance_keywords: List[str], query_type: str,
                                          enterprise_context: str) -> float:
        """Calculate advanced relevance score considering enterprise patterns."""
        chunk_text = chunk["text"].lower()
        score = 0.0
        
        # Basic keyword matching
        for keyword in relevance_keywords:
            if keyword in chunk_text:
                score += 1.0
        
        # Function name matching with higher weight
        for func_name in expected_functions:
            if func_name in chunk_text:
                if f"def {func_name}" in chunk_text or f"function {func_name}" in chunk_text:
                    score += 5.0  # Function definition
                else:
                    score += 2.0  # Function reference
        
        # File pattern matching
        for file_pattern in expected_files:
            if file_pattern in chunk.get("inferred_file_type", "").lower():
                score += 1.5
        
        # Query type specific scoring
        if query_type == "architectural_pattern":
            score += len(chunk.get("architectural_keywords", [])) * 0.5
        elif query_type == "security_pattern" and chunk.get("has_security", False):
            score += 2.0
        elif query_type == "performance_pattern" and chunk.get("has_caching", False):
            score += 2.0
        elif query_type == "testing_pattern" and chunk.get("has_testing", False):
            score += 2.0
        
        # Enterprise context matching
        if enterprise_context:
            context_keywords = enterprise_context.split()
            for keyword in context_keywords:
                if keyword in chunk_text:
                    score += 0.5
        
        # Bonus for completeness and complexity
        score += chunk.get("completeness_score", 0) * 0.5
        score += chunk.get("complexity_score", 0) * 0.3
        
        # Cross-reference bonus for architectural queries
        if "architectural" in query_type or "cross" in query_type:
            score += chunk.get("cross_reference_count", 0) * 0.1
        
        return score
    
    def _calculate_enterprise_recall(self, retrieved_chunks: List[Dict], query: Dict, k: int) -> float:
        """Calculate recall considering enterprise query complexity."""
        if not retrieved_chunks:
            return 0.0
        
        highly_relevant = sum(
            1 for chunk in retrieved_chunks 
            if chunk.get("relevance_score", 0) >= 3.0  # Higher threshold for enterprise queries
        )
        
        # Expected relevant based on query difficulty and type
        difficulty = query.get("difficulty", "medium")
        query_type = query.get("type", "")
        
        if difficulty == "easy":
            expected = 3
        elif difficulty == "medium":
            expected = 5
        elif difficulty == "hard":
            expected = 8
        else:  # expert
            expected = 12
        
        # Adjust based on query type
        if "architectural" in query_type or "cross" in query_type:
            expected = int(expected * 1.5)
        
        return min(highly_relevant / expected, 1.0)
    
    def _calculate_enterprise_precision(self, retrieved_chunks: List[Dict], query: Dict, k: int) -> float:
        """Calculate precision for enterprise queries."""
        if not retrieved_chunks:
            return 0.0
        
        relevant_count = sum(
            1 for chunk in retrieved_chunks 
            if chunk.get("relevance_score", 0) >= 1.5  # Moderate threshold for precision
        )
        
        return relevant_count / len(retrieved_chunks)
    
    def _calculate_f1_score(self, recall: float, precision: float) -> float:
        """Calculate F1 score."""
        if recall + precision == 0:
            return 0.0
        return 2 * (recall * precision) / (recall + precision)
    
    def _calculate_architectural_relevance(self, chunks: List[Dict], query: Dict) -> float:
        """Calculate how well chunks capture architectural patterns."""
        if not chunks:
            return 0.0
        
        query_type = query.get("type", "")
        
        # Count chunks with relevant architectural indicators
        relevant_count = 0
        for chunk in chunks:
            if query_type == "architectural_pattern" and chunk.get("architectural_keywords"):
                relevant_count += 1
            elif query_type == "security_pattern" and chunk.get("has_security"):
                relevant_count += 1
            elif query_type == "messaging_pattern" and chunk.get("has_messaging"):
                relevant_count += 1
            elif query_type == "performance_pattern" and chunk.get("has_caching"):
                relevant_count += 1
            elif chunk.get("complexity_score", 0) > 0.5:  # Complex chunks likely contain patterns
                relevant_count += 1
        
        return relevant_count / len(chunks)
    
    def _calculate_cross_file_coherence(self, chunks: List[Dict]) -> float:
        """Calculate coherence across multiple files/modules."""
        if not chunks:
            return 0.0
        
        # Count chunks with cross-references
        cross_ref_chunks = sum(1 for chunk in chunks if chunk.get("cross_reference_count", 0) > 0)
        
        # Calculate diversity of file types
        file_types = set(chunk.get("inferred_file_type", "unknown") for chunk in chunks)
        type_diversity = len(file_types) / max(len(chunks), 1)
        
        # Calculate import coherence
        avg_completeness = statistics.mean(chunk.get("completeness_score", 0) for chunk in chunks)
        
        return (cross_ref_chunks / len(chunks) + type_diversity + avg_completeness) / 3
    
    def _calculate_result_completeness(self, chunks: List[Dict], query: Dict) -> float:
        """Calculate how complete the search results are."""
        if not chunks:
            return 0.0
        
        # Average completeness score of retrieved chunks
        avg_completeness = statistics.mean(chunk.get("completeness_score", 0) for chunk in chunks)
        
        # Check for presence of expected patterns
        expected_functions = query.get("expected_functions", [])
        found_functions = set()
        
        for chunk in chunks:
            chunk_text = chunk["text"].lower()
            for func in expected_functions:
                if func.lower() in chunk_text:
                    found_functions.add(func)
        
        function_coverage = len(found_functions) / max(len(expected_functions), 1)
        
        return (avg_completeness + function_coverage) / 2
    
    def _aggregate_search_results(self, query_results: List[Dict], approach: str) -> Dict[str, Any]:
        """Aggregate search results across all queries."""
        if not query_results:
            return {"approach": approach}
        
        aggregated = {"approach": approach}
        
        # Standard metrics
        k_values = [1, 3, 5, 10, 20]
        for k in k_values:
            recalls = [qr["metrics"][f"recall@{k}"] for qr in query_results]
            precisions = [qr["metrics"][f"precision@{k}"] for qr in query_results]
            f1_scores = [qr["metrics"][f"f1@{k}"] for qr in query_results]
            
            aggregated[f"avg_recall@{k}"] = statistics.mean(recalls)
            aggregated[f"avg_precision@{k}"] = statistics.mean(precisions)
            aggregated[f"avg_f1@{k}"] = statistics.mean(f1_scores)
        
        # Enterprise-specific metrics
        for k in k_values:
            arch_relevances = [qr["metrics"][f"architectural_relevance@{k}"] for qr in query_results]
            coherences = [qr["metrics"][f"cross_file_coherence@{k}"] for qr in query_results]
            completenesses = [qr["metrics"][f"completeness_score@{k}"] for qr in query_results]
            
            aggregated[f"avg_architectural_relevance@{k}"] = statistics.mean(arch_relevances)
            aggregated[f"avg_cross_file_coherence@{k}"] = statistics.mean(coherences)
            aggregated[f"avg_completeness@{k}"] = statistics.mean(completenesses)
        
        # Query type performance
        type_performance = defaultdict(list)
        for qr in query_results:
            query_type = qr["query_type"]
            type_performance[query_type].append(qr["metrics"]["f1@5"])
        
        aggregated["performance_by_type"] = {
            qtype: statistics.mean(scores) 
            for qtype, scores in type_performance.items()
        }
        
        return aggregated
    
    def _calculate_advanced_comparison(self, ast_results: Dict, naive_results: Dict) -> Dict[str, Any]:
        """Calculate advanced comparison metrics."""
        comparison = {}
        
        k_values = [1, 3, 5, 10, 20]
        
        for k in k_values:
            # Standard metrics
            comparison[f"recall_improvement@{k}"] = (
                ast_results.get(f"avg_recall@{k}", 0) - 
                naive_results.get(f"avg_recall@{k}", 0)
            )
            comparison[f"precision_improvement@{k}"] = (
                ast_results.get(f"avg_precision@{k}", 0) - 
                naive_results.get(f"avg_precision@{k}", 0)
            )
            comparison[f"f1_improvement@{k}"] = (
                ast_results.get(f"avg_f1@{k}", 0) - 
                naive_results.get(f"avg_f1@{k}", 0)
            )
            
            # Enterprise metrics
            comparison[f"architectural_improvement@{k}"] = (
                ast_results.get(f"avg_architectural_relevance@{k}", 0) - 
                naive_results.get(f"avg_architectural_relevance@{k}", 0)
            )
            comparison[f"coherence_improvement@{k}"] = (
                ast_results.get(f"avg_cross_file_coherence@{k}", 0) - 
                naive_results.get(f"avg_cross_file_coherence@{k}", 0)
            )
            comparison[f"completeness_improvement@{k}"] = (
                ast_results.get(f"avg_completeness@{k}", 0) - 
                naive_results.get(f"avg_completeness@{k}", 0)
            )
        
        # Query type comparison
        ast_by_type = ast_results.get("performance_by_type", {})
        naive_by_type = naive_results.get("performance_by_type", {})
        
        type_improvements = {}
        for query_type in set(ast_by_type.keys()) | set(naive_by_type.keys()):
            ast_score = ast_by_type.get(query_type, 0)
            naive_score = naive_by_type.get(query_type, 0)
            type_improvements[query_type] = ast_score - naive_score
        
        comparison["query_type_improvements"] = type_improvements
        
        return comparison
    
    def _analyze_architectural_patterns(self, ast_chunks: List[Dict], naive_chunks: List[Dict]) -> Dict[str, Any]:
        """Analyze how well each approach captures architectural patterns."""
        analysis = {
            "ast_patterns": self._extract_pattern_statistics(ast_chunks),
            "naive_patterns": self._extract_pattern_statistics(naive_chunks),
            "pattern_preservation_comparison": {}
        }
        
        # Compare pattern preservation
        ast_patterns = analysis["ast_patterns"]
        naive_patterns = analysis["naive_patterns"]
        
        for pattern_type in ast_patterns.keys():
            if pattern_type in naive_patterns:
                ast_count = ast_patterns[pattern_type]["total_chunks"]
                naive_count = naive_patterns[pattern_type]["total_chunks"]
                
                analysis["pattern_preservation_comparison"][pattern_type] = {
                    "ast_chunks": ast_count,
                    "naive_chunks": naive_count,
                    "improvement_ratio": ast_count / max(naive_count, 1)
                }
        
        return analysis
    
    def _extract_pattern_statistics(self, chunks: List[Dict]) -> Dict[str, Dict]:
        """Extract statistics about architectural patterns in chunks."""
        patterns = {
            "authentication": {"total_chunks": 0, "avg_completeness": 0},
            "database_access": {"total_chunks": 0, "avg_completeness": 0},
            "api_endpoints": {"total_chunks": 0, "avg_completeness": 0},
            "error_handling": {"total_chunks": 0, "avg_completeness": 0},
            "configuration": {"total_chunks": 0, "avg_completeness": 0},
            "testing": {"total_chunks": 0, "avg_completeness": 0},
            "caching": {"total_chunks": 0, "avg_completeness": 0},
            "messaging": {"total_chunks": 0, "avg_completeness": 0},
            "security": {"total_chunks": 0, "avg_completeness": 0}
        }
        
        for chunk in chunks:
            completeness = chunk.get("completeness_score", 0)
            
            if chunk.get("has_authentication"):
                patterns["authentication"]["total_chunks"] += 1
                patterns["authentication"]["avg_completeness"] += completeness
            
            if chunk.get("has_database_access"):
                patterns["database_access"]["total_chunks"] += 1
                patterns["database_access"]["avg_completeness"] += completeness
            
            if chunk.get("has_api_endpoints"):
                patterns["api_endpoints"]["total_chunks"] += 1
                patterns["api_endpoints"]["avg_completeness"] += completeness
            
            if chunk.get("has_error_handling"):
                patterns["error_handling"]["total_chunks"] += 1
                patterns["error_handling"]["avg_completeness"] += completeness
            
            if chunk.get("has_configuration"):
                patterns["configuration"]["total_chunks"] += 1
                patterns["configuration"]["avg_completeness"] += completeness
            
            if chunk.get("has_testing"):
                patterns["testing"]["total_chunks"] += 1
                patterns["testing"]["avg_completeness"] += completeness
            
            if chunk.get("has_caching"):
                patterns["caching"]["total_chunks"] += 1
                patterns["caching"]["avg_completeness"] += completeness
            
            if chunk.get("has_messaging"):
                patterns["messaging"]["total_chunks"] += 1
                patterns["messaging"]["avg_completeness"] += completeness
            
            if chunk.get("has_security"):
                patterns["security"]["total_chunks"] += 1
                patterns["security"]["avg_completeness"] += completeness
        
        # Calculate averages
        for pattern_type, stats in patterns.items():
            if stats["total_chunks"] > 0:
                stats["avg_completeness"] /= stats["total_chunks"]
        
        return patterns
    
    def _analyze_scale_effects(self, ast_chunks: List[Dict], naive_chunks: List[Dict]) -> Dict[str, Any]:
        """Analyze how scale affects chunking quality."""
        analysis = {
            "ast_scale_metrics": self._calculate_scale_metrics(ast_chunks),
            "naive_scale_metrics": self._calculate_scale_metrics(naive_chunks),
            "scale_conclusions": {}
        }
        
        ast_metrics = analysis["ast_scale_metrics"]
        naive_metrics = analysis["naive_scale_metrics"]
        
        # Compare scale effects
        analysis["scale_conclusions"] = {
            "cross_reference_advantage": (
                ast_metrics["avg_cross_references"] - 
                naive_metrics["avg_cross_references"]
            ),
            "complexity_preservation": (
                ast_metrics["avg_complexity"] - 
                naive_metrics["avg_complexity"]
            ),
            "architectural_density": (
                ast_metrics["architectural_keyword_density"] - 
                naive_metrics["architectural_keyword_density"]
            )
        }
        
        return analysis
    
    def _calculate_scale_metrics(self, chunks: List[Dict]) -> Dict[str, float]:
        """Calculate metrics that show scale effects."""
        if not chunks:
            return {}
        
        total_cross_refs = sum(chunk.get("cross_reference_count", 0) for chunk in chunks)
        total_complexity = sum(chunk.get("complexity_score", 0) for chunk in chunks)
        total_arch_keywords = sum(len(chunk.get("architectural_keywords", [])) for chunk in chunks)
        total_size = sum(chunk.get("size", 0) for chunk in chunks)
        
        return {
            "avg_cross_references": total_cross_refs / len(chunks),
            "avg_complexity": total_complexity / len(chunks),
            "architectural_keyword_density": total_arch_keywords / max(total_size, 1) * 1000,  # per 1000 chars
            "chunk_count": len(chunks),
            "total_content_size": total_size
        }


def main():
    """Main function for large-scale search evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Large-scale search quality evaluation")
    parser.add_argument("--data-dir", type=str, default="large_scale_evaluation", 
                       help="Directory containing cloned repositories")
    parser.add_argument("--queries", type=str, default="benchmarks/complex_queries.json",
                       help="Complex queries file")
    parser.add_argument("--output", type=str, default="large_scale_search_results.json",
                       help="Output file")
    parser.add_argument("--max-files", type=int, default=100,
                       help="Maximum files to process per repository")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Load documents from repository directories
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            logger.error(f"Data directory {data_dir} does not exist")
            return 1
        
        all_documents = []
        for repo_dir in data_dir.iterdir():
            if repo_dir.is_dir() and not repo_dir.name.startswith('.'):
                logger.info(f"Loading documents from {repo_dir.name}")
                
                # Load code files from repository
                code_extensions = [".py", ".js", ".ts", ".tsx", ".java", ".go", ".rs", ".cpp", ".c", ".cs"]
                file_count = 0
                
                for ext in code_extensions:
                    for file_path in repo_dir.rglob(f"*{ext}"):
                        if file_count >= args.max_files:
                            break
                        
                        # Skip common non-source directories
                        if any(skip in str(file_path) for skip in ['node_modules', '.git', '__pycache__', 'target', 'build']):
                            continue
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                if len(content.strip()) > 100:  # Skip very small files
                                    doc = Document(
                                        text=content,
                                        metadata={
                                            "file_path": str(file_path),
                                            "repository": repo_dir.name,
                                            "language": ext[1:]
                                        }
                                    )
                                    all_documents.append(doc)
                                    file_count += 1
                        except Exception as e:
                            logger.warning(f"Could not read {file_path}: {e}")
                    
                    if file_count >= args.max_files:
                        break
        
        if not all_documents:
            logger.error("No documents found to evaluate")
            return 1
        
        logger.info(f"Loaded {len(all_documents)} documents for evaluation")
        
        # Run evaluation
        evaluator = EnterpriseSearchEvaluator(all_documents, args.queries)
        results = evaluator.evaluate_large_scale_search_quality()
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("LARGE-SCALE SEARCH QUALITY EVALUATION SUMMARY")
        print("="*80)
        
        metadata = results["evaluation_metadata"]
        comparison = results["comparison"]
        
        print(f"Documents Evaluated: {metadata['document_count']}")
        print(f"Total Content Size: {metadata['total_content_size']:,} characters")
        print(f"Queries Tested: {metadata['query_count']}")
        
        # Key improvements
        print("\n📈 Key Search Quality Improvements:")
        if "recall_improvement@5" in comparison:
            print(f"  Recall@5:        {comparison['recall_improvement@5']:+.3f}")
        if "precision_improvement@5" in comparison:
            print(f"  Precision@5:     {comparison['precision_improvement@5']:+.3f}")
        if "architectural_improvement@5" in comparison:
            print(f"  Architectural:   {comparison['architectural_improvement@5']:+.3f}")
        if "coherence_improvement@5" in comparison:
            print(f"  Coherence:       {comparison['coherence_improvement@5']:+.3f}")
        
        # Query type performance
        type_improvements = comparison.get("query_type_improvements", {})
        if type_improvements:
            print("\n🎯 Query Type Improvements:")
            for qtype, improvement in type_improvements.items():
                print(f"  {qtype:<20}: {improvement:+.3f}")
        
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