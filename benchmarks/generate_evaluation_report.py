#!/usr/bin/env python3
"""
Comprehensive Evaluation Report Generator

This script generates a comprehensive report comparing AST-aware vs naive chunking
across all evaluation dimensions with visualizations and actionable insights.
"""

import json
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List
import subprocess
import logging

logger = logging.getLogger(__name__)


class EvaluationReportGenerator:
    """Generates comprehensive evaluation reports with recommendations."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.performance_results = None
        self.search_quality_results = None
        
    def run_full_evaluation(self, test_data_dir: str = "data/code_samples") -> Dict[str, Any]:
        """Run complete evaluation pipeline and generate comprehensive report."""
        logger.info("Starting full evaluation pipeline...")
        
        # Run performance evaluation
        logger.info("Running performance evaluation...")
        self.performance_results = self._run_performance_evaluation(test_data_dir)
        
        # Run search quality evaluation
        logger.info("Running search quality evaluation...")
        self.search_quality_results = self._run_search_quality_evaluation(test_data_dir)
        
        # Generate comprehensive report
        logger.info("Generating comprehensive report...")
        report = self._generate_comprehensive_report()
        
        # Save report
        self._save_report(report)
        
        return report
        
    def _run_performance_evaluation(self, test_data_dir: str) -> Dict[str, Any]:
        """Run the AST vs naive performance evaluation."""
        try:
            # Import and run the performance evaluator
            sys.path.insert(0, str(Path(__file__).parent))
            from ast_vs_naive_evaluation import ChunkingEvaluator
            
            evaluator = ChunkingEvaluator(test_data_dir, str(self.output_dir))
            documents = evaluator.load_test_documents()
            results = evaluator.evaluate_chunking_performance(documents)
            
            return results
            
        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            return {"error": str(e)}
            
    def _run_search_quality_evaluation(self, test_data_dir: str) -> Dict[str, Any]:
        """Run the search quality evaluation."""
        try:
            # Import and run the search quality evaluator
            from search_quality_evaluator import SearchQualityEvaluator
            
            queries_file = Path(__file__).parent / "code_understanding_queries.json"
            evaluator = SearchQualityEvaluator(test_data_dir, str(queries_file))
            results = evaluator.evaluate_search_quality()
            
            return results
            
        except Exception as e:
            logger.error(f"Search quality evaluation failed: {e}")
            return {"error": str(e)}
            
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        report = {
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_version": "1.0",
                "evaluator": "AST vs Naive Chunking Comparison"
            },
            "executive_summary": {},
            "detailed_analysis": {},
            "recommendations": {},
            "raw_results": {
                "performance": self.performance_results,
                "search_quality": self.search_quality_results
            }
        }
        
        # Generate executive summary
        report["executive_summary"] = self._generate_executive_summary()
        
        # Generate detailed analysis
        report["detailed_analysis"] = self._generate_detailed_analysis()
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations()
        
        return report
        
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of findings."""
        summary = {
            "overall_recommendation": "unknown",
            "key_findings": [],
            "performance_overview": {},
            "search_quality_overview": {}
        }
        
        # Performance overview
        if self.performance_results and "comparison" in self.performance_results:
            perf_comp = self.performance_results["comparison"]
            ast_perf = self.performance_results.get("ast_chunking", {})
            naive_perf = self.performance_results.get("naive_chunking", {})
            
            summary["performance_overview"] = {
                "speed_ratio": perf_comp.get("speed_ratio", 1.0),
                "memory_ratio": perf_comp.get("memory_ratio", 1.0),
                "completeness_improvement": perf_comp.get("completeness_improvement", 0),
                "coherence_improvement": perf_comp.get("coherence_improvement", 0),
                "ast_chunks": ast_perf.get("chunk_count", 0),
                "naive_chunks": naive_perf.get("chunk_count", 0)
            }
            
        # Search quality overview
        if self.search_quality_results and "comparison" in self.search_quality_results:
            search_comp = self.search_quality_results["comparison"]
            
            summary["search_quality_overview"] = {
                "recall_improvement_at_5": search_comp.get("recall_improvement@5", 0),
                "precision_improvement_at_5": search_comp.get("precision_improvement@5", 0),
                "f1_improvement_at_5": search_comp.get("f1_improvement@5", 0),
                "overall_relevance_improvement": search_comp.get("avg_relevance_improvement", 0)
            }
            
        # Generate key findings
        summary["key_findings"] = self._extract_key_findings()
        
        # Overall recommendation
        summary["overall_recommendation"] = self._determine_overall_recommendation()
        
        return summary
        
    def _extract_key_findings(self) -> List[str]:
        """Extract key findings from the evaluation results."""
        findings = []
        
        # Performance findings
        if self.performance_results:
            perf_comp = self.performance_results.get("comparison", {})
            
            speed_ratio = perf_comp.get("speed_ratio", 1.0)
            if speed_ratio > 2.0:
                findings.append(f"AST chunking is {speed_ratio:.1f}x slower than naive chunking")
            elif speed_ratio < 0.8:
                findings.append(f"AST chunking is {1/speed_ratio:.1f}x faster than naive chunking")
                
            completeness_improvement = perf_comp.get("completeness_improvement", 0)
            if completeness_improvement > 0.1:
                findings.append(f"AST chunking improves function completeness by {completeness_improvement:.1%}")
                
            coherence_improvement = perf_comp.get("coherence_improvement", 0)
            if coherence_improvement > 0.05:
                findings.append(f"AST chunking improves chunk coherence by {coherence_improvement:.1%}")
                
        # Search quality findings
        if self.search_quality_results:
            search_comp = self.search_quality_results.get("comparison", {})
            
            recall_improvement = search_comp.get("recall_improvement@5", 0)
            if recall_improvement > 0.05:
                findings.append(f"AST chunking improves Recall@5 by {recall_improvement:.1%}")
            elif recall_improvement < -0.05:
                findings.append(f"Naive chunking has {abs(recall_improvement):.1%} better Recall@5")
                
            precision_improvement = search_comp.get("precision_improvement@5", 0)
            if precision_improvement > 0.05:
                findings.append(f"AST chunking improves Precision@5 by {precision_improvement:.1%}")
                
            f1_improvement = search_comp.get("f1_improvement@5", 0)
            if f1_improvement > 0.05:
                findings.append(f"AST chunking improves F1@5 score by {f1_improvement:.1%}")
                
        # Code structure findings
        if self.performance_results:
            ast_results = self.performance_results.get("ast_chunking", {})
            naive_results = self.performance_results.get("naive_chunking", {})
            
            ast_complete = ast_results.get("complete_functions", 0)
            naive_complete = naive_results.get("complete_functions", 0)
            
            if ast_complete > naive_complete:
                findings.append(f"AST chunking preserves {ast_complete - naive_complete} more complete functions")
                
        if not findings:
            findings.append("No significant differences found between approaches")
            
        return findings
        
    def _determine_overall_recommendation(self) -> str:
        """Determine overall recommendation based on results."""
        # Scoring system
        ast_score = 0
        naive_score = 0
        
        # Performance factors
        if self.performance_results:
            perf_comp = self.performance_results.get("comparison", {})
            
            # Speed penalty for AST
            speed_ratio = perf_comp.get("speed_ratio", 1.0)
            if speed_ratio > 3.0:
                naive_score += 2  # Significant speed penalty
            elif speed_ratio > 1.5:
                naive_score += 1  # Moderate speed penalty
                
            # Completeness bonus for AST
            completeness_improvement = perf_comp.get("completeness_improvement", 0)
            if completeness_improvement > 0.1:
                ast_score += 2
            elif completeness_improvement > 0.05:
                ast_score += 1
                
            # Coherence bonus for AST
            coherence_improvement = perf_comp.get("coherence_improvement", 0)
            if coherence_improvement > 0.1:
                ast_score += 1
                
        # Search quality factors
        if self.search_quality_results:
            search_comp = self.search_quality_results.get("comparison", {})
            
            # Recall improvement
            recall_improvement = search_comp.get("recall_improvement@5", 0)
            if recall_improvement > 0.1:
                ast_score += 2
            elif recall_improvement > 0.05:
                ast_score += 1
            elif recall_improvement < -0.05:
                naive_score += 1
                
            # Precision improvement
            precision_improvement = search_comp.get("precision_improvement@5", 0)
            if precision_improvement > 0.1:
                ast_score += 2
            elif precision_improvement > 0.05:
                ast_score += 1
                
        # Make recommendation
        if ast_score > naive_score + 1:
            return "strongly_recommend_ast"
        elif ast_score > naive_score:
            return "recommend_ast"
        elif naive_score > ast_score + 1:
            return "recommend_naive"
        elif naive_score > ast_score:
            return "slight_preference_naive"
        else:
            return "no_clear_preference"
            
    def _generate_detailed_analysis(self) -> Dict[str, Any]:
        """Generate detailed analysis of results."""
        analysis = {
            "performance_analysis": {},
            "search_quality_analysis": {},
            "use_case_analysis": {}
        }
        
        # Performance analysis
        if self.performance_results:
            analysis["performance_analysis"] = self._analyze_performance_results()
            
        # Search quality analysis
        if self.search_quality_results:
            analysis["search_quality_analysis"] = self._analyze_search_quality_results()
            
        # Use case analysis
        analysis["use_case_analysis"] = self._analyze_use_cases()
        
        return analysis
        
    def _analyze_performance_results(self) -> Dict[str, Any]:
        """Analyze performance evaluation results in detail."""
        if not self.performance_results:
            return {"error": "No performance results available"}
            
        ast_results = self.performance_results.get("ast_chunking", {})
        naive_results = self.performance_results.get("naive_chunking", {})
        comparison = self.performance_results.get("comparison", {})
        
        analysis = {
            "processing_speed": {
                "ast_time": ast_results.get("processing_time", 0),
                "naive_time": naive_results.get("processing_time", 0),
                "speed_ratio": comparison.get("speed_ratio", 1.0),
                "efficiency_ast": comparison.get("ast_efficiency", 0),
                "efficiency_naive": comparison.get("naive_efficiency", 0)
            },
            "memory_usage": {
                "ast_memory_mb": ast_results.get("peak_memory_mb", 0),
                "naive_memory_mb": naive_results.get("peak_memory_mb", 0),
                "memory_ratio": comparison.get("memory_ratio", 1.0)
            },
            "chunk_characteristics": {
                "ast_chunk_count": ast_results.get("chunk_count", 0),
                "naive_chunk_count": naive_results.get("chunk_count", 0),
                "ast_avg_length": ast_results.get("avg_chunk_length", 0),
                "naive_avg_length": naive_results.get("avg_chunk_length", 0),
                "chunk_count_ratio": comparison.get("chunk_count_ratio", 1.0)
            },
            "code_quality": {
                "ast_completeness": ast_results.get("function_completeness_ratio", 0),
                "naive_completeness": naive_results.get("function_completeness_ratio", 0),
                "completeness_improvement": comparison.get("completeness_improvement", 0),
                "coherence_improvement": comparison.get("coherence_improvement", 0)
            }
        }
        
        return analysis
        
    def _analyze_search_quality_results(self) -> Dict[str, Any]:
        """Analyze search quality evaluation results in detail."""
        if not self.search_quality_results:
            return {"error": "No search quality results available"}
            
        ast_results = self.search_quality_results.get("ast_chunking", {})
        naive_results = self.search_quality_results.get("naive_chunking", {})
        comparison = self.search_quality_results.get("comparison", {})
        
        analysis = {
            "recall_metrics": {},
            "precision_metrics": {},
            "f1_metrics": {},
            "query_type_performance": {}
        }
        
        # Recall metrics
        for k in [1, 3, 5, 10]:
            analysis["recall_metrics"][f"k_{k}"] = {
                "ast": ast_results.get(f"avg_recall@{k}", 0),
                "naive": naive_results.get(f"avg_recall@{k}", 0),
                "improvement": comparison.get(f"recall_improvement@{k}", 0),
                "ratio": comparison.get(f"recall_ratio@{k}", 1.0)
            }
            
        # Precision metrics
        for k in [1, 3, 5, 10]:
            analysis["precision_metrics"][f"k_{k}"] = {
                "ast": ast_results.get(f"avg_precision@{k}", 0),
                "naive": naive_results.get(f"avg_precision@{k}", 0),
                "improvement": comparison.get(f"precision_improvement@{k}", 0),
                "ratio": comparison.get(f"precision_ratio@{k}", 1.0)
            }
            
        # F1 metrics
        for k in [1, 3, 5, 10]:
            analysis["f1_metrics"][f"k_{k}"] = {
                "ast": ast_results.get(f"avg_f1@{k}", 0),
                "naive": naive_results.get(f"avg_f1@{k}", 0),
                "improvement": comparison.get(f"f1_improvement@{k}", 0)
            }
            
        # Query type performance
        ast_by_type = ast_results.get("performance_by_type", {})
        naive_by_type = naive_results.get("performance_by_type", {})
        
        for query_type in ast_by_type.keys():
            if query_type in naive_by_type:
                analysis["query_type_performance"][query_type] = {
                    "ast_recall@5": ast_by_type[query_type].get("avg_recall@5", 0),
                    "naive_recall@5": naive_by_type[query_type].get("avg_recall@5", 0)
                }
                
        return analysis
        
    def _analyze_use_cases(self) -> Dict[str, Any]:
        """Analyze different use cases and provide specific recommendations."""
        use_cases = {
            "code_heavy_repositories": {
                "description": "Repositories with primarily code files",
                "recommendation": "ast",
                "reasoning": "Better function boundary preservation and semantic understanding"
            },
            "mixed_content": {
                "description": "Repositories with code and documentation",
                "recommendation": "conditional",
                "reasoning": "Use AST for code files, naive for documentation"
            },
            "performance_critical": {
                "description": "Applications where speed is paramount",
                "recommendation": "naive",
                "reasoning": "Lower processing overhead and memory usage"
            },
            "semantic_search": {
                "description": "Applications requiring semantic code understanding",
                "recommendation": "ast",
                "reasoning": "Better preservation of code structure and context"
            },
            "large_scale_indexing": {
                "description": "Indexing very large codebases",
                "recommendation": "conditional",
                "reasoning": "Depends on available resources and quality requirements"
            }
        }
        
        # Adjust recommendations based on actual results
        if self.performance_results:
            speed_ratio = self.performance_results.get("comparison", {}).get("speed_ratio", 1.0)
            completeness_improvement = self.performance_results.get("comparison", {}).get("completeness_improvement", 0)
            
            if speed_ratio > 5.0:  # Very slow
                use_cases["performance_critical"]["recommendation"] = "strongly_naive"
            elif completeness_improvement > 0.2:  # Very good completeness
                use_cases["code_heavy_repositories"]["recommendation"] = "strongly_ast"
                
        return use_cases
        
    def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate actionable recommendations."""
        recommendations = {
            "immediate_actions": [],
            "configuration_guidelines": {},
            "implementation_strategy": {},
            "monitoring_metrics": []
        }
        
        # Immediate actions based on results
        overall_rec = self._determine_overall_recommendation()
        
        if overall_rec in ["strongly_recommend_ast", "recommend_ast"]:
            recommendations["immediate_actions"].extend([
                "Enable AST chunking for code-heavy repositories",
                "Configure appropriate AST chunk sizes (512-768 characters)",
                "Set up fallback to naive chunking for unsupported languages"
            ])
        elif overall_rec in ["recommend_naive", "slight_preference_naive"]:
            recommendations["immediate_actions"].extend([
                "Continue using naive chunking for most use cases",
                "Consider AST chunking only for specialized code analysis tasks",
                "Monitor performance metrics if implementing AST chunking"
            ])
        else:
            recommendations["immediate_actions"].extend([
                "Implement both approaches with dynamic selection",
                "Use AST chunking for code files, naive for documentation",
                "A/B test both approaches in production"
            ])
            
        # Configuration guidelines
        if self.performance_results:
            ast_results = self.performance_results.get("ast_chunking", {})
            
            recommendations["configuration_guidelines"] = {
                "ast_chunk_size": "512-768 characters for optimal balance",
                "ast_chunk_overlap": "64-96 characters for good continuity", 
                "fallback_enabled": True,
                "memory_limit": f"Plan for {ast_results.get('peak_memory_mb', 100):.0f}MB+ memory usage",
                "timeout_settings": "Set 3-5x longer timeouts for AST processing"
            }
            
        # Implementation strategy
        recommendations["implementation_strategy"] = {
            "phase_1": "Implement AST chunking as optional feature",
            "phase_2": "A/B test with subset of users/repositories",
            "phase_3": "Roll out based on repository characteristics",
            "rollback_plan": "Keep naive chunking as fallback option"
        }
        
        # Monitoring metrics
        recommendations["monitoring_metrics"] = [
            "Processing time per document",
            "Memory usage during chunking",
            "Search result relevance scores",
            "User satisfaction with search results",
            "Function boundary preservation rate"
        ]
        
        return recommendations
        
    def _save_report(self, report: Dict[str, Any]):
        """Save the comprehensive report to files."""
        # Save JSON report
        json_file = self.output_dir / "comprehensive_evaluation_report.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Save markdown report
        markdown_file = self.output_dir / "evaluation_report.md"
        markdown_content = self._generate_markdown_report(report)
        with open(markdown_file, 'w') as f:
            f.write(markdown_content)
            
        logger.info(f"Reports saved to {json_file} and {markdown_file}")
        
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate a markdown version of the report."""
        md = []
        
        # Header
        md.append("# AST-Aware vs Naive Chunking Evaluation Report")
        md.append("")
        md.append(f"**Generated:** {report['metadata']['generated_at']}")
        md.append(f"**Version:** {report['metadata']['evaluation_version']}")
        md.append("")
        
        # Executive Summary
        summary = report["executive_summary"]
        md.append("## Executive Summary")
        md.append("")
        
        # Overall recommendation
        rec = summary.get("overall_recommendation", "unknown")
        rec_text = {
            "strongly_recommend_ast": "🚀 **Strongly Recommend AST Chunking**",
            "recommend_ast": "✅ **Recommend AST Chunking**", 
            "no_clear_preference": "⚖️ **No Clear Preference - Use Case Dependent**",
            "slight_preference_naive": "🔄 **Slight Preference for Naive Chunking**",
            "recommend_naive": "⚡ **Recommend Naive Chunking**"
        }.get(rec, "❓ **Unknown Recommendation**")
        
        md.append(f"### Overall Recommendation: {rec_text}")
        md.append("")
        
        # Key findings
        md.append("### Key Findings")
        for finding in summary.get("key_findings", []):
            md.append(f"- {finding}")
        md.append("")
        
        # Performance Overview
        perf_overview = summary.get("performance_overview", {})
        if perf_overview:
            md.append("### Performance Overview")
            md.append("")
            md.append("| Metric | Value |")
            md.append("|--------|-------|")
            md.append(f"| Speed Ratio (AST/Naive) | {perf_overview.get('speed_ratio', 0):.2f}x |")
            md.append(f"| Memory Ratio (AST/Naive) | {perf_overview.get('memory_ratio', 0):.2f}x |")
            md.append(f"| Completeness Improvement | {perf_overview.get('completeness_improvement', 0):+.2%} |")
            md.append(f"| Coherence Improvement | {perf_overview.get('coherence_improvement', 0):+.2%} |")
            md.append("")
            
        # Search Quality Overview
        search_overview = summary.get("search_quality_overview", {})
        if search_overview:
            md.append("### Search Quality Overview")
            md.append("")
            md.append("| Metric | Improvement |")
            md.append("|--------|-------------|")
            md.append(f"| Recall@5 | {search_overview.get('recall_improvement_at_5', 0):+.2%} |")
            md.append(f"| Precision@5 | {search_overview.get('precision_improvement_at_5', 0):+.2%} |") 
            md.append(f"| F1@5 | {search_overview.get('f1_improvement_at_5', 0):+.2%} |")
            md.append("")
            
        # Recommendations
        recommendations = report.get("recommendations", {})
        if recommendations:
            md.append("## Recommendations")
            md.append("")
            
            immediate_actions = recommendations.get("immediate_actions", [])
            if immediate_actions:
                md.append("### Immediate Actions")
                for action in immediate_actions:
                    md.append(f"1. {action}")
                md.append("")
                
            config_guidelines = recommendations.get("configuration_guidelines", {})
            if config_guidelines:
                md.append("### Configuration Guidelines")
                md.append("")
                for key, value in config_guidelines.items():
                    md.append(f"- **{key.replace('_', ' ').title()}**: {value}")
                md.append("")
                
        # Detailed Analysis
        detailed = report.get("detailed_analysis", {})
        if detailed:
            md.append("## Detailed Analysis")
            md.append("")
            
            # Performance analysis
            perf_analysis = detailed.get("performance_analysis", {})
            if perf_analysis and "processing_speed" in perf_analysis:
                speed = perf_analysis["processing_speed"]
                md.append("### Processing Performance")
                md.append("")
                md.append("| Approach | Time (s) | Efficiency (chunks/s) |")
                md.append("|----------|----------|----------------------|")
                md.append(f"| AST Chunking | {speed.get('ast_time', 0):.3f} | {speed.get('efficiency_ast', 0):.1f} |")
                md.append(f"| Naive Chunking | {speed.get('naive_time', 0):.3f} | {speed.get('efficiency_naive', 0):.1f} |")
                md.append("")
                
        # Use Cases
        use_cases = detailed.get("use_case_analysis", {})
        if use_cases:
            md.append("### Use Case Recommendations")
            md.append("")
            for use_case, details in use_cases.items():
                rec = details.get("recommendation", "unknown")
                rec_emoji = {"ast": "🧠", "naive": "⚡", "conditional": "🔄", "strongly_ast": "🚀", "strongly_naive": "💨"}.get(rec, "❓")
                md.append(f"**{use_case.replace('_', ' ').title()}** {rec_emoji}")
                md.append(f"- {details.get('description', '')}")
                md.append(f"- *Recommendation: {rec.replace('_', ' ').title()}*")
                md.append(f"- *Reasoning: {details.get('reasoning', '')}*")
                md.append("")
                
        return "\n".join(md)


def main():
    """Main function for running the comprehensive evaluation."""
    parser = argparse.ArgumentParser(description="Generate comprehensive AST vs Naive chunking evaluation report")
    parser.add_argument(
        "--test-data-dir",
        type=str,
        default="data/code_samples",
        help="Directory containing test code files"
    )
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="benchmark_results",
        help="Directory to save results"
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
        generator = EvaluationReportGenerator(args.output_dir)
        report = generator.run_full_evaluation(args.test_data_dir)
        
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION COMPLETE")
        print("="*80)
        
        # Print executive summary
        summary = report.get("executive_summary", {})
        rec = summary.get("overall_recommendation", "unknown")
        
        print(f"\nOverall Recommendation: {rec.replace('_', ' ').title()}")
        
        print("\nKey Findings:")
        for finding in summary.get("key_findings", []):
            print(f"  • {finding}")
            
        print(f"\nDetailed reports saved to: {args.output_dir}/")
        print("  • comprehensive_evaluation_report.json")
        print("  • evaluation_report.md")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())