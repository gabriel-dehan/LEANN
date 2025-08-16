#!/usr/bin/env python3
"""
Quick AST vs Naive Evaluation Demo

This script runs a streamlined evaluation to quickly demonstrate the 
performance differences between AST-aware and naive chunking approaches.
Perfect for PR demonstrations and quick assessments.
"""

import sys
import time
from pathlib import Path

# Add apps directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

def main():
    print("🧪 AST vs Naive Chunking Quick Evaluation")
    print("=" * 50)
    
    try:
        # Import evaluation modules
        from generate_evaluation_report import EvaluationReportGenerator
        
        # Run quick evaluation
        print("📊 Running comprehensive evaluation...")
        start_time = time.time()
        
        generator = EvaluationReportGenerator("quick_evaluation_results")
        results = generator.run_full_evaluation("data/code_samples")
        
        total_time = time.time() - start_time
        
        # Extract key metrics for display
        summary = results.get("executive_summary", {})
        performance = summary.get("performance_overview", {})
        search_quality = summary.get("search_quality_overview", {})
        
        print(f"\n✅ Evaluation completed in {total_time:.1f} seconds")
        print("\n" + "=" * 50)
        print("📈 QUICK RESULTS SUMMARY")
        print("=" * 50)
        
        # Performance Results
        print("\n🚀 Performance Metrics:")
        print(f"  Speed Ratio (AST/Naive):     {performance.get('speed_ratio', 0):.2f}x")
        print(f"  Memory Ratio (AST/Naive):    {performance.get('memory_ratio', 0):.2f}x")
        print(f"  Completeness Improvement:    {performance.get('completeness_improvement', 0):+.1%}")
        print(f"  Coherence Improvement:       {performance.get('coherence_improvement', 0):+.1%}")
        print(f"  AST Chunks Generated:        {performance.get('ast_chunks', 0)}")
        print(f"  Naive Chunks Generated:      {performance.get('naive_chunks', 0)}")
        
        # Search Quality Results
        print("\n🔍 Search Quality Metrics:")
        print(f"  Recall@5 Improvement:       {search_quality.get('recall_improvement_at_5', 0):+.1%}")
        print(f"  Precision@5 Improvement:    {search_quality.get('precision_improvement_at_5', 0):+.1%}")
        print(f"  F1@5 Improvement:           {search_quality.get('f1_improvement_at_5', 0):+.1%}")
        print(f"  Overall Relevance Boost:    {search_quality.get('overall_relevance_improvement', 0):+.1%}")
        
        # Overall Recommendation
        recommendation = summary.get("overall_recommendation", "unknown")
        rec_display = {
            "strongly_recommend_ast": "🚀 STRONGLY RECOMMEND AST CHUNKING",
            "recommend_ast": "✅ RECOMMEND AST CHUNKING",
            "no_clear_preference": "⚖️ NO CLEAR PREFERENCE - CONTEXT DEPENDENT",
            "slight_preference_naive": "🔄 SLIGHT PREFERENCE FOR NAIVE CHUNKING",
            "recommend_naive": "⚡ RECOMMEND NAIVE CHUNKING"
        }.get(recommendation, "❓ UNKNOWN")
        
        print(f"\n🎯 Overall Recommendation: {rec_display}")
        
        # Key Findings
        print(f"\n💡 Key Findings:")
        for finding in summary.get("key_findings", []):
            print(f"  • {finding}")
            
        print(f"\n📄 Detailed reports available in: quick_evaluation_results/")
        print(f"  • evaluation_report.md")
        print(f"  • comprehensive_evaluation_report.json")
        
        # Quick interpretation for PR
        print("\n" + "=" * 50)
        print("📋 QUICK INTERPRETATION FOR PR")
        print("=" * 50)
        
        speed_ratio = performance.get('speed_ratio', 1.0)
        completeness_improvement = performance.get('completeness_improvement', 0)
        recall_improvement = search_quality.get('recall_improvement_at_5', 0)
        
        if speed_ratio > 2.0 and (completeness_improvement < 0.1 and recall_improvement < 0.05):
            print("❌ AST chunking shows performance overhead without significant quality gains")
            print("   Recommendation: Continue with naive chunking for most use cases")
        elif completeness_improvement > 0.1 or recall_improvement > 0.05:
            print("✅ AST chunking provides measurable quality improvements")
            print("   Recommendation: Use AST chunking for code-heavy repositories")
            if speed_ratio > 2.0:
                print("   Note: Accept performance trade-off for better code understanding")
        else:
            print("⚖️ Mixed results - quality vs performance trade-offs")
            print("   Recommendation: Implement both with user/use-case selection")
            
        return 0
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())