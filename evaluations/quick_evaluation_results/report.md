# AST-Aware vs Naive Chunking Evaluation Report

**Generated:** 2025-08-16 20:16:49
**Version:** 1.0

## Individual analysis

### 1. benchmarks/ast_vs_naive_evaluation.py

```
============================================================
EVALUATION SUMMARY
============================================================
AST Chunking:   135 chunks in 3.77s
Naive Chunking: 142 chunks in 0.14s
Speed Ratio:    0.04x (AST vs Naive)
AST Completeness:   0.50
Naive Completeness: 0.48
Improvement:        +0.03
```

### 2. benchmarks/run_quick_evaluation.py

```
==================================================
📈 QUICK RESULTS SUMMARY
==================================================

🚀 Performance Metrics:
  Speed Ratio (AST/Naive):     0.04x
  Memory Ratio (AST/Naive):    0.00x
  Completeness Improvement:    +2.9%
  Coherence Improvement:       +13.4%
  AST Chunks Generated:        135
  Naive Chunks Generated:      142

🔍 Search Quality Metrics:
  Recall@5 Improvement:       +0.0%
  Precision@5 Improvement:    +0.0%
  F1@5 Improvement:           +0.0%
  Overall Relevance Boost:    -41.1%

🎯 Overall Recommendation: ✅ RECOMMEND AST CHUNKING

💡 Key Findings:
  • AST chunking is 26.4x faster than naive chunking
  • AST chunking improves chunk coherence by 13.4%
```

### 3. benchmarks/search_quality_evaluator.py

```
============================================================
SEARCH QUALITY EVALUATION SUMMARY
============================================================

Recall@5:
  AST Chunking:   0.958
  Naive Chunking: 0.958
  Improvement:    +0.000

Precision@5:
  AST Chunking:   1.000
  Naive Chunking: 1.000
  Improvement:    +0.000

F1@5:
  AST Chunking:   0.977
  Naive Chunking: 0.977
  Improvement:    +0.000
```
