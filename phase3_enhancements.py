#!/usr/bin/env python3
"""
phase3_enhancements.py - Phase 3 LLM Classifier Enhancements

Implements:
1. DynamicFewShotSelector - Semantic similarity-based example selection
2. ClassifierCalibrator - Threshold adjustment based on observed FPR
3. ClassificationMetrics - Performance tracking (latency, entropy, invalid_rate)

These components enhance classifier_v2.py according to Phase 3 requirements.
"""

import json
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ClassificationMetrics:
    """
    Track classification performance metrics across a batch.
    
    Metrics:
    - latencies: Response times for each classification
    - invalid_count: Number of failed/invalid classifications
    - label_distribution: Count of each severity_bucket
    - total_processed: Total findings processed
    """
    latencies: List[float] = field(default_factory=list)
    invalid_count: int = 0
    label_distribution: Dict[str, int] = field(default_factory=dict)
    total_processed: int = 0
    
    def add_classification(self, latency: float, label: Optional[str] = None, is_valid: bool = True) -> None:
        """
        Record a single classification result.
        
        Args:
            latency: Time taken for classification (seconds)
            label: Severity bucket label (if classification succeeded)
            is_valid: Whether classification was valid JSON with required fields
        """
        self.latencies.append(latency)
        self.total_processed += 1
        
        if not is_valid:
            self.invalid_count += 1
        elif label:
            self.label_distribution[label] = self.label_distribution.get(label, 0) + 1
    
    def calculate_tp95(self) -> float:
        """Calculate 95th percentile latency (tail latency)."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(0.95 * len(sorted_latencies))
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]
    
    def calculate_entropy(self) -> float:
        """
        Calculate Shannon entropy of label distribution.
        
        High entropy = diverse labels (good for balanced dataset)
        Low entropy = concentrated labels (may indicate bias)
        
        Returns:
            Entropy in bits (0 to log2(num_labels))
        """
        if not self.label_distribution or self.total_processed == 0:
            return 0.0
        
        total = sum(self.label_distribution.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in self.label_distribution.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def calculate_invalid_rate(self) -> float:
        """Calculate percentage of invalid/failed classifications."""
        if self.total_processed == 0:
            return 0.0
        return (self.invalid_count / self.total_processed) * 100.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate summary dictionary with all metrics."""
        tp95 = self.calculate_tp95()
        entropy = self.calculate_entropy()
        invalid_rate = self.calculate_invalid_rate()
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
        
        return {
            "total_processed": self.total_processed,
            "avg_latency_seconds": round(avg_latency, 3),
            "tp95_latency_seconds": round(tp95, 3),
            "invalid_rate_percent": round(invalid_rate, 2),
            "label_entropy_bits": round(entropy, 3),
            "label_distribution": self.label_distribution,
            "invalid_count": self.invalid_count
        }
    
    def print_summary(self) -> None:
        """Print formatted metrics summary to stdout."""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("CLASSIFICATION PERFORMANCE METRICS")
        print("=" * 70)
        print(f"Total Processed:      {summary['total_processed']}")
        print(f"Invalid Count:        {summary['invalid_count']} ({summary['invalid_rate_percent']}%)")
        print(f"Avg Latency:          {summary['avg_latency_seconds']}s")
        print(f"P95 Latency:          {summary['tp95_latency_seconds']}s")
        print(f"Label Entropy:        {summary['label_entropy_bits']} bits")
        print("\nLabel Distribution:")
        for label, count in sorted(summary['label_distribution'].items(), key=lambda x: -x[1]):
            percentage = (count / summary['total_processed']) * 100
            print(f"  {label:12s}: {count:3d} ({percentage:5.1f}%)")
        print("=" * 70 + "\n")


class ClassifierCalibrator:
    """
    Adjust classification thresholds based on observed false positive rate.
    
    Formula: θ_adjusted = θ_base + α·(FPR_target - FPR_observed)
    
    Where:
    - θ_base: Base confidence threshold (default 0.5)
    - α: Learning rate (default 0.1)
    - FPR_target: Target false positive rate (default 0.05 = 5%)
    - FPR_observed: Measured FPR from ground truth
    """
    
    def __init__(self, 
                 base_threshold: float = 0.5,
                 learning_rate: float = 0.1,
                 target_fpr: float = 0.05,
                 save_path: Optional[Path] = None):
        """
        Initialize calibrator.
        
        Args:
            base_threshold: Initial confidence threshold
            learning_rate: Adjustment step size (α)
            target_fpr: Target false positive rate
            save_path: Path to save/load calibration state
        """
        self.base_threshold = base_threshold
        self.learning_rate = learning_rate
        self.target_fpr = target_fpr
        self.adjusted_threshold = base_threshold
        self.calibration_history: List[Dict[str, float]] = []
        self.save_path = save_path or Path("calibration_state.json")
        
        # Try to load existing calibration
        if self.save_path.exists():
            self.load()
    
    def update_threshold(self, predictions: List[bool], ground_truth: List[bool]) -> float:
        """
        Update threshold based on observed FPR.
        
        Args:
            predictions: List of automation_candidate predictions (True/False)
            ground_truth: List of ground truth labels (True/False)
            
        Returns:
            Updated threshold value
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("predictions and ground_truth must have same length")
        
        if len(predictions) == 0:
            return self.adjusted_threshold
        
        # Calculate observed FPR
        fp = sum(1 for pred, truth in zip(predictions, ground_truth) 
                if pred and not truth)
        tn = sum(1 for pred, truth in zip(predictions, ground_truth)
                if not pred and not truth)
        
        if fp + tn == 0:
            observed_fpr = 0.0  # No negative cases
        else:
            observed_fpr = fp / (fp + tn)
        
        # Adjust threshold using calibration formula
        delta = self.learning_rate * (self.target_fpr - observed_fpr)
        self.adjusted_threshold = self.base_threshold + delta
        
        # Clamp to valid range [0.0, 1.0]
        self.adjusted_threshold = max(0.0, min(1.0, self.adjusted_threshold))
        
        # Record history
        self.calibration_history.append({
            "observed_fpr": observed_fpr,
            "threshold": self.adjusted_threshold,
            "delta": delta
        })
        
        # Auto-save
        self.save()
        
        return self.adjusted_threshold
    
    def apply_threshold(self, llm_confidence: float) -> bool:
        """
        Apply calibrated threshold to LLM confidence score.
        
        Args:
            llm_confidence: Confidence from LLM (0.0-1.0)
            
        Returns:
            True if confidence exceeds adjusted threshold
        """
        return llm_confidence >= self.adjusted_threshold
    
    def get_state(self) -> Dict[str, Any]:
        """Get current calibration state as dictionary."""
        return {
            "base_threshold": self.base_threshold,
            "adjusted_threshold": self.adjusted_threshold,
            "learning_rate": self.learning_rate,
            "target_fpr": self.target_fpr,
            "calibration_history": self.calibration_history
        }
    
    def save(self) -> None:
        """Save calibration state to disk."""
        with open(self.save_path, 'w') as f:
            json.dump(self.get_state(), f, indent=2)
    
    def load(self) -> None:
        """Load calibration state from disk."""
        if not self.save_path.exists():
            return
        
        with open(self.save_path, 'r') as f:
            state = json.load(f)
        
        self.base_threshold = state.get("base_threshold", self.base_threshold)
        self.adjusted_threshold = state.get("adjusted_threshold", self.base_threshold)
        self.learning_rate = state.get("learning_rate", self.learning_rate)
        self.target_fpr = state.get("target_fpr", self.target_fpr)
        self.calibration_history = state.get("calibration_history", [])
    
    def reset(self) -> None:
        """Reset calibration to base threshold."""
        self.adjusted_threshold = self.base_threshold
        self.calibration_history = []
        self.save()


class DynamicFewShotSelector:
    """
    Select relevant few-shot examples based on semantic similarity.
    
    Uses sentence-transformers to embed vulnerability descriptions and
    find the k most similar examples from a labeled dataset.
    """
    
    def __init__(self, examples_path: Path = Path("examples.json"),
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize few-shot selector.
        
        Args:
            examples_path: Path to examples.json with labeled examples
            model_name: Sentence-transformers model name
        """
        self.examples_path = examples_path
        self.model_name = model_name
        self.examples: List[Dict[str, Any]] = []
        self.embeddings: Optional[Any] = None
        self.model: Optional[Any] = None
        
        # Load examples
        if not examples_path.exists():
            raise FileNotFoundError(f"Examples file not found: {examples_path}")
        
        with open(examples_path, 'r') as f:
            self.examples = json.load(f)
        
        if not self.examples:
            raise ValueError("Examples file is empty")
        
        # Lazy load model and embeddings (only when needed)
        self._initialized = False
    
    def _initialize_model(self) -> None:
        """Lazy initialization of sentence-transformers model."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
        
        # Load model
        self.model = SentenceTransformer(self.model_name)
        
        # Pre-compute embeddings for all examples
        descriptions = [ex['description'] for ex in self.examples]
        self.embeddings = self.model.encode(descriptions, convert_to_tensor=True)
        
        self._initialized = True
    
    def select_examples(self, description: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Select k most similar examples to given description.
        
        Args:
            description: Vulnerability description to match
            k: Number of examples to return
            
        Returns:
            List of k example dictionaries with 'description' and 'labels' keys
        """
        self._initialize_model()
        
        if not description or not description.strip():
            # Return random examples if no description
            return self.examples[:k]
        
        # Encode query description
        query_embedding = self.model.encode(description, convert_to_tensor=True)
        
        # Compute cosine similarities
        from sentence_transformers import util
        similarities = util.cos_sim(query_embedding, self.embeddings)[0]
        
        # Get top-k indices
        top_k_indices = similarities.argsort(descending=True)[:k].tolist()
        
        # Return corresponding examples
        return [self.examples[i] for i in top_k_indices]
    
    def format_examples_for_prompt(self, examples: List[Dict[str, Any]]) -> str:
        """
        Format examples as text for inclusion in LLM prompt.
        
        Args:
            examples: List of example dictionaries
            
        Returns:
            Formatted string for prompt
        """
        if not examples:
            return ""
        
        formatted = "EXAMPLE CLASSIFICATIONS:\n\n"
        
        for i, ex in enumerate(examples, 1):
            labels = ex.get('labels', {})
            formatted += f"Example {i}:\n"
            formatted += f"Description: {ex.get('description', 'N/A')}\n"
            formatted += f"Service: {ex.get('service', 'N/A')} | Port: {ex.get('port', 0)} | CVSS: {ex.get('cvss', 0.0)}\n"
            formatted += f"Classification:\n"
            formatted += json.dumps(labels, indent=2)
            formatted += "\n\n"
        
        formatted += "Now classify the following finding:\n\n"
        return formatted


# ============================================================================
# Integration Test / Demo
# ============================================================================

def test_few_shot_selector():
    """Test DynamicFewShotSelector with sample descriptions."""
    print("[*] Testing DynamicFewShotSelector\n")
    
    selector = DynamicFewShotSelector()
    
    test_cases = [
        "Remote code execution in Apache Tomcat via deserialization",
        "SSL certificate validation issues in mobile application",
        "SQL injection in user login form"
    ]
    
    for description in test_cases:
        print(f"Query: {description}")
        examples = selector.select_examples(description, k=3)
        print(f"Selected {len(examples)} examples:")
        for ex in examples:
            print(f"  - {ex['title']} (CVSS: {ex['cvss']})")
        print()


def test_calibrator():
    """Test ClassifierCalibrator with synthetic data."""
    print("[*] Testing ClassifierCalibrator\n")
    
    calibrator = ClassifierCalibrator(save_path=Path("test_calibration.json"))
    
    # Simulate predictions with high FPR
    predictions = [True] * 20 + [False] * 80  # 20% predicted positive
    ground_truth = [True] * 10 + [False] * 90  # 10% actually positive
    # FPR = 10 / 90 = 11.1% (high)
    
    print(f"Initial threshold: {calibrator.adjusted_threshold}")
    new_threshold = calibrator.update_threshold(predictions, ground_truth)
    print(f"Updated threshold: {new_threshold}")
    print(f"Target FPR: {calibrator.target_fpr * 100}%")
    print()


def test_metrics():
    """Test ClassificationMetrics with sample data."""
    print("[*] Testing ClassificationMetrics\n")
    
    metrics = ClassificationMetrics()
    
    # Simulate classifications
    metrics.add_classification(1.2, "Critical", True)
    metrics.add_classification(0.9, "High", True)
    metrics.add_classification(1.5, "Medium", True)
    metrics.add_classification(2.1, "High", True)
    metrics.add_classification(0.8, None, False)  # Invalid
    metrics.add_classification(1.1, "Low", True)
    metrics.add_classification(3.2, "Critical", True)
    metrics.add_classification(1.0, "Medium", True)
    
    metrics.print_summary()


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 3 ENHANCEMENTS TEST SUITE")
    print("=" * 70)
    print()
    
    try:
        test_metrics()
        test_calibrator()
        test_few_shot_selector()
        print("[+] All tests completed successfully!")
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
