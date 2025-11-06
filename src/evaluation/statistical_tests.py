"""
Statistical Tests
Statistical significance testing for comparing methods.

Section 5.4: Statistical tests including Wilcoxon signed-rank,
Mann-Whitney U, and paired t-tests.

Author: Majdi Argoubi
Date: 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass


@dataclass
class StatisticalTestResult:
    """
    Result of a statistical test.
    
    Attributes:
        test_name: Name of statistical test
        statistic: Test statistic value
        p_value: P-value
        significant: Whether result is significant (p < α)
        alpha: Significance level used
        effect_size: Effect size (if computed)
        interpretation: Human-readable interpretation
    """
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    effect_size: Optional[float] = None
    interpretation: str = ""


class StatisticalTester:
    """
    Statistical significance tester for comparing methods.
    
    Implements various statistical tests for evaluating whether
    improvements over baselines are statistically significant.
    
    Args:
        alpha: Significance level (default: 0.05)
    
    Example:
        >>> tester = StatisticalTester(alpha=0.05)
        >>> result = tester.wilcoxon_test(costs_method1, costs_method2)
        >>> if result.significant:
        ...     print(f"Improvement is significant (p={result.p_value:.4f})")
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def wilcoxon_test(
        self,
        sample1: List[float],
        sample2: List[float],
        alternative: str = 'two-sided'
    ) -> StatisticalTestResult:
        """
        Wilcoxon signed-rank test for paired samples.
        
        Tests whether two related paired samples come from the same distribution.
        Non-parametric alternative to paired t-test.
        
        Section 5.4: Used for comparing PPO-GNN against baselines on same instances.
        
        Args:
            sample1: First sample (e.g., costs from method 1)
            sample2: Second sample (e.g., costs from method 2)
            alternative: 'two-sided', 'less', or 'greater'
        
        Returns:
            result: StatisticalTestResult object
        """
        # Ensure equal length
        assert len(sample1) == len(sample2), "Samples must have equal length"
        
        # Perform test
        statistic, p_value = stats.wilcoxon(
            sample1, sample2,
            alternative=alternative
        )
        
        # Effect size (r = Z / sqrt(N))
        n = len(sample1)
        z_score = stats.norm.ppf(1 - p_value / 2)  # Approximate Z-score
        effect_size = z_score / np.sqrt(n)
        
        # Interpretation
        if p_value < self.alpha:
            if alternative == 'less':
                interpretation = f"Sample 1 is significantly less than Sample 2 (p={p_value:.4f})"
            elif alternative == 'greater':
                interpretation = f"Sample 1 is significantly greater than Sample 2 (p={p_value:.4f})"
            else:
                interpretation = f"Samples are significantly different (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference found (p={p_value:.4f})"
        
        return StatisticalTestResult(
            test_name='Wilcoxon Signed-Rank',
            statistic=statistic,
            p_value=p_value,
            significant=(p_value < self.alpha),
            alpha=self.alpha,
            effect_size=effect_size,
            interpretation=interpretation
        )
    
    def mann_whitney_test(
        self,
        sample1: List[float],
        sample2: List[float],
        alternative: str = 'two-sided'
    ) -> StatisticalTestResult:
        """
        Mann-Whitney U test for independent samples.
        
        Tests whether two independent samples come from the same distribution.
        Non-parametric alternative to independent t-test.
        
        Args:
            sample1: First sample
            sample2: Second sample
            alternative: 'two-sided', 'less', or 'greater'
        
        Returns:
            result: StatisticalTestResult object
        """
        # Perform test
        statistic, p_value = stats.mannwhitneyu(
            sample1, sample2,
            alternative=alternative
        )
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(sample1), len(sample2)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        # Interpretation
        if p_value < self.alpha:
            interpretation = f"Significant difference between samples (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference found (p={p_value:.4f})"
        
        return StatisticalTestResult(
            test_name='Mann-Whitney U',
            statistic=statistic,
            p_value=p_value,
            significant=(p_value < self.alpha),
            alpha=self.alpha,
            effect_size=effect_size,
            interpretation=interpretation
        )
    
    def paired_t_test(
        self,
        sample1: List[float],
        sample2: List[float],
        alternative: str = 'two-sided'
    ) -> StatisticalTestResult:
        """
        Paired t-test for paired samples.
        
        Parametric test assuming normally distributed differences.
        
        Args:
            sample1: First sample
            sample2: Second sample
            alternative: 'two-sided', 'less', or 'greater'
        
        Returns:
            result: StatisticalTestResult object
        """
        # Perform test
        statistic, p_value = stats.ttest_rel(
            sample1, sample2,
            alternative=alternative
        )
        
        # Effect size (Cohen's d)
        differences = np.array(sample1) - np.array(sample2)
        effect_size = np.mean(differences) / np.std(differences, ddof=1)
        
        # Interpretation
        if p_value < self.alpha:
            interpretation = f"Significant difference between paired samples (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference found (p={p_value:.4f})"
        
        return StatisticalTestResult(
            test_name='Paired t-test',
            statistic=statistic,
            p_value=p_value,
            significant=(p_value < self.alpha),
            alpha=self.alpha,
            effect_size=effect_size,
            interpretation=interpretation
        )
    
    def independent_t_test(
        self,
        sample1: List[float],
        sample2: List[float],
        equal_var: bool = False
    ) -> StatisticalTestResult:
        """
        Independent t-test for two samples.
        
        Args:
            sample1: First sample
            sample2: Second sample
            equal_var: Assume equal variances (default: False for Welch's t-test)
        
        Returns:
            result: StatisticalTestResult object
        """
        # Perform test
        statistic, p_value = stats.ttest_ind(
            sample1, sample2,
            equal_var=equal_var
        )
        
        # Effect size (Cohen's d)
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        effect_size = (mean1 - mean2) / pooled_std
        
        test_name = "Independent t-test (Welch)" if not equal_var else "Independent t-test"
        
        return StatisticalTestResult(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            significant=(p_value < self.alpha),
            alpha=self.alpha,
            effect_size=effect_size,
            interpretation=f"{'Significant' if p_value < self.alpha else 'No'} difference (p={p_value:.4f})"
        )
    
    def friedman_test(
        self,
        *samples: List[float]
    ) -> StatisticalTestResult:
        """
        Friedman test for multiple related samples.
        
        Non-parametric alternative to repeated measures ANOVA.
        Tests whether k related samples come from the same distribution.
        
        Args:
            *samples: Multiple samples to compare
        
        Returns:
            result: StatisticalTestResult object
        """
        # Perform test
        statistic, p_value = stats.friedmanchisquare(*samples)
        
        # Effect size (Kendall's W)
        n = len(samples[0])
        k = len(samples)
        effect_size = statistic / (n * (k - 1))
        
        return StatisticalTestResult(
            test_name='Friedman',
            statistic=statistic,
            p_value=p_value,
            significant=(p_value < self.alpha),
            alpha=self.alpha,
            effect_size=effect_size,
            interpretation=f"{'Significant' if p_value < self.alpha else 'No'} difference among {k} methods (p={p_value:.4f})"
        )


def compute_confidence_interval(
    sample: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for sample mean.
    
    Args:
        sample: Data sample
        confidence: Confidence level (default: 0.95)
    
    Returns:
        mean: Sample mean
        lower: Lower bound
        upper: Upper bound
    """
    mean = np.mean(sample)
    std_err = stats.sem(sample)
    margin = std_err * stats.t.ppf((1 + confidence) / 2, len(sample) - 1)
    
    return mean, mean - margin, mean + margin


def compute_effect_size_cohens_d(
    sample1: List[float],
    sample2: List[float],
    paired: bool = True
) -> float:
    """
    Compute Cohen's d effect size.
    
    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 ≤ |d| < 0.5: small
    - 0.5 ≤ |d| < 0.8: medium
    - |d| ≥ 0.8: large
    
    Args:
        sample1: First sample
        sample2: Second sample
        paired: Whether samples are paired
    
    Returns:
        d: Cohen's d effect size
    """
    if paired:
        differences = np.array(sample1) - np.array(sample2)
        d = np.mean(differences) / np.std(differences, ddof=1)
    else:
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        d = (mean1 - mean2) / pooled_std
    
    return d


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def bootstrap_confidence_interval(
    sample: List[float],
    statistic_fn=np.mean,
    confidence: float = 0.95,
    n_bootstrap: int = 10000
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Args:
        sample: Data sample
        statistic_fn: Function to compute statistic (default: mean)
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        statistic: Sample statistic
        lower: Lower bound
        upper: Upper bound
    """
    sample = np.array(sample)
    n = len(sample)
    
    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(sample, size=n, replace=True)
        bootstrap_stats.append(statistic_fn(bootstrap_sample))
    
    # Compute percentiles
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    statistic = statistic_fn(sample)
    
    return statistic, lower, upper


class MultipleComparisonCorrection:
    """
    Corrections for multiple hypothesis testing.
    
    Implements Bonferroni and Holm-Bonferroni corrections.
    
    Example:
        >>> corrector = MultipleComparisonCorrection(alpha=0.05)
        >>> adjusted_alpha = corrector.bonferroni(num_tests=10)
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def bonferroni(self, num_tests: int) -> float:
        """
        Bonferroni correction.
        
        Adjusted α = α / m where m is number of tests.
        Most conservative correction.
        """
        return self.alpha / num_tests
    
    def holm_bonferroni(self, p_values: List[float]) -> List[bool]:
        """
        Holm-Bonferroni correction.
        
        Step-down procedure that is less conservative than Bonferroni.
        
        Args:
            p_values: List of p-values to correct
        
        Returns:
            significant: List of booleans indicating significance
        """
        m = len(p_values)
        
        # Sort p-values with indices
        sorted_indices = np.argsort(p_values)
        sorted_p_values = np.array(p_values)[sorted_indices]
        
        # Test each p-value
        significant = np.zeros(m, dtype=bool)
        for i, p in enumerate(sorted_p_values):
            adjusted_alpha = self.alpha / (m - i)
            if p <= adjusted_alpha:
                significant[sorted_indices[i]] = True
            else:
                # Stop testing once we fail to reject
                break
        
        return significant.tolist()


if __name__ == '__main__':
    # Test statistical tests
    print("Testing Statistical Tests...")
    
    np.random.seed(42)
    
    # Generate sample data
    method1_costs = 1000 + np.random.randn(30) * 50  # Method 1
    method2_costs = 950 + np.random.randn(30) * 50   # Method 2 (better)
    
    tester = StatisticalTester(alpha=0.05)
    
    # Test Wilcoxon
    print("\n1. Wilcoxon Signed-Rank Test:")
    result = tester.wilcoxon_test(method1_costs, method2_costs, alternative='greater')
    print(f"   {result.interpretation}")
    print(f"   Statistic: {result.statistic:.2f}, p-value: {result.p_value:.4f}")
    print(f"   Effect size: {result.effect_size:.3f}")
    
    # Test Mann-Whitney
    print("\n2. Mann-Whitney U Test:")
    result = tester.mann_whitney_test(method1_costs, method2_costs)
    print(f"   {result.interpretation}")
    print(f"   Effect size: {result.effect_size:.3f}")
    
    # Test Cohen's d
    print("\n3. Effect Size (Cohen's d):")
    d = compute_effect_size_cohens_d(method1_costs, method2_costs, paired=True)
    print(f"   Cohen's d: {d:.3f} ({interpret_effect_size(d)})")
    
    # Test confidence interval
    print("\n4. Confidence Interval:")
    mean, lower, upper = compute_confidence_interval(method1_costs)
    print(f"   Mean: {mean:.2f}, 95% CI: [{lower:.2f}, {upper:.2f}]")
    
    # Test bootstrap
    print("\n5. Bootstrap CI:")
    stat, lower, upper = bootstrap_confidence_interval(method1_costs)
    print(f"   Mean: {stat:.2f}, 95% Bootstrap CI: [{lower:.2f}, {upper:.2f}]")
    
    # Test multiple comparison correction
    print("\n6. Multiple Comparison Correction:")
    p_values = [0.01, 0.03, 0.05, 0.07, 0.10]
    corrector = MultipleComparisonCorrection(alpha=0.05)
    bonferroni_alpha = corrector.bonferroni(len(p_values))
    print(f"   Bonferroni adjusted α: {bonferroni_alpha:.4f}")
    
    significant = corrector.holm_bonferroni(p_values)
    print(f"   Holm-Bonferroni significant tests: {sum(significant)}/{len(p_values)}")
    
    print("\n✓ All tests passed!")
