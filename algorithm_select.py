from __future__ import annotations

import time
import random
import math
from typing import Protocol, TypeVar, Callable, Iterable, Any, Generic, cast
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import threading

# Generic type variables for input and output
I = TypeVar("I", contravariant=True)
O = TypeVar("O", covariant=True)


class SelectionStrategy(ABC):
    """Abstract base class for algorithm selection strategies"""
    
    @abstractmethod
    def select(self, selector: AdaptiveSelector) -> int:
        """Select which algorithm to use based on current statistics"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the strategy for display purposes"""
        pass


@dataclass
class EpsilonGreedy(SelectionStrategy):
    """ε-greedy selection strategy"""
    epsilon: float = 0.1
    
    def select(self, selector: AdaptiveSelector) -> int:
        """ε-greedy selection: exploit best, explore randomly with probability ε"""
        if random.random() < self.epsilon or selector.total_calls < len(selector.callables):
            # Explore: choose randomly, but ensure each algorithm gets tried at least once
            if selector.total_calls < len(selector.callables):
                # Initially, try each algorithm once
                untried = [
                    i for i, stat in enumerate(selector.stats) if stat.total_calls == 0
                ]
                if untried:
                    return random.choice(untried)
            return random.randint(0, len(selector.callables) - 1)
        else:
            # Exploit: choose the algorithm with best recent performance
            best_idx = min(
                range(len(selector.callables)),
                key=lambda i: selector.stats[i].recent_average_time,
            )
            return best_idx
    
    @property
    def name(self) -> str:
        return f"epsilon_greedy(ε={self.epsilon})"


@dataclass
class UCB(SelectionStrategy):
    """Upper Confidence Bound selection strategy"""
    c: float = 1.0
    
    def select(self, selector: AdaptiveSelector) -> int:
        """Upper Confidence Bound selection"""
        if selector.total_calls < len(selector.callables):
            # Initially try each algorithm once
            untried = [i for i, stat in enumerate(selector.stats) if stat.total_calls == 0]
            if untried:
                return random.choice(untried)

        # UCB formula: mean + c * sqrt(ln(total_trials) / trials_for_arm)
        ucb_values = []
        for i, stat in enumerate(selector.stats):
            if stat.total_calls == 0:
                ucb_values.append(float("-inf"))  # Will be selected first
            else:
                # For timing, we want LOWER values, so we negate
                mean_time = stat.recent_average_time
                confidence = self.c * math.sqrt(
                    math.log(selector.total_calls) / stat.total_calls
                )
                # UCB for minimization: choose the one with lowest lower confidence bound
                ucb_values.append(mean_time - confidence)

        return min(range(len(ucb_values)), key=lambda i: ucb_values[i])
    
    @property
    def name(self) -> str:
        return f"ucb(c={self.c})"


@dataclass
class SlidingWindow(SelectionStrategy):
    """Sliding window selection strategy"""
    window_size: int = 50
    epsilon: float = 0.1
    
    def select(self, selector: AdaptiveSelector) -> int:
        """Sliding window: choose based on recent performance in a window"""
        if len(selector.recent_choices) < self.window_size // 2:
            # Not enough data, choose randomly
            return random.randint(0, len(selector.callables) - 1)

        # Count recent performance
        recent_performance = defaultdict(list)
        for choice_idx, exec_time in selector.recent_choices:
            recent_performance[choice_idx].append(exec_time)

        # Find the algorithm with best recent average
        best_avg = float("inf")
        best_idx = 0

        for i in range(len(selector.callables)):
            if i in recent_performance:
                avg_time = sum(recent_performance[i]) / len(recent_performance[i])
                if avg_time < best_avg:
                    best_avg = avg_time
                    best_idx = i

        # Add some exploration
        if random.random() < self.epsilon:
            return random.randint(0, len(selector.callables) - 1)

        return best_idx
    
    @property
    def name(self) -> str:
        return f"sliding_window(size={self.window_size}, ε={self.epsilon})"


@dataclass
class ThompsonSampling(SelectionStrategy):
    """Thompson Sampling selection strategy"""
    
    def select(self, selector: 'AdaptiveSelector') -> int:
        """Thompson Sampling: sample from posterior distributions"""
        # Sample from Beta distributions for each algorithm
        samples = []

        # Define a reasonable threshold for "success" (e.g., median of all recent times)
        all_recent_times = []
        for stat in selector.stats:
            all_recent_times.extend(stat.recent_times)

        if all_recent_times:
            threshold = sorted(all_recent_times)[len(all_recent_times) // 2]
        else:
            threshold = 1.0  # Default threshold

        for stat in selector.stats:
            # Beta distribution parameters
            alpha = stat.successes
            beta = stat.failures

            # Sample probability of being "fast" (execution time <= threshold)
            prob_fast = random.betavariate(alpha, beta)
            samples.append(prob_fast)

        # Choose the algorithm with highest sampled probability of being fast
        return max(range(len(samples)), key=lambda i: samples[i])
    
    @property
    def name(self) -> str:
        return "thompson_sampling"


@dataclass
class PerformanceStats:
    """Track performance statistics for an algorithm"""

    total_calls: int = 0
    total_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))

    # For Thompson Sampling (Beta distribution parameters)
    successes: float = 1.0  # Start with weak priors
    failures: float = 1.0

    def add_measurement(self, execution_time: float, threshold: float | None = None):
        """Add a new timing measurement"""
        self.total_calls += 1
        self.total_time += execution_time
        self.recent_times.append(execution_time)

        # For Thompson Sampling: consider "success" if faster than threshold
        if threshold is not None:
            if execution_time <= threshold:
                self.successes += 1
            else:
                self.failures += 1

    @property
    def average_time(self) -> float:
        """Overall average execution time"""
        return (
            self.total_time / self.total_calls if self.total_calls > 0 else float("inf")
        )

    @property
    def recent_average_time(self) -> float:
        """Recent average execution time"""
        if not self.recent_times:
            return float("inf")
        return sum(self.recent_times) / len(self.recent_times)

    @property
    def confidence_width(self) -> float:
        """UCB confidence interval width"""
        if self.total_calls == 0:
            return float("inf")
        # Simple confidence based on number of samples
        return math.sqrt(2 * math.log(self.total_calls + 1) / self.total_calls)


@dataclass
class AlgorithmStatistics:
    """Statistics for a single algorithm"""
    algorithm_name: str
    total_calls: int
    average_time: float
    recent_average_time: float
    success_rate: float
    
    def __str__(self) -> str:
        """Nicely formatted string representation"""
        return (
            f"{self.algorithm_name}: "
            f"{self.total_calls} calls, "
            f"avg: {self.average_time:.4f}s, "
            f"recent: {self.recent_average_time:.4f}s, "
            f"success: {self.success_rate:.1%}"
        )


@dataclass
class SelectionStatistics:
    """Complete statistics for the adaptive selector"""
    total_calls: int
    algorithm_stats: list[AlgorithmStatistics]
    strategy_name: str
    
    @property
    def best_algorithm_name(self) -> str:
        """Name of the algorithm with the best recent average time"""
        if not self.algorithm_stats:
            return "None"
        return min(self.algorithm_stats, key=lambda x: x.recent_average_time).algorithm_name
    
    @property
    def most_used_algorithm_name(self) -> str:
        """Name of the algorithm that has been called most often"""
        if not self.algorithm_stats:
            return "None"
        return max(self.algorithm_stats, key=lambda x: x.total_calls).algorithm_name
    
    def __str__(self) -> str:
        """Nicely formatted string representation"""
        lines = [
            f"=== Adaptive Selection Statistics ===",
            f"Strategy: {self.strategy_name}",
            f"Total calls: {self.total_calls}",
        ]
        
        if self.algorithm_stats:
            best_algo = min(self.algorithm_stats, key=lambda x: x.recent_average_time)
            lines.append(f"Best performing algorithm: {best_algo.algorithm_name} (recent avg: {best_algo.recent_average_time:.4f}s)")
            lines.append(f"Most used algorithm: {self.most_used_algorithm_name}")
        
        lines.extend(["", "Algorithm Performance:"])
        
        # Sort algorithms by recent average time for clearer display
        sorted_stats = sorted(self.algorithm_stats, key=lambda x: x.recent_average_time)
        for stat in sorted_stats:
            lines.append(f"  {stat}")
        
        return "\n".join(lines)


@dataclass
class AdaptiveSelector(Generic[I, O]):
    """Core class implementing the adaptive selection strategies"""

    callables: list[Callable[[I], O]]
    strategy: SelectionStrategy

    # Performance tracking (initialized in __post_init__)
    stats: list[PerformanceStats] = field(init=False)
    total_calls: int = field(default=0, init=False)
    lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    recent_choices: deque = field(init=False)

    def __post_init__(self):
        """Initialize fields that depend on other fields"""
        self.callables = list(self.callables)
        self.stats = [PerformanceStats() for _ in self.callables]
        # Set window size based on strategy if it's SlidingWindow
        window_size = self.strategy.window_size if isinstance(self.strategy, SlidingWindow) else 50
        self.recent_choices = deque(maxlen=window_size)

    def select_and_execute(self, *args: I) -> O:
        """Select an algorithm and execute it, tracking performance"""
        with self.lock:
            # Select which algorithm to use (polymorphically)
            chosen_idx = self.strategy.select(self)
            self.total_calls += 1

        # Execute the chosen algorithm and measure time
        start_time = time.perf_counter()
        try:
            result = self.callables[chosen_idx](*args)
            success = True
        except Exception as e:
            success = False
            raise e
        finally:
            execution_time = time.perf_counter() - start_time

            with self.lock:
                # Update statistics
                all_recent_times = []
                for stat in self.stats:
                    all_recent_times.extend(stat.recent_times)

                threshold = None
                if all_recent_times:
                    threshold = sorted(all_recent_times)[len(all_recent_times) // 2]

                self.stats[chosen_idx].add_measurement(execution_time, threshold)

                # For sliding window approach
                if isinstance(self.strategy, SlidingWindow):
                    self.recent_choices.append((chosen_idx, execution_time))

        return result

    def get_statistics(self) -> SelectionStatistics:
        """Get current performance statistics"""
        algorithm_stats = []
        for i, stat in enumerate(self.stats):
            algorithm_stats.append(AlgorithmStatistics(
                algorithm_name=self.callables[i].__name__,
                total_calls=stat.total_calls,
                average_time=stat.average_time,
                recent_average_time=stat.recent_average_time,
                success_rate=stat.successes / (stat.successes + stat.failures),
            ))
        
        return SelectionStatistics(
            total_calls=self.total_calls,
            algorithm_stats=algorithm_stats,
            strategy_name=self.strategy.name
        )

class AdaptiveFunction(Protocol, Generic[I, O]):
    def __call__(self, *args: I) -> O: ...
    def get_statistics(self) -> SelectionStatistics: ...

def select_fastest(
    callables: Iterable[Callable[[I], O]],
    strategy: SelectionStrategy,
) -> AdaptiveFunction[I, O]:
    """
    Create an adaptive function selector that learns which algorithm is fastest.

    Args:
        callables: Iterable of functions with identical signatures
        strategy: Selection strategy to use (instance of a SelectionStrategy subclass)

    Returns:
        A function with the same signature that adaptively selects the best algorithm
    """
    selector = AdaptiveSelector(
        callables=list(callables),
        strategy=strategy,
    )

    def adaptive_function(*args: I) -> O:
        return selector.select_and_execute(*args)

    # Add method to access statistics
    setattr(adaptive_function, 'get_statistics', selector.get_statistics)

    return cast(AdaptiveFunction[I, O], adaptive_function) 


# Example usage and testing
if __name__ == "__main__":
    # Example: Two sorting algorithms with different performance characteristics
    def quick_sort(arr):
        """Fast for random data, slow for already sorted"""
        time.sleep(0.0001 * len(arr) if sorted(arr) != arr else 0.001 * len(arr))
        return sorted(arr)

    def merge_sort(arr):
        """Consistent O(n log n) performance"""
        time.sleep(0.0005 * len(arr))
        return sorted(arr)

    strategies = [
        EpsilonGreedy(epsilon=0.1),
        UCB(c=1.0),
        SlidingWindow(window_size=50, epsilon=0.1),
        ThompsonSampling(),
    ]

    for strategy in strategies:
        adaptive_sort = select_fastest([quick_sort, merge_sort], strategy=strategy)

        # Test with different types of data
        print(f"{'*'*80}\n* Testing selection strategy {strategy.name}")

        print("random data (quick_sort should be faster):")
        for i in range(20):
            data = random.sample(range(100), 50)
            result = adaptive_sort(data)
            if i % 5 == 0:
                stats = adaptive_sort.get_statistics()
                print(f"Call {i}: Best algorithm: {stats.best_algorithm_name}, Total calls: {stats.total_calls}")

        print("\nsorted data (merge_sort should be faster):")
        for i in range(20):
            data = list(range(50))
            result = adaptive_sort(data)
            if i % 5 == 0:
                stats = adaptive_sort.get_statistics()
                print(f"Call {i}: Best algorithm: {stats.best_algorithm_name}, Total calls: {stats.total_calls}")

        print(f"\n{adaptive_sort.get_statistics()}")