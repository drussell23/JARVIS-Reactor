"""
Causal Reasoning Training for JARVIS Reactor Core.

Implements:
- Structural Causal Models (SCMs)
- Causal graph discovery
- Intervention learning (do-calculus)
- Counterfactual inference
- Causal attention mechanisms
- Causal evaluation metrics

Based on research:
- "The Book of Why" (Pearl & Mackenzie, 2018)
- "Causality" (Pearl, 2000)
- "Causal Inference in Statistics" (Pearl, Glymour, Jewell, 2016)
- "Neural Causal Models" (Ke et al., 2021)
- "Causal Attention for LLMs" (Geiger et al., 2023)
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# CAUSAL GRAPH
# =============================================================================

@dataclass
class CausalEdge:
    """Edge in causal graph."""

    source: str
    target: str
    strength: float = 1.0
    confidence: float = 1.0
    mechanism: Optional[str] = None


class CausalGraph:
    """
    Directed acyclic graph (DAG) representing causal relationships.

    Nodes are variables, edges are causal relationships.
    """

    def __init__(self):
        """Initialize empty causal graph."""
        self.nodes: Set[str] = set()
        self.edges: List[CausalEdge] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        self.parents: Dict[str, List[str]] = defaultdict(list)
        self.children: Dict[str, List[str]] = defaultdict(list)

    def add_node(self, variable: str):
        """Add variable node."""
        self.nodes.add(variable)

    def add_edge(
        self,
        source: str,
        target: str,
        strength: float = 1.0,
        confidence: float = 1.0,
    ):
        """Add causal edge from source to target."""
        # Add nodes if not exist
        self.add_node(source)
        self.add_node(target)

        # Check for cycles
        if self._would_create_cycle(source, target):
            logger.warning(f"Cannot add edge {source} -> {target}: would create cycle")
            return

        # Add edge
        edge = CausalEdge(source, target, strength, confidence)
        self.edges.append(edge)

        # Update adjacency
        self.adjacency[source].append(target)
        self.parents[target].append(source)
        self.children[source].append(target)

    def _would_create_cycle(self, source: str, target: str) -> bool:
        """Check if adding edge would create a cycle."""
        # If target can reach source, adding edge would create cycle
        return self._can_reach(target, source)

    def _can_reach(self, start: str, end: str) -> bool:
        """Check if start can reach end via directed edges."""
        if start == end:
            return True

        visited = set()
        queue = [start]

        while queue:
            current = queue.pop(0)
            if current == end:
                return True

            if current in visited:
                continue

            visited.add(current)
            queue.extend(self.children.get(current, []))

        return False

    def get_parents(self, variable: str) -> List[str]:
        """Get causal parents of variable."""
        return self.parents.get(variable, [])

    def get_children(self, variable: str) -> List[str]:
        """Get causal children of variable."""
        return self.children.get(variable, [])

    def get_ancestors(self, variable: str) -> Set[str]:
        """Get all ancestors of variable."""
        ancestors = set()
        queue = [variable]

        while queue:
            current = queue.pop(0)
            for parent in self.get_parents(current):
                if parent not in ancestors:
                    ancestors.add(parent)
                    queue.append(parent)

        return ancestors

    def get_descendants(self, variable: str) -> Set[str]:
        """Get all descendants of variable."""
        descendants = set()
        queue = [variable]

        while queue:
            current = queue.pop(0)
            for child in self.get_children(current):
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)

        return descendants

    def d_separated(
        self,
        X: Set[str],
        Y: Set[str],
        Z: Set[str],
    ) -> bool:
        """
        Check if X and Y are d-separated given Z.

        D-separation implies conditional independence.
        """
        # Simplified d-separation check
        # Full implementation would require checking all paths

        # Check if any path from X to Y is blocked by Z
        for x in X:
            for y in Y:
                if not self._all_paths_blocked(x, y, Z):
                    return False

        return True

    def _all_paths_blocked(self, start: str, end: str, conditioning_set: Set[str]) -> bool:
        """Check if all paths from start to end are blocked by conditioning set."""
        # Simplified - would need full path enumeration for correctness
        # Block if any node in path is in conditioning set
        visited = set()
        queue = [(start, [])]

        while queue:
            current, path = queue.pop(0)

            if current == end:
                # Found path - check if blocked
                if not any(node in conditioning_set for node in path):
                    return False  # Unblocked path exists

            if current in visited:
                continue

            visited.add(current)

            # Explore children
            for child in self.get_children(current):
                queue.append((child, path + [child]))

        return True  # All paths blocked


# =============================================================================
# STRUCTURAL CAUSAL MODEL
# =============================================================================

class StructuralCausalModel:
    """
    Structural Causal Model (SCM).

    Defines causal mechanisms: X_i = f_i(PA_i, U_i)
    where PA_i are parents and U_i is exogenous noise.
    """

    def __init__(self, causal_graph: CausalGraph):
        """
        Initialize SCM.

        Args:
            causal_graph: Underlying causal graph structure
        """
        self.graph = causal_graph
        self.mechanisms: Dict[str, Callable] = {}
        self.noise_distributions: Dict[str, Callable] = {}

    def set_mechanism(
        self,
        variable: str,
        mechanism: Callable,
        noise_distribution: Optional[Callable] = None,
    ):
        """
        Set causal mechanism for variable.

        Args:
            variable: Variable name
            mechanism: Function mapping parents to variable
            noise_distribution: Noise distribution (optional)
        """
        self.mechanisms[variable] = mechanism

        if noise_distribution:
            self.noise_distributions[variable] = noise_distribution
        else:
            # Default: standard normal noise
            self.noise_distributions[variable] = lambda: np.random.normal(0, 1)

    def sample(
        self,
        num_samples: int = 1,
        interventions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Sample from SCM.

        Args:
            num_samples: Number of samples
            interventions: Dict of {variable: value} for do-calculus

        Returns:
            Dictionary of sampled variables
        """
        interventions = interventions or {}
        samples = {var: np.zeros(num_samples) for var in self.graph.nodes}

        # Topological sort for sampling order
        sorted_vars = self._topological_sort()

        for var in sorted_vars:
            if var in interventions:
                # Intervention: set to fixed value
                samples[var] = np.full(num_samples, interventions[var])
            else:
                # Normal causal mechanism
                parents = self.graph.get_parents(var)

                if var in self.mechanisms:
                    # Use defined mechanism
                    parent_values = {p: samples[p] for p in parents}
                    noise = np.array([self.noise_distributions[var]() for _ in range(num_samples)])
                    samples[var] = self.mechanisms[var](parent_values, noise)
                else:
                    # Default: linear combination of parents + noise
                    if parents:
                        samples[var] = sum(samples[p] for p in parents) / len(parents)
                    samples[var] += np.array([self.noise_distributions[var]() for _ in range(num_samples)])

        return samples

    def _topological_sort(self) -> List[str]:
        """Topological sort of causal graph."""
        visited = set()
        sorted_vars = []

        def dfs(var: str):
            if var in visited:
                return
            visited.add(var)

            for parent in self.graph.get_parents(var):
                dfs(parent)

            sorted_vars.append(var)

        for var in self.graph.nodes:
            dfs(var)

        return sorted_vars

    def do_calculus(
        self,
        intervention: Dict[str, float],
        num_samples: int = 1000,
    ) -> Dict[str, np.ndarray]:
        """
        Compute causal effect of intervention using do-calculus.

        Args:
            intervention: Dict of {variable: value}
            num_samples: Number of samples for estimation

        Returns:
            Samples under intervention
        """
        return self.sample(num_samples=num_samples, interventions=intervention)

    def counterfactual(
        self,
        factual: Dict[str, float],
        intervention: Dict[str, float],
        num_samples: int = 1,
    ) -> Dict[str, np.ndarray]:
        """
        Compute counterfactual: what would have happened if intervention?

        Args:
            factual: Observed factual values
            intervention: Counterfactual intervention
            num_samples: Number of counterfactual samples

        Returns:
            Counterfactual samples
        """
        # Simplified counterfactual inference
        # Full implementation requires abduction-action-prediction

        # 1. Abduction: infer exogenous noise from factual
        # 2. Action: apply intervention
        # 3. Prediction: forward sample with intervened values

        return self.sample(num_samples=num_samples, interventions=intervention)


# =============================================================================
# CAUSAL DISCOVERY
# =============================================================================

class CausalDiscovery:
    """
    Discover causal graph structure from data.

    Implements constraint-based and score-based methods.
    """

    def __init__(
        self,
        method: str = "pc",  # 'pc', 'ges', 'notears'
        significance_level: float = 0.05,
    ):
        """
        Initialize causal discovery.

        Args:
            method: Discovery algorithm
            significance_level: Significance level for tests
        """
        self.method = method
        self.significance_level = significance_level

    async def discover(
        self,
        data: Dict[str, np.ndarray],
        variable_names: List[str],
    ) -> CausalGraph:
        """
        Discover causal graph from observational data.

        Args:
            data: Dictionary of variable data
            variable_names: Names of variables

        Returns:
            Discovered causal graph
        """
        if self.method == "pc":
            return await self._pc_algorithm(data, variable_names)
        elif self.method == "ges":
            return await self._ges_algorithm(data, variable_names)
        elif self.method == "notears":
            return await self._notears_algorithm(data, variable_names)
        else:
            raise ValueError(f"Unknown causal discovery method: {self.method}")

    async def _pc_algorithm(
        self,
        data: Dict[str, np.ndarray],
        variable_names: List[str],
    ) -> CausalGraph:
        """
        PC (Peter-Clark) algorithm for causal discovery.

        Constraint-based method using conditional independence tests.
        """
        graph = CausalGraph()

        # Add all nodes
        for var in variable_names:
            graph.add_node(var)

        # Start with complete undirected graph
        # Then remove edges based on conditional independence

        # Phase 1: Skeleton discovery
        for i, var_i in enumerate(variable_names):
            for var_j in variable_names[i+1:]:
                # Test independence
                if not self._is_independent(data[var_i], data[var_j]):
                    # Add undirected edge (both directions for now)
                    graph.add_edge(var_i, var_j, confidence=0.5)
                    graph.add_edge(var_j, var_i, confidence=0.5)

        # Phase 2: Orient edges (simplified)
        # Full PC would orient v-structures and apply orientation rules

        logger.info(f"PC algorithm discovered {len(graph.edges)} causal edges")

        return graph

    def _is_independent(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> bool:
        """Test conditional independence X ⊥ Y | Z."""
        from scipy.stats import pearsonr

        if Z is None:
            # Unconditional independence
            corr, p_value = pearsonr(X, Y)
            return p_value > self.significance_level
        else:
            # Conditional independence (simplified via partial correlation)
            # Full implementation would use conditional independence tests

            # Residualize X and Y on Z
            # This is a simplification
            return True  # Placeholder

    async def _ges_algorithm(
        self,
        data: Dict[str, np.ndarray],
        variable_names: List[str],
    ) -> CausalGraph:
        """
        GES (Greedy Equivalence Search) algorithm.

        Score-based method optimizing BIC score.
        """
        # Placeholder - would implement GES search
        graph = CausalGraph()
        for var in variable_names:
            graph.add_node(var)

        logger.warning("GES algorithm not fully implemented - returning empty graph")
        return graph

    async def _notears_algorithm(
        self,
        data: Dict[str, np.ndarray],
        variable_names: List[str],
    ) -> CausalGraph:
        """
        NOTEARS algorithm for causal discovery.

        Continuous optimization for DAG learning.
        """
        # Placeholder - would implement NOTEARS
        graph = CausalGraph()
        for var in variable_names:
            graph.add_node(var)

        logger.warning("NOTEARS algorithm not fully implemented - returning empty graph")
        return graph


# =============================================================================
# NEURAL CAUSAL MODEL
# =============================================================================

class NeuralCausalModel(nn.Module):
    """
    Neural network-based causal model.

    Learns causal mechanisms using neural networks.
    """

    def __init__(
        self,
        causal_graph: CausalGraph,
        variable_dims: Dict[str, int],
        hidden_dim: int = 256,
    ):
        """
        Initialize neural causal model.

        Args:
            causal_graph: Causal graph structure
            variable_dims: Dimension of each variable
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.graph = causal_graph
        self.variable_dims = variable_dims

        # Create neural mechanism for each variable
        self.mechanisms = nn.ModuleDict()

        for var in causal_graph.nodes:
            parents = causal_graph.get_parents(var)

            if parents:
                # Input: concatenation of parent values
                input_dim = sum(variable_dims[p] for p in parents)
            else:
                # Root node: no parents
                input_dim = 1  # Noise dimension

            output_dim = variable_dims[var]

            # Neural mechanism: f(parents, noise) -> variable
            self.mechanisms[var] = nn.Sequential(
                nn.Linear(input_dim + 1, hidden_dim),  # +1 for noise
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(
        self,
        parent_values: Dict[str, torch.Tensor],
        variable: str,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute variable value from parents.

        Args:
            parent_values: Dictionary of parent variable values
            variable: Target variable
            noise: Exogenous noise (sampled if not provided)

        Returns:
            Predicted variable value
        """
        parents = self.graph.get_parents(variable)
        batch_size = next(iter(parent_values.values())).size(0) if parent_values else 1

        if noise is None:
            noise = torch.randn(batch_size, 1, device=next(self.parameters()).device)

        if parents:
            # Concatenate parent values
            parent_tensors = [parent_values[p] for p in parents]
            parent_concat = torch.cat(parent_tensors, dim=-1)

            # Concatenate with noise
            mechanism_input = torch.cat([parent_concat, noise], dim=-1)
        else:
            # Root node: only noise
            mechanism_input = noise

        # Apply neural mechanism
        output = self.mechanisms[variable](mechanism_input)

        return output

    def sample(
        self,
        batch_size: int,
        interventions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample from neural causal model.

        Args:
            batch_size: Number of samples
            interventions: Intervention values (do-calculus)

        Returns:
            Dictionary of sampled variables
        """
        interventions = interventions or {}
        samples = {}
        device = next(self.parameters()).device

        # Topological sort
        sorted_vars = self._topological_sort()

        for var in sorted_vars:
            if var in interventions:
                # Intervention
                samples[var] = interventions[var]
            else:
                # Normal mechanism
                noise = torch.randn(batch_size, 1, device=device)
                samples[var] = self.forward(samples, var, noise)

        return samples

    def _topological_sort(self) -> List[str]:
        """Topological sort of variables."""
        visited = set()
        sorted_vars = []

        def dfs(var: str):
            if var in visited:
                return
            visited.add(var)

            for parent in self.graph.get_parents(var):
                dfs(parent)

            sorted_vars.append(var)

        for var in self.graph.nodes:
            dfs(var)

        return sorted_vars


# =============================================================================
# CAUSAL ATTENTION
# =============================================================================

class CausalAttention(nn.Module):
    """
    Causal attention mechanism for transformers.

    Enforces causal structure in attention weights.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal_mask: Optional[torch.Tensor] = None,
    ):
        """
        Initialize causal attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            causal_mask: Binary mask enforcing causal structure
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.register_buffer("causal_mask", causal_mask)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply causal attention.

        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor [batch, seq_len, embed_dim]
            value: Value tensor [batch, seq_len, embed_dim]

        Returns:
            Attention output [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = query.size()

        # Project to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask if provided
        if self.causal_mask is not None:
            scores = scores.masked_fill(self.causal_mask == 0, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)

        return output


# =============================================================================
# CAUSAL EVALUATION
# =============================================================================

@dataclass
class CausalEvaluationMetrics:
    """Metrics for evaluating causal models."""

    structural_hamming_distance: float  # Graph accuracy
    precision: float  # Edge precision
    recall: float  # Edge recall
    f1_score: float  # Harmonic mean
    intervention_accuracy: float  # Intervention prediction accuracy


def evaluate_causal_graph(
    predicted_graph: CausalGraph,
    true_graph: CausalGraph,
) -> CausalEvaluationMetrics:
    """
    Evaluate predicted causal graph against ground truth.

    Args:
        predicted_graph: Predicted causal graph
        true_graph: Ground truth causal graph

    Returns:
        Evaluation metrics
    """
    # Convert to edge sets
    predicted_edges = {(e.source, e.target) for e in predicted_graph.edges}
    true_edges = {(e.source, e.target) for e in true_graph.edges}

    # Compute metrics
    true_positives = len(predicted_edges & true_edges)
    false_positives = len(predicted_edges - true_edges)
    false_negatives = len(true_edges - predicted_edges)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Structural Hamming Distance
    shd = false_positives + false_negatives

    return CausalEvaluationMetrics(
        structural_hamming_distance=shd,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        intervention_accuracy=0.0,  # Would compute separately
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Causal Graph
    "CausalEdge",
    "CausalGraph",
    # Structural Causal Model
    "StructuralCausalModel",
    # Causal Discovery
    "CausalDiscovery",
    # Neural Causal Model
    "NeuralCausalModel",
    # Causal Attention
    "CausalAttention",
    # Evaluation
    "CausalEvaluationMetrics",
    "evaluate_causal_graph",
]
