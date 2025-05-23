# Copyright 2024 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance_matrix


class CKMeans:
    """A simple clustering method that forms k clusters by iteratively reassigning
    samples to the closest centroids such that the total weight of the
    cluster stays within the capacity.

    Args:
        k: The number of clusters the algorithm will form.
        max_iterations: The number of iterations the algorithm will run for
            if it does not converge before that.
    """

    def __init__(self, k: int = 2, max_iterations: int = 500) -> None:
        self.k = k
        self.max_iterations = max_iterations

    def _init_random_centroids(self, X: NDArray) -> NDArray:
        """Initialize the centroids as ``k`` random samples of ``X``."""
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))

        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid

        return centroids

    def _sort_centroids(self, sample: NDArray, centroids: NDArray) -> NDArray:
        """Return the index of the closest centroids in order

        In a normal KMeans algorithm, the closest center is return. Here we
        return all in order to be able to iterate through them.
        """
        return np.argsort(np.sqrt(np.sum(np.square(sample - centroids), axis=1)))

    def _create_clusters(
        self, centroids: NDArray, X: NDArray, demand: Sequence, capacities: Sequence
    ):
        """Assign the samples to the closest centroids to create clusters.

        If adding the each point to the cluster would result in surpassing
        the cluster capacity, the sample is added to the next best cluster.
        """
        clusters = [[] for _ in range(self.k)]
        clusters_filled = [0 for _ in range(self.k)]

        for sample_i, sample in enumerate(X):
            sorted_centroids = self._sort_centroids(sample, centroids)

            for centroid_i in sorted_centroids:
                if clusters_filled[centroid_i] + demand[sample_i] <= capacities[centroid_i]:
                    clusters[centroid_i].append(sample_i)
                    clusters_filled[centroid_i] += demand[sample_i]
                    break

        return clusters

    def _calculate_centroids(self, clusters, X):
        """Calculate new centroids as the means of the samples in each cluster"""
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))

        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid

        return centroids

    def _get_cluster_labels(self, clusters: Sequence, X: NDArray):
        """Classify samples as the index of their clusters"""
        # One prediction for each sample
        y_pred = np.zeros(np.shape(X)[0])

        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i

        return y_pred

    def _get_score(self, X, assignments):
        assignments = np.array(assignments)
        d = distance_matrix(X, X)
        max_clusters = int(np.max(assignments) + 1)

        score = 0
        for k in range(max_clusters):
            ind = assignments == k
            score += np.sum(d[ind, :][:, ind])

        return score

    def predict_once(self, X: NDArray, demand: Sequence, capacities: Sequence) -> NDArray:
        """K-Means clustering subject to capacity constraint

        Args:
            X: 2-d numpy.array (each row is a sample, each column is feature/coordinate.
            demand: Demand of each sample.
            capacities: Capacity of each cluster (must be same length as the number of clusters).

        Returns:
            array[float]: An array with the index of cluster for each sample.

        """
        if len(demand) != X.shape[0]:
            raise ValueError(
                "The length of the demand array must be " "the same as the number of samples"
            )

        if len(capacities) != self.k:
            raise ValueError("You must specify the capacity of each clusters")

        if sum(demand) > sum(capacities):
            raise ValueError("Constraints cannot be satisfied")

        # Initialize centroids as k random samples from X
        centroids = self._init_random_centroids(X)

        clusters = []
        for _ in range(self.max_iterations):
            clusters = self._create_clusters(centroids, X, demand, capacities)
            prev_centroids = centroids
            centroids = self._calculate_centroids(clusters, X)
            diff = centroids - prev_centroids

            if not diff.any():
                break

        if len(clusters) == 0:
            return np.array([])

        return self._get_cluster_labels(clusters, X)

    def predict(
        self, X: NDArray, demand: Sequence, capacities: Sequence, time_limit: float = 10.0
    ) -> NDArray:
        """K-Means clustering subject to capacity constraint.

        Args:
            X: 2-d numpy.array (each row is a sample, each column is feature/coordinate.
            demand: The demand of each sample.
            capacities: Capacity of each cluster (must be same length as the number of clusters).
            time_limit: Maximum time in seconds the stochastic K-Means lgorithm
                can be repeated before returning solution.

        Returns:
            array: An array with the index of cluster for each sample.

        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        best_score = 0
        start = time.time()

        best_assignments = np.array([])
        while time.time() - start < time_limit:
            assignments = self.predict_once(X, demand, capacities)
            score = self._get_score(X, assignments)

            if score > best_score:
                best_score = score
                best_assignments = assignments

        return best_assignments
