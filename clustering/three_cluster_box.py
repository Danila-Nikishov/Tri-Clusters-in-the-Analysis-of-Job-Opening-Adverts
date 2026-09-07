import numpy as np
from itertools import product
from tqdm import tqdm


class tri_cluster_box:
    def __init__(self, corelevance_matrix):
        self.corelevance_matrix = corelevance_matrix

    def make_tri_clusters_on_updated_matrix(self, lambda_0, max_size=None):
        tri_clusters = list()

        dim_1, dim_2, dim_3 = self.corelevance_matrix.shape

        shifted_corelevance_matrix = self.corelevance_matrix - lambda_0

        free_dim_1 = set(range(dim_1))
        free_dim_2 = set(range(dim_2))
        free_dim_3 = set(range(dim_3))

        for i in tqdm(range(dim_1)):
            for j in range(dim_2):
                for k in range(dim_3):
                    if (
                        i not in free_dim_1
                        or j not in free_dim_2
                        or k not in free_dim_3
                        or self.corelevance_matrix[i, j, k] == 0
                    ):
                        continue
                    new_cluster = {"dim_1": {i}, "dim_2": {j}, "dim_3": {k}}

                    max_difference = 1

                    while max_difference > 0:
                        x_indices = list(new_cluster["dim_1"])
                        y_indices = list(new_cluster["dim_2"])
                        z_indices = list(new_cluster["dim_3"])
                        r_XYZ = np.sum(
                            shifted_corelevance_matrix[
                                np.ix_(x_indices, y_indices, z_indices)
                            ]
                        )

                        R1 = shifted_corelevance_matrix[:, y_indices][
                            :, :, z_indices
                        ].sum(axis=(1, 2))

                        R2 = shifted_corelevance_matrix[x_indices, :][
                            :, :, z_indices
                        ].sum(axis=(0, 2))

                        R3 = shifted_corelevance_matrix[x_indices][:, y_indices, :].sum(
                            axis=(0, 1)
                        )

                        max_difference = -1

                        if max_size is None:
                            search_space_1 = free_dim_1 | new_cluster["dim_1"]
                            search_space_2 = free_dim_2 | new_cluster["dim_2"]
                            search_space_3 = free_dim_3 | new_cluster["dim_3"]
                        else:
                            if len(new_cluster["dim_1"]) < max_size:
                                search_space_1 = free_dim_1 | new_cluster["dim_1"]
                            else:
                                search_space_1 = new_cluster["dim_1"]
                            if len(new_cluster["dim_2"]) < max_size:
                                search_space_2 = free_dim_2 | new_cluster["dim_2"]
                            else:
                                search_space_2 = new_cluster["dim_2"]
                            if len(new_cluster["dim_3"]) < max_size:
                                search_space_3 = free_dim_3 | new_cluster["dim_3"]
                            else:
                                search_space_3 = new_cluster["dim_3"]

                        for d_1_element in search_space_1:
                            new_difference = self.calculate_difference(
                                new_cluster,
                                d_1_element,
                                r_XYZ,
                                R1,
                                R2,
                                R3,
                                position="dim_1",
                            )
                            if new_difference > max_difference:
                                max_difference = new_difference
                                best_element = d_1_element
                                best_position = "dim_1"
                        for d_2_element in search_space_2:
                            new_difference = self.calculate_difference(
                                new_cluster,
                                d_2_element,
                                r_XYZ,
                                R1,
                                R2,
                                R3,
                                position="dim_2",
                            )
                            if new_difference > max_difference:
                                max_difference = new_difference
                                best_element = d_2_element
                                best_position = "dim_2"
                        for d_3_element in search_space_3:
                            new_difference = self.calculate_difference(
                                new_cluster,
                                d_3_element,
                                r_XYZ,
                                R1,
                                R2,
                                R3,
                                position="dim_3",
                            )
                            if new_difference > max_difference:
                                max_difference = new_difference
                                best_element = d_3_element
                                best_position = "dim_3"
                        if max_difference > 0:
                            if best_element not in new_cluster[best_position]:
                                new_cluster[best_position].add(best_element)
                            else:
                                new_cluster[best_position].discard(best_element)

                    tri_clusters.append(new_cluster)
                    free_dim_1 -= new_cluster["dim_1"]
                    free_dim_2 -= new_cluster["dim_2"]
                    free_dim_3 -= new_cluster["dim_3"]

        return tri_clusters

    def make_tri_clusters_classical(
        self,
        lambda_0,
        decrease_intensity=False,
        discard_conditions=False,
        discard_responsibilities=False,
        discard_requirements=False,
        early_stop=False,
        max_size=None,
    ):
        tri_clusters = list()

        dim_1, dim_2, dim_3 = self.corelevance_matrix.shape
        free_requirements = set(range(dim_1))
        free_responsibilities = set(range(dim_2))
        free_conditions = set(range(dim_3))
        shifted_corelevance_matrix = self.corelevance_matrix - lambda_0

        for i in tqdm(range(dim_1)):
            for j in range(dim_2):
                for k in range(dim_3):
                    if (
                        self.corelevance_matrix[i, j, k] == 0
                        or k not in free_conditions
                        or j not in free_responsibilities
                        or i not in free_requirements
                    ):
                        continue

                    new_cluster = {"dim_1": {i}, "dim_2": {j}, "dim_3": {k}}

                    max_difference = 1
                    n_iteration = 0

                    while max_difference > 0:
                        if early_stop:
                            n_iteration += 1
                            if n_iteration > 100:
                                break
                        x_indices = list(new_cluster["dim_1"])
                        y_indices = list(new_cluster["dim_2"])
                        z_indices = list(new_cluster["dim_3"])
                        r_XYZ = np.sum(
                            shifted_corelevance_matrix[
                                np.ix_(x_indices, y_indices, z_indices)
                            ]
                        )

                        R1 = shifted_corelevance_matrix[:, y_indices][
                            :, :, z_indices
                        ].sum(axis=(1, 2))

                        R2 = shifted_corelevance_matrix[x_indices, :][
                            :, :, z_indices
                        ].sum(axis=(0, 2))

                        R3 = shifted_corelevance_matrix[x_indices][:, y_indices, :].sum(
                            axis=(0, 1)
                        )

                        max_difference = -1

                        if max_size is None:
                            search_space_1 = range(dim_1)
                            search_space_2 = range(dim_2)
                            search_space_3 = range(dim_3)
                        else:
                            if len(new_cluster["dim_1"]) < max_size:
                                search_space_1 = range(dim_1)
                            else:
                                search_space_1 = new_cluster["dim_1"]
                            if len(new_cluster["dim_2"]) < max_size:
                                search_space_2 = range(dim_2)
                            else:
                                search_space_2 = new_cluster["dim_2"]
                            if len(new_cluster["dim_3"]) < max_size:
                                search_space_3 = range(dim_3)
                            else:
                                search_space_3 = new_cluster["dim_3"]

                        for d_1_element in search_space_1:
                            if (
                                discard_requirements
                                and d_1_element not in free_requirements
                                and d_1_element not in new_cluster["dim_1"]
                            ):
                                continue
                            new_difference = self.calculate_difference(
                                new_cluster,
                                d_1_element,
                                r_XYZ,
                                R1,
                                R2,
                                R3,
                                position="dim_1",
                            )
                            if new_difference > max_difference:
                                max_difference = new_difference
                                best_element = d_1_element
                                best_position = "dim_1"
                        for d_2_element in search_space_2:
                            if (
                                discard_responsibilities
                                and d_2_element not in free_responsibilities
                                and d_2_element not in new_cluster["dim_2"]
                            ):
                                continue
                            new_difference = self.calculate_difference(
                                new_cluster,
                                d_2_element,
                                r_XYZ,
                                R1,
                                R2,
                                R3,
                                position="dim_2",
                            )
                            if new_difference > max_difference:
                                max_difference = new_difference
                                best_element = d_2_element
                                best_position = "dim_2"
                        for d_3_element in search_space_3:
                            if (
                                discard_conditions
                                and d_3_element not in free_conditions
                                and d_3_element not in new_cluster["dim_3"]
                            ):
                                continue
                            new_difference = self.calculate_difference(
                                new_cluster,
                                d_3_element,
                                r_XYZ,
                                R1,
                                R2,
                                R3,
                                position="dim_3",
                            )
                            if new_difference > max_difference:
                                max_difference = new_difference
                                best_element = d_3_element
                                best_position = "dim_3"
                        if max_difference > 0:
                            if best_element not in new_cluster[best_position]:
                                new_cluster[best_position].add(best_element)
                            else:
                                new_cluster[best_position].discard(best_element)

                    tri_clusters.append(new_cluster)
                    if decrease_intensity == True:
                        cluster_indices = np.ix_(
                            list(new_cluster["dim_1"]),
                            list(new_cluster["dim_2"]),
                            list(new_cluster["dim_3"]),
                        )
                        submatrix = shifted_corelevance_matrix[cluster_indices]
                        cluster_density = np.mean(submatrix)
                        shifted_corelevance_matrix[cluster_indices] = (
                            shifted_corelevance_matrix[cluster_indices]
                            - cluster_density
                        )
                    if discard_conditions == True:
                        free_conditions -= new_cluster["dim_3"]
                    if discard_responsibilities == True:
                        free_responsibilities -= new_cluster["dim_2"]
                    if discard_requirements == True:
                        free_requirements -= new_cluster["dim_1"]

        unique_clusters = []
        seen = set()

        for cluster in tri_clusters:
            key = (
                frozenset(cluster["dim_1"]),
                frozenset(cluster["dim_2"]),
                frozenset(cluster["dim_3"]),
            )

            if key not in seen:
                seen.add(key)
                unique_clusters.append(cluster)

        return unique_clusters

    def calculate_difference(self, cluster, element, r_XYZ, R1, R2, R3, position):
        X = len(cluster["dim_1"])
        Y = len(cluster["dim_2"])
        Z = len(cluster["dim_3"])

        if element in cluster[position]:
            z = -1
        else:
            z = 1

        if position == "dim_1":
            if X + z == 0:
                return -1
            r = R1[element]
            difference = (r**2 + 2 * z * r * r_XYZ - z * (r_XYZ**2 / X)) / (
                (X + z) * Y * Z
            )
        elif position == "dim_2":
            if Y + z == 0:
                return -1
            r = R2[element]
            difference = (r**2 + 2 * z * r * r_XYZ - z * (r_XYZ**2 / Y)) / (
                X * (Y + z) * Z
            )
        else:
            if Z + z == 0:
                return -1
            r = R3[element]
            difference = (r**2 + 2 * z * r * r_XYZ - z * (r_XYZ**2 / Z)) / (
                X * Y * (Z + z)
            )

        return difference

    def calculate_intensity_and_contribution(self, clusters):
        data_scatter = np.sum(self.corelevance_matrix**2)
        contributions = []
        intensities = []
        for cluster in clusters:
            box = self.corelevance_matrix[
                np.ix_(
                    list(cluster["dim_1"]),
                    list(cluster["dim_2"]),
                    list(cluster["dim_3"]),
                )
            ]
            box_intensity = np.mean(box)
            box_contribution = np.round(
                (((np.sum(box) ** 2) / box.size) / data_scatter) * 100, 2
            )
            contributions.append(box_contribution)
            intensities.append(box_intensity)

        return intensities, contributions
