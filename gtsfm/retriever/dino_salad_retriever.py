from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.retriever.retriever_base import ImageMatchingRegime, RetrieverBase

logger = logger_utils.get_logger()


class DinoSaladRetriever(RetrieverBase):

    def __init__(self,
                 num_matched: int,
                 min_score: float = 0.5,
                 blocksize: int = 50) -> None:
        """
        Args:
            num_matched: Number of K potential matches to provide per query. These are the top "K" matches per query.
            min_score: Minimum allowed similarity score to accept a match.
            blocksize: Size of matching sub-blocks when creating similarity matrix.
        """
        super().__init__(matching_regime=ImageMatchingRegime.RETRIEVAL)
        self._num_matched = num_matched
        self._blocksize = blocksize
        self._min_score = min_score
        print('DinoSaladRetriever initialized')

    def __repr__(self) -> str:
        return f"""
        DinoSaladRetriever:
            Num. frames matched: {self._num_matched}
            Block size: {self._blocksize}
            Minimum score: {self._min_score}
        """

    def get_image_pairs(
        self,
        global_descriptors: Optional[List[np.ndarray]],
        image_fnames: List[str],
        plots_output_dir: Optional[Path] = None,
    ) -> List[Tuple[int, int]]:
        print(f'Get image pairs by dino_salad')
        if global_descriptors is None:
            raise ValueError("Global descriptors need to be provided")
        score_matrix = self.compute_similarity_matrix(global_descriptors)

        return self.compute_pairs_from_similarity_matrix(score_matrix)

    def compute_similarity_matrix(self, features):
        cosine_distances = np.array([
            np.dot(features, feature) /
            (np.linalg.norm(features) * np.linalg.norm(feature))
            for feature in features
        ])
        cosine_distances /= cosine_distances[0, 0]

        return cosine_distances

    def compute_pairs_from_similarity_matrix(
            self, score_matrix) -> List[Tuple[int, int]]:
        indices = np.where(score_matrix > self._min_score)
        indices = list(zip(indices[0], indices[1]))

        return indices
