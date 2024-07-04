import faiss
import faiss.contrib.torch_utils
import matplotlib.pyplot as plt
import numpy as np
import torch

# from prettytable import PrettyTable


def calculate_distance(query_feature, database_features):
    cosine_distances = np.array([
        np.dot(query_feature, feature) /
        (np.linalg.norm(query_feature) * np.linalg.norm(feature))
        for feature in database_features
    ])
    cosine_distances /= cosine_distances[0, 0]
    return cosine_distances


def find_image_pairs(descriptors, faiss_gpu=False, topk=10):
    embed_size = descriptors.shape[1]
    if faiss_gpu:
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = True
        flat_config.device = 0
        faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
    else:
        faiss_index = faiss.IndexFlatL2(embed_size)

    print(faiss_index.d)
    faiss_index.add(descriptors)

    scores, predictions = faiss_index.search(descriptors, topk)
    # print(f'predictions: {predictions.shape}')
    # print(f'scores:\n{scores}')
    print(f'predictions:\n{predictions}')


def main():
    descriptors = np.load("descriptors.npy")
    print(descriptors.shape)
    # find_image_pairs(descriptors)

    scores = calculate_distance(descriptors, descriptors)
    np.set_printoptions(precision=3,
                        suppress=True,
                        floatmode='fixed',
                        linewidth=200)
    print(f'scores:\n{scores}')

    np.fill_diagonal(scores, 0.0)
    scores = np.triu(scores)
    scores[scores < 0.6] = 0.0
    print(f'scores:\n{scores}')

    sc = plt.imshow(np.triu(scores))
    plt.colorbar(sc)
    plt.title("Image Similarity Matrix")
    plt.savefig("netvlad_similarity_matrix.jpg", dpi=300)
    plt.close("all")


if __name__ == '__main__':
    main()
