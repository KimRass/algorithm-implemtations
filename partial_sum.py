import numpy as np

# img = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)
# block_size = 5
# padded_img = np.pad(
#     img,
#     # pad_width=(block_size // 2 + 1, 0),
#     pad_width=(block_size // 2 + 1, block_size // 2),
#     mode="constant",
#     constant_values=0,
# ).astype(np.float64)
# # padded_img
# # for row in range(1, img.shape[0] + 1):
# #     print(np.sum(padded_img[row: row + block_size]), end=" ")
# cum_sum = np.cumsum(padded_img)[block_size:]
# rev_cum_sum = np.cumsum(-padded_img)[: -block_size]
# img_sum = (cum_sum + rev_cum_sum).astype(np.uint8)
# img_sum
# num_neighbors = np.full_like(img_sum, fill_value=block_size)
# num_neighbors
# -padded_img
