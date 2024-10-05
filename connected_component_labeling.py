# References:
    # https://en.wikipedia.org/wiki/Connected-component_labeling

import numpy as np
from itertools import product
import sys

sys.setrecursionlimit(10 ** 9)


class ConnectedComponentLabeling(object):
    def __init__(self, img, connectivity=4):
        assert connectivity in [4, 8]

        self.img = img
        self.connectivity = connectivity
        self.h, self.w = self.img.shape
        self.out = np.zeros_like(self.img, dtype=np.int32)
        self.cur_label = 0

    def get_neighbors(self, node):
        row, col = node
        neighbors = []
        if row > 0:
            neighbors.append((row - 1, col)) # Up
        if row < self.h - 1:
            neighbors.append((row + 1, col)) # Down
        if col > 0:
            neighbors.append((row, col - 1)) # Left
        if col < self.w - 1:
            neighbors.append((row, col + 1)) # Right

        if self.connectivity == 8:
            if row > 0:
                if col > 0:
                    neighbors.append((row - 1, col - 1)) # Top-left
                if col < self.w - 1:
                    neighbors.append((row - 1, col + 1)) # Top-right
            if row < self.h - 1:
                if col > 0:
                    neighbors.append((row + 1, col - 1)) # Bottom-left
                if col < self.w - 1:
                    neighbors.append((row + 1, col + 1)) # Bottom-right
        return neighbors

    def dfs(self, cur_node):
        if self.img[cur_node] != 0 and self.out[cur_node] == 0:
            self.out[cur_node] = self.cur_label
            for neighbor in self.get_neighbors(cur_node):
                self.dfs(neighbor)

    def __call__(self):
        for cur_node in product(range(self.h), range(self.w)):
            if self.img[cur_node] != 0 and self.out[cur_node] == 0:
                self.cur_label += 1
            self.dfs(cur_node)
        return self.cur_label + 1, self.out


def perform_ccl(img, connectivity=4):
    ccl = ConnectedComponentLabeling(img=img, connectivity=connectivity)
    return ccl()


if __name__ == "__main__":
    import cv2

    img_path = "./resources/j.webp"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    connectivity = 4
    out11, out12 = cv2.connectedComponents(img, connectivity=connectivity)
    out21, out22 = perform_ccl(img=img, connectivity=connectivity)
    print(out11, out21) # 3 3
    print(np.array_equal(out12, out22)) # True
