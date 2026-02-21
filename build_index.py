import faiss
import numpy as np
from config import *

def build_index(features=None):
    if features is None:
        features = np.load(CACHE_DIR / "features.npy")

    print(f"\n构建索引: {len(features)} 向量")
    # 使用 Flat 索引（精确搜索，CPU 也很快）
    index = faiss.IndexFlatIP(features.shape[1])  # 内积 = 余弦相似度
    index.add(features)
    faiss.write_index(index, str(CACHE_DIR / "faiss.index"))
    print("索引完成")
    return index

def load_index():
    path = CACHE_DIR / "faiss.index"
    if not path.exists():
        return None
    return faiss.read_index(str(path))

if __name__ == "__main__":
    build_index()