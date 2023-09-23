import random
from sentence_transformers import SentenceTransformer
import numpy as np

random.seed(42)

model_name = "sentence-transformers/all-mpnet-base-v2"

words = ["Any", "shuffled", "words", "form", "a", "sentence", "."]
texts = [" ".join([random.choice(words) for _ in range(10)]) for _ in range(10)]

st = SentenceTransformer(model_name)

e0 = st.encode(texts, batch_size=32)
print(f"==>> type(e0): {type(e0)}")
print(f"==>> e0: {e0}")
e1 = st.encode(texts, batch_size=1)
print(f"==>> e1: {e1}")
e2 = st.encode(texts, batch_size=32)
print(f"==>> e2: {e2}")

np.testing.assert_allclose(e0, e1, atol=1e-5)
np.testing.assert_allclose(e0, e2, atol=1e-5)
np.testing.assert_allclose(e1, e2, atol=1e-5)
