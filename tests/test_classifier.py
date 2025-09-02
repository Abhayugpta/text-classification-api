---

#### 5. `tests/test_classifier.py`
```python
import pytest
from classifier import TextClassifier

def test_training_and_prediction():
    clf = TextClassifier()
    texts = ["good", "bad", "awesome", "terrible"]
    labels = ["pos", "neg", "pos", "neg"]
    clf.train(texts, labels)

    assert clf.predict("good") in ["pos", "neg"]
    assert clf.predict("terrible") == "neg"

def test_untrained_model():
    clf = TextClassifier()
    assert clf.predict("hello") == "Model not trained yet"
