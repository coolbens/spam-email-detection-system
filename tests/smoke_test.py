from __future__ import annotations

import sys
from pathlib import Path

# Kuhaon ang root directory sa project.
#
# __file__
#   -> current file location
#
# .resolve()
#   -> himuon nga absolute path
#
# .parents[1]
#   -> mosaka ug usa ka level gikan sa current folder
#
# Example:
# project/
# ├── model/
# ├── tests/
# │   └── test_predict.py
#
# ROOT_DIR mahimong project/
ROOT_DIR = Path(__file__).resolve().parents[1]

# Siguraduhon nga ang root directory naa sa Python import path.
#
# Kini aron maka-import ta ug modules gikan
# sa ubang folders sulod sa project.
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import sa predict_text function nga gigamit
# para mag-predict kung spam o ham ang text.
from model.predict import predict_text


# Unit test function.
#
# Ang katuyoan ani nga test mao ang pagsiguro nga
# ang predict_text() function mobalik sa expected nga data.
def test_predict_text_returns_expected_keys():

    # Sample spam-like message nga ipredict sa model.
    result = predict_text("Win a free cash prize now")

    # Siguraduhon nga ang result dictionary adunay
    # tanan importanteng keys.
    #
    # Expected keys:
    # - label             -> spam o ham
    # - confidence        -> confidence score sa prediction
    # - spam_probability  -> probability nga spam
    # - ham_probability   -> probability nga ham
    #
    # .issubset() nag-check kung ang tanan expected keys
    # naa sulod sa result.keys().
    assert set([
        "label",
        "confidence",
        "spam_probability",
        "ham_probability"
    ]).issubset(result.keys())