from __future__ import annotations

import sys
from pathlib import Path

# Kuhaon ang root directory sa project.
#
# __file__
#   -> current nga file
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
# ├── scripts/
# │   └── healthcheck.py
#
# Kung naa ta sa healthcheck.py,
# ROOT_DIR mahimong project/
ROOT_DIR = Path(__file__).resolve().parents[1]

# Siguraduhon nga ang ROOT_DIR naa sa Python import path.
#
# Kinahanglan ni aron maka-import ta ug modules
# gikan sa ubang folders sulod sa project.
#
# Kung wala pa siya sa sys.path,
# ibutang siya sa pinakauna nga posisyon.
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import sa function nga mo-load sa model artifacts.
#
# Kasagaran kini nga artifacts mao ang:
# - trained model (.pkl)
# - vectorizer (.pkl)
# - metrics.json
from model.predict import load_artifacts


# Main function sa script.
def main() -> None:

    # Sulayan pag-load ang tanan model artifacts.
    #
    # Kung adunay kulang nga file,
    # kasagaran mo-raise ni ug error.
    load_artifacts()

    # Kung walay error,
    # pasabot successful ang pag-load.
    print("OK: model artifacts loaded successfully")


# Entry point sa program.
#
# Ang code sulod niini modagan ra
# kung direkta nimo gi-run ang file.
#
# Example:
# python healthcheck.py
#
# Dili kini modagan kung gi-import lang
# ang file sa laing module.
if __name__ == "__main__":
    main()