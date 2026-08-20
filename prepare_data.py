"""Download the public datasets used by the FedKAPT experiments.

The helper stores raw files locally under ``data/``. Third-party datasets are
not redistributed with this repository; users remain responsible for
complying with the terms on the official dataset pages.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen


CWRU_URLS = {
    "97": "https://engineering.case.edu/sites/default/files/97.mat",
    "98": "https://engineering.case.edu/sites/default/files/98.mat",
    "99": "https://engineering.case.edu/sites/default/files/99.mat",
    "100": "https://engineering.case.edu/sites/default/files/100.mat",
    "105": "https://engineering.case.edu/sites/default/files/105.mat",
    "106": "https://engineering.case.edu/sites/default/files/106.mat",
    "107": "https://engineering.case.edu/sites/default/files/107.mat",
    "108": "https://engineering.case.edu/sites/default/files/108.mat",
    "118": "https://engineering.case.edu/sites/default/files/118.mat",
    "119": "https://engineering.case.edu/sites/default/files/119.mat",
    "120": "https://engineering.case.edu/sites/default/files/120.mat",
    "121": "https://engineering.case.edu/sites/default/files/121.mat",
    "130": "https://engineering.case.edu/sites/default/files/130.mat",
    "131": "https://engineering.case.edu/sites/default/files/131.mat",
    "132": "https://engineering.case.edu/sites/default/files/132.mat",
    "133": "https://engineering.case.edu/sites/default/files/133.mat",
}
WDBC_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/wdbc.data"
)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"exists: {destination}")
        return
    request = Request(url, headers={"User-Agent": "FedKAPT-research-code"})
    print(f"download: {destination.name}")
    with urlopen(request, timeout=120) as response, destination.open("wb") as fh:
        while chunk := response.read(1024 * 1024):
            fh.write(chunk)


def prepare_cwru(root: Path) -> None:
    for stem, url in CWRU_URLS.items():
        download(url, root / f"{stem}.mat")


def prepare_wdbc(root: Path) -> None:
    download(WDBC_URL, root / "wdbc.data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("cwru", "wdbc", "all"), default="all"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    if args.dataset in {"cwru", "all"}:
        prepare_cwru(args.data_root / "cwru")
    if args.dataset in {"wdbc", "all"}:
        prepare_wdbc(args.data_root / "wdbc")


if __name__ == "__main__":
    main()
