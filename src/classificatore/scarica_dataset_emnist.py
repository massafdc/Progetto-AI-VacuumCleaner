from pathlib import Path
from urllib.request import Request, urlopen
import zipfile


DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "emnist"
    / "raw"
)

ZIP_PATH = DATA_DIR / "gzip.zip"

URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"


def download_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        print("[SKIP] gzip.zip già presente")
    else:
        print("[DOWNLOAD] EMNIST")

        request = Request(
            URL,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urlopen(request) as response:
            with open(ZIP_PATH, "wb") as file:
                file.write(response.read())

        print("[OK] Download completato")

    print("[EXTRACT] Estrazione archivio...")

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extractall(DATA_DIR)

    print("[OK] Estrazione completata")


if __name__ == "__main__":
    download_dataset()