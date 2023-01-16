import os
from gcsfs import GCSFileSystem
from config import GCP_PROJ, BUCKET


def write_to_gcs(data: str, filename: str):
    fs = GCSFileSystem(project=GCP_PROJ)

    filepath = os.path.join(BUCKET, filename)
    with fs.open(filepath, "w") as f:
        f.write(data)

    print(fs.ls(BUCKET))


if __name__ == "__main__":
    write_to_gcs("hello world", "hello.txt")
