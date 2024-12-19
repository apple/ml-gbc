import argparse
import os
import math
import io
import base64
import shlex
from typing import Optional
from dataclasses import dataclass

import duckdb
from async_lru import alru_cache
from httpx import AsyncClient
from PIL import Image

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from headers import headers_dict


parser = argparse.ArgumentParser(exit_on_error="API_ARGS" not in os.environ)
parser.add_argument(
    "--path", type=str, help="Path to the directory containing parquet files"
)
parser.add_argument(
    "--low_ram", action="store_true", help="Use low RAM mode, slower but less RAM usage"
)
parser.add_argument(
    "--db", default="", type=str, help="Path to the duckdb database file"
)
parser.add_argument(
    "--frontend_path",
    default="",
    type=str,
    help="Path to the dist folder of built frontend",
)
parser.add_argument(
    "--img_root_dir",
    default="",
    type=str,
    help=(
        "Root directory containing image files. When 'img_path' is present in "
        "the input data, the full image path is constructed by joining this "
        "directory with the value of 'img_path'. Default to the current directory."
    ),
)
parser.add_argument("--host", default="127.0.0.1", type=str)
parser.add_argument("--port", default=5050, type=int)

if "API_ARGS" in os.environ:
    args = parser.parse_args(shlex.split(os.environ["API_ARGS"]))
else:
    args = parser.parse_args()
path = args.path


img_client = AsyncClient(http2=True, headers=headers_dict)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@dataclass
class SearchRequest(object):
    target: str


def resize_to_max_size(img, max_size):
    if all(i < max_size for i in img.size):
        return img
    if img.size[0] > img.size[1]:
        return img.resize((max_size, int(img.size[1] * max_size / img.size[0])))
    else:
        return img.resize((int(img.size[0] * max_size / img.size[1]), max_size))


if args.db:
    conn = duckdb.connect(args.db)
    args.low_ram = False
elif args.low_ram:
    print("Using Low Ram Mode, use VIEW instead of create TABLE")
    conn = duckdb.connect(":memory:")
    conn.execute(
        f"""
        CREATE VIEW gbc AS
        SELECT
            ROW_NUMBER() OVER () - 1 AS id,
            *
        FROM read_parquet('{os.path.join(path, "*.parquet")}')
        """
    )
else:
    conn = duckdb.connect(":memory:")

if not args.low_ram:
    print("Using High Ram(or file) Mode, use TABLE and INDEX to speed up queries")
    print("Loading data into memory or file...")
    conn.execute(
        f"""
        CREATE TABLE gbc AS
        SELECT
            ROW_NUMBER() OVER () - 1 AS id,
            *
        FROM read_parquet('{os.path.join(path, "*.parquet")}')
        """
    )
    print("Creating index...")
    conn.execute("CREATE INDEX id_idx ON gbc(id)")
    print("Done!")
col_names = [i[0] for i in conn.execute("SELECT * FROM gbc WHERE 1=0").description]
col_index = {col_names[i]: i for i in range(len(col_names))}
total = conn.execute("SELECT COUNT(*) FROM gbc").fetchall()[0][0]
print(total)


def search_query(target: str):
    target = "%".join(target.split(" "))
    return (
        "Where "
        f"CAST(original_caption AS VARCHAR) LIKE '%{target}%'"
        " OR "
        f"CAST(short_caption AS VARCHAR) LIKE '%{target}%'"
        " OR "
        f"CAST(detail_caption AS VARCHAR) LIKE '%{target}%'"
    )


@app.get("/api/graph/{item_id}")
@alru_cache(maxsize=1024)
async def get_item(item_id: int):
    if args.low_ram:
        data = conn.execute(f"SELECT * FROM gbc OFFSET {item_id} LIMIT 1").fetchone()
    else:
        data = conn.execute(f"SELECT * FROM gbc WHERE id={item_id}").fetchone()
    data = {col: data[i] for i, col in enumerate(col_names)}

    # Fetch image on server side.
    if "img_path" in data and data["img_path"] is not None:
        img_path = os.path.join(args.img_root_dir, data["img_path"])
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            img = resize_to_max_size(img, 768)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="webp", quality=80)
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
            data["img_url"] = f"data:image/webp;base64,{img_base64}"
    else:
        pass
        # # Uncomment below codes if you want the server to fetch image from internet
        # # Can help in some CORS case, but slower.
        # try:
        #     response = await img_client.get(df.loc[item_id]["img_url"])
        #     # PIL to webp than base64
        #     img = Image.open(io.BytesIO(response.content)).convert("RGB")
        #     img = resize_to_max_size(img, 768)
        #     img_bytes = io.BytesIO()
        #     img.save(img_bytes, format="webp", quality=80)
        #     img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
        #     data["img_url"] = f"data:image/webp;base64,{img_base64}"
        # except:
        #     print("Error loading image: ", df.loc[item_id]["img_url"])
        #     data["img_url"] = "none"
    return data


@app.get("/api/index")
async def list_items(target: Optional[str] = None):
    if target is None:
        return math.ceil(total / 10)
    else:
        query = "SELECT COUNT(id) FROM gbc " + search_query(target) + ";"
        search_total = conn.execute(query).fetchall()[0][0]
        return math.ceil(search_total / 10)


@app.get("/api/index/{item_id}")
async def get_page(item_id: int, target: Optional[str] = None):
    if target is None:
        datas = conn.execute(
            f"SELECT * FROM gbc OFFSET {item_id * 10} LIMIT 10"
        ).fetchall()
    else:
        query = (
            "SELECT * "
            + "FROM gbc "
            + search_query(target)
            + f" OFFSET {item_id * 10} LIMIT 10"
        )
        datas = conn.execute(query).fetchall()
    return [
        (
            data[col_index["id"]],
            {"short_caption": data[col_index["short_caption"]]},
        )
        for i, data in enumerate(datas)
    ]


if args.frontend_path:
    print("Serving frontend from:", args.frontend_path)
    app.mount("/", StaticFiles(directory=args.frontend_path, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
