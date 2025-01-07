# GBC Viewer

The GBC Viewer is an interactive tool for exploring GBC-annotated data. It supports both reading images locally and from the internet. 

<p align="left">
<img src="../assets/GBC_viewer.png" width=65% height=65% class="center">
</p>


Download the released datasets [GBC1M](https://huggingface.co/datasets/graph-based-captions/GBC1M/tree/main/data/parquet)/[GBC10M](https://huggingface.co/datasets/graph-based-captions/GBC10M/tree/main/data/parquet) into the [data](data) folder and explore them using the viewer!

## Requirements

- **Node.js**: `>= 20`
- **Python**: `>= 3.10`

## Installation

Install the required dependencies:

```bash
npm install
python3 -m pip install -r ./server/requirements.txt
```

## Usage

### Build and Run the Viewer

To build the website and start the Python server with both frontend and backend running together:

```bash
npm run build
python ./server/api.py --path ../data/gbc/wiki --img_root_dir .. --frontend_path dist --port 5050
```

In this example:
- The server reads data from [wiki_gbc_graphs.parquet](../data/gbc/wiki/wiki_gbc_graphs.parquet) in the `../data/gbc/wiki` directory.
- The website will be available at [http://localhost:5050/](http://localhost:5050/).

**Command-Line Arguments**

The `server/api.py` script supports the following arguments:

- `--path`: Path to the directory containing parquet files. Only immediate parquet files are used (non-recursive), and other formats like json or jsonl are ignored.
- `--img_root_dir`: Root directory for resolving relative paths in the `img_path` field.
- `--frontend_path`: Path to the folder containing the built frontend (e.g., `dist`).
- `--port`: Port where the server will run (defaults to `5050`).
- `--low_ram`: Enables low RAM mode, which reduces memory usage but may be slower. Recommended for large datasets like GBC1M or GBC10M.

### Image Preview

- **Local Images**: The viewer looks for images locally using the sample's `img_path` field.
- **Fallback to Internet**: If `img_path` is `None` or the local image is missing, the viewer fetches the image using `img_url`.


## Development

For development, you can run the viewer in different modes:

- **Frontend Only**: Start the Vite dev server for the frontend.
  ```bash
  npm run dev:vite
  ```

- **Backend Only**: Start the API dev server. The frontend will be available at [http://localhost:5173/](http://localhost:5173/).
  ```bash
  npm run dev:api
  ```

- **Frontend and Backend Together**: Start both servers simultaneously.
  ```bash
  npm run dev
  ```

By default, the development server reads data from the [data](data) folder. You can modify the arguments of `npm run dev:api` in [package.json](package.json) to change this behavior.
