# Hunyuan3D API Server Test

This directory contains a comprehensive test script for the Hunyuan3D API server, managed with `uv` for fast and reliable dependency management.

## Features

-  **Health Check**: Verifies the API server is running
-  **Synchronous Generation**: Tests immediate 3D model generation
-  **Asynchronous Generation**: Tests background processing with status tracking
-  **Secure Configuration**: API URL loaded from `.env` file (not committed)
-  **Organized Output**: Generated models saved to `output/` folder
-  **Flexible Options**: Tests both textured and non-textured generation
-  **Modern Tooling**: Uses `uv` for fast dependency management

## Setup

1. Make sure you have `uv` installed:
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Configure the API URL in `.env`:
   ```bash
   # Create or modify the .env file with your API server URL:
   API_BASE_URL=http://your-api-server:port
   ```

3. Add test images to the `images/` folder (PNG, JPG, JPEG supported)

## Usage

Run the test script using `uv` (automatically installs dependencies):
```bash
uv run main.py
```

The script will:
1. Test the health endpoint
2. Generate a 3D model synchronously (fast, no texture)
3. Generate a 3D model asynchronously (with texture)
4. Save all generated models to the `output/` folder

## Dependencies

Dependencies are managed in `pyproject.toml` and automatically installed by `uv`:
- `requests>=2.31.0` - HTTP client for API calls
- `python-dotenv>=1.0.0` - Environment variable loading
- `pillow>=10.0.0` - Image processing

## Output

Generated 3D models are saved as GLB files in the `output/` directory with descriptive names:
- `{image_name}_sync_no_texture_nobg.glb` - Synchronous generation
- `{image_name}_async_textured_nobg.glb` - Asynchronous generation

## API Endpoints Tested

- `GET /health` - Health check
- `POST /generate` - Synchronous 3D generation
- `POST /send` - Start asynchronous generation
- `GET /status/{uid}` - Check generation status

## Configuration Options

The test script supports various generation parameters:
- `texture`: Enable/disable texture generation
- `remove_background`: Automatic background removal
- `seed`: Random seed for reproducible results
- `octree_resolution`: Mesh resolution (64-512)
- `num_inference_steps`: Generation steps (1-20)
- `guidance_scale`: Generation guidance (0.1-20.0)
- `num_chunks`: Processing chunks (1000-20000)
- `face_count`: Max faces for texture (1000-100000)

## Development

To add new dependencies:
```bash
uv add package-name
```

To run in development mode:
```bash
uv run --dev main.py
```

## Security

The `.env` file containing the API URL is included in `.gitignore` to prevent accidental commits of sensitive URLs.
