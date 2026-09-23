from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Run the API with settings that work locally and on container platforms."""
    host = os.getenv("GRIDTOEV_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("gridtoev.api:app", host=host, port=port)


if __name__ == "__main__":
    main()
