"""Serve the built hash-routed frontend after every API route is registered."""

from pathlib import Path

from fastapi import FastAPI
from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles


class FrontendFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        # Never reinterpret an unknown API route as a frontend asset or HTML.
        normalized = path.replace("\\", "/").lstrip("/").casefold()
        if normalized == "api" or normalized.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")
        return await super().get_response(path, scope)


def mount_static_site(app: FastAPI, directory: Path | None = None) -> bool:
    target = Path(directory) if directory is not None else Path(__file__).resolve().parents[1] / "dist"
    if not target.is_dir() or not (target / "index.html").is_file():
        return False
    app.mount("/", FrontendFiles(directory=str(target.resolve()), html=True, follow_symlink=False), name="frontend")
    return True
