from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlencode, urlparse

import requests

try:
    import websocket  # type: ignore
except Exception:  # pragma: no cover
    websocket = None


@dataclass(frozen=True)
class UploadedFile:
    name: str
    subfolder: str = ""
    type: str = "input"


class ComfyUIError(RuntimeError):
    pass


class ComfyUIClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def new_client_id(self) -> str:
        return str(uuid.uuid4())

    def get_system_stats(self, timeout_s: float = 10.0) -> dict[str, Any]:
        r = requests.get(f"{self.base_url}/system_stats", timeout=timeout_s)
        r.raise_for_status()
        return r.json()

    def get_object_info(self, timeout_s: float = 30.0) -> dict[str, Any]:
        r = requests.get(f"{self.base_url}/object_info", timeout=timeout_s)
        r.raise_for_status()
        return r.json()

    def upload_image(self, path: str, timeout_s: float = 60.0) -> UploadedFile:
        with open(path, "rb") as f:
            files = {"image": (os.path.basename(path), f, "application/octet-stream")}
            r = requests.post(f"{self.base_url}/upload/image", files=files, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        return UploadedFile(name=data.get("name") or data.get("filename") or data.get("image"), subfolder=data.get("subfolder", ""), type=data.get("type", "input"))

    def upload_mask(self, path: str, original_ref: Optional[UploadedFile] = None, timeout_s: float = 60.0) -> UploadedFile:
        """
        Upload a mask image via ComfyUI's `/upload/mask` endpoint.

        Notes:
        - `/upload/mask` is not a generic "upload a standalone mask" route; it expects an `original_ref`
          so the server can apply the uploaded alpha channel to the original image.
        - If `original_ref` is omitted, we fall back to `/upload/image` so workflows can load the mask
          separately (e.g. via `LoadImageMask`).
        """
        if original_ref is None:
            return self.upload_image(path, timeout_s=timeout_s)
        with open(path, "rb") as f:
            files = {"image": (os.path.basename(path), f, "application/octet-stream")}
            data = {
                "original_ref": json.dumps(
                    {
                        "filename": original_ref.name,
                        "subfolder": original_ref.subfolder,
                        "type": original_ref.type,
                    }
                )
            }
            r = requests.post(f"{self.base_url}/upload/mask", data=data, files=files, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        return UploadedFile(name=data.get("name") or data.get("filename") or data.get("image"), subfolder=data.get("subfolder", ""), type=data.get("type", "input"))

    def submit_prompt(self, prompt_graph: dict[str, Any], client_id: str, timeout_s: float = 30.0) -> str:
        payload = {"prompt": prompt_graph, "client_id": client_id}
        r = requests.post(f"{self.base_url}/prompt", json=payload, timeout=timeout_s)
        try:
            data = r.json()
        except Exception:
            data = None
        if r.status_code != 200:
            if isinstance(data, dict):
                raise ComfyUIError(f"ComfyUI rejected prompt ({r.status_code}): error={data.get('error')} node_errors={data.get('node_errors')}")
            r.raise_for_status()
        if not isinstance(data, dict):
            raise ComfyUIError(f"ComfyUI returned non-JSON response for /prompt: status={r.status_code}")
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError(f"Missing prompt_id in response: {data}")
        return str(prompt_id)

    def interrupt(self, timeout_s: float = 5.0) -> None:
        try:
            r = requests.post(f"{self.base_url}/interrupt", timeout=timeout_s)
            if r.status_code not in (200, 204, 404):
                r.raise_for_status()
        except Exception:
            return

    def get_history(self, prompt_id: str, timeout_s: float = 30.0) -> dict[str, Any]:
        r = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=timeout_s)
        if r.status_code == 404:
            r = requests.get(f"{self.base_url}/history", timeout=timeout_s)
            r.raise_for_status()
            data = r.json()
            return data.get(prompt_id) or {}
        r.raise_for_status()
        return r.json()

    def download_view(self, filename: str, subfolder: str = "", type_: str = "output", timeout_s: float = 60.0) -> bytes:
        params = {"filename": filename, "type": type_}
        if subfolder:
            params["subfolder"] = subfolder
        url = f"{self.base_url}/view?{urlencode(params)}"
        r = requests.get(url, timeout=timeout_s)
        r.raise_for_status()
        return r.content

    def wait_for_completion(
        self,
        prompt_id: str,
        client_id: str,
        timeout_s: float = 600.0,
        poll_s: float = 1.0,
    ) -> None:
        if websocket is not None:
            try:
                self._wait_ws(prompt_id=prompt_id, client_id=client_id, timeout_s=timeout_s)
                return
            except Exception:
                pass
        self._wait_poll(prompt_id=prompt_id, timeout_s=timeout_s, poll_s=poll_s)

    def _wait_poll(self, prompt_id: str, timeout_s: float, poll_s: float) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            history = self.get_history(prompt_id)
            if history:
                return
            time.sleep(poll_s)
        raise ComfyUIError(f"Timed out waiting for prompt completion (poll) after {timeout_s}s: {prompt_id}")

    def _wait_ws(self, prompt_id: str, client_id: str, timeout_s: float) -> None:
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        host = parsed.netloc
        ws_url = f"{scheme}://{host}/ws?clientId={client_id}"
        ws = websocket.WebSocket()  # type: ignore[attr-defined]
        ws.settimeout(timeout_s)
        ws.connect(ws_url)
        deadline = time.time() + timeout_s
        try:
            while time.time() < deadline:
                raw = ws.recv()
                if not raw:
                    continue
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="ignore")
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if msg.get("type") == "executing":
                    data = msg.get("data") or {}
                    if str(data.get("prompt_id")) == str(prompt_id) and data.get("node") is None:
                        return
        finally:
            try:
                ws.close()
            except Exception:
                pass
        raise ComfyUIError(f"Timed out waiting for prompt completion (ws) after {timeout_s}s: {prompt_id}")
