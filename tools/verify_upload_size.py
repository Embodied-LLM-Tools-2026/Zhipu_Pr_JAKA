import argparse
import os
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from typing import Optional

from upload_image import get_upload_policy, upload_file_to_oss  # type: ignore


def read_local_image_stats(image_path: Path) -> dict:
    with Image.open(image_path) as img:
        width, height = img.size
    size_bytes = image_path.stat().st_size
    return {
        "width": width,
        "height": height,
        "bytes": size_bytes,
    }


def read_remote_image_stats(url: str) -> dict:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.content
    with Image.open(BytesIO(data)) as img:
        width, height = img.size
    return {
        "width": width,
        "height": height,
        "bytes": len(data),
    }


def get_signed_download_url(api_key: str, oss_url: str) -> str:
    endpoint = "https://dashscope.aliyuncs.com/api/v1/uploads"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    candidates = [oss_url]
    if oss_url.startswith("oss://"):
        candidates.append(oss_url[len("oss://"):])

    last_error: Optional[str] = None

    for candidate in candidates:
        params = {"action": "getUrl", "ossUrl": candidate}
        resp = requests.get(endpoint, headers=headers, params=params, timeout=10)
        if resp.ok:
            data = resp.json().get("data", {})
            url = data.get("url")
            if url:
                return url
            last_error = "响应中缺少 url 字段"
        else:
            last_error = f"HTTP {resp.status_code}: {resp.text}"

    if last_error:
        raise RuntimeError(f"获取下载链接失败: {last_error}")
    raise RuntimeError("获取下载链接失败：未知错误")


def upload_and_get_remote_url(api_key: str, model_name: str, image_path: Path) -> str:
    policy = get_upload_policy(api_key, model_name)
    oss_url = upload_file_to_oss(policy, str(image_path))
    return get_signed_download_url(api_key, oss_url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify image size before and after upload")
    parser.add_argument("image", type=Path, help="Path to the local image file")
    parser.add_argument("--api-key", dest="api_key", default=os.getenv("Zhipu_real_demo_API_KEY"), help="DashScope API key")
    parser.add_argument("--model", dest="model_name", default="qwen3-vl-plus", help="Model name for the upload")
    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError("DashScope API key not provided. Use --api-key or set Zhipu_real_demo_API_KEY")

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    local_stats = read_local_image_stats(args.image)
    remote_url = upload_and_get_remote_url(args.api_key, args.model_name, args.image)
    try:
        remote_stats = read_remote_image_stats(remote_url)
    except requests.HTTPError as err:
        print(f"下载上传后的图片失败: {err}")
        print("如403错误，可检查是否取得了带签名的下载URL")
        raise

    print("Local image:")
    print(f"  size: {local_stats['width']}x{local_stats['height']}")
    print(f"  bytes: {local_stats['bytes']}")

    print("Uploaded image:")
    print(f"  url: {remote_url}")
    print(f"  size: {remote_stats['width']}x{remote_stats['height']}")
    print(f"  bytes: {remote_stats['bytes']}")

    diff_bytes = remote_stats['bytes'] - local_stats['bytes']
    print(f"Byte difference: {diff_bytes}")


if __name__ == "__main__":
    main()
