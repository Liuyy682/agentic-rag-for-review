import base64
import mimetypes
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import unquote

import httpx
from PIL import Image

import config


IMAGE_MARKDOWN_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

IMAGE_ANALYSIS_PROMPT = """你在为 RAG 知识库生成图片检索文本，不是在写视觉审美描述。

附近正文上下文：
{context_text}

判断规则：
- 如果图片只是装饰、logo、图标、页眉页脚、简单箭头、分隔线，或没有独立知识价值，严格只输出：SKIP_IMAGE
- 如果图片内容已经被附近正文完整表达，严格只输出：SKIP_IMAGE
- 只描述图片对理解本文档主题有帮助的信息。
- 优先结合附近正文解释图片的专业含义，不要只描述“左边是三角形、右边是圆形”这类几何外观。
- 不要编造图片中不存在的信息。

输出规则：
- 不要使用 Markdown 标题。
- 不要输出寒暄、分析过程或解释。
- 使用以下纯文本格式：
OCR: 图片中的原始可见文字；没有则写“无”
RAG_SUMMARY: 1-3 句面向检索的知识摘要，说明图表达的概念、流程、关系或结论
KEY_TERMS: 用逗号分隔的关键词、缩写、协议名、设备名、数值
"""


@dataclass(frozen=True)
class ImageAnalysisTask:
    match_start: int
    match_end: int
    markdown_image: str
    image_path: Path
    context_text: str


class LocalVLMImageDescriber:
    def __init__(
        self,
        base_url: str = config.LOCAL_VLM_BASE_URL,
        api_key: str = config.LOCAL_VLM_API_KEY,
        model: str = config.LOCAL_VLM_MODEL,
        timeout_seconds: float = config.LOCAL_VLM_TIMEOUT_SECONDS,
        max_tokens: int = config.LOCAL_VLM_MAX_TOKENS,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens

    def describe_image(self, image_path: Path, context_text: str = "") -> str:
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        prompt = IMAGE_ANALYSIS_PROMPT.format(context_text=context_text.strip() or "无")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


def enhance_markdown_image_references(
    markdown_text: str,
    markdown_dir: Path | None = None,
    image_root: Path | None = None,
    describe_image: Callable[[Path, str], str] | None = None,
) -> str:
    if describe_image is None:
        describe_image = LocalVLMImageDescriber().describe_image

    tasks = _collect_image_analysis_tasks(markdown_text, markdown_dir, image_root)
    if not tasks:
        return markdown_text

    descriptions = _analyze_images(tasks, describe_image)
    replacements = {}
    for task in tasks:
        description = descriptions.get(task.image_path, "").strip()
        if not description or description == "SKIP_IMAGE":
            continue
        replacements[(task.match_start, task.match_end)] = (
            f"{task.markdown_image}\n\n"
            f"<!-- image-analysis:start\n"
            f"{description}\n"
            f"image-analysis:end -->"
        )

    if not replacements:
        return markdown_text

    output_parts = []
    cursor = 0
    for (start, end), replacement in sorted(replacements.items()):
        output_parts.append(markdown_text[cursor:start])
        output_parts.append(replacement)
        cursor = end
    output_parts.append(markdown_text[cursor:])
    return "".join(output_parts)


def _collect_image_analysis_tasks(
    markdown_text: str,
    markdown_dir: Path | None = None,
    image_root: Path | None = None,
) -> list[ImageAnalysisTask]:
    tasks: list[ImageAnalysisTask] = []
    seen_paths: set[Path] = set()

    for match in IMAGE_MARKDOWN_RE.finditer(markdown_text):
        image_ref = match.group(1)
        image_path = resolve_image_path(image_ref, markdown_dir, image_root)
        if image_path is None:
            continue
        if image_path in seen_paths:
            continue
        if len(tasks) >= config.VLM_IMAGE_MAX_PER_DOC:
            break
        if should_skip_image_by_size(image_path):
            continue
        tasks.append(
            ImageAnalysisTask(
                match_start=match.start(),
                match_end=match.end(),
                markdown_image=match.group(0),
                image_path=image_path,
                context_text=extract_image_context(markdown_text, match.start(), match.end()),
            )
        )
        seen_paths.add(image_path)

    return tasks


def _analyze_images(
    tasks: list[ImageAnalysisTask],
    describe_image: Callable[[Path, str], str],
) -> dict[Path, str]:
    worker_count = min(max(1, config.VLM_IMAGE_ANALYSIS_WORKERS), 4)
    if worker_count == 1 or len(tasks) == 1:
        results = {}
        for task in tasks:
            result = _safe_describe_image(task, describe_image)
            if result is not None:
                results[task.image_path] = result
        return results

    results = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_task = {
            executor.submit(_safe_describe_image, task, describe_image): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            if result is not None:
                results[task.image_path] = result
    return results


def _safe_describe_image(
    task: ImageAnalysisTask,
    describe_image: Callable[[Path, str], str],
) -> str | None:
    try:
        return describe_image(task.image_path, task.context_text)
    except Exception as exc:
        print(f"Warning: could not analyze image {task.image_path}: {exc}")
        return None


def should_skip_image_by_size(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception:
        return False
    return width < config.VLM_IMAGE_MIN_WIDTH or height < config.VLM_IMAGE_MIN_HEIGHT


def extract_image_context(markdown_text: str, start: int, end: int) -> str:
    context_chars = config.VLM_IMAGE_CONTEXT_CHARS
    before = markdown_text[max(0, start - context_chars):start]
    after = markdown_text[end:min(len(markdown_text), end + context_chars)]
    context = f"{before}\n{after}"
    context = IMAGE_MARKDOWN_RE.sub("", context)
    context = re.sub(r"<!--.*?-->", "", context, flags=re.DOTALL)
    context = re.sub(r"\s+", " ", context)
    return context.strip()


def resolve_image_path(
    image_ref: str,
    markdown_dir: Path | None = None,
    image_root: Path | None = None,
) -> Path | None:
    raw_ref = _clean_image_ref(image_ref)
    if raw_ref.startswith(("http://", "https://", "data:")):
        return None

    ref_path = Path(raw_ref)
    candidates = []
    if ref_path.is_absolute():
        candidates.append(ref_path)
    if markdown_dir is not None:
        candidates.append(markdown_dir / ref_path)
    if image_root is not None:
        candidates.append(image_root / ref_path)
        candidates.append(image_root / ref_path.name)
    candidates.append(ref_path)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _clean_image_ref(image_ref: str) -> str:
    cleaned = image_ref.strip()
    if cleaned.startswith("<") and ">" in cleaned:
        cleaned = cleaned[1:cleaned.index(">")]
    else:
        title_match = re.match(r"(.+?)\s+['\"].*['\"]$", cleaned)
        if title_match:
            cleaned = title_match.group(1)
    return unquote(cleaned)
