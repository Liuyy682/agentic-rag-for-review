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


class PaddleOcrImageDescriber:
    """使用 PaddleOCR 从图片中提取文字，输出与 VLM 兼容的三字段格式。

    延迟加载 PaddleOCR 模型（首次调用时初始化），避免在未启用 OCR 时
    引入不必要的内存开销。无文字图片返回 "SKIP_IMAGE" 以与 VLM 的跳过
    语义保持一致。
    """

    def __init__(self, lang: str = "ch", use_gpu: bool = False):
        self._ocr = None
        self._lang = lang
        self._use_gpu = use_gpu

    @property
    def ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(lang=self._lang, use_gpu=self._use_gpu)
        return self._ocr

    def describe_image(self, image_path: Path, context_text: str = "") -> str:
        """对单张图片执行 OCR 并返回三字段格式结果。

        Args:
            image_path: 图片文件路径。
            context_text: 附近正文上下文（OCR 不使用，保留以兼容接口）。

        Returns:
            "SKIP_IMAGE" 如果未检测到文字，否则返回:
            OCR: <原始可见文字>
            RAG_SUMMARY: <前 200 字符摘要>
            KEY_TERMS: <jieba TextRank 关键词>
        """
        try:
            results = self.ocr.ocr(str(image_path))
        except Exception:
            return "SKIP_IMAGE"

        texts: list[str] = []
        if results and results[0]:
            for line in results[0]:
                text = line[1][0].strip()
                if text:
                    texts.append(text)

        if not texts:
            return "SKIP_IMAGE"

        ocr_text = "\n".join(texts)
        summary = ocr_text[:200].replace("\n", " ")
        keywords = self._extract_keywords(ocr_text)
        return f"OCR: {ocr_text}\nRAG_SUMMARY: {summary}\nKEY_TERMS: {keywords}"

    def _extract_keywords(self, text: str, topk: int = 10) -> str:
        """使用 jieba TextRank 从 OCR 文本中提取关键词。"""
        try:
            import jieba.analyse
            keywords = jieba.analyse.textrank(text, topK=topk)
            return ", ".join(keywords)
        except Exception:
            return ""


def create_image_describer() -> Callable[[Path, str], str] | None:
    """根据 IMAGE_ANALYSIS_ENGINE 配置创建图片描述器。

    Returns:
        描述器 callable，如果引擎为 "none" 则返回 None。

    Raises:
        ValueError: 未知的引擎名称。
    """
    engine = config.IMAGE_ANALYSIS_ENGINE
    if engine == "paddleocr":
        describer = PaddleOcrImageDescriber(
            lang=config.PADDLEOCR_LANG,
            use_gpu=config.PADDLEOCR_USE_GPU,
        )
        return describer.describe_image
    elif engine == "vlm":
        return LocalVLMImageDescriber().describe_image
    elif engine == "none":
        return None
    else:
        raise ValueError(f"Unknown IMAGE_ANALYSIS_ENGINE: {engine}")


def _resolve_default_describer() -> Callable[[Path, str], str]:
    """解析默认的图片描述器（当调用者未显式传入时）。"""
    describer = create_image_describer()
    if describer is not None:
        return describer
    # 兜底：未配置时回退到 VLM（保持历史行为）
    return LocalVLMImageDescriber().describe_image


def _is_ocr_engine() -> bool:
    """当前引擎是否为 OCR 引擎（使用 OCR 配置阈值）。"""
    return config.IMAGE_ANALYSIS_ENGINE == "paddleocr"


def _default_max_per_doc() -> int:
    if _is_ocr_engine():
        return config.OCR_IMAGE_MAX_PER_DOC
    return config.VLM_IMAGE_MAX_PER_DOC


def _default_min_width() -> int:
    if _is_ocr_engine():
        return config.OCR_IMAGE_MIN_WIDTH
    return config.VLM_IMAGE_MIN_WIDTH


def _default_min_height() -> int:
    if _is_ocr_engine():
        return config.OCR_IMAGE_MIN_HEIGHT
    return config.VLM_IMAGE_MIN_HEIGHT


def _default_worker_count() -> int:
    if _is_ocr_engine():
        return config.OCR_IMAGE_ANALYSIS_WORKERS
    return config.VLM_IMAGE_ANALYSIS_WORKERS


def enhance_markdown_image_references(
    markdown_text: str,
    markdown_dir: Path | None = None,
    image_root: Path | None = None,
    describe_image: Callable[[Path, str], str] | None = None,
    max_per_doc: int | None = None,
    min_width: int | None = None,
    min_height: int | None = None,
    worker_count: int | None = None,
) -> str:
    if describe_image is None:
        describe_image = _resolve_default_describer()

    max_per_doc = max_per_doc if max_per_doc is not None else _default_max_per_doc()
    min_width = min_width if min_width is not None else _default_min_width()
    min_height = min_height if min_height is not None else _default_min_height()
    worker_count = worker_count if worker_count is not None else _default_worker_count()

    tasks = _collect_image_analysis_tasks(
        markdown_text, markdown_dir, image_root,
        max_per_doc=max_per_doc, min_width=min_width, min_height=min_height,
    )
    if not tasks:
        return markdown_text

    descriptions = _analyze_images(tasks, describe_image, worker_count=worker_count)
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
    max_per_doc: int | None = None,
    min_width: int | None = None,
    min_height: int | None = None,
) -> list[ImageAnalysisTask]:
    tasks: list[ImageAnalysisTask] = []
    seen_paths: set[Path] = set()

    max_per_doc = max_per_doc if max_per_doc is not None else _default_max_per_doc()
    min_width = min_width if min_width is not None else _default_min_width()
    min_height = min_height if min_height is not None else _default_min_height()

    for match in IMAGE_MARKDOWN_RE.finditer(markdown_text):
        image_ref = match.group(1)
        image_path = resolve_image_path(image_ref, markdown_dir, image_root)
        if image_path is None:
            continue
        if image_path in seen_paths:
            continue
        if len(tasks) >= max_per_doc:
            break
        if should_skip_image_by_size(image_path, min_width=min_width, min_height=min_height):
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
    worker_count: int | None = None,
) -> dict[Path, str]:
    worker_count = min(max(1, worker_count if worker_count is not None else _default_worker_count()), 4)
    if worker_count <= 1 or len(tasks) <= 1:
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


def should_skip_image_by_size(
    image_path: Path,
    min_width: int | None = None,
    min_height: int | None = None,
) -> bool:
    min_width = min_width if min_width is not None else _default_min_width()
    min_height = min_height if min_height is not None else _default_min_height()
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception:
        return False
    return width < min_width or height < min_height


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
