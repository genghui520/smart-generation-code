from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils import write_jsonl


DEFAULT_FOCAS_BASE_URL = "https://www.woody.vip/fanuc/"

# 这些是函数列表之外的总览页，提供 FOCAS 的连接、句柄、错误码等背景知识。
DEFAULT_FOCAS_PAGES = [
    "overview.htm",
    "general.htm",
    "dnc1.htm",
    "handle.htm",
    "log.htm",
    "errcode.htm",
]
MAX_REFERENCE_CHARS = 5000


@dataclass(slots=True)
class FocasFunctionRef:
    # 从 All/flist_All.xml 中解析出来的一条函数索引记录。
    name: str
    category: str
    explanation: str
    page_path: str
    xml_path: str


def build_focas_chunks(
    output_path: Path,
    *,
    base_url: str = DEFAULT_FOCAS_BASE_URL,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch FOCAS reference pages and write RAG chunks as JSONL."""

    # 统一 base_url 末尾的斜杠，避免 urljoin 拼接时吞掉路径。
    base_url = ensure_trailing_slash(base_url)
    chunks: list[dict[str, Any]] = []

    # 第一步：抓取总览页，例如 overview/general/errcode。
    # 这些页面不是具体函数，但会告诉 RAG 系统 FOCAS 的整体使用规则。
    for page_path in DEFAULT_FOCAS_PAGES:
        url = urllib.parse.urljoin(base_url, page_path)
        try:
            html = fetch_text(url)
        except OSError:
            continue
        chunks.extend(chunks_from_reference_page(page_path, url, html))

    # 第二步：读取函数总表 All/flist_All.xml。
    # 它只负责告诉我们有哪些函数、函数属于什么分类、详情页在哪里。
    function_refs = fetch_function_refs(base_url)
    if limit is not None:
        # 调试用：只处理前 N 个函数，避免每次测试都下载全部 717 个函数页。
        function_refs = function_refs[:limit]

    # 第三步：逐个函数下载详情 XML，并拆成多个 RAG chunk。
    for ref in function_refs:
        # catalog chunk 来自总表，至少包含函数名、分类、简短说明。
        # 即使详情页下载失败，也保留这个函数索引信息。
        chunks.append(function_catalog_chunk(ref, base_url))
        url = urllib.parse.urljoin(base_url, ref.xml_path)
        try:
            xml_bytes = fetch_bytes(url)
            chunks.extend(chunks_from_function_xml(ref, url, xml_bytes))
        except (OSError, ET.ParseError, UnicodeError):
            continue

    # 第四步：写出 JSONL。后续 embedding 阶段会逐行读取这个文件。
    write_jsonl(output_path, chunks)
    return chunks


def fetch_function_refs(base_url: str = DEFAULT_FOCAS_BASE_URL) -> list[FocasFunctionRef]:
    # All/flist_All.xml 是函数参考的总目录。
    base_url = ensure_trailing_slash(base_url)
    xml_bytes = fetch_bytes(urllib.parse.urljoin(base_url, "All/flist_All.xml"))
    return parse_function_refs(xml_bytes)


def parse_function_refs(xml_bytes: bytes) -> list[FocasFunctionRef]:
    # 将函数总表 XML 解析成 FocasFunctionRef 列表。
    root = parse_xml(xml_bytes)
    refs: list[FocasFunctionRef] = []

    # chapter 是函数分类，例如 Handle、Position、Program、PMC。
    for chapter in root.findall(".//chapter"):
        category = clean_text(text_content(chapter.find("title")))
        for item in chapter.findall("item"):
            # item 是具体函数索引，包含函数名、HTML 页面路径、简短说明。
            name = clean_text(text_content(item.find("fname")))
            page_path = clean_text(text_content(item.find("fpage")))
            explanation = clean_text(text_content(item.find("explanation")))
            if not name or not page_path:
                continue
            # 列表里给的是 .htm 框架页，真正可解析的详情在同名 .xml。
            xml_path = function_page_to_xml_path(page_path)
            refs.append(
                FocasFunctionRef(
                    name=name,
                    category=category,
                    explanation=explanation,
                    page_path=page_path,
                    xml_path=xml_path,
                )
            )

    return refs


def chunks_from_function_xml(
    ref: FocasFunctionRef, source_url: str, xml_bytes: bytes
) -> list[dict[str, Any]]:
    # 个别 URL 可能返回空内容或占位文件，直接跳过。
    if len(xml_bytes.strip()) < 20:
        return []

    # 函数详情 XML 里通常有 declare/doc/argument/errcode/option/example。
    root = parse_xml(xml_bytes)
    function_name = clean_text(text_content(root.find(".//title"))) or ref.name
    chunks: list[dict[str, Any]] = []

    # 函数原型 chunk：用于回答“怎么调用、参数顺序是什么”。
    prototype = clean_text(text_content(root.find(".//declare")))
    if prototype:
        chunks.append(
            make_chunk(
                ref,
                source_url,
                "prototype",
                f"Function: {function_name}\nCategory: {ref.category}\nDeclaration:\n{prototype}",
            )
        )

    # 功能说明 chunk：用于回答“这个函数做什么、什么时候用”。
    doc = clean_text(text_content(root.find(".//doc")))
    if doc:
        chunks.append(
            make_chunk(
                ref,
                source_url,
                "overview",
                f"Function: {function_name}\nCategory: {ref.category}\nPurpose:\n{doc}",
            )
        )

    # 参数说明 chunk：用于回答“每个参数输入/输出含义是什么”。
    arguments = argument_text(root)
    if arguments:
        chunks.append(
            make_chunk(
                ref,
                source_url,
                "arguments",
                f"Function: {function_name}\nArguments:\n{arguments}",
            )
        )

    # 错误码 chunk：用于回答“返回 EW_xxx 时怎么处理”。
    errcodes = named_items_text(root.find(".//errcode"))
    if errcodes:
        chunks.append(
            make_chunk(
                ref,
                source_url,
                "error_codes",
                f"Function: {function_name}\nReturn/error codes:\n{errcodes}",
            )
        )

    # 选项/注意事项 chunk：用于回答“需要哪些 CNC option 或使用限制”。
    option = clean_text(text_content(root.find(".//option")))
    if option:
        chunks.append(
            make_chunk(
                ref,
                source_url,
                "options",
                f"Function: {function_name}\nRequired options and notes:\n{option}",
            )
        )

    # 示例代码 chunk：不是每个函数都有，但对生成调用脚本很有价值。
    example = clean_text(text_content(root.find(".//example")))
    if example:
        chunks.append(
            make_chunk(
                ref,
                source_url,
                "example",
                f"Function: {function_name}\nExample:\n{example}",
            )
        )

    return chunks


def chunks_from_reference_page(page_path: str, source_url: str, html: str) -> list[dict[str, Any]]:
    # 将普通 HTML 总览页去标签、清洗文本，再按长度拆成 reference_page chunks。
    text = clean_text(strip_html(html))
    if not text:
        return []

    title = Path(page_path).stem
    parts = split_text(text, max_chars=MAX_REFERENCE_CHARS)
    chunks = []
    for index, part in enumerate(parts, start=1):
        chunks.append({
            "chunk_id": f"focas-reference-{title}-{index:03d}",
            "chunk_index": index,
            "chunk_total": len(parts),
            "protocol": "focas",
            "source_type": "html_reference",
            "source_url": source_url,
            "source_path": page_path,
            "doc_type": title,
            "chunk_type": "reference_page",
            "text": part,
        })
    return chunks


def function_catalog_chunk(ref: FocasFunctionRef, base_url: str) -> dict[str, Any]:
    # catalog chunk 是函数总表的轻量摘要，方便粗粒度检索。
    return make_chunk(
        ref,
        urllib.parse.urljoin(ensure_trailing_slash(base_url), ref.xml_path),
        "catalog",
        (
            f"Function: {ref.name}\n"
            f"Category: {ref.category}\n"
            f"Summary: {ref.explanation}"
        ),
    )


def make_chunk(
    ref: FocasFunctionRef, source_url: str, chunk_type: str, text: str
) -> dict[str, Any]:
    # 所有函数类 chunk 使用统一 metadata，后续 embedding 和检索都靠这些字段过滤。
    return {
        "chunk_id": f"focas-{safe_id(ref.name)}-{chunk_type}",
        "protocol": "focas",
        "source_type": "focas_xml",
        "source_url": source_url,
        "source_path": ref.xml_path,
        "category": ref.category,
        "function": ref.name,
        "chunk_type": chunk_type,
        "text": text,
    }


def argument_text(root: ET.Element) -> str:
    # 把 XML 参数表转成适合 embedding 的短文本列表。
    argument = root.find(".//argument")
    if argument is None:
        return ""

    rows: list[str] = []
    for item in argument.findall(".//item"):
        # 常见字段：name/type/content，例如 FlibHndl(in)、position(out)。
        name = clean_text(text_content(item.find("name")))
        direction = clean_text(text_content(item.find("type")))
        content = clean_text(text_content(item.find("content")))
        if name or content:
            prefix = name
            if direction:
                prefix = f"{prefix} ({direction})" if prefix else direction
            rows.append(f"- {prefix}: {content}".strip())

    return "\n".join(rows) if rows else clean_text(text_content(argument))


def named_items_text(element: ET.Element | None) -> str:
    # errcode 这类节点通常也是 item/name/content 结构，这里统一转成文本。
    if element is None:
        return ""

    rows: list[str] = []
    for item in element.findall(".//item"):
        name = clean_text(text_content(item.find("name")))
        content = clean_text(text_content(item.find("content")))
        if name or content:
            rows.append(f"- {name}: {content}".strip())

    return "\n".join(rows) if rows else clean_text(text_content(element))


def function_page_to_xml_path(page_path: str) -> str:
    # 函数总表给出的是 ../Position/cnc_rdposition.htm。
    # 实际详情内容在 Position/cnc_rdposition.xml。
    path = page_path.replace("\\", "/")
    while path.startswith("../"):
        path = path[3:]
    if path.lower().endswith(".htm"):
        path = path[:-4] + ".xml"
    return path


def text_content(element: ET.Element | None) -> str:
    # 提取 XML 节点下所有文本，包含嵌套 table/pre/list 的内容。
    if element is None:
        return ""
    return "".join(element.itertext())


def strip_html(html: str) -> str:
    # 粗略 HTML 转文本：去掉脚本/样式，保留换行边界。
    text = re.sub(r"(?is)<script.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?</style>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|li|tr|h[1-6]|td|th)>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return text


def clean_text(text: str) -> str:
    # 归一化空白，避免生成的 chunk 里出现大量多余空格和空行。
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, *, max_chars: int) -> list[str]:
    # 总览页可能很长；这里按段落切成较小块，避免单个 chunk 过长。
    if len(text) <= max_chars:
        return [text]

    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if current and current_len + len(paragraph) + 1 > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0

        if len(paragraph) > max_chars:
            for start in range(0, len(paragraph), max_chars):
                piece = paragraph[start : start + max_chars].strip()
                if piece:
                    chunks.append(piece)
            continue

        current.append(paragraph)
        current_len += len(paragraph) + 1

    if current:
        chunks.append("\n".join(current))

    return chunks


def safe_id(value: str) -> str:
    # 把函数名或标题变成适合放进 chunk_id 的安全字符串。
    return re.sub(r"[^a-zA-Z0-9_]+", "-", value).strip("-").lower()


def ensure_trailing_slash(url: str) -> str:
    # urljoin 对不带尾斜杠的 base URL 行为不同，所以统一补齐。
    return url if url.endswith("/") else f"{url}/"


def fetch_bytes(url: str) -> bytes:
    # 下载原始字节。XML 需要保留原始编码，后面再统一 decode。
    request = urllib.request.Request(url, headers={"User-Agent": "smart-traffic-agent/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def fetch_text(url: str) -> str:
    # 下载 HTML 文本页并自动尝试常见编码。
    data = fetch_bytes(url)
    return decode_bytes(data)


def parse_xml(xml_bytes: bytes) -> ET.Element:
    # FOCAS XML 声明是 Shift_JIS；先手动解码成 Unicode，再交给 ElementTree。
    text = decode_bytes(xml_bytes)
    text = re.sub(r"^\s*<\?xml[^>]*\?>", "", text)
    return ET.fromstring(text)


def decode_bytes(data: bytes) -> str:
    # FOCAS 网页/XML 混用了 utf-8、Shift_JIS、iso-8859-1，这里按顺序尝试。
    for encoding in ("utf-8", "shift_jis", "iso-8859-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")
