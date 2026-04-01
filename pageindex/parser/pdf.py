import pymupdf
import litellm
from pathlib import Path
from .protocol import ContentNode, ParsedDocument


class PdfParser:
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        path = Path(file_path)
        model = kwargs.get("model")
        nodes = []

        with pymupdf.open(str(path)) as doc:
            for i, page in enumerate(doc):
                text = page.get_text()
                tokens = litellm.token_counter(model=model, text=text) if text else 0
                nodes.append(ContentNode(
                    content=text or "",
                    tokens=tokens,
                    index=i + 1,  # 1-based
                ))

        doc_name = path.stem
        return ParsedDocument(doc_name=doc_name, nodes=nodes)
