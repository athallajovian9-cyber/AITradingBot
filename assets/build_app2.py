"""
Builds the advanced single-file app: app2_template.html + engine2.js + market2.json
                                   -> AITradingBot.html

Everything is inlined so the page makes no external requests and works offline.

RUN
    python3 build_app2.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "AITradingBot.html"


def main():
    tpl = (HERE / "app2_template.html").read_text(encoding="utf-8")
    engine = (HERE / "engine2.js").read_text(encoding="utf-8")
    market = json.loads((HERE / "market2.json").read_text(encoding="utf-8"))

    engine_block = "/* ==== engine2.js, inlined verbatim ==== */\n" + engine + \
                   "\n/* ==== end engine ==== */"
    data_block = ("/* Real daily OHLCV, fetched at build time from Yahoo Finance.\n"
                  "   Embedded so the page needs no network. */\n"
                  "const DATA = " + json.dumps(market, separators=(",", ":")) + ";")

    assert "/*__ENGINE__*/" in tpl, "engine slot missing"
    assert "/*__DATA__*/" in tpl, "data slot missing"
    html = tpl.replace("/*__ENGINE__*/", engine_block, 1)
    html = html.replace("/*__DATA__*/", data_block, 1)

    ext = re.findall(r'(?:src|href)\s*=\s*["\'](?:https?:)?//[^"\']+', html)
    assert not ext, f"external references remain: {ext}"

    OUT.write_text(html, encoding="utf-8")
    print(f"  wrote {OUT}")
    print(f"  size  {OUT.stat().st_size:,} bytes")
    print(f"  assets: {len(market)}  external refs: {len(ext)}")
    return OUT


if __name__ == "__main__":
    main()
