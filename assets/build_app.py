"""
Builds the single-file app: app_template.html + engine.js + market.json -> AITradingBot.html

Why inline everything: the page it replaces pulled Chart.js from a CDN, so it broke
without a network and could not be opened offline. This one makes no external requests.

RUN
    python3 build_app.py
"""
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "AITradingBot.html"


def main():
    tpl = (HERE / "app_template.html").read_text(encoding="utf-8")
    engine = (HERE / "engine.js").read_text(encoding="utf-8")
    market = json.loads((HERE / "market.json").read_text(encoding="utf-8"))

    # a guard comment so a future edit cannot silently re-introduce a CDN dependency
    engine_block = ("/* ==== engine.js, inlined verbatim ==== */\n" + engine +
                    "\n/* ==== end engine ==== */")
    data_block = ("/* Real daily closes, fetched once at build time.\n"
                  "   Sources: CoinGecko (crypto), Yahoo Finance (equities).\n"
                  "   Embedded so the page needs no network and works offline. */\n"
                  "const DATA = " + json.dumps(market, separators=(",", ":")) + ";")

    assert "/*__ENGINE__*/" in tpl, "engine slot missing"
    assert "/*__DATA__*/" in tpl, "data slot missing"
    html = tpl.replace("/*__ENGINE__*/", engine_block, 1)
    html = html.replace("/*__DATA__*/", data_block, 1)

    # no external anything
    ext = re.findall(r'(?:src|href)\s*=\s*["\'](https?:)?//[^"\']+', html)
    assert not ext, f"external references remain: {ext}"

    OUT.write_text(html, encoding="utf-8")
    print(f"  wrote {OUT}")
    print(f"  size  {OUT.stat().st_size:,} bytes")
    print(f"  series embedded: {list(market.keys())}")
    print(f"  external refs  : {len(ext)}")
    return OUT


if __name__ == "__main__":
    main()
