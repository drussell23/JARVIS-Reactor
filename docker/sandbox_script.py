#!/usr/bin/env python3
"""
Safe Scout Sandbox Script.

This script runs inside the Docker container to safely fetch web content.
It's designed to be minimal and secure.

Usage:
    python3 sandbox_script.py <url> [--output output.json] [--timeout 30]

Output:
    JSON object with:
    - success: bool
    - url: str
    - html_content: str (if success)
    - text_content: str (if success)
    - title: str (if success)
    - status_code: int
    - error: str (if failed)
    - timing: dict
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Limit resource usage
import resource

# Set memory limit (512MB)
MEMORY_LIMIT = 512 * 1024 * 1024
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, hard))
except (ValueError, resource.error):
    pass  # May fail in some environments


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Safe Scout Sandbox - Fetch web content safely",
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="URL to fetch",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file (JSON)",
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=30,
        help="Timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--wait-for",
        choices=["load", "domcontentloaded", "networkidle"],
        default="domcontentloaded",
        help="Wait condition (default: domcontentloaded)",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Take screenshot and save to file",
    )
    parser.add_argument(
        "--user-agent",
        default="Mozilla/5.0 (compatible; SafeScout/1.0; +https://github.com/drussell23/reactor-core)",
        help="Custom user agent",
    )
    parser.add_argument(
        "--no-javascript",
        action="store_true",
        help="Disable JavaScript execution",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )

    return parser.parse_args()


async def fetch_url(
    url: str,
    timeout: int = 30,
    wait_for: str = "domcontentloaded",
    user_agent: str = "",
    disable_js: bool = False,
    screenshot_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Fetch URL content using Playwright.

    Returns:
        Dict with success status, content, and timing info
    """
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

    result: Dict[str, Any] = {
        "success": False,
        "url": url,
        "html_content": None,
        "text_content": None,
        "title": None,
        "status_code": 0,
        "error": None,
        "timing": {
            "start": time.time(),
            "browser_launch": 0,
            "page_load": 0,
            "content_extract": 0,
            "total": 0,
        },
    }

    try:
        async with async_playwright() as p:
            # Launch browser with security settings
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--disable-setuid-sandbox",
                    "--no-sandbox",
                    "--disable-extensions",
                    "--disable-background-networking",
                    "--disable-default-apps",
                    "--disable-sync",
                    "--disable-translate",
                    "--metrics-recording-only",
                    "--mute-audio",
                    "--no-first-run",
                    "--safebrowsing-disable-auto-update",
                ],
            )
            result["timing"]["browser_launch"] = time.time() - result["timing"]["start"]

            # Create context with security settings
            context = await browser.new_context(
                user_agent=user_agent,
                java_script_enabled=not disable_js,
                bypass_csp=False,
                ignore_https_errors=False,
                # Block potentially dangerous content
                extra_http_headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                },
            )

            # Block tracking/ads
            await context.route(
                "**/*",
                lambda route: route.abort()
                if any(x in route.request.url for x in [
                    "analytics", "tracking", "ads", "pixel", "beacon",
                    "facebook.com", "twitter.com", "linkedin.com",
                ])
                else route.continue_()
            )

            page = await context.new_page()

            # Navigate with timeout
            page_load_start = time.time()
            response = await page.goto(
                url,
                timeout=timeout * 1000,
                wait_until=wait_for,
            )
            result["timing"]["page_load"] = time.time() - page_load_start

            if response:
                result["status_code"] = response.status

                if response.ok:
                    extract_start = time.time()

                    # Get content
                    result["html_content"] = await page.content()
                    result["title"] = await page.title()

                    # Extract text content
                    try:
                        result["text_content"] = await page.evaluate(
                            "() => document.body.innerText"
                        )
                    except Exception:
                        result["text_content"] = ""

                    result["timing"]["content_extract"] = time.time() - extract_start

                    # Take screenshot if requested
                    if screenshot_path:
                        await page.screenshot(path=str(screenshot_path))

                    result["success"] = True
                else:
                    result["error"] = f"HTTP {response.status}: {response.status_text}"

            await browser.close()

    except PlaywrightTimeout:
        result["error"] = f"Timeout after {timeout}s"
    except Exception as e:
        result["error"] = str(e)

    result["timing"]["total"] = time.time() - result["timing"]["start"]
    return result


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.version:
        print("Safe Scout Sandbox v1.0.0")
        return 0

    if not args.url:
        print("Error: URL is required", file=sys.stderr)
        print("Usage: python3 sandbox_script.py <url>", file=sys.stderr)
        return 1

    # Validate URL
    if not args.url.startswith(("http://", "https://")):
        print(f"Error: Invalid URL scheme: {args.url}", file=sys.stderr)
        return 1

    # Run fetch
    result = asyncio.run(fetch_url(
        url=args.url,
        timeout=args.timeout,
        wait_for=args.wait_for,
        user_agent=args.user_agent,
        disable_js=args.no_javascript,
        screenshot_path=args.screenshot,
    ))

    # Output result
    output_json = json.dumps(result, indent=2, default=str)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Output written to: {args.output}")
    else:
        print(output_json)

    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
