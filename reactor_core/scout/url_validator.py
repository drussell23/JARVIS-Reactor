"""
URL Validator for Safe Scout.

Provides:
- Domain allowlist management
- robots.txt compliance checking
- URL pattern validation
- DNS blocklist integration
- Safe Browsing API integration
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

logger = logging.getLogger(__name__)


class URLSafetyLevel(Enum):
    """Safety classification for URLs."""
    TRUSTED = "trusted"        # On explicit allowlist
    ALLOWED = "allowed"        # Passes all checks
    UNKNOWN = "unknown"        # Not on allowlist, needs verification
    RESTRICTED = "restricted"  # On greylist, limited access
    BLOCKED = "blocked"        # On blocklist or failed safety check
    MALICIOUS = "malicious"    # Detected as malware/phishing


class BlockReason(Enum):
    """Reasons a URL might be blocked."""
    NONE = "none"
    ROBOTS_TXT = "robots_txt"
    DOMAIN_BLOCKED = "domain_blocked"
    MALWARE_DETECTED = "malware_detected"
    PHISHING_DETECTED = "phishing_detected"
    PAYWALL_DETECTED = "paywall_detected"
    LOGIN_REQUIRED = "login_required"
    RATE_LIMITED = "rate_limited"
    INVALID_URL = "invalid_url"
    UNKNOWN_DOMAIN = "unknown_domain"


@dataclass
class URLValidationResult:
    """Result of URL validation."""
    url: str
    is_valid: bool
    safety_level: URLSafetyLevel
    block_reason: BlockReason = BlockReason.NONE
    message: str = ""
    domain: str = ""
    robots_txt_crawl_delay: Optional[float] = None
    checked_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "is_valid": self.is_valid,
            "safety_level": self.safety_level.value,
            "block_reason": self.block_reason.value,
            "message": self.message,
            "domain": self.domain,
            "robots_txt_crawl_delay": self.robots_txt_crawl_delay,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class DomainTrustLevel:
    """Trust configuration for a domain."""
    domain: str
    trust_level: URLSafetyLevel
    max_pages_per_session: int = 50
    crawl_delay_seconds: float = 2.0
    allowed_paths: List[str] = field(default_factory=list)  # Empty = all allowed
    blocked_paths: List[str] = field(default_factory=list)
    notes: str = ""


# Default trusted technical documentation domains
DEFAULT_TRUSTED_DOMAINS: List[DomainTrustLevel] = [
    # Official Python
    DomainTrustLevel("docs.python.org", URLSafetyLevel.TRUSTED, 100, 1.0),
    DomainTrustLevel("peps.python.org", URLSafetyLevel.TRUSTED, 50, 1.0),
    DomainTrustLevel("packaging.python.org", URLSafetyLevel.TRUSTED, 50, 1.0),

    # JavaScript/TypeScript
    DomainTrustLevel("developer.mozilla.org", URLSafetyLevel.TRUSTED, 100, 1.0),
    DomainTrustLevel("nodejs.org", URLSafetyLevel.TRUSTED, 50, 1.0),
    DomainTrustLevel("typescriptlang.org", URLSafetyLevel.TRUSTED, 50, 1.0),
    DomainTrustLevel("react.dev", URLSafetyLevel.TRUSTED, 100, 1.0),
    DomainTrustLevel("vuejs.org", URLSafetyLevel.TRUSTED, 50, 1.0),
    DomainTrustLevel("nextjs.org", URLSafetyLevel.TRUSTED, 50, 1.0),

    # Cloud Providers
    DomainTrustLevel("cloud.google.com", URLSafetyLevel.TRUSTED, 100, 2.0),
    DomainTrustLevel("firebase.google.com", URLSafetyLevel.TRUSTED, 50, 2.0),
    DomainTrustLevel("docs.aws.amazon.com", URLSafetyLevel.TRUSTED, 100, 2.0),
    DomainTrustLevel("learn.microsoft.com", URLSafetyLevel.TRUSTED, 100, 2.0),
    DomainTrustLevel("docs.github.com", URLSafetyLevel.TRUSTED, 50, 1.0),

    # ML/AI
    DomainTrustLevel("huggingface.co", URLSafetyLevel.TRUSTED, 100, 2.0),
    DomainTrustLevel("pytorch.org", URLSafetyLevel.TRUSTED, 50, 1.0),
    DomainTrustLevel("tensorflow.org", URLSafetyLevel.TRUSTED, 50, 1.0),
    DomainTrustLevel("keras.io", URLSafetyLevel.TRUSTED, 50, 1.0),
    DomainTrustLevel("langchain.com", URLSafetyLevel.TRUSTED, 50, 2.0),
    DomainTrustLevel("docs.anthropic.com", URLSafetyLevel.TRUSTED, 50, 2.0),
    DomainTrustLevel("platform.openai.com", URLSafetyLevel.TRUSTED, 50, 2.0),

    # Databases
    DomainTrustLevel("www.postgresql.org", URLSafetyLevel.TRUSTED, 50, 1.0),
    DomainTrustLevel("redis.io", URLSafetyLevel.TRUSTED, 50, 1.0),
    DomainTrustLevel("www.mongodb.com", URLSafetyLevel.TRUSTED, 50, 2.0),

    # DevOps
    DomainTrustLevel("docs.docker.com", URLSafetyLevel.TRUSTED, 50, 1.0),
    DomainTrustLevel("kubernetes.io", URLSafetyLevel.TRUSTED, 50, 1.0),
    DomainTrustLevel("www.terraform.io", URLSafetyLevel.TRUSTED, 50, 1.0),

    # Community (with restrictions)
    DomainTrustLevel("stackoverflow.com", URLSafetyLevel.ALLOWED, 30, 3.0,
                     allowed_paths=["/questions/"], notes="Q&A only"),
    DomainTrustLevel("github.com", URLSafetyLevel.ALLOWED, 50, 2.0,
                     blocked_paths=["/login", "/signup", "/settings"]),
    DomainTrustLevel("arxiv.org", URLSafetyLevel.ALLOWED, 20, 5.0,
                     allowed_paths=["/abs/", "/pdf/"]),
]

# Domains that should never be accessed
DEFAULT_BLOCKED_DOMAINS: List[str] = [
    # Social media (not technical)
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "tiktok.com",
    "linkedin.com",

    # Paywalled content
    "medium.com",  # Many articles are paywalled
    "substack.com",

    # Malware/suspicious
    "bit.ly",
    "tinyurl.com",
    "t.co",

    # Adult content
    # (additional patterns would be loaded from external blocklist)
]


@dataclass
class URLValidatorConfig:
    """Configuration for URL validator."""
    # Cache settings
    cache_robots_txt_hours: int = 24
    cache_validation_hours: int = 1

    # User agent
    user_agent: str = field(
        default_factory=lambda: os.getenv(
            "NIGHTSHIFT_USER_AGENT",
            "NightShift-Scout/1.0 (+https://github.com/drussell23/JARVIS-Reactor; training-data-collection)"
        )
    )

    # Safety settings
    require_https: bool = True
    max_redirects: int = 3
    timeout_seconds: int = 10

    # Domain lists
    additional_trusted_domains: List[str] = field(default_factory=list)
    additional_blocked_domains: List[str] = field(default_factory=list)

    # External integrations
    use_google_safe_browsing: bool = field(
        default_factory=lambda: bool(os.getenv("GOOGLE_SAFE_BROWSING_API_KEY"))
    )
    google_safe_browsing_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_SAFE_BROWSING_API_KEY")
    )

    # Rate limiting
    global_rate_limit_per_minute: int = 60


class RobotsTxtCache:
    """Cache for robots.txt files."""

    def __init__(self, cache_hours: int = 24):
        self._cache: Dict[str, Tuple[RobotFileParser, datetime]] = {}
        self._cache_duration = timedelta(hours=cache_hours)
        self._lock = asyncio.Lock()

    async def get_parser(
        self,
        domain: str,
        user_agent: str,
    ) -> Optional[RobotFileParser]:
        """Get or fetch robots.txt parser for domain."""
        async with self._lock:
            now = datetime.now()

            # Check cache
            if domain in self._cache:
                parser, cached_at = self._cache[domain]
                if now - cached_at < self._cache_duration:
                    return parser

            # Fetch robots.txt
            parser = await self._fetch_robots_txt(domain, user_agent)
            if parser:
                self._cache[domain] = (parser, now)

            return parser

    async def _fetch_robots_txt(
        self,
        domain: str,
        user_agent: str,
    ) -> Optional[RobotFileParser]:
        """Fetch and parse robots.txt."""
        try:
            import aiohttp

            robots_url = f"https://{domain}/robots.txt"

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    robots_url,
                    timeout=aiohttp.ClientTimeout(total=10),
                    headers={"User-Agent": user_agent},
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        parser = RobotFileParser()
                        parser.parse(content.splitlines())
                        logger.debug(f"Fetched robots.txt for {domain}")
                        return parser
                    elif response.status == 404:
                        # No robots.txt = all allowed
                        parser = RobotFileParser()
                        parser.allow_all = True
                        return parser
                    else:
                        logger.warning(
                            f"Failed to fetch robots.txt for {domain}: {response.status}"
                        )
                        return None

        except Exception as e:
            logger.error(f"Error fetching robots.txt for {domain}: {e}")
            return None

    def invalidate(self, domain: str) -> None:
        """Invalidate cache for domain."""
        self._cache.pop(domain, None)


class URLValidator:
    """
    Validates URLs for safe scouting.

    Performs:
    - Domain allowlist/blocklist checking
    - robots.txt compliance
    - URL pattern validation
    - Optional Safe Browsing API integration
    """

    def __init__(
        self,
        config: Optional[URLValidatorConfig] = None,
    ):
        self.config = config or URLValidatorConfig()

        # Initialize domain mappings
        self._trusted_domains: Dict[str, DomainTrustLevel] = {}
        self._blocked_domains: Set[str] = set()

        # Load defaults
        for domain_config in DEFAULT_TRUSTED_DOMAINS:
            self._trusted_domains[domain_config.domain] = domain_config

        for domain in DEFAULT_BLOCKED_DOMAINS:
            self._blocked_domains.add(domain)

        # Add additional domains from config
        for domain in self.config.additional_trusted_domains:
            if domain not in self._trusted_domains:
                self._trusted_domains[domain] = DomainTrustLevel(
                    domain, URLSafetyLevel.ALLOWED
                )

        for domain in self.config.additional_blocked_domains:
            self._blocked_domains.add(domain)

        # Initialize caches
        self._robots_cache = RobotsTxtCache(self.config.cache_robots_txt_hours)
        self._validation_cache: Dict[str, Tuple[URLValidationResult, datetime]] = {}
        self._cache_duration = timedelta(hours=self.config.cache_validation_hours)

    def add_trusted_domain(self, domain_config: DomainTrustLevel) -> None:
        """Add a trusted domain."""
        self._trusted_domains[domain_config.domain] = domain_config
        logger.info(f"Added trusted domain: {domain_config.domain}")

    def block_domain(self, domain: str, reason: str = "") -> None:
        """Block a domain."""
        self._blocked_domains.add(domain)
        logger.info(f"Blocked domain: {domain} ({reason})")

    async def validate(self, url: str) -> URLValidationResult:
        """
        Validate a URL for safe scouting.

        Args:
            url: The URL to validate

        Returns:
            URLValidationResult with safety assessment
        """
        # Check cache
        cache_key = hashlib.md5(url.encode()).hexdigest()
        if cache_key in self._validation_cache:
            result, cached_at = self._validation_cache[cache_key]
            if datetime.now() - cached_at < self._cache_duration:
                return result

        # Perform validation
        result = await self._validate_url(url)

        # Cache result
        self._validation_cache[cache_key] = (result, datetime.now())

        return result

    async def _validate_url(self, url: str) -> URLValidationResult:
        """Internal URL validation logic."""
        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            return URLValidationResult(
                url=url,
                is_valid=False,
                safety_level=URLSafetyLevel.BLOCKED,
                block_reason=BlockReason.INVALID_URL,
                message=f"Failed to parse URL: {e}",
            )

        # Extract domain
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain_normalized = domain[4:]
        else:
            domain_normalized = domain

        # Check for required HTTPS
        if self.config.require_https and parsed.scheme != "https":
            return URLValidationResult(
                url=url,
                is_valid=False,
                safety_level=URLSafetyLevel.BLOCKED,
                block_reason=BlockReason.INVALID_URL,
                message="HTTPS required",
                domain=domain,
            )

        # Check blocked domains
        if self._is_blocked_domain(domain):
            return URLValidationResult(
                url=url,
                is_valid=False,
                safety_level=URLSafetyLevel.BLOCKED,
                block_reason=BlockReason.DOMAIN_BLOCKED,
                message="Domain is on blocklist",
                domain=domain,
            )

        # Check trusted domains
        domain_config = self._get_domain_config(domain)
        if domain_config:
            # Check path restrictions
            path = parsed.path

            if domain_config.blocked_paths:
                for blocked_path in domain_config.blocked_paths:
                    if path.startswith(blocked_path):
                        return URLValidationResult(
                            url=url,
                            is_valid=False,
                            safety_level=URLSafetyLevel.RESTRICTED,
                            block_reason=BlockReason.ROBOTS_TXT,
                            message=f"Path {blocked_path} is blocked for this domain",
                            domain=domain,
                        )

            if domain_config.allowed_paths:
                allowed = False
                for allowed_path in domain_config.allowed_paths:
                    if path.startswith(allowed_path):
                        allowed = True
                        break

                if not allowed:
                    return URLValidationResult(
                        url=url,
                        is_valid=False,
                        safety_level=URLSafetyLevel.RESTRICTED,
                        block_reason=BlockReason.ROBOTS_TXT,
                        message="Path not in allowed list for this domain",
                        domain=domain,
                    )

            # Check robots.txt
            robots_result = await self._check_robots_txt(url, domain)
            if not robots_result[0]:
                return URLValidationResult(
                    url=url,
                    is_valid=False,
                    safety_level=URLSafetyLevel.BLOCKED,
                    block_reason=BlockReason.ROBOTS_TXT,
                    message="Disallowed by robots.txt",
                    domain=domain,
                    robots_txt_crawl_delay=robots_result[1],
                )

            # Trusted domain passes
            return URLValidationResult(
                url=url,
                is_valid=True,
                safety_level=domain_config.trust_level,
                message="Validated against trusted domain list",
                domain=domain,
                robots_txt_crawl_delay=domain_config.crawl_delay_seconds,
            )

        # Unknown domain - check with Safe Browsing if available
        if self.config.use_google_safe_browsing:
            safe_browsing_result = await self._check_safe_browsing(url)
            if not safe_browsing_result[0]:
                return URLValidationResult(
                    url=url,
                    is_valid=False,
                    safety_level=URLSafetyLevel.MALICIOUS,
                    block_reason=safe_browsing_result[1],
                    message=safe_browsing_result[2],
                    domain=domain,
                )

        # Unknown domain with no negative signals
        return URLValidationResult(
            url=url,
            is_valid=False,  # Conservative: don't allow unknown domains
            safety_level=URLSafetyLevel.UNKNOWN,
            block_reason=BlockReason.UNKNOWN_DOMAIN,
            message="Domain not on allowlist - add explicitly to scout",
            domain=domain,
        )

    def _is_blocked_domain(self, domain: str) -> bool:
        """Check if domain is blocked."""
        # Normalize
        if domain.startswith("www."):
            domain = domain[4:]

        # Direct match
        if domain in self._blocked_domains:
            return True

        # Check parent domains
        parts = domain.split(".")
        for i in range(1, len(parts) - 1):
            parent = ".".join(parts[i:])
            if parent in self._blocked_domains:
                return True

        return False

    def _get_domain_config(self, domain: str) -> Optional[DomainTrustLevel]:
        """Get trust configuration for domain."""
        # Normalize
        if domain.startswith("www."):
            domain = domain[4:]

        # Direct match
        if domain in self._trusted_domains:
            return self._trusted_domains[domain]

        # Check with www prefix
        www_domain = f"www.{domain}"
        if www_domain in self._trusted_domains:
            return self._trusted_domains[www_domain]

        # Check parent domains
        parts = domain.split(".")
        for i in range(1, len(parts) - 1):
            parent = ".".join(parts[i:])
            if parent in self._trusted_domains:
                return self._trusted_domains[parent]

        return None

    async def _check_robots_txt(
        self,
        url: str,
        domain: str,
    ) -> Tuple[bool, Optional[float]]:
        """Check if URL is allowed by robots.txt."""
        parser = await self._robots_cache.get_parser(
            domain, self.config.user_agent
        )

        if parser is None:
            # Conservative: if we can't fetch robots.txt, allow
            logger.warning(f"Could not fetch robots.txt for {domain}, allowing")
            return (True, None)

        # Check if allowed
        allowed = parser.can_fetch(self.config.user_agent, url)

        # Get crawl delay
        try:
            crawl_delay = parser.crawl_delay(self.config.user_agent)
        except Exception:
            crawl_delay = None

        return (allowed, crawl_delay)

    async def _check_safe_browsing(
        self,
        url: str,
    ) -> Tuple[bool, BlockReason, str]:
        """Check URL against Google Safe Browsing API."""
        if not self.config.google_safe_browsing_api_key:
            return (True, BlockReason.NONE, "")

        try:
            import aiohttp

            api_url = (
                f"https://safebrowsing.googleapis.com/v4/threatMatches:find"
                f"?key={self.config.google_safe_browsing_api_key}"
            )

            payload = {
                "client": {
                    "clientId": "nightshift-scout",
                    "clientVersion": "1.0.0",
                },
                "threatInfo": {
                    "threatTypes": [
                        "MALWARE",
                        "SOCIAL_ENGINEERING",
                        "UNWANTED_SOFTWARE",
                    ],
                    "platformTypes": ["ANY_PLATFORM"],
                    "threatEntryTypes": ["URL"],
                    "threatEntries": [{"url": url}],
                },
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("matches"):
                            threat_type = data["matches"][0].get("threatType", "UNKNOWN")
                            if threat_type == "MALWARE":
                                return (
                                    False,
                                    BlockReason.MALWARE_DETECTED,
                                    "URL flagged as malware by Safe Browsing",
                                )
                            elif threat_type == "SOCIAL_ENGINEERING":
                                return (
                                    False,
                                    BlockReason.PHISHING_DETECTED,
                                    "URL flagged as phishing by Safe Browsing",
                                )
                            else:
                                return (
                                    False,
                                    BlockReason.MALWARE_DETECTED,
                                    f"URL flagged as {threat_type} by Safe Browsing",
                                )

                        return (True, BlockReason.NONE, "")
                    else:
                        logger.warning(f"Safe Browsing API error: {response.status}")
                        return (True, BlockReason.NONE, "")

        except Exception as e:
            logger.error(f"Safe Browsing API error: {e}")
            return (True, BlockReason.NONE, "")

    async def validate_batch(
        self,
        urls: List[str],
        max_concurrent: int = 10,
    ) -> List[URLValidationResult]:
        """Validate multiple URLs concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def validate_with_limit(url: str) -> URLValidationResult:
            async with semaphore:
                return await self.validate(url)

        results = await asyncio.gather(
            *[validate_with_limit(url) for url in urls]
        )

        return list(results)

    def get_trusted_domains(self) -> List[DomainTrustLevel]:
        """Get list of trusted domains."""
        return list(self._trusted_domains.values())

    def get_blocked_domains(self) -> List[str]:
        """Get list of blocked domains."""
        return list(self._blocked_domains)


# Convenience exports
__all__ = [
    "URLValidator",
    "URLValidatorConfig",
    "URLValidationResult",
    "URLSafetyLevel",
    "BlockReason",
    "DomainTrustLevel",
    "RobotsTxtCache",
    "DEFAULT_TRUSTED_DOMAINS",
    "DEFAULT_BLOCKED_DOMAINS",
]
