"""
SSRF (Server-Side Request Forgery) Protection Utilities

This module provides protection against SSRF attacks by validating URLs
and blocking requests to internal/private networks and dangerous protocols.
"""

import ipaddress
import socket
from urllib.parse import urlparse
from typing import Optional, Set, List
import re
from dataclasses import dataclass
from loguru import logger


@dataclass
class SSRFValidationResult:
    """Result of SSRF validation"""
    is_safe: bool
    reason: Optional[str] = None
    resolved_ip: Optional[str] = None


class SSRFProtectionError(Exception):
    """Raised when SSRF protection blocks a request"""


class SSRFProtector:
    """
    Comprehensive SSRF protection for outbound HTTP requests.
    
    Blocks requests to:
    - Private/internal IP ranges
    - Localhost/loopback addresses
    - Link-local addresses
    - Multicast addresses
    - Dangerous protocols
    - Specific hostnames/ports
    """
    
    def __init__(self):
        # Allowed protocols
        self.allowed_protocols: Set[str] = {'http', 'https'}
        
        # Blocked hostnames (case-insensitive)
        self.blocked_hostnames: Set[str] = {
            'localhost',
            'metadata.google.internal',  # Google Cloud metadata
            '169.254.169.254',           # AWS/Azure metadata
            'metadata',
            'internal',
            'admin',
            'administrator'
        }
        
        # Blocked ports (commonly used for internal services)
        self.blocked_ports: Set[int] = {
            22,    # SSH
            23,    # Telnet
            25,    # SMTP
            53,    # DNS
            110,   # POP3
            143,   # IMAP
            993,   # IMAPS
            995,   # POP3S
            1433,  # MSSQL
            1521,  # Oracle
            3306,  # MySQL
            3389,  # RDP
            5432,  # PostgreSQL
            5984,  # CouchDB
            6379,  # Redis
            8086,  # InfluxDB
            9042,  # Cassandra
            9200,  # Elasticsearch
            11211, # Memcached
            27017, # MongoDB
        }
        
        # Additional dangerous port ranges
        self.dangerous_port_ranges: List[tuple] = [
            (1, 1023),      # System/privileged ports
            (6000, 6010),   # X11
            (7000, 7010),   # Cassandra inter-node
        ]
        
    def validate_url(self, url: str, allow_private_ips: bool = False) -> SSRFValidationResult:
        """
        Validate a URL for SSRF safety.
        
        Args:
            url: The URL to validate
            allow_private_ips: Whether to allow private IP addresses
            
        Returns:
            SSRFValidationResult with validation outcome
        """
        try:
            parsed = urlparse(url)
            
            # Check protocol
            if parsed.scheme.lower() not in self.allowed_protocols:
                return SSRFValidationResult(
                    is_safe=False,
                    reason=f"Protocol '{parsed.scheme}' not allowed. Allowed: {self.allowed_protocols}"
                )
            
            # Check for missing hostname
            if not parsed.hostname:
                return SSRFValidationResult(
                    is_safe=False,
                    reason="Missing hostname in URL"
                )
            
            # Check blocked hostnames
            if parsed.hostname.lower() in self.blocked_hostnames:
                return SSRFValidationResult(
                    is_safe=False,
                    reason=f"Hostname '{parsed.hostname}' is blocked"
                )
            
            # Check for suspicious patterns in hostname
            if self._is_suspicious_hostname(parsed.hostname):
                return SSRFValidationResult(
                    is_safe=False,
                    reason=f"Suspicious hostname pattern: {parsed.hostname}"
                )
            
            # Resolve hostname to IP
            try:
                resolved_ip = socket.gethostbyname(parsed.hostname)
            except socket.gaierror as e:
                return SSRFValidationResult(
                    is_safe=False,
                    reason=f"Failed to resolve hostname '{parsed.hostname}': {e}"
                )
            
            # Validate the resolved IP
            ip_validation = self._validate_ip_address(resolved_ip, allow_private_ips)
            if not ip_validation.is_safe:
                return SSRFValidationResult(
                    is_safe=False,
                    reason=f"Resolved IP {resolved_ip} is not safe: {ip_validation.reason}",
                    resolved_ip=resolved_ip
                )
            
            # Check port if specified
            if parsed.port:
                port_validation = self._validate_port(parsed.port)
                if not port_validation.is_safe:
                    return SSRFValidationResult(
                        is_safe=False,
                        reason=f"Port {parsed.port} is not safe: {port_validation.reason}",
                        resolved_ip=resolved_ip
                    )
            
            # Check for URL manipulation attempts
            if self._has_url_manipulation(url):
                return SSRFValidationResult(
                    is_safe=False,
                    reason="URL contains potential manipulation patterns"
                )
            
            return SSRFValidationResult(
                is_safe=True,
                resolved_ip=resolved_ip
            )
            
        except Exception as e:
            logger.warning(f"SSRF validation error for URL {url}: {e}")
            return SSRFValidationResult(
                is_safe=False,
                reason=f"Validation error: {e}"
            )
    
    def _validate_ip_address(self, ip_str: str, allow_private_ips: bool = False) -> SSRFValidationResult:
        """Validate an IP address for SSRF safety"""
        try:
            ip = ipaddress.ip_address(ip_str)
            
            # Check for loopback addresses (127.0.0.0/8, ::1)
            if ip.is_loopback:
                return SSRFValidationResult(
                    is_safe=False,
                    reason="Loopback address not allowed"
                )
            
            # Check for link-local addresses (169.254.0.0/16, fe80::/10)
            if ip.is_link_local:
                return SSRFValidationResult(
                    is_safe=False,
                    reason="Link-local address not allowed"
                )
            
            # Check for multicast addresses
            if ip.is_multicast:
                return SSRFValidationResult(
                    is_safe=False,
                    reason="Multicast address not allowed"
                )
            
            # Check for unspecified addresses (0.0.0.0, ::)
            if ip.is_unspecified:
                return SSRFValidationResult(
                    is_safe=False,
                    reason="Unspecified address not allowed"
                )
            
            # Check for private addresses unless explicitly allowed
            if not allow_private_ips and ip.is_private:
                return SSRFValidationResult(
                    is_safe=False,
                    reason="Private IP address not allowed"
                )
            
            # Check for reserved addresses
            if ip.is_reserved:
                return SSRFValidationResult(
                    is_safe=False,
                    reason="Reserved IP address not allowed"
                )
            
            # Additional checks for IPv4
            if isinstance(ip, ipaddress.IPv4Address):
                # Check for broadcast address
                if ip == ipaddress.IPv4Address('255.255.255.255'):
                    return SSRFValidationResult(
                        is_safe=False,
                        reason="Broadcast address not allowed"
                    )
                
                # Check for specific dangerous ranges
                dangerous_ranges = [
                    ipaddress.IPv4Network('0.0.0.0/8'),      # "This" network
                    ipaddress.IPv4Network('224.0.0.0/4'),   # Multicast
                    ipaddress.IPv4Network('240.0.0.0/4'),   # Reserved
                ]
                
                for network in dangerous_ranges:
                    if ip in network:
                        return SSRFValidationResult(
                            is_safe=False,
                            reason=f"IP in dangerous range {network}"
                        )
            
            return SSRFValidationResult(is_safe=True)
            
        except ValueError as e:
            return SSRFValidationResult(
                is_safe=False,
                reason=f"Invalid IP address: {e}"
            )
    
    def _validate_port(self, port: int) -> SSRFValidationResult:
        """Validate a port number for SSRF safety"""
        # Check blocked ports
        if port in self.blocked_ports:
            return SSRFValidationResult(
                is_safe=False,
                reason=f"Port {port} is blocked"
            )
        
        # Check dangerous port ranges
        for start, end in self.dangerous_port_ranges:
            if start <= port <= end:
                return SSRFValidationResult(
                    is_safe=False,
                    reason=f"Port {port} is in dangerous range {start}-{end}"
                )
        
        return SSRFValidationResult(is_safe=True)
    
    def _is_suspicious_hostname(self, hostname: str) -> bool:
        """Check for suspicious hostname patterns"""
        suspicious_patterns = [
            r'^\d+$',                    # All numeric (suspicious)
            r'.*\.local$',               # .local domains
            r'.*\.internal$',            # .internal domains
            r'.*\.corp$',                # .corp domains
            r'.*\.(test|dev|stage)$',    # Test/dev domains
            r'.*-admin.*',               # Admin-related hostnames
            r'.*metadata.*',             # Metadata services
        ]
        
        hostname_lower = hostname.lower()
        
        for pattern in suspicious_patterns:
            if re.match(pattern, hostname_lower):
                return True
        
        # Check for IP address masquerading as hostname
        try:
            ipaddress.ip_address(hostname)
            return False  # Valid IP, not suspicious as hostname
        except ValueError:
            pass
        
        # Check for URL-encoded characters
        if '%' in hostname:
            return True
        
        return False
    
    def _has_url_manipulation(self, url: str) -> bool:
        """Check for URL manipulation attempts"""
        suspicious_patterns = [
            r'@',                    # User info in URL
            r'%[0-9a-fA-F]{2}',     # URL encoding
            r'\\',                   # Backslashes
            r'\.\.',                 # Directory traversal
            r'[:;]',                 # Additional protocols/ports
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, url):
                return True
        
        return False
    
    def safe_request_url(self, url: str, allow_private_ips: bool = False) -> str:
        """
        Validate URL and return it if safe, otherwise raise SSRFProtectionError.
        
        Args:
            url: URL to validate
            allow_private_ips: Whether to allow private IP addresses
            
        Returns:
            The validated URL
            
        Raises:
            SSRFProtectionError: If URL is not safe
        """
        validation = self.validate_url(url, allow_private_ips)
        
        if not validation.is_safe:
            logger.warning(f"SSRF protection blocked URL {url}: {validation.reason}")
            raise SSRFProtectionError(f"URL blocked by SSRF protection: {validation.reason}")
        
        logger.debug(f"SSRF validation passed for URL {url} (resolved to {validation.resolved_ip})")
        return url


# Global SSRF protector instance
ssrf_protector = SSRFProtector()


def validate_webhook_url(url: str) -> SSRFValidationResult:
    """
    Validate a webhook URL for SSRF safety.
    
    Args:
        url: Webhook URL to validate
        
    Returns:
        SSRFValidationResult with validation outcome
    """
    return ssrf_protector.validate_url(url, allow_private_ips=False)


def validate_external_url(url: str, allow_private_ips: bool = False) -> SSRFValidationResult:
    """
    Validate an external URL for SSRF safety.
    
    Args:
        url: External URL to validate
        allow_private_ips: Whether to allow private IP addresses
        
    Returns:
        SSRFValidationResult with validation outcome
    """
    return ssrf_protector.validate_url(url, allow_private_ips)


def safe_webhook_url(url: str) -> str:
    """
    Validate webhook URL and return if safe, otherwise raise exception.
    
    Args:
        url: Webhook URL to validate
        
    Returns:
        The validated URL
        
    Raises:
        SSRFProtectionError: If URL is not safe
    """
    return ssrf_protector.safe_request_url(url, allow_private_ips=False)


def safe_external_url(url: str, allow_private_ips: bool = False) -> str:
    """
    Validate external URL and return if safe, otherwise raise exception.
    
    Args:
        url: External URL to validate
        allow_private_ips: Whether to allow private IP addresses
        
    Returns:
        The validated URL
        
    Raises:
        SSRFProtectionError: If URL is not safe
    """
    return ssrf_protector.safe_request_url(url, allow_private_ips)


# Configuration for common use cases
WEBHOOK_VALIDATION_CONFIG = {
    'allow_private_ips': False,
    'description': 'Strict validation for webhook URLs - blocks all internal networks'
}

EXTERNAL_URL_VALIDATION_CONFIG = {
    'allow_private_ips': False,
    'description': 'Standard validation for external URLs - blocks internal networks'
}

DEVELOPMENT_VALIDATION_CONFIG = {
    'allow_private_ips': True,
    'description': 'Relaxed validation for development - allows private IPs'
}