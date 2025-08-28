import re


def param_input_validator(param):
    """
    Checks for any security vulnerabilities in the input parameter including SQL Injection, XSS, and Command Injection.
    """

    # Normalize input
    if not isinstance(param, str):
        raise ValueError("Input parameter must be a string.")

    param = param.strip()

    # Patterns for common attacks
    sql_injection_pattern = re.compile(
        r"('|--|;|/\*|\*/|xp_|union\s+select|select\s+\*|information_schema|sleep\(|benchmark\(|or\s+1=1|and\s+1=1)",
        re.IGNORECASE,
    )
    xss_pattern = re.compile(
        r"(<script.*?>.*?</script>|<.*?on\w+\s*=|javascript:|<iframe|<img|<svg|document\.cookie|alert\s*\(|eval\s*\()",
        re.IGNORECASE,
    )
    command_injection_pattern = re.compile(
        r"(\||&|;|\$|\`|>|<|&&|\n|\r|cat\s|ls\s|rm\s|wget\s|curl\s|nc\s|python\s|perl\s|bash\s|sh\s)",
        re.IGNORECASE,
    )
    path_traversal_pattern = re.compile(
        r"(\.\./|\.\.\\|/etc/passwd|/windows/win.ini)", re.IGNORECASE
    )
    hex_encoding_pattern = re.compile(r"(0x[0-9a-fA-F]+)")

    # Check for SQL Injection
    if sql_injection_pattern.search(param):
        raise ValueError("Potential SQL Injection detected.")

    # Check for XSS
    if xss_pattern.search(param):
        raise ValueError("Potential XSS attack detected.")

    # Check for Command Injection
    if command_injection_pattern.search(param):
        raise ValueError("Potential Command Injection detected.")

    # Check for Path Traversal
    if path_traversal_pattern.search(param):
        raise ValueError("Potential Path Traversal attack detected.")

    # Check for Hex Encoding (obfuscation)
    if hex_encoding_pattern.search(param):
        raise ValueError("Potential obfuscated attack detected.")

    # Optionally, restrict allowed characters (uncomment if needed)
    # if not re.fullmatch(r"[\w\s\-@.,:]+", param):
    #     raise ValueError("Input contains disallowed characters.")

    return True
