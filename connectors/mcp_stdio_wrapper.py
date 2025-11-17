"""
MCP Stdio Wrapper - Suppress MCP server output to prevent UI conflicts

This module provides a wrapper around MCP stdio_client that modifies
server parameters to suppress MCP server logs from interfering with the
main program's UI.
"""

import os
from contextlib import asynccontextmanager
from mcp.client.stdio import stdio_client, StdioServerParameters
from dataclasses import replace


@asynccontextmanager
async def quiet_stdio_client(server_params: StdioServerParameters):
    """
    Wrapper that creates a stdio client with suppressed subprocess output.

    This prevents MCP server initialization messages from interfering with
    the main program's UI by setting environment variables and modifying
    the server parameters.

    Args:
        server_params: StdioServerParameters for the MCP server

    Yields:
        Same as stdio_client - (read_stream, write_stream)
    """
    # Get the environment, defaulting to os.environ if not set
    env = server_params.env if server_params.env is not None else os.environ.copy()

    # Suppress common MCP server output by setting environment variables
    quiet_env = {
        **env,
        "NODE_NO_WARNINGS": "1",
        "NPM_CONFIG_LOGLEVEL": "silent",
        "NPM_CONFIG_UPDATE_NOTIFIER": "false",
        "SUPPRESS_NO_CONFIG_WARNING": "1",
    }

    # Create new server params with quiet environment
    quiet_params = replace(server_params, env=quiet_env)

    # Use the original stdio_client with modified params
    async with stdio_client(quiet_params) as (read_stream, write_stream):
        yield (read_stream, write_stream)
