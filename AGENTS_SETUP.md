# Agent Setup Guide

This guide explains how to set up and configure the multi-agent system.

## Quick Start

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Add your API keys to `.env`:**
   ```bash
   # Required for all LLM operations
   GOOGLE_API_KEY=your_gemini_api_key_here

   # Required for specific agents (see below)
   SLACK_BOT_TOKEN=your_slack_bot_token
   SLACK_TEAM_ID=your_slack_team_id
   JIRA_URL=https://your-domain.atlassian.net
   JIRA_API_TOKEN=your_jira_token
   GITHUB_TOKEN=your_github_personal_access_token
   NOTION_TOKEN=your_notion_integration_token
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the system:**
   ```bash
   python main.py
   ```

## Available Agents

### Core Agent (Always Available)
- **code_reviewer** - Code analysis, security checks, best practices review
  - No external dependencies
  - Uses Gemini LLM directly

### MCP Agents (Require Setup)
All MCP agents require `npx` (Node.js package runner) to be installed:
```bash
# Install Node.js and npm first, then npx is available
npm install -g npx
```

- **slack** - Slack workspace integration
  - Requires: `SLACK_BOT_TOKEN`, `SLACK_TEAM_ID`
  - Get tokens: https://api.slack.com/apps

- **jira** - Jira issue tracking
  - Requires: `JIRA_URL`, `JIRA_API_TOKEN`
  - Get token: Your Jira Settings → Security → API Tokens

- **github** - GitHub repositories and PRs
  - Requires: `GITHUB_TOKEN`
  - Get token: GitHub Settings → Developer settings → Personal access tokens

- **notion** - Notion workspace management
  - Requires: `NOTION_TOKEN`
  - Get token: https://www.notion.so/profile/integrations

- **browser** - Web automation
  - Requires: npx installed
  - No additional API keys needed

- **scraper** - Web scraping
  - Requires: npx installed
  - No additional API keys needed

## Disabling Agents

If you don't need certain agents, you can disable them by setting the `DISABLED_AGENTS` environment variable:

```bash
# In .env file
DISABLED_AGENTS=slack,jira,github
```

This will skip loading those agents entirely, speeding up startup time.

## Troubleshooting

### Agents Fail to Load

**Problem:** "⚠ Missing required environment variables"
- **Solution:** Add the required API keys to your `.env` file
- Check the error message for which variables are missing

**Problem:** "⚠ npx/npm not installed"
- **Solution:** Install Node.js which includes npm and npx
- Download from: https://nodejs.org/

**Problem:** "⚠ Network/connection error"
- **Solution:** Check your internet connection
- Verify API endpoints are accessible
- Check firewall settings

**Problem:** Agent times out during initialization
- **Solution:** This is normal for some MCP agents
- The system will continue with loaded agents
- Try increasing timeout in orchestrator.py if needed

### Verbose Mode

To see detailed error messages and debug information:
```bash
python main.py --verbose
```

This will show:
- Detailed initialization steps
- Error stack traces
- Agent capability lists
- Performance metrics

## Agent Loading Behavior

### Default Behavior (Current)
- All agents are enabled by default
- Agents that fail to load are skipped
- System continues with successfully loaded agents
- 5-second timeout per agent
- 30-second global timeout for all agents

### Error Handling
- Missing credentials → Clear error message
- Missing dependencies → Helpful instructions
- Network issues → Retry suggestions
- Timeout → System continues anyway

## Architecture Notes

### MCP (Model Context Protocol)
MCP agents spawn subprocess servers using `npx`. This is why they require:
1. Node.js/npm/npx installed
2. Network access to download packages
3. API credentials to authenticate with services

### Non-MCP Agents
The `code_reviewer` agent uses the Gemini LLM directly without spawning subprocesses:
- Faster initialization
- No Node.js dependency
- Only requires `GOOGLE_API_KEY`

### Resilience Features
- **Timeouts:** Prevent hanging during initialization
- **Graceful Degradation:** System works with partial agent loading
- **Error Recovery:** Clear messages for common failure modes
- **Output Suppression:** MCP server logs don't interfere with UI

## Configuration Reference

See `.env.example` for complete list of configuration options including:
- Confirmation settings (which operations require approval)
- Timeout values
- Retry configuration
- Logging settings
- Security settings

## Need Help?

1. Run with `--verbose` flag for detailed logs
2. Check `.env` file has all required keys
3. Verify `npx --version` works
4. Check agent-specific documentation in `connectors/` directory
