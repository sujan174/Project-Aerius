# Notion MCP Self-Hosted Setup Guide

## Overview

The chatbot now uses the **official self-hosted Notion MCP server** (`@notionhq/notion-mcp-server`) instead of the hosted OAuth version. This provides:

- **Better reliability** - No dependency on Notion's hosted service
- **Full control** - Direct access with your integration token
- **BYOT support** - Ready for Bring Your Own Token scaling
- **Security** - Token-based authentication with fine-grained permissions
- **Cost efficiency** - No rate limits from hosted proxy

## Quick Setup

### 1. Create Notion Integration

1. Go to [Notion Integrations](https://www.notion.so/profile/integrations)
2. Click **"New Integration"** (or "+ New integration")
3. Fill in the details:
   - **Name**: `Chatbot Integration` (or your preferred name)
   - **Logo**: Optional
   - **Associated workspace**: Select your workspace

4. **Configure Capabilities**:
   - ✅ **Read content** - Required for searching and reading pages
   - ✅ **Update content** - Required for editing pages
   - ✅ **Insert content** - Required for creating new pages
   - ⚠️ **Read comments** - Optional (if you want comment access)
   - ⚠️ **Insert comments** - Optional (if you want to add comments)

   **For Read-Only Access** (safer for testing):
   - ✅ **Read content** only
   - ❌ All other capabilities disabled

5. Click **"Submit"** or **"Save"**

6. **Copy your Integration Token**:
   - You'll see a token that starts with `secret_`
   - Click **"Show"** and **"Copy"**
   - ⚠️ **Keep this secret!** Never commit to git

### 2. Share Pages with Integration

The integration can only access pages/databases you explicitly share with it:

1. Open any Notion page or database you want the bot to access
2. Click the **"···"** (three dots) in the top right
3. Scroll to **"Connections"** (or "Connect to")
4. Search for your integration name (e.g., "Chatbot Integration")
5. Click to connect it

**Tip**: Share your main workspace or top-level pages to give access to everything underneath.

### 3. Configure Environment Variable

Add your token to the `.env` file:

```bash
# In your .env file
NOTION_TOKEN=secret_your_actual_token_here
```

**Security Notes**:
- Never commit `.env` to version control
- The `.env.example` file shows the format without real credentials
- Use different tokens for development, staging, and production

### 4. Restart the Chatbot

```bash
# Stop the currently running chatbot (Ctrl+C)
python orchestrator.py
```

The Notion agent will now use the self-hosted MCP server with your integration token.

## Verification

To verify the setup is working:

1. Start the chatbot
2. Ask it to list your Notion pages:
   ```
   List my Notion pages
   ```
3. Or create a test page:
   ```
   Create a test page in Notion titled "MCP Test"
   ```

If you see errors about missing `NOTION_TOKEN`, double-check your `.env` file.

## Architecture

### Before (Hosted)
```
Chatbot → mcp-remote → https://mcp.notion.com/sse → OAuth → Notion API
```
- Required OAuth browser popup
- Dependent on Notion's hosted service
- Session-based authentication
- Not suitable for BYOT

### After (Self-Hosted)
```
Chatbot → @notionhq/notion-mcp-server → Notion API
          (with NOTION_TOKEN)
```
- Direct connection
- Token-based authentication
- Full control and reliability
- BYOT-ready architecture

## MCP Server Details

**Package**: `@notionhq/notion-mcp-server`
**Repository**: https://github.com/makenotion/notion-mcp-server
**Documentation**: https://developers.notion.com/docs/mcp
**Version**: Latest (auto-updated via `npx -y`)

### Key Features

1. **Full Notion API Support**:
   - Create, read, update pages
   - Work with databases and entries
   - Search across workspace
   - Manage page properties

2. **Security**:
   - Token-based authentication
   - Fine-grained permissions via Capabilities
   - Per-page access control
   - Audit trail in Notion

3. **Reliability**:
   - Direct API calls (no proxy)
   - Automatic retries with exponential backoff
   - Timeout protection (60s per operation)
   - Graceful error handling

4. **Performance**:
   - Metadata caching (1-hour TTL)
   - Prefetch optimization
   - Parallel operation support

## Troubleshooting

### Error: "NOTION_TOKEN environment variable is required"

**Solution**:
1. Check `.env` file exists in project root
2. Verify `NOTION_TOKEN=secret_...` is present
3. Ensure no extra spaces: `NOTION_TOKEN=token` (not `NOTION_TOKEN = token`)
4. Restart the chatbot after adding the token

### Error: "Authentication error" or "unauthorized"

**Solution**:
1. Verify your token is valid (starts with `secret_`)
2. Check if integration still exists at https://www.notion.so/profile/integrations
3. Regenerate token if needed (from integration settings)

### Error: "Permission denied" or "not found"

**Solution**:
1. Share the page/database with your integration:
   - Open the page in Notion
   - Click "···" → "Connections"
   - Add your integration
2. Check integration capabilities (needs "Read content" at minimum)

### Error: "Connection timeout" or "server unavailable"

**Solution**:
1. Check internet connection
2. Verify `npx` is installed: `npx --version`
3. Try manually installing: `npm install -g @notionhq/notion-mcp-server`
4. Check Notion API status: https://status.notion.so/

### Pages not showing up in search

**Solution**:
1. Verify pages are shared with integration
2. Wait ~30 seconds for Notion's search index to update
3. Try searching by exact title first
4. Check that pages aren't archived

## Advanced Configuration

### Read-Only Mode

For safety in production, create a read-only integration:

1. When creating integration, enable only:
   - ✅ **Read content**
   - ❌ All other capabilities disabled
2. Use a separate `NOTION_TOKEN_READONLY` for read operations
3. Use a full-access token only for write operations

### Multiple Workspaces

To support multiple Notion workspaces:

1. Create separate integrations for each workspace
2. Use different tokens:
   ```bash
   NOTION_TOKEN_WORKSPACE_1=secret_...
   NOTION_TOKEN_WORKSPACE_2=secret_...
   ```
3. Modify `notion_agent.py` to select token based on context

### BYOT (Bring Your Own Token)

See [BYOT_SCALING.md](./BYOT_SCALING.md) for multi-tenant architecture.

## Security Best Practices

1. **Token Storage**:
   - ✅ Store tokens in `.env` file
   - ✅ Add `.env` to `.gitignore`
   - ❌ Never commit tokens to git
   - ❌ Never log tokens in code

2. **Integration Permissions**:
   - ✅ Grant minimum required capabilities
   - ✅ Use read-only for testing
   - ✅ Review integration access regularly
   - ❌ Don't grant "Insert comments" unless needed

3. **Page Sharing**:
   - ✅ Share only necessary pages
   - ✅ Use page-level permissions
   - ✅ Audit integration access monthly
   - ❌ Don't share entire workspace unless required

4. **Token Rotation**:
   - ✅ Rotate tokens every 90 days
   - ✅ Use different tokens per environment
   - ✅ Revoke unused integrations
   - ❌ Don't reuse tokens across projects

## Migration from Hosted

If you were previously using the hosted `mcp-remote` setup:

1. **No data migration needed** - Same Notion workspace
2. **Update code** - Already done in `notion_agent.py`
3. **Add NOTION_TOKEN** - Follow setup steps above
4. **Restart chatbot** - Will use new self-hosted server
5. **Test functionality** - Verify all operations work
6. **Remove old auth** - OAuth tokens no longer needed

## Support

- **Notion MCP Issues**: https://github.com/makenotion/notion-mcp-server/issues
- **Notion API Docs**: https://developers.notion.com/
- **Notion API Status**: https://status.notion.so/
- **MCP Protocol**: https://github.com/anthropics/mcp

## Next Steps

After completing setup:

1. ✅ Test basic operations (create, read, search)
2. ✅ Share relevant pages with integration
3. ✅ Review integration permissions
4. 📖 Read [BYOT_SCALING.md](./BYOT_SCALING.md) for multi-tenant support
5. 📖 Explore advanced features in agent code
