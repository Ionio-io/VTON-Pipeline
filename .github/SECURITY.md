# Security Policy

## Reporting Security Issues

If you discover a security vulnerability in this project, please do **NOT** create a public GitHub issue. Instead:

1. Email your findings to: [your-email@example.com]
2. Include a detailed description of the vulnerability
3. Provide steps to reproduce if possible
4. Allow reasonable time for the maintainers to respond

## API Key Security

### Important ⚠️

Never commit your API keys to GitHub:

1. **FAL.AI Keys**: Keep your `key_id:key_secret` in `.env` (not committed)
2. **RunPod Keys**: Keep your endpoint ID and API key in `.env` (not committed)
3. **`.env` file**: Make sure `.env` is in `.gitignore`

### Best Practices

- Use `.env.example` as a template
- Regenerate exposed keys immediately
- Use environment variables in production
- Rotate keys periodically
- Never share keys in issues or discussions

## Deployment Security

### For Vercel/Netlify/etc:

1. Set environment variables in the platform dashboard
2. Never hardcode credentials
3. Use secrets/environment variables features
4. Review access logs regularly

### For Self-Hosted Servers:

1. Use HTTPS (SSL certificates)
2. Set environment variables on the server
3. Restrict API key access to server-side only
4. Keep server software updated

## API Usage Limits

- **FAL.AI**: Monitor your API usage to prevent unexpected charges
- **RunPod**: Set up spending limits and alerts
- Implement rate limiting if deploying publicly

## Version Updates

Keep dependencies updated:

- Check for security updates in browser APIs
- Update links to latest documentation
- Follow upstream security announcements

## Questions?

For security-related questions, please reach out privately rather than creating public issues.
