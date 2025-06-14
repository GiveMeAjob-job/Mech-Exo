# Operations Guide - Mech-Exo Risk Control

## 🔐 Secrets Management

### Overview
The Risk Control system requires several secrets and environment variables for production operation. This guide covers setup, rotation, and troubleshooting.

### Required Secrets

#### 🚨 Critical (System will not function without these)
| Secret | Purpose | Where to Get |
|--------|---------|--------------|
| `TELEGRAM_BOT_TOKEN` | Risk alert delivery | Create bot via @BotFather |
| `TELEGRAM_CHAT_ID` | Alert destination | Use @userinfobot in target chat |

#### 📈 Optional (Enhanced functionality)
| Secret | Purpose | Impact if Missing |
|--------|---------|-------------------|
| `AWS_ACCESS_KEY_ID` | S3 audit storage | Reports not archived |
| `AWS_SECRET_ACCESS_KEY` | S3 audit storage | Reports not archived |
| `PREFECT_API_KEY` | Workflow orchestration | Local execution only |

### Setup Instructions

#### GitHub Actions
```bash
# Run interactive setup script
.github/actions/set_secrets.sh

# Or set manually via GitHub CLI
echo "YOUR_BOT_TOKEN" | gh secret set TELEGRAM_BOT_TOKEN
echo "YOUR_CHAT_ID" | gh secret set TELEGRAM_CHAT_ID
```

#### Kubernetes
```bash
# Apply base secret manifest
kubectl apply -f deploy/k8s/secret.yaml

# Set actual secret values
kubectl patch secret mech-exo-secrets -p '{"stringData":{"TELEGRAM_BOT_TOKEN":"1234567890:ABC-DEF..."}}'
kubectl patch secret mech-exo-secrets -p '{"stringData":{"TELEGRAM_CHAT_ID":"-1001234567890"}}'
```

#### Docker Compose
```bash
# Create .env file
cp .env.example .env
# Edit .env with your values
vim .env

# Or export directly
export TELEGRAM_BOT_TOKEN="1234567890:ABC-DEF..."
export TELEGRAM_CHAT_ID="-1001234567890"
docker compose up
```

### Secret Rotation

#### Telegram Bot Token
1. **Create new bot**: Message @BotFather → `/newbot`
2. **Update secret**: 
   ```bash
   echo "NEW_TOKEN" | gh secret set TELEGRAM_BOT_TOKEN
   kubectl patch secret mech-exo-secrets -p '{"stringData":{"TELEGRAM_BOT_TOKEN":"NEW_TOKEN"}}'
   ```
3. **Test**: Run `python scripts/test_telegram_alerts.py`
4. **Revoke old**: Contact @BotFather → `/revoke`

#### AWS Keys
1. **Create new key**: AWS IAM → Users → Security Credentials
2. **Update secrets**:
   ```bash
   echo "NEW_ACCESS_KEY" | gh secret set AWS_ACCESS_KEY_ID
   echo "NEW_SECRET_KEY" | gh secret set AWS_SECRET_ACCESS_KEY
   ```
3. **Test**: Run AWS CLI commands
4. **Delete old**: AWS IAM → Delete old access key

#### Prefect API Key
1. **Generate new**: Prefect Cloud → Settings → API Keys
2. **Update secret**: `echo "NEW_KEY" | gh secret set PREFECT_API_KEY`
3. **Test**: `prefect cloud login --key NEW_KEY`
4. **Revoke old**: Prefect Cloud → Revoke old key

### Validation & Testing

#### Quick Health Check
```bash
# Test all critical secrets
python scripts/test_secrets_health.py

# Test individual components
python scripts/test_telegram_alerts.py    # Telegram integration
python scripts/test_aws_connection.py     # AWS integration
python scripts/test_prefect_auth.py       # Prefect integration
```

#### CI Validation
The `risk_master.yml` workflow includes secret validation:
```yaml
- name: Validate Secrets
  run: |
    echo "Checking TELEGRAM_BOT_TOKEN..."
    echo $TELEGRAM_BOT_TOKEN | md5sum  # Check length/format only
    echo "Checking TELEGRAM_CHAT_ID..."
    [[ $TELEGRAM_CHAT_ID =~ ^-?[0-9]+$ ]] || exit 1
```

### Troubleshooting

#### Common Issues

**"TELEGRAM_BOT_TOKEN not found"**
- Check secret is set: `gh secret list`
- Verify workflow has access to secrets
- Ensure repo has correct permissions

**"Telegram API Error 401"**
- Token may be revoked or invalid
- Regenerate token via @BotFather
- Update secret and test

**"Chat not found"**
- Verify chat ID format (starts with -)
- Ensure bot is added to the chat
- Bot must be admin for group chats

**"AWS credentials invalid"**
- Check key format and permissions
- Verify IAM policy includes S3 access
- Test with AWS CLI

#### Emergency Procedures

**Complete Secret Compromise**
1. **Immediate**: Revoke all compromised secrets
2. **Regenerate**: Create new secrets from scratch
3. **Update**: Apply to all environments (GitHub, K8s, etc.)
4. **Verify**: Run full test suite
5. **Document**: Update incident log

**Partial Service Degradation**
1. **Identify**: Which secrets are affected
2. **Isolate**: Disable affected features temporarily
3. **Fix**: Rotate affected secrets only
4. **Restore**: Re-enable features after testing

### Security Best Practices

#### Secret Lifecycle
- **Rotation**: Rotate secrets every 90 days
- **Monitoring**: Set up alerts for secret expiration
- **Access**: Limit secret access to necessary personnel only
- **Audit**: Log all secret access and modifications

#### Development vs Production
- **Separation**: Use different secrets for dev/staging/prod
- **Testing**: Never use production secrets in tests
- **Local**: Use `.env` files (add to `.gitignore`)
- **CI**: Use repository secrets, not environment variables

### Contact Information

#### Escalation Path
1. **On-call Engineer**: Immediate secret issues
2. **Lead Engineer**: Secret rotation planning
3. **Security Team**: Suspected compromise
4. **CTO**: Major security incidents

#### External Contacts
- **Telegram Support**: @BotSupport
- **AWS Support**: AWS Support Console
- **Prefect Support**: support@prefect.io

---

*Last Updated: 2024-12-13*  
*Version: 1.0*  
*Owner: Trading Operations Team*