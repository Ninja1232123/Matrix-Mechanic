# 🛡️ Deploy-Shield

Validate deployments before they fail. Auto-fix config issues, check environment variables, validate permissions, and ensure your app will actually run in production.

## Features

- 🔧 **Environment validation** - Check env vars, detect placeholders
- 🔌 **Port validation** - Ensure port consistency  
- 📁 **Permission checks** - Validate file permissions
- 🗄️ **Database testing** - Test connections before deploy
- 🔒 **SSL validation** - Check certificates
- 📊 **Resource limits** - Validate memory/CPU configs
- ⚙️ **Config validation** - Check YAML/JSON syntax

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Validate everything
deploy-shield validate

# Auto-fix issues
deploy-shield fix

# Test deployment  
deploy-shield test

# Pre-deploy checklist
deploy-shield checklist

# Generate configs
deploy-shield generate --dockerfile
deploy-shield generate --docker-compose
```

## Usage

### Validate Deployment

```bash
deploy-shield validate              # Check everything
deploy-shield validate --env        # Check env vars only
deploy-shield validate --database   # Check database only
```

### Fix Issues

```bash
deploy-shield fix --mode=auto      # Auto-fix all
deploy-shield fix --mode=review    # Review each fix
deploy-shield fix --dry-run        # Preview changes
```

### Test

```bash
deploy-shield test                 # Run all tests
```

### Generate Configs

```bash
deploy-shield generate --dockerfile      # Generate Dockerfile
deploy-shield generate --docker-compose  # Generate docker-compose.yml
deploy-shield generate --env            # Generate .env
deploy-shield generate --k8s            # Generate K8s manifests
```

## What It Checks

### Environment Variables
- Missing variables
- Placeholder values
- Invalid formats
- Localhost in production

### Port Configuration
- Port consistency across files
- Port conflicts
- Docker port mappings

### File Permissions
- Writable directories
- Readable config files
- Certificate permissions

### Database
- Connection testing
- Localhost detection
- Pool configuration

### SSL/TLS
- Certificate expiration
- Hostname matching
- Self-signed detection

### Resources
- Memory limits
- CPU limits
- Health checks
- Restart policies

## Integration with DevMaster

Deploy-Shield is Tool #5 in the DevMaster suite:

```bash
devmaster deploy  # Runs all tools + Deploy-Shield
```

## License

MIT

---

**Part of the DevMaster suite of autonomous debugging tools.**
