GraphAI can be configured using environment variables to customize behavior and output.

## Environment Variables

### GRAPHAI_LOG_LEVEL

Controls the logging output level for GraphAI components. This is useful for debugging or reducing log verbosity in production environments.

**Available levels:**
- `DEBUG` - Show all messages (most verbose)
- `INFO` - Show info, warning, error, critical messages (default)
- `WARNING` - Show warning, error, critical messages  
- `ERROR` - Show only error and critical messages
- `CRITICAL` - Show only critical messages

**Usage:**

```bash
# Hide warning messages (e.g., missing docstrings in functions)
export GRAPHAI_LOG_LEVEL=ERROR

# Show debug information for troubleshooting
export GRAPHAI_LOG_LEVEL=DEBUG

# Set for a single command
GRAPHAI_LOG_LEVEL=WARNING python my_script.py
```

**Example:**

If you're seeing warnings like:
```
2025-08-16 13:41:54 WARNING graphai.utils Function start has no docstring
```

You can suppress them by setting:
```bash
export GRAPHAI_LOG_LEVEL=ERROR
```

## Default Values

| Environment Variable | Default Value | Description |
|---------------------|---------------|-------------|
| `GRAPHAI_LOG_LEVEL` | `INFO` | Controls logging verbosity |

## Best Practices

- **Development**: Use `DEBUG` or `INFO` level to see detailed information
- **Production**: Use `WARNING` or `ERROR` level to reduce log noise
- **CI/CD**: Use `ERROR` level to only show important issues
