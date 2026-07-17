# Configuration Access Pattern

When accessing configuration parameters in YAML configs:

**ALWAYS use direct dictionary access:**
```python
# CORRECT - Direct access
log_frequency = metrics_config['log_frequency']
batch_size = metrics_config['batch_size']
```

**NEVER use `.get()` with default values:**
```python
# INCORRECT - Using .get() with defaults
log_frequency = metrics_config.get('log_frequency', 100)
batch_size = metrics_config.get('batch_size', 32)
```

**Rationale:**
- Direct access will raise a clear KeyError if a required parameter is missing
- This forces explicit configuration and catches missing parameters early
- Using `.get()` with defaults can hide configuration issues
- If a parameter is truly optional, use direct access with a pre-defined default in the config schema, not at the call site

**Exception:** Only use `.get()` when the parameter is truly optional and the code has logic to handle its absence.