# Project Conventions

## Code Style Rules

### Configuration Access
- **Always access config attributes directly**, never use `getattr()` or `get()` methods
- **Example:**
  ```python
  # ✅ Correct - Direct access
  original_tokens = config.original_tokens

  # ❌ Wrong - Using getattr
  original_tokens = getattr(config, 'original_tokens', 'mask')

  # ❌ Wrong - Using get
  original_tokens = config.get('original_tokens', 'mask')
  ```

- **Rationale:** Direct access is more explicit, type-safe, and makes it clear what configuration is expected. If a config field is missing, it should be added explicitly rather than hidden behind a fallback value.
