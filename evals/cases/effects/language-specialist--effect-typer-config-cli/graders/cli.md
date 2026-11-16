---
type: llm
focus: { source: file, path: mytool.py }
---

All claims are required.
PASS if the missing-key path writes its error to stderr and exits with status 1.
PASS if TOML is written with a writer package such as tomli_w (read with tomllib), and the file never calls tomllib.dump or tomllib.dumps.
PASS if the config directory comes from platformdirs (user_config_dir) or equivalent logic that covers macOS Application Support and Linux XDG paths.
FAIL if the existing keys in config.toml are overwritten rather than loaded and merged before writing.
