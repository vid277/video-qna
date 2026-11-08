import sys
from pathlib import Path

import yaml

base, override, out = map(Path, sys.argv[1:4])
merged = yaml.safe_load(base.read_text()) | yaml.safe_load(override.read_text())

out.write_text(yaml.safe_dump(merged, sort_keys=False))
