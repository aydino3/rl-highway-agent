
set -euo pipefail

python3 -m src.agents.smoke_test_env
python3 -m src.agents.smoke_test_wrapped_env
