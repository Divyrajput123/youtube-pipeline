"""Allow the pipeline package to be invoked directly as a module.

    python -m pipeline --config config.json [--batch-size N] [--resume-run-id UUID]
"""

from pipeline.cli import main

main()
