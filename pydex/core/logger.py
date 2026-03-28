import sys
from io import TextIOWrapper


class Logger:
    def __init__(self, file_path: str) -> None:
        self.terminal: TextIOWrapper = sys.stdout  # type: ignore[assignment]
        self.log: TextIOWrapper = open(file_path, "a")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        pass
