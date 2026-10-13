"""minisql: see SPEC.md. Replace this stub with the real implementation."""
import sys


class SQLError(Exception):
    def __init__(self, code, msg=""):
        super().__init__(f"{code}: {msg}")
        self.code, self.msg = code, msg


class Result:
    def __init__(self, rows=None, columns=None, rowcount=-1):
        self.rows, self.columns, self.rowcount = rows or [], columns or [], rowcount


class Database:
    def __init__(self, path=None):
        raise NotImplementedError

    def execute(self, sql, params=()):
        raise NotImplementedError

    def close(self):
        pass


if __name__ == "__main__":
    sys.exit(2)
