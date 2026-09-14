import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SOURCE_DIRECTORIES = ("configs", "scripts", "src", "tests")
FORBIDDEN_PATTERNS = (
    re.compile(r"/Users/[^/<\s]+/"),
    re.compile(r"/home/[^/<\s]+/"),
    re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
)


class PortablePathTests(unittest.TestCase):
    def test_source_files_do_not_embed_user_home_paths(self):
        # Slurm templates are intentionally site-specific and audited separately.
        paths = [
            PROJECT_ROOT / ".gitignore",
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "requirements.txt",
        ]
        for directory_name in SOURCE_DIRECTORIES:
            directory = PROJECT_ROOT / directory_name
            if directory.is_dir():
                paths.extend(
                    path
                    for path in directory.rglob("*")
                    if path.is_file() and path.suffix in TEXT_SUFFIXES
                )

        violations = []
        for path in paths:
            if path == Path(__file__).resolve():
                continue
            text = path.read_text(errors="ignore")
            for pattern in FORBIDDEN_PATTERNS:
                if pattern.search(text):
                    violations.append(str(path.relative_to(PROJECT_ROOT)))
                    break

        self.assertEqual(
            violations,
            [],
            "Source files contain machine-specific home-directory paths.",
        )


if __name__ == "__main__":
    unittest.main()
