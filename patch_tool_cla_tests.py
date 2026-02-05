#!/usr/bin/env python3
"""
Tests for the standalone V4A apply_patch tool.

Run with: python test_apply_patch.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import apply_patch
sys.path.insert(0, str(Path(__file__).parent))

from patch_tool_cla import apply_diff, DiffApplicationError, parse_v4a_diff


def test_create_simple_file():
    """Test creating a new file from scratch."""
    diff = """@@
+def hello():
+    print("Hello, world!")
"""
    result = apply_diff("", diff, mode="create")
    expected = """def hello():
    print("Hello, world!")"""
    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"
    print("✓ test_create_simple_file passed")


def test_create_multi_hunk():
    """Test creating a file with multiple hunks."""
    diff = """@@
+def hello():
+    print("Hello")
@@
+def goodbye():
+    print("Goodbye")
"""
    result = apply_diff("", diff, mode="create")
    expected = """def hello():
    print("Hello")
def goodbye():
    print("Goodbye")"""
    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"
    print("✓ test_create_multi_hunk passed")


def test_update_simple_change():
    """Test updating a single line."""
    current = """def hello():
    print('Hi')
    return True"""
    
    diff = """@@
 def hello():
-    print('Hi')
+    print("Hello, world!")
     return True"""
    
    result = apply_diff(current, diff)
    expected = """def hello():
    print("Hello, world!")
    return True"""
    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"
    print("✓ test_update_simple_change passed")


def test_update_add_lines():
    """Test adding lines to existing file."""
    current = """def hello():
    pass"""
    
    diff = """@@
 def hello():
-    pass
+    print("Hello")
+    print("World")
+    return True"""
    
    result = apply_diff(current, diff)
    expected = """def hello():
    print("Hello")
    print("World")
    return True"""
    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"
    print("✓ test_update_add_lines passed")


def test_update_remove_lines():
    """Test removing lines from file."""
    current = """def hello():
    print("Line 1")
    print("Line 2")
    print("Line 3")
    return True"""
    
    diff = """@@
 def hello():
     print("Line 1")
-    print("Line 2")
     print("Line 3")
     return True"""
    
    result = apply_diff(current, diff)
    expected = """def hello():
    print("Line 1")
    print("Line 3")
    return True"""
    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"
    print("✓ test_update_remove_lines passed")


def test_multi_hunk_update():
    """Test updating multiple separate locations in a file."""
    current = """def func1():
    print("old1")

def func2():
    print("old2")

def func3():
    print("old3")"""
    
    diff = """@@
 def func1():
-    print("old1")
+    print("new1")
@@
 def func3():
-    print("old3")
+    print("new3")"""
    
    result = apply_diff(current, diff)
    expected = """def func1():
    print("new1")

def func2():
    print("old2")

def func3():
    print("new3")"""
    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"
    print("✓ test_multi_hunk_update passed")


def test_context_descriptions():
    """Test hunks with context descriptions."""
    current = """def hello():
    return "hi\""""
    
    diff = """@@ Update greeting message
 def hello():
-    return "hi"
+    return "Hello, world!\""""
    
    result = apply_diff(current, diff)
    expected = """def hello():
    return "Hello, world!\""""
    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"
    print("✓ test_context_descriptions passed")


def test_empty_diff():
    """Test that empty diff returns original content."""
    current = "def hello():\n    pass"
    result = apply_diff(current, "")
    assert result == current
    print("✓ test_empty_diff passed")


def test_whitespace_only_diff():
    """Test diff with only whitespace."""
    current = "def hello():\n    pass"
    result = apply_diff(current, "   \n\n   ")
    assert result == current
    print("✓ test_whitespace_only_diff passed")


def test_context_mismatch_error():
    """Test that mismatched context raises an error."""
    current = """def hello():
    print("actual")"""
    
    diff = """@@
 def hello():
-    print("expected")
+    print("new")"""
    
    try:
        apply_diff(current, diff)
        assert False, "Should have raised DiffApplicationError"
    except DiffApplicationError as e:
        assert "Context mismatch" in str(e) or "Removal mismatch" in str(e)
        print("✓ test_context_mismatch_error passed")


def test_removal_in_create_mode_error():
    """Test that removal lines in create mode raise an error."""
    diff = """@@
+def hello():
-    pass"""
    
    try:
        apply_diff("", diff, mode="create")
        assert False, "Should have raised DiffApplicationError"
    except DiffApplicationError as e:
        assert "create mode" in str(e).lower()
        print("✓ test_removal_in_create_mode_error passed")


def test_parse_v4a_diff():
    """Test the diff parser."""
    diff = """@@
+line1
+line2
@@ with description
 context
-remove
+add"""
    
    hunks = parse_v4a_diff(diff)
    assert len(hunks) == 2
    assert hunks[0].context_description is None
    assert len(hunks[0].lines) == 2
    assert hunks[1].context_description == "with description"
    assert len(hunks[1].lines) == 3
    print("✓ test_parse_v4a_diff passed")


def test_realistic_python_refactor():
    """Test a realistic Python refactoring scenario."""
    current = """def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

def main():
    print(fib(10))

if __name__ == "__main__":
    main()"""
    
    # Rename fib to fibonacci
    diff = """@@
-def fib(n):
+def fibonacci(n):
     if n <= 1:
         return n
-    return fib(n-1) + fib(n-2)
+    return fibonacci(n-1) + fibonacci(n-2)
@@
 def main():
-    print(fib(10))
+    print(fibonacci(10))"""
    
    result = apply_diff(current, diff)
    expected = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    print(fibonacci(10))

if __name__ == "__main__":
    main()"""
    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"
    print("✓ test_realistic_python_refactor passed")


def test_preserve_trailing_lines():
    """Test that lines after all hunks are preserved."""
    current = """def func1():
    pass

def func2():
    pass

# trailing comment
# another comment"""
    
    diff = """@@
 def func1():
-    pass
+    return True"""
    
    result = apply_diff(current, diff)
    expected = """def func1():
    return True

def func2():
    pass

# trailing comment
# another comment"""
    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"
    print("✓ test_preserve_trailing_lines passed")


def test_add_at_end():
    """Test adding lines at the end of file."""
    current = """def hello():
    pass"""
    
    diff = """@@
 def hello():
     pass
+
+def goodbye():
+    pass"""
    
    result = apply_diff(current, diff)
    expected = """def hello():
    pass

def goodbye():
    pass"""
    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"
    print("✓ test_add_at_end passed")


def run_all_tests():
    """Run all tests."""
    print("Running apply_patch tests...\n")
    
    tests = [
        test_create_simple_file,
        test_create_multi_hunk,
        test_update_simple_change,
        test_update_add_lines,
        test_update_remove_lines,
        test_multi_hunk_update,
        test_context_descriptions,
        test_empty_diff,
        test_whitespace_only_diff,
        test_context_mismatch_error,
        test_removal_in_create_mode_error,
        test_parse_v4a_diff,
        test_realistic_python_refactor,
        test_preserve_trailing_lines,
        test_add_at_end,
    ]
    
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {len(tests) - failed}/{len(tests)} tests passed")
    if failed == 0:
        print("All tests passed! ✓")
        return 0
    else:
        print(f"{failed} test(s) failed ✗")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())