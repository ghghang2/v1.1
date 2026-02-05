#!/usr/bin/env python3
"""
Practical Examples of Using the V4A Apply Patch Tool

This file demonstrates various real-world use cases for the standalone apply_patch tool.
"""

from pathlib import Path
from patch_tool_cla import apply_diff, apply_patch_to_file, DiffApplicationError


def example_1_create_new_file():
    """Example 1: Create a new Python file from scratch."""
    print("\n" + "="*60)
    print("Example 1: Creating a new Python file")
    print("="*60)
    
    diff = """@@
+#!/usr/bin/env python3
+\"\"\"Simple calculator module.\"\"\"
+
+def add(a, b):
+    return a + b
+
+def subtract(a, b):
+    return a - b
+
+def multiply(a, b):
+    return a * b
+
+def divide(a, b):
+    if b == 0:
+        raise ValueError("Cannot divide by zero")
+    return a / b
"""
    
    result = apply_diff("", diff, mode="create")
    print("\nGenerated file content:")
    print("-" * 60)
    print(result)
    print("-" * 60)


def example_2_refactor_function_name():
    """Example 2: Refactor a function name throughout a file."""
    print("\n" + "="*60)
    print("Example 2: Refactoring function name 'calc' to 'calculate'")
    print("="*60)
    
    original = """def calc(x, y):
    return x + y

def process_data():
    result = calc(5, 3)
    print(f"Result: {result}")
    return calc(10, 20)

class Calculator:
    def run(self):
        return calc(1, 2)
"""
    
    diff = """@@
-def calc(x, y):
+def calculate(x, y):
     return x + y
@@
 def process_data():
-    result = calc(5, 3)
+    result = calculate(5, 3)
     print(f"Result: {result}")
-    return calc(10, 20)
+    return calculate(10, 20)
@@
 class Calculator:
     def run(self):
-        return calc(1, 2)
+        return calculate(1, 2)
"""
    
    result = apply_diff(original, diff)
    print("\nOriginal:")
    print("-" * 60)
    print(original)
    print("\nRefactored:")
    print("-" * 60)
    print(result)


def example_3_add_error_handling():
    """Example 3: Add error handling to existing code."""
    print("\n" + "="*60)
    print("Example 3: Adding error handling to a function")
    print("="*60)
    
    original = """def read_config(filename):
    with open(filename) as f:
        return json.load(f)
"""
    
    diff = """@@
 def read_config(filename):
+    try:
-    with open(filename) as f:
-        return json.load(f)
+        with open(filename) as f:
+            return json.load(f)
+    except FileNotFoundError:
+        print(f"Config file not found: {filename}")
+        return {}
+    except json.JSONDecodeError as e:
+        print(f"Invalid JSON in config: {e}")
+        return {}
"""
    
    result = apply_diff(original, diff)
    print("\nOriginal:")
    print("-" * 60)
    print(original)
    print("\nWith error handling:")
    print("-" * 60)
    print(result)


def example_4_add_docstrings():
    """Example 4: Add documentation to functions."""
    print("\n" + "="*60)
    print("Example 4: Adding docstrings to functions")
    print("="*60)
    
    original = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
"""
    
    diff = """@@
 def fibonacci(n):
+    \"\"\"
+    Calculate the nth Fibonacci number using recursion.
+    
+    Args:
+        n: The position in the Fibonacci sequence
+        
+    Returns:
+        The nth Fibonacci number
+    \"\"\"
     if n <= 1:
         return n
     return fibonacci(n-1) + fibonacci(n-2)
@@
 def factorial(n):
+    \"\"\"
+    Calculate the factorial of n using recursion.
+    
+    Args:
+        n: A non-negative integer
+        
+    Returns:
+        The factorial of n
+    \"\"\"
     if n <= 1:
         return 1
     return n * factorial(n-1)
"""
    
    result = apply_diff(original, diff)
    print("\nOriginal:")
    print("-" * 60)
    print(original)
    print("\nWith docstrings:")
    print("-" * 60)
    print(result)


def example_5_fix_bug():
    """Example 5: Fix a bug in existing code."""
    print("\n" + "="*60)
    print("Example 5: Fixing off-by-one bug")
    print("="*60)
    
    original = """def get_last_n_items(items, n):
    # Bug: This doesn't handle n=0 correctly
    if n > len(items):
        return items
    return items[-n:]

def process_batch(data, batch_size=10):
    results = []
    for i in range(0, len(data), batch_size):
        # Bug: This might go past the end
        batch = data[i:i+batch_size+1]
        results.append(batch)
    return results
"""
    
    diff = """@@
 def get_last_n_items(items, n):
-    # Bug: This doesn't handle n=0 correctly
-    if n > len(items):
+    # Handle edge cases
+    if n <= 0:
+        return []
+    if n >= len(items):
         return items
     return items[-n:]
@@
 def process_batch(data, batch_size=10):
     results = []
     for i in range(0, len(data), batch_size):
-        # Bug: This might go past the end
-        batch = data[i:i+batch_size+1]
+        # Fixed: Correct slicing
+        batch = data[i:i+batch_size]
         results.append(batch)
     return results
"""
    
    result = apply_diff(original, diff)
    print("\nOriginal (with bugs):")
    print("-" * 60)
    print(original)
    print("\nFixed:")
    print("-" * 60)
    print(result)


def example_6_command_line_usage():
    """Example 6: Using the tool from command line."""
    print("\n" + "="*60)
    print("Example 6: Command-line usage examples")
    print("="*60)
    
    examples = """
# Create a new file from a diff
python apply_patch.py --diff new_feature.diff --output src/feature.py --create

# Update an existing file in-place (creates backup)
python apply_patch.py --file src/main.py --diff fixes.diff

# Update with custom output location (no backup needed)
python apply_patch.py --file src/main.py --diff changes.diff --output src/main_v2.py

# Preview changes without modifying files
python apply_patch.py --file src/main.py --diff changes.diff --dry-run

# Update without creating backup
python apply_patch.py --file src/main.py --diff changes.diff --no-backup

# Apply diff from stdin (useful for piping)
cat changes.diff | python apply_patch.py --file src/main.py --diff -

# Using in a script
for diff_file in patches/*.diff; do
    echo "Applying $diff_file..."
    python apply_patch.py --file src/code.py --diff "$diff_file" || exit 1
done
"""
    
    print(examples)


def example_7_programmatic_usage():
    """Example 7: Using the tool programmatically in Python."""
    print("\n" + "="*60)
    print("Example 7: Programmatic usage in Python")
    print("="*60)
    
    code_example = '''
from apply_patch import apply_diff, DiffApplicationError
from pathlib import Path

# Method 1: Apply diff to string content
current_code = Path("myfile.py").read_text()
diff_text = """@@
-old_line
+new_line
"""
try:
    new_code = apply_diff(current_code, diff_text)
    Path("myfile.py").write_text(new_code)
    print("Patch applied successfully!")
except DiffApplicationError as e:
    print(f"Failed to apply patch: {e}")

# Method 2: Use the file helper
from apply_patch import apply_patch_to_file

try:
    apply_patch_to_file(
        file_path=Path("src/main.py"),
        diff_path=Path("patches/fix.diff"),
        backup=True  # Creates .bak file
    )
except DiffApplicationError as e:
    print(f"Patch failed: {e}")

# Method 3: Create new file
diff = """@@
+print("Hello, world!")
"""
content = apply_diff("", diff, mode="create")
Path("hello.py").write_text(content)
'''
    
    print(code_example)


def example_8_integration_with_llm():
    """Example 8: How to use this with LLM-generated diffs."""
    print("\n" + "="*60)
    print("Example 8: Integration with LLM-generated diffs")
    print("="*60)
    
    workflow = """
WORKFLOW: Using this tool with GPT/Claude for code editing

1. Provide context to the LLM:
   "Here's my current code: [paste code]"
   "Please generate a V4A diff to [describe change]"

2. LLM generates a V4A diff:
   @@
   -old code
   +new code

3. Save the diff to a file:
   with open('change.diff', 'w') as f:
       f.write(llm_response)

4. Apply the patch:
   python apply_patch.py --file src/code.py --diff change.diff

5. Review and test the changes

TIPS:
- Ask the LLM to use V4A format explicitly
- Always review LLM-generated diffs before applying
- Use --dry-run to preview changes
- Keep backups enabled (default)
- Test thoroughly after applying patches

EXAMPLE PROMPT FOR LLM:
"I have this Python code:
[your code]

Please generate a V4A format diff to add type hints to all functions.
Use the format:
@@
 existing line
-line to remove
+line to add

Do not include file headers, just the @@ hunks."
"""
    
    print(workflow)


def run_all_examples():
    """Run all examples."""
    print("\n" + "="*70)
    print(" "*15 + "V4A APPLY PATCH TOOL - EXAMPLES")
    print("="*70)
    
    example_1_create_new_file()
    example_2_refactor_function_name()
    example_3_add_error_handling()
    example_4_add_docstrings()
    example_5_fix_bug()
    example_6_command_line_usage()
    example_7_programmatic_usage()
    example_8_integration_with_llm()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")


if __name__ == '__main__':
    run_all_examples()