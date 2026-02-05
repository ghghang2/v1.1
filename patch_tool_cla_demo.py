#!/usr/bin/env python3
"""
Quick demo of the apply_patch tool in action
"""

from patch_tool_cla import apply_diff

print("="*70)
print("V4A APPLY PATCH TOOL - LIVE DEMO")
print("="*70)

# Demo 1: Simple function update
print("\n📝 Demo 1: Update a function")
print("-" * 70)

original = """def greet(name):
    return f'Hi {name}'
"""

diff = """@@
 def greet(name):
-    return f'Hi {name}'
+    return f'Hello, {name}!'
"""

print("BEFORE:")
print(original)
print("\nDIFF:")
print(diff)
print("\nAFTER:")
result = apply_diff(original, diff)
print(result)

# Demo 2: Create a new file
print("\n\n📄 Demo 2: Create a new file from scratch")
print("-" * 70)

diff = """@@
+class Calculator:
+    def add(self, a, b):
+        return a + b
+    
+    def multiply(self, a, b):
+        return a * b
"""

print("DIFF:")
print(diff)
print("\nCREATED FILE:")
result = apply_diff("", diff, mode="create")
print(result)

# Demo 3: Refactoring
print("\n\n🔧 Demo 3: Multi-location refactoring")
print("-" * 70)

original = """def process():
    data = fetch_data()
    result = transform(data)
    return result

def main():
    result = process()
    print(result)
"""

diff = """@@
-def process():
-    data = fetch_data()
-    result = transform(data)
-    return result
+def process_pipeline():
+    \"\"\"Process data through the pipeline.\"\"\"
+    data = fetch_data()
+    result = transform(data)
+    return result
@@
 def main():
-    result = process()
+    result = process_pipeline()
     print(result)
"""

print("BEFORE:")
print(original)
print("\nDIFF:")
print(diff)
print("\nAFTER:")
result = apply_diff(original, diff)
print(result)

print("\n" + "="*70)
print("Demo complete! ✓")
print("="*70)