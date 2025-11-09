import tempfile
import subprocess

def code_check(code: str):
  with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
      tmp.write(code)
      tmp_path = tmp.name
  result = subprocess.run(
    ["python", tmp_path],
    capture_output=True,
    text=True,
    timeout=20
  )
  return result.returncode

def code_save(code: str,
              name: str):
  file_path = f"{name}.py"
  with open(file_path, "w", encoding="utf-8") as f:
      f.write(code)
