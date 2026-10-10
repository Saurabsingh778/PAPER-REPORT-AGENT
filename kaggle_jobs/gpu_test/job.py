import subprocess, torch
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
open("/kaggle/working/result.txt", "w").write(f"cuda={torch.cuda.is_available()}\n")
