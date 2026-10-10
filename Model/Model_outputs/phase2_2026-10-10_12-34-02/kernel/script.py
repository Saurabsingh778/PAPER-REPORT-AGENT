
import subprocess, json, time, os, sys
PLAN = json.loads('{"setup_commands":["pip install -r requirements.txt"],"steps":[{"claim_index":0,"command":"python exp_4/l_j_baseline_5_fold.py","expected_output_hint":"95.10","timeout_min":20},{"claim_index":1,"command":"python glass_topology_test.py","expected_output_hint":"52","timeout_min":10},{"claim_index":2,"command":"python glass_strain_normalized.py","expected_output_hint":"94.5","timeout_min":15},{"claim_index":3,"command":"python exp_4/ka_new_val.py","expected_output_hint":"96.23","timeout_min":20},{"claim_index":4,"command":"python exp_4/ka_new_val.py","expected_output_hint":"0.9935","timeout_min":20},{"claim_index":5,"command":"python exp_4/ka_new_val.py","expected_output_hint":"0.9621","timeout_min":20},{"claim_index":6,"command":"python exp_4/ka_new_val.py","expected_output_hint":"96.23","timeout_min":20},{"claim_index":7,"command":"python exp_5/test_v1.py","expected_output_hint":"-0.929","timeout_min":30},{"claim_index":8,"command":"python exp_5/test_v1.py","expected_output_hint":"8.6","timeout_min":30},{"claim_index":9,"command":"python exp_4/l_j_baseline_5_fold.py","expected_output_hint":"0.9906","timeout_min":20}],"unreproducible":[]}')
REPO = 'https://github.com/Saurabsingh778/Thermal_history_in_glass'
W = "/kaggle/working"
out = {"setup": [], "steps": []}

def run(cmd, cwd, timeout):
    t = time.time()
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {"cmd": cmd, "rc": p.returncode, "sec": round(time.time()-t),
                "stdout": p.stdout[-6000:], "stderr": p.stderr[-3000:]}
    except subprocess.TimeoutExpired as e:
        so = e.stdout if isinstance(e.stdout, str) else ""
        return {"cmd": cmd, "rc": -9, "sec": timeout, "stdout": so[-3000:], "stderr": "TIMEOUT"}

print(run("nvidia-smi -L", W, 60)["stdout"])
out["clone"] = run(f"git clone --depth 1 {REPO} repo", W, 600)
RD = W + "/repo"
for c in PLAN["setup_commands"]:
    out["setup"].append(run(c, RD, 1800))
for s in PLAN["steps"]:
    r = run(s["command"], RD, s["timeout_min"] * 60)
    r["claim_index"] = s["claim_index"]
    out["steps"].append(r)
    json.dump(out, open(W + "/results.json", "w"), indent=1)  # save progress
json.dump(out, open(W + "/results.json", "w"), indent=1)
