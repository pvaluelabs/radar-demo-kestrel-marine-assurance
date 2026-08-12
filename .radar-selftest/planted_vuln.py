# RADAR self-test fixture: known planted vulnerabilities for the ensemble SARIF demo.
import subprocess

def run_cmd(cmd):
    subprocess.run(cmd, shell=True)   # planted: unsafe shell=True

def dynamic(expr):
    eval(expr)                        # planted: unsafe eval
