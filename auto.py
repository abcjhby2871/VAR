import subprocess 
import re
import json
import os

MODEL_LIST=[16,20,24,30]
QUALITY_PY=os.getenv('QUALITY_PY','python')
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
for model in MODEL_LIST:
    task1=subprocess.Popen(f"python ../sample.py {model}",shell=True,env=os.environ) #采样
    task1.wait()
    task2=subprocess.Popen(f"""python -c "from utils.misc import create_npz_from_sample_folder; create_npz_from_sample_folder('result/d{model}')" """,shell=True,env=os.environ) #转成npz格式
    task2.wait()
    task3=subprocess.Popen(f"{QUALITY_PY} evaluator.py data/imagenetnpz/VIRTUAL_imagenet256_labeled.npz result/d{model}.npz",stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,shell=True,env=os.environ)
    task3.wait()
    output = task3.stdout.read()
    error_output = task3.stderr.read()
    print(output)
    if error_output:
        print(error_output)
    results = {}
    inception_score = re.search(r"Inception Score: ([\d\.]+)", output)
    fid = re.search(r"FID: ([\d\.]+)", output)
    precision = re.search(r"Precision: ([\d\.]+)", output)
    recall = re.search(r"Recall: ([\d\.]+)", output)
    results['inception_score'] = float(inception_score.group(1))
    results['fid'] = float(fid.group(1))
    results['precision'] = float(precision.group(1))
    results['recall'] = float(recall.group(1))
    if os.path.exists('log/quality.json'):
        with open('log/quality.json','r') as f:
            T = json.load(f)
    else:
        T={}
    T[model]=results
    with open("log/quality.json", "w") as f:
        json.dump(T,f)
    print("Results saved to output_results.json")
    task4=subprocess.Popen(f"python ../scale.py {model}",shell=True,env=os.environ)
    task4.wait()




