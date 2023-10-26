import os


# MSMT17 BoT_R50 is not invovled with below
yaml_number = [1, 4, 9, 10]
dataset_name = ['MSMT17'] #, ]'CUHK03', 'DukeMTMC'
for number in yaml_number:
    if number == 1:
        for name in dataset_name:
            yaml_path = f"./configs/{name}/bagtricks_R50.yml"
            shell_call = f"python3 tools/train.py --config-file {yaml_path} --num-gpus 1 --eval-only TEST.RERANK.ENABLED True"
            print(shell_call)
            os.system(shell_call)
    elif number == 4 or number == 9:
        for name in dataset_name:
            yaml_path = f"./configs/{name}/bagtricks_R50-{number}.yml"
            shell_call = f"python3 tools/train.py --config-file {yaml_path} --num-gpus 1 --eval-only TEST.RERANK.ENABLED True"
            print(shell_call)
            os.system(shell_call)
    else:
        for name in dataset_name:
            yaml_path = f"./configs/{name}/bagtricks_S50-{number}.yml"
            shell_call = f"python3 tools/train.py --config-file {yaml_path} --num-gpus 1 --eval-only TEST.RERANK.ENABLED True"
            print(shell_call)
            os.system(shell_call)
print(shell_call)
# os.system("python3 tools/train.py --config-file ./configs/MSMT17/bagtricks_R50.yml --eval-only TEST.RERANK.ENABLE True")