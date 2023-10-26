import os

log_path = "/data/code/lorenzo/ReID/lorenzo-reid-baseline/logs/market1501"
dir_names = os.listdir(log_path)
for dir_name in dir_names:
    old_log_path = os.path.join(log_path, dir_name)
    new_log_path = os.path.join('./logs', dir_name)
    if not os.path.exists(new_log_path):
        os.mkdir(new_log_path)
    cp_cmd = f"cp {os.path.join(old_log_path, 'model_best.pth')} {os.path.join(new_log_path, 'model_best.pth')}"
    os.system(cp_cmd)