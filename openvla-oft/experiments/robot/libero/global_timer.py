# global_timer.py
import torch

# 定义全局变量
start_event_act = None

def record_start():
    global start_event_act
    start_event_act = torch.cuda.Event(enable_timing=True)
    start_event_act.record()

def get_time_elapsed():
    global start_event_act
    if start_event_act is None:
        return 0.0
    
    end_event_act = torch.cuda.Event(enable_timing=True)
    end_event_act.record()
    torch.cuda.synchronize() # 等待 GPU 跑完
    
    return start_event_act.elapsed_time(end_event_act) # 返回毫秒