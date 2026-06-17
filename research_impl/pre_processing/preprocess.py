# -*- coding: utf-8 -*-
import random
import os
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from tqdm import tqdm
import argparse
import copy
import geohash2
from collections import defaultdict
from multiprocessing import Pool

# ==========================================
# 1. Utility Functions (formerly utils/util.py)
# ==========================================

def get_workspace():
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    file = os.path.dirname(file)
    return file

ws = get_workspace()

def dir_check(path):
    dir_path = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir_path): os.makedirs(dir_path)
    return path

def dict_merge(dict_list=[]):
    dict_ = {}
    for dic in dict_list:
        dict_ = {**dict_, **dic}
    return dict_

def multi_thread_work(parameter_queue, function_name, thread_number=5):
    pool = Pool(thread_number)
    result = pool.map(function_name, parameter_queue)
    pool.close()
    pool.join()
    return result

def write_list_list(fp, list_, model="a", sep=","):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, mode=model, encoding="utf-8") as f:
        for line in list_:
            a_line = sep.join(str(l) for l in line)
            f.write(f"{a_line}\n")

# ==========================================
# 2. Preprocessing Logic (formerly data/preprocess.py)
# ==========================================

def idx(df, col_name):
    return list(df.columns).index(col_name)

def time2min(t):
    if pd.isna(t) or t == 'nan' or not isinstance(t, str): return 0, 0
    try:
        M, d = t.split(' ')[0].split('-')
        h, m, s = t.split(' ')[1].split(':')
        return int(f'{M}{d}'), 60 * int(h) + int(m) + int(s) / 60
    except:
        return 0, 0

def split_trajectory(df):
    courier_l = []
    if len(df) == 0: return courier_l
    temp = df.values[0]
    c_idx = idx(df, 'courier_id')
    ds_idx = idx(df, 'ds')
    f = 0
    t = 0
    for row in df.values:
        if row[c_idx] != temp[c_idx] or row[ds_idx] != temp[ds_idx]:
            courier_l.append(df[f:t])
            f = t
        t = t + 1
        temp = row
    courier_l.append(df[f:t])
    return courier_l

def list2str(l):
    return '.'.join(map(str, l))

def courier_info(df):
    couriers = list(set(df['courier_id']))
    feature_dict = {}
    keys = ['index', 'id', 'order_sum', 'dis_sum', 'work_days', 'order_avg_day', 'dis_avg_day', 'time_avg_order', 'dis_avg_order', 'speed_avg_order']
    for key in keys: feature_dict[key] = {}
    
    for idx_val, c in enumerate(couriers):
        c_df = df[df['courier_id'] == c]
        feature_dict['index'][c] = idx_val
        feature_dict['id'][c] = c
        feature_dict['order_sum'][c] = c_df.shape[0]
        feature_dict['dis_sum'][c] = sum(c_df['dis_to_last_package'])
        feature_dict['work_days'][c] = len(set(c_df['ds']))
        feature_dict['order_avg_day'][c] = feature_dict['order_sum'][c] / feature_dict['work_days'][c]
        feature_dict['dis_avg_day'][c] = feature_dict['dis_sum'][c] / feature_dict['work_days'][c]
        feature_dict['time_avg_order'][c] = np.mean(c_df['time_to_last_package'])
        feature_dict['dis_avg_order'][c] = np.mean(c_df['dis_to_last_package'])
        feature_dict['speed_avg_order'][c] = feature_dict['dis_sum'][c] / (sum(c_df['time_to_last_package'])) if sum(c_df['time_to_last_package']) != 0 else 5
    return couriers, feature_dict

def process_traj_kernel(args={}):
    c_lst = args['c_lst']
    result = {}
    for c in c_lst:
        c_v = c.reset_index()
        for n, row in c_v.iterrows():
            # Delivery Context: Depot -> Delivery
            # Using delivery_time as the primary completion metric
            date_dt, dt = time2min(row['delivery_time'])
            date_at, at = time2min(row['accept_time'])
            if date_dt != date_at: at = at - 60 * 24
            et = 1440  # LaDe-D delivery has no promised time window
            
            last_idx = max(0, n - 1)
            last_ft = c_v.iloc[last_idx]['finish_time_minute']
            last_lon_ = c_v.iloc[last_idx]['lng']
            last_lat_ = c_v.iloc[last_idx]['lat']
            
            o_id = row['order_id']
            result[(o_id, 'accept_time_minute')] = at
            result[(o_id, 'expect_finish_time_minute')] = et
            result[(o_id, 'time_to_last_package')] = row['finish_time_minute'] - last_ft
            result[(o_id, 'dis_to_last_package')] = int(geodesic((last_lat_, last_lon_), (row['lat'], row['lng'])).meters)
    return result

# LaDe-D delivery columns already use the target names
# (courier_id, lng, lat, aoi_type, accept_time, delivery_time, ds, order_id),
# so no renaming is required. Kept for forward-compatibility with other schemas.
delivery_name_dict = {}

def pre_process_delivery(fin, fout, is_test=False, thread_num=20):
    df = pd.read_csv(fin, sep=',', encoding='utf-8')
    df = df.rename(columns=delivery_name_dict)
    
    # Context: Transition to Delivery focus (Depot -> Delivery -> Depot)
    df['finish_time_minute'] = df['delivery_time'].apply(lambda t: time2min(t)[1])
    df = df.sort_values(by=['ds', 'courier_id', 'finish_time_minute'])
    
    courier_l = split_trajectory(df)
    n = len(courier_l)
    task_num = max(1, n // thread_num)
    args_lst = [{'c_lst': courier_l[i: min(i + task_num, n)]} for i in range(0, n, task_num)]
    
    results = multi_thread_work(args_lst, process_traj_kernel, thread_num)
    result_dict = dict_merge(results)
    
    for col in ['accept_time_minute', 'expect_finish_time_minute', 'time_to_last_package', 'dis_to_last_package']:
        df[col] = df['order_id'].apply(lambda x: result_dict.get((x, col), 0))
    
    couriers, couriers_feature = courier_info(df)
    df.insert(0, 'index', range(1, df.shape[0] + 1))
    
    if fout:
        dir_check(fout)
        df.to_csv(os.path.join(fout, 'package_feature.csv'), index=False)
        pd.DataFrame(couriers_feature).to_csv(os.path.join(fout, 'courier_feature.csv'), index=False)
    
    return df, pd.DataFrame(couriers_feature)

# ==========================================
# 3. Entry point: raw delivery logs -> package_feature.csv (per city)
#    Dataset tensor construction now lives in research_impl/dataset/dataset.py.
# ==========================================

from research_impl.pre_processing.utils import CITIES, CITY_CODE


def main():
    parser = argparse.ArgumentParser(description="Delivery preprocessing -> package_feature.csv")
    parser.add_argument('--cities', nargs='+', default=CITIES)
    parser.add_argument('--raw_dir', type=str,
                        default=os.path.join(ws, 'research_impl', 'dataset', 'csv'))
    parser.add_argument('--is_test', type=bool, default=False)
    parser.add_argument('--thread_num', type=int, default=20)
    args = parser.parse_args()

    for city in args.cities:
        code = CITY_CODE.get(city, city)
        fin = os.path.join(args.raw_dir, f'delivery_{code}.csv')
        if not os.path.exists(fin):
            print(f"[!] Skipping {city}: raw file not found at {fin}")
            continue
        fout = os.path.join(ws, 'research_impl', 'dataset', 'tmp', city)
        print(f"[*] Preprocessing {city} ({fin}) ...")
        pre_process_delivery(fin, fout, args.is_test, args.thread_num)
        print(f"[+] {city}: wrote package_feature.csv + courier_feature.csv to {fout}")


if __name__ == "__main__":
    main()
