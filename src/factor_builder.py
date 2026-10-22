import pandas as pd
import numpy as np
import glob
import os
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_all_stock_data(data_path=None):
    """读取所有股票CSV，合并成一个大的DataFrame"""
    if data_path is None:
        data_path=os.path.join(BASE_DIR,"data","raw","*.csv")
    files=glob.glob(data_path)
    if not files:
        raise FileNotFoundError("没有找到数据文件，请先运行data_loader.py")

    dfs=[]
    for file in files:
        if os.path.getsize(file)<1000:
            print(f"跳过异常文件:{file}")
            continue
        code=os.path.basename(file).replace(".csv","")
        try:
            df=pd.read_csv(file)
        except pd.errors.EmptyDataError:
            print(f"跳过无法解析的文件:{file}")
            continue
        if df.empty:
            print(f"跳过空数据:{file}")
            continue
        
        df["股票代码"]=code
        dfs.append(df)

    all_data=pd.concat(dfs,ignore_index=True)
    all_data["日期"]=pd.to_datetime(all_data["日期"])
    all_data=all_data.sort_values(["股票代码","日期"])
    return all_data

def calculate_technical_factors(df):
    """基于日频行情计算技术因子，并重采样为月频"""
    df["日收益率"]=df.groupby("股票代码")["收盘"].pct_change()
    grouped=df.groupby("股票代码")
    df["动量20"]=grouped["收盘"].pct_change(20)
    df["反转5"]=-grouped["收盘"].pct_change(5)
    df["波动率20"]=grouped["日收益率"].rolling(20).std().reset_index(0,drop=True)
    df["年月"]=df["日期"].dt.to_period("M")
    monthly=df.groupby(["股票代码","年月"]).last().reset_index()
    return monthly

def save_factors(df,output_path=None):
    """保存因子数据"""
    if output_path is None:
        output_path=os.path.join(BASE_DIR,"data","processed","factors.csv")
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    df.to_csv(output_path,index=False)
    print(f"因子数据已保存到{output_path},共{len(df)}行")

if(__name__=="__main__"):
    print("加载数据...")
    data=load_all_stock_data()
    print(f"共加载{data['股票代码'].nunique()}只股票，{len(data)}行数据")
    print("计算技术因子...")
    factors=calculate_technical_factors(data)
    print("保存因子数据...")
    save_factors(factors)
    print("完成") 