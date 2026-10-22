import akshare as ak
import pandas as pd
import os
import time

def get_stock_pool():
    """获取沪深300成分股"""
    print("获取股票池...")
    df=ak.index_stock_cons_csindex(symbol="000300")
    stock_list=df['成分券代码'].tolist()
    return stock_list

def download_stock_data(stock_list,start_date='20180101',end_date='20241231'):
    """下载历史行情（后复权）"""
    os.makedirs("data/raw",exist_ok=True)

    for code in stock_list:
        file_path=f"data/raw/{code}.csv"
        if os.path.exists(file_path) and os.path.getsize(file_path)>1000:
            print(f"{code}数据已存在，跳过")
            continue

        print(f"正在下载{code}...")
        try:
            df=ak.stock_zh_a_hist(
                symbol=code,period="daily",
                start_date=start_date,end_date=end_date,adjust="hfq"
            )
            if df is None or df.empty:
                print(f"{code}返回空数据,跳过")
                continue
            df.to_csv(file_path,index=False)
            time.sleep(1)
        except Exception as e:
            print(f"{code}下载失败:{e}")

if __name__=="__main__":
    pool=get_stock_pool()
    download_stock_data(pool)
    print("下载完成")