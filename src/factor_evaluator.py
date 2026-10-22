import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.sans-serif']=['SimHei','Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_factors(path=None):
    """读取月频因子数据"""
    if path is None:
        path=os.path.join(BASE_DIR,"data","processed","factors.csv")
    df=pd.read_csv(path)
    df["年月"]=pd.to_datetime(df["年月"])
    df=df.sort_values(["股票代码","年月"])
    return df

def compute_next_month_return(df):
    """计算每只股票的下月收益率"""
    df["下月收盘"]=df.groupby("股票代码")["收盘"].shift(-1)
    df["下月收益"]=df["下月收盘"]/df["收盘"]-1
    return df

def evaluate_factor(df,factor_col,month_col="年月",return_col="下月收益"):
    """计算单因子的IC序列、IC均值、IC_IR、IC胜率"""
    ic_series=[]
    for month,group in df.groupby(month_col):
        group=group.dropna(subset=[factor_col,return_col])
        if len(group)<10:
            continue
        ic,_=spearmanr(group[factor_col],group[return_col])
        ic_series.append({"年月":month,"IC":ic})
    ic_df=pd.DataFrame(ic_series)
    if ic_df.empty:
        return None
    ic_mean=ic_df["IC"].mean()
    ic_std=ic_df["IC"].std()
    ic_ir=ic_mean/ic_std if ic_std!=0 else np.nan
    win_rate=(ic_df["IC"]>0).mean()
    return {
        "因子":factor_col,
        "IC均值":ic_mean,
        "IC标准差":ic_std,
        "IC_IR":ic_ir,
        "IC胜率":win_rate,
        "IC序列":ic_df
    }

def layered_backtest(df,factor_col,return_col="下月收益",n_layers=5):
    """分层回测：每月按因子值分n层，计算每层下月平均收益"""
    def group_factor(x):
        if x.nunique()<n_layers:
            return pd.Series([np.nan]*len(x),index=x.index)
        return pd.qcut(x,n_layers,labels=False,duplicates='drop')

    df["因子分层"]=df.groupby("年月")[factor_col].transform(group_factor)
    layer_returns=df.groupby(["年月","因子分层"])[return_col].mean().unstack()
    layer_returns=layer_returns.fillna(0)
    if n_layers-1 in layer_returns.columns and 0 in layer_returns.columns:
        layer_returns["多空"]=layer_returns[n_layers-1]-layer_returns[0]
    return layer_returns

def save_results(ic_result,layer_returns,factor_name):
    """保存IC图和分层累计收益图"""
    result_dir=os.path.join(BASE_DIR,"results")
    os.makedirs(result_dir,exist_ok=True)

    plt.figure(figsize=(10,4))
    plt.plot(ic_result["IC序列"]["年月"],ic_result["IC序列"]["IC"],marker='o',linestyle='-')
    plt.axhline(0,color='blue',linestyle='--')
    plt.title(f"{factor_name}IC序列")
    plt.xlabel("年月")
    plt.ylabel("IC")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir,f"{factor_name}_IC.png"))
    plt.close()

    plt.figure(figsize=(10,4))
    for col in layer_returns.columns:
        if col!="多空":
            cum=(1+layer_returns[col]).cumprod()-1
            plt.plot(layer_returns.index,cum,label=f"第{col+1}层")
    if "多空" in layer_returns.columns:
        cum_long_short=(1+layer_returns["多空"]).cumprod()-1
        plt.plot(layer_returns.index,cum_long_short,label="多空",linewidth=2,color='red')
    plt.title(f"{factor_name}分层累计收益")
    plt.xlabel("年月")
    plt.ylabel("累计收益")
    plt.legend()
    ax=plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir,f"{factor_name}_layered.png"))
    plt.close()


if __name__=="__main__":
    print("加载因子数据...")
    df=load_factors()
    df=compute_next_month_return(df)
    factors_to_test=["动量20","反转5","波动率20"]
    all_results=[]
    for factor in factors_to_test:
        print(f"\n评估因子:{factor}")
        ic_result=evaluate_factor(df,factor)
        if ic_result is None:
            print(f"因子{factor}数据不足，跳过")
            continue
        print(f"IC均值:{ic_result['IC均值']:.4f},IC_IR:{ic_result['IC_IR']:.4f},胜率:{ic_result['IC胜率']:.2%}")
        layer_returns=layered_backtest(df,factor)
        save_results(ic_result,layer_returns,factor)
        all_results.append({
            "因子":factor,
            "IC均值":ic_result["IC均值"],
            "IC_IR":ic_result["IC_IR"],
            "IC胜率":ic_result["IC胜率"]
        })
    summary=pd.DataFrame(all_results)
    summary_path=os.path.join(BASE_DIR,"results","factor_summary.csv")
    os.makedirs(os.path.dirname(summary_path),exist_ok=True)
    summary.to_csv(summary_path,index=False)
    print(f"\n汇总结果已保存到{summary_path}")

    