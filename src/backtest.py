import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.sans-serif']=['Simhei','Microsoft YaHei']
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
    """计算下月收益率"""
    df["下月收盘"]=df.groupby("股票代码")["收盘"].shift(-1)
    df["下月收益"]=df["下月收盘"]/df["收盘"]-1
    return df

def standardize_factor(df,factor_col):
    """每个月底对因子做去极值和标准化"""
    def _std(x):
        if x.nunique()<2:
            return pd.Series([0]*len(x),index=x.index)
        lower=x.quantile(0.01)
        upper=x.quantile(0.99)
        x=x.clip(lower,upper)
        std=x.std()
        if std==0:
            return pd.Series([0]*len(x),index=x.index)
        return (x-x.mean())/x.std()
    return df.groupby("年月")[factor_col].transform(_std)

def build_composite_score(df,factor_cols):
    """等权合成多因子综合得分"""
    for col in factor_cols:
        df[f"{col}_std"]=standardize_factor(df,col)
    df["综合得分"]=df[[f"{col}_std" for col in factor_cols]].mean(axis=1)
    return df

def build_portfolio(df,score_col="综合得分",top_pct=0.2,return_col="下月收益"):
    """每月按综合得分选前top_pct的股票，等权持有"""
    selected_idx=[]
    for month,group in df.groupby("年月"):
        n=max(int(len(group)*top_pct),1)
        selected=group.nlargest(n,score_col).index
        selected_idx.extend(selected)
    portfolio=df.loc[selected_idx].copy()
    monthly_return=portfolio.groupby("年月")[return_col].mean()
    return monthly_return

def apply_transaction_cost(monthly_return,turnover=0.3,cost_rate=0.002):
    """扣除交易成本：换手率×单边成本×2（买+卖）"""
    cost=turnover*cost_rate*2
    return monthly_return-cost

def performance(monthly_return,rf=0.02):
    """计算绩效指标"""
    monthly_return=monthly_return.dropna()
    cum_nav=(1+monthly_return).cumprod()
    total_return=cum_nav.iloc[-1]-1
    n_months=len(monthly_return)
    ann_return=(1+total_return)**(12/n_months)-1
    ann_vol=monthly_return.std()*np.sqrt(12)
    sharpe=(ann_return-rf)/ann_vol if ann_vol!=0 else np.nan
    rolling_max=cum_nav.cummax()
    drawdown=(cum_nav-rolling_max)/rolling_max
    max_dd=drawdown.min()
    calmar=ann_return/abs(max_dd) if max_dd!=0 else np.nan
    win_rate=(monthly_return>0).mean()
    return {
        "累计收益":total_return,
        "年化收益":ann_return,
        "年化波动":ann_vol,
        "夏普比率":sharpe,
        "最大回撤":max_dd,
        "卡玛比率":calmar,
        "月胜率":win_rate
    },cum_nav

def save_results(monthly_return,cum_nav,perf,prefix="strategy"):
    """保存净值和绩效"""
    result_dir=os.path.join(BASE_DIR,"results")
    os.makedirs(result_dir,exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.plot(cum_nav.index,cum_nav.values,marker='o')
    plt.title("策略累计净值")
    plt.xlabel("年月")
    plt.ylabel("净值")
    ax=plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir,f"{prefix}_nav.png"))
    plt.close()
    perf_df=pd.DataFrame([perf])
    perf_df.to_csv(os.path.join(result_dir,f"{prefix}_performance.csv"),index=False)
    print(f"绩效结果已保存到{result_dir}")

if __name__=="__main__":
    print("加载因子数据...")
    base_df=load_factors()
    base_df=compute_next_month_return(base_df)
    versions=[
        {"name":"V1_raw","desc":"V1线性版","nonlinear":False},
        {"name":"V2_nonlinear","desc":"V2非线性版","nonlinear":True}
    ]
    factor_cols=["动量20","反转5","波动率20"]
    for ver in versions:
        print(f"正在运行策略:{ver['desc']}")
        df=base_df.copy()
        if ver["nonlinear"]:
            df["动量20"]=abs(df["动量20"]-df["动量20"].median())
            df["反转5"]=abs(df["反转5"]-df["反转5"].median())
        print("合成综合得分...")
        df=build_composite_score(df,factor_cols)
        print("构建组合...")
        monthly_return=build_portfolio(df,top_pct=0.2)
        print("扣除交易成本...")
        monthly_return_net=apply_transaction_cost(monthly_return,turnover=0.3,cost_rate=0.002)
        print("计算绩效...")
        perf,cum_nav=performance(monthly_return_net)
        for k,v in perf.items():
            print(f"{k}:{v:.4f}")
        print("保存结果...")
        save_results(monthly_return_net,cum_nav,perf,prefix=f"strategy_{ver['name']}")
    print("回测完成")
