# %%
import yfinance as yf
from pandas.core.frame import DataFrame
import pandas
import datetime

import csv
from pathlib import Path

# %%

def download_ticker(ticker: str, start: datetime.datetime, days=5, interval="1m"):
    # start, end = YYYY-MM-DD
    datefmt = "%Y-%m-%d"
    if isinstance(start, str):
        start = datetime.datetime.strptime(start, datefmt)
    end = start + datetime.timedelta(days=days)
    startstr = start.strftime(datefmt)

    ticker = ticker.upper()
    path = Path(f"yf-data/{ticker}-{startstr}-{days}d-{interval}.csv")
    if path.exists():
        return pandas.read_csv(path)

    print(f"make file {path}")    
    df: DataFrame = None

    while start < end:
        period = min(5, days)
        req_end = start + datetime.timedelta(days=period)
        print(f"request start {start.strftime(datefmt)}, req_end {req_end.strftime(datefmt)}, period {period}")
        thisdf: DataFrame = yf.download(tickers="MSFT", period=f"{period}d", interval=interval,
                                        start=start.strftime(datefmt), end=req_end.strftime(datefmt))
        if len(thisdf) == 0:
            raise Exception("didn't get what we wanted: {thisdf=}")
        if df is None:
            df = thisdf
        else:
            df = pandas.concat([df, thisdf])
        
        # print(f"  {len(thisdf)=} {len(df)=}")
        
        start = req_end
        days -= period

    df.to_csv(path)
    return df
    
data = download_ticker("MSFT", "2023-01-23", 30, "1m")
data

# %%
