import pandas as pd
from src import WINDOWS

def get_year_prediction(company, year='2019'):
    date_range = (f"01.01.{year}", f"31.12.{year}")
    data = pd.read_csv(f'data/{company}.csv')
    data.drop(labels=['<PER>', '<TIME>'], axis=1, inplace=True)
    data.rename({'<TICKER>': 'company', '<CLOSE>': 'close', '<DATE>': 'date'}, axis=1, inplace=True)
    data['date'] = pd.to_datetime(data['date'], dayfirst=True)
    trade_data = data.loc[(data["date"] >= pd.to_datetime(date_range[0])) &
                          (data["date"] <= pd.to_datetime(date_range[1]))]
    win_size = WINDOWS[company]
    return trade_data


def trade_sym(x, y, long_short=False):
    s = 0
    min_cost = x[0]
    for i in range(len(y) - 1):
        if y[i + 1] == 0:
            if y[i] == 1:
                s += x[i] - min_cost
            else:
                min_cost = x[i + 1]
    return s


print(get_year_prediction('gazp'))

# x = [1, 3, 5, 3, 2, 1, 0, 6] # stock costs
# y = get_year_prediction(x) # model predicts
#
# res = trade_sym(x, y)
# print(f"Конечный капитал: {s}")
# print(f"Прирост: {100 * (s - x[0]) / x[0]:.2f}%")
