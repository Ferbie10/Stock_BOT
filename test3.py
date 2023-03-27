import os
symbol = aapl
path = '/root/home/git'
pathname = os.path.join(path,symbol)
filename = os.path.join(path, f'{symbol}.csv')
if not os.path.exists(pathname):
    os.mkdir(pathname)
else:
    tik_history.to_csv(filename)
