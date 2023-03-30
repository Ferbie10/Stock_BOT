
FROM vsc_stock:vsc_working
COPY . /root/home/git
WORKDIR /root/home/git

COPY requirements.txt /root/home/git
RUN apt-get update 
RUN apt-get install --no-install-recommends python3 python3-pip
RUN apt install -y git
RUN pip install -r requirements.txt
COPY . /root/home/git



EXPOSE 8888

