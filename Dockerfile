FROM stockbot/latest:1.2
COPY . /root/home/git
WORKDIR /root/home/git

COPY requirements.txt /root/home/git
WORKDIR /opt/app
RUN pip install -r requirements.txt
COPY . /root/home/git



EXPOSE 8888


RUN echo "/usr/lib/python3.x/site-packages" >> /usr/local/lib/python3.x/dist-packages/site-packages.pth
