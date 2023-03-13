
FROM tensorflow/tensorflow:latest-gpu
COPY . /root/home/git
WORKDIR /root/home/git

COPY requirements.txt /root/home/git
RUN apt-get update && apt-get upgrade
RUN apt-get install --no-install-recommends python3 python3-pip
RUN pip install -r requirements.txt
COPY . /root/home/git



EXPOSE 8888

CMD ["pwd"]
CMD [ "cd", 'Stock_BOT' ]
