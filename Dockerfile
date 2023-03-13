
FROM tensorflow/tensorflow:devel-gpu
COPY . /root/home/git
WORKDIR /root/home/git

COPY requirements.txt /root/home/git

RUN pip install -r requirements.txt
COPY . /root/home/git



EXPOSE 8888

CMD ["pwd"]
CMD [ "cd", 'Stock_BOT' ]
