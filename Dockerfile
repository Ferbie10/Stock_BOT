FROM stockbot/latest:1.2
COPY . /root/home/git
WORKDIR /root/home/git


RUN pip install -r requirements.txt


EXPOSE 8888

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root","--no-browser"]
RUN echo "/usr/lib/python3.x/site-packages" >> /usr/local/lib/python3.x/dist-packages/site-packages.pth
