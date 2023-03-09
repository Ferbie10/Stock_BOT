FROM stockbot/latest:1.1
WORKDIR /home/git


EXPOSE 8888

CMD ["git clone https://github.com/Ferbie10/Stock_BOT.git"]