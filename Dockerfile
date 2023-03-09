FROM stockbot/latest:1.2
WORKDIR /home/git


EXPOSE 8888

CMD ["git" ,"clone", "https://github.com/Ferbie10/Stock_BOT.git"]
CMD [ "cd", 'Stock_BOT' ]
RUN pip install -r requirements.txt