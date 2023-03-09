FROM stockbot/latest:1.2
WORKDIR /home/git


EXPOSE 8888

CMD ["pwd"]
CMD [ "cd", 'Stock_BOT' ]
