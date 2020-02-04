from Review import *
text = 'This is a bad movie.'
print(predict_sentiment(text, vocab, tokenizer, naive_bayes_classifier))

text = 'This is awesome and have great story line '
print(predict_sentiment(text, vocab, tokenizer, naive_bayes_classifier))

com = []
a = 0
while a!=1:
    comment = input("Enter Your Comment")
    pre = predict_sentiment(comment, vocab, tokenizer, naive_bayes_classifier)
    com.append(pre[0])
    a = int(input('Do You Want To Enter More Comment It Will Increase Accuracy (0: For More And 1: To exit)'))
print("The Rating Is:- {}".format(((np.count_nonzero(com))/(len(com))/0.2)))   
