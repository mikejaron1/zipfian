def summarize(article, sent_detector, n):
    '''
    Choose top n the sentences based on max tf-idf score.
    '''
    sentences = sent_detector.tokenize(article.strip())
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences).toarray()
    # We are summing on axis=1 (total per row)
    total = np.sum(vectors, 1)
    lengths = np.array([len(sent) for sent in sentences])
    return get_top_values(total / lengths.astype(float), n, sentences)


def summarization(article, categories, n):
    '''
    Print top n sentences from the article.
    Print top n sentences from each category.
    '''
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    # summarize an article
    print summarize(article, sent_detector, n)
    print

    for cat in categories:
        newsgroup = fetch_20newsgroups(subset='train', categories=[cat])
        print cat
        # combine all articles into one string to summarize.
        print summarize("\n".join(newsgroup.data), sent_detector, n)
