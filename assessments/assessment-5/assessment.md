# Assessment 5: Profit Curves, Web Scraping, NLP, Naive Bayes and Clustering

1. Complete this function that will give the number of jobs on indeed from a search result.

    ```python
    import requests
    from bs4 import BeautifulSoup

    def number_of_jobs(query):
        '''
        INPUT: string
        OUTPUT: int

        Return the number of jobs on the indeed.com for the search query.
        '''

        url = "http://www.indeed.com/jobs?q=%s" % query.replace(' ', '+')
        html = requests.get(url).text
        soup = BeautifulSoup(html, 'html.parser')
        search_count = soup.find('div', id='searchCount')
        return int(search_count.text.split('of ')[-1].replace(',', ''))
    ```

2. Say I am detecting fraud. If I identify a user as fraud, I will call them to confirm their identity. This costs $10. Catching fraud saves us $100. What does my cost benefit matrix look like?

    ```
    Cost-Benefit Matrix:
                    Actual
                    Y    N
                  -----------
               Y |  90 | -10 |
    Predicted     -----------
               N |   0 |   0 |
                  -----------
    ```

3. We've built two different models which result in the following two confusion matrices.

    ```
            Model 1:                          Model 2:
                    Actual                            Actual
                    Y    N                            Y    N
                  -----------                       -----------
               Y | 150 | 150 |                   Y | 200 | 500 |
    Predicted     -----------         Predicted     -----------
               N |  50 | 650 |                   N |   0 | 300 |
                  -----------                       ----------- 
    ```

    ```
    Model 1 profit: 150 * 90 + 150 * -10 = 12,000
    Model 2 profit: 200 * 90 + 500 * -10 = 13,000
    ```

    *Model 2 yields a higher profit, so we should use that one.*

4. Consider a corpus made up of the following four documents:

    ```
    Doc 1: Dogs like dogs more than cats.
    Doc 2: The dog chased the bicycle.
    Doc 3: The cat rode in the bicycle basket.
    Doc 4: I have a fast bicycle.
    ```

    We assume that we are lowercasing everything, stemming and removing stop words and punctuation. These are the features you should have:

    ```dog, like, cat, chase, bicycle, ride, basket, fast```

    * What is the term frequency vector for Document 1?

        ```
        tf = (2, 1, 1, 0, 0, 0, 0, 0)
        ```

    * What is the document frequency vector for all the words in the corpus?

        ```
        df = (3, 1, 2, 1, 3, 1, 1, 1)
        ```

5. Given the same documents, use python to build the tf-idf vectors and calculate the cosine similarity of each document with each other document. For your convenience, here's the data in a python list:
    
    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    documents = ["Dogs like dogs more than cats.",
                 "The dog chased the bicycle.",
                 "The cat rode in the bicycle basket.",
                 "I have a fast bicycle."]
    vect = TfidfVectorizer(stop_words='english')
    X = vect.fit_transform(documents)
    M = linear_kernel(X, X)
    print M
    ```

    Output:

    ```
    array([[ 1.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  1.        ,  0.14224755,  0.22133323],
           [ 0.        ,  0.14224755,  1.        ,  0.18604135],
           [ 0.        ,  0.22133323,  0.18604135,  1.        ]])
    ```

    Which two documents are the most similar?
    
    *Documents 2 and 4 are the most similar.*
    
    *Answers will vary depending if you used stopwords and/or stemming.*

6. What is wrong with this approach to building my feature matrix?

    We assume that `documents` is a list of the text of emails, each as a string. `y` is an array of 0, 1 labels of whether or not the email is spam.

    ```python
    vect = TfidfVectorizer(stop_words='english')
    X = vect.fit_transform(documents)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    print "Accuracy on test set:", log_reg.score(X_test, y_test)
    ```

    *You should not fit your `TfidfVectorizer` using both your train and test set. This is using your test set as part of your training! This is what the code should look like:*

    ```python
    docs_train, docs_test, y_train, y_test = train_test_split(documents, y)
    vect = TfidfVectorizer(stop_words='english')
    X_train = vect.fit_transform(docs_train)
    X_test = vect.transform(docs_test)
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    print "Accuracy on test set:", log_reg.score(X_test, y_test)
    ```

7. Why do we need to do Laplace Smoothing in Naive Bayes?

    *If we've never seen a word in a certain class before, we would say the probability is 0. This would essentially make all our probabilities 0 and we would not be able to compare them. Laplace Smoothing makes these probabilities small but nonzero.*

8. Suppose N = 100 represents a dense sample for a three dimensional feature space. To achieve same density for 8 inputs, how many points would we need?

    *Sampling density is proportional to N^(1/p) where N is the number of points and p is the number of dimensions in the input space.*

    *For 3 dim, we have 100^(1/3) = 4.641589.*
    
    *For 8 dim, we set X^(1/8) = 4.641589 to achieve the same density.*
    
    *X = 215443.5 so we need about 215443 points.*


9. The first step in the K-means algorithm involves randomly assigning data points to clusters, and as such, only finds local minimums. How do we typically deal with this?

  *Try multiple initializations and pick one with lowest minimum within cluster variation.  One could use K-means++ to get a "smarter" set of multiple initializations (see http://en.wikipedia.org/wiki/K-means%2B%2B)*

10. Describe the process of varying K in K-means. Contrast this with the process of varying K in the hierarchical clustering setting.

  *In K-means, you first choose a K. One way to find which K is optimal is to increase K until you reach an elbow point in the plot of within-cluster variation against the number of clusters. One could also use the gap statistic or compute the silhouette coefficient for each K.  The GAP statistic is explicit about how to choose K.  The silhouette coefficient is less explicit (see lecture notes)*  
  
  *In hierarchical clustering, unlike K-means, you don't have to choose a K a priori. You can cut your dendrogram at various  points, corresponding to different K, and then use any of the methods above to measure how good K is.  A single dendrogram provides choices of K=1 up to K=n*
