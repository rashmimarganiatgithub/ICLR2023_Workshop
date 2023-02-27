# ICLR2023_KAGGLE
We detailed in this repository, how we did we obtained social media data, constructed NLP model for predicting arabic social media texts as positive/negative/neutral. The nature of our work is a binary one 

# Challenges Faced during the project:
    1.  Data collection challenges;
        1.  Access to the Twitter API have rate limits on the number of requests that can be made within a certain period of time. This             made it difficult to collect large amounts of data quickly.

        2.  Data volume: Twitter generates a large volume of data, with millions of tweets being posted every day. Storing this bigdata             it requires a lot of storage space and processing power.

        3.  Data quality: Not all tweets are relevant or useful for a particular analysis. Filtering out irrelevant or spam tweets canbe           challenging, especially when dealing with large volumes of data.

        4.  Privacy concerns: Some Twitter data can contain sensitive information, such as personal data or identifying information.                 Collecting and storing such data can raise privacy concerns. It was demanding to anonymize the data to mask off the real                 identity of the Twitter handles.

        5.  Bias: Twitter users are not be representative of the general African population, and the data collected from Twitter may be             biased towards certain groups or opinions--for example, it only covers the educated. In our next project, we planning to                inegrate more social media platforms.

        6.  Ethics: Collecting and using Twitter data raises ethical questions, such as the need for informed consent from users, the               protection of user privacy, and the potential for harm or misuse of the data.
     2. EDA challenges:
     
        1.  Data preprocessing: The herein Twitter data contains noisy and unstructured text, including hashtags, emojis, and URLs.                 Preprocessing these data requires domain knowledge and understanding of Twitter-specific features.
        2.  Data quality: Filtering the Twitter data that can contain spam, irrelevant tweets, and irrelevant user profiles was                     challenging and might have affected the quality of the data.

        3.  Text representation: Text data needs to be represented in a format that can be analyzed. This requires the use of natural               language processing (NLP) techniques such as tokenization, stemming, and lemmatization. Choosing the right text                         representation techniques and tools was challenging and required collections of domain knowledge.
        4.  Language-specific challenges: Arabic language has its own set of challenges for NLP, such as the presence of diacritics,                 variations in spelling, and complex grammatical structures. These challenges required specialized techniques for text                   representation and analysis.
      3. MLOP deployment:
         MLOps required careful planning, technical expertise, and domain knowledge in both NLP and software engineering. Successful              deployment requires a focus on scalability, reliability, and performance, as well as ongoing monitoring and optimization to              ensure that models are performing as expected.
